from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd

from fraud_detection.config import load_config
from fraud_detection.feature_pipeline import FeaturePipeline
from fraud_detection.feature_store import DuckDBFeatureStore
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.logger import configure_logging
from fraud_detection.pipeline.explainability import (
    EXPLAINABILITY_METADATA_KEY,
    build_explainability_metadata,
)
from fraud_detection.pipeline.metrics import (
    collect_metrics,
    score_frame,
    write_metrics,
)
from fraud_detection.pipeline.modeling import build_cv_pipeline, build_stacked_model, save_model
from fraud_detection.schemas import SCHEMA_VERSION, read_transactions

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument(
        "--config",
        default="base",
        help="Config name from fraud_detection/configs or explicit .yml path.",
    )
    parser.add_argument(
        "--mode",
        choices=["individual", "stacked"],
        default=None,
        help="Training mode. Overrides model.training_mode.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for individual mode. Overrides model.components.champion.",
    )
    return parser.parse_args()


def _chronological_split(
    df: pd.DataFrame,
    test_window_days: int,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("transactionDateTime").reset_index(drop=True)
    timestamp = pd.to_datetime(ordered["transactionDateTime"], errors="coerce")
    cutoff = timestamp.max() - pd.Timedelta(days=test_window_days)
    train_df = ordered.loc[timestamp < cutoff].copy()
    test_df = ordered.loc[timestamp >= cutoff].copy()
    if train_df.empty or test_df.empty:
        split_at = max(1, min(len(ordered) - 1, int(len(ordered) * train_ratio)))
        train_df = ordered.iloc[:split_at].copy()
        test_df = ordered.iloc[split_at:].copy()
    return train_df, test_df


def _build_train_test_features(config: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_path = str(config.resolve_path(config.data["training_path"]))
    LOGGER.info("Reading training data from %s", training_path)
    raw_df = read_transactions(training_path)
    feature_pipeline = FeaturePipeline()
    valid_df = feature_pipeline.validate_raw(raw_df)

    train_raw, test_raw = _chronological_split(
        valid_df,
        test_window_days=int(config.model.get("test_window_days", 60)),
        train_ratio=float(config.model["train_ratio"]),
    )
    train_df = feature_pipeline.fit_transform(train_raw, with_class_weights=True)
    test_df = feature_pipeline.transform(test_raw, history_df=train_raw)
    train_df.attrs["entity_encoders"] = feature_pipeline.entity_encoders
    return train_df, test_df


def _metadata(
    config: Any,
    metrics: dict[str, float],
    train_df: pd.DataFrame,
    model_family: str,
    training_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": float(config.streaming["fraud_probability_threshold"]),
        "metrics": metrics,
        "training_rows": int(len(train_df)),
        "model_family": model_family,
        "training_mode": training_mode,
        "entity_encoders": train_df.attrs.get("entity_encoders", {}),
        EXPLAINABILITY_METADATA_KEY: build_explainability_metadata(
            train_df,
            seed=int(config.model["seed"]),
            feature_columns=FEATURE_COLUMNS,
        ),
    }


def _train_individual_model(config: Any, train_df: pd.DataFrame, model_name: str) -> Any:
    cv_pipeline = build_cv_pipeline(
        seed=int(config.model["seed"]),
        max_bins=int(config.model.get("max_bins", 0)),
        max_depths=[max(2, int(config.model["max_depth"]) - 2), int(config.model["max_depth"])],
        num_trees_list=[int(config.model["num_trees"]), int(config.model["num_trees"]) + 50],
        model_name=model_name,
    )
    cv_pipeline.fit(
        train_df[FEATURE_COLUMNS],
        train_df["isFraud"],
        classifier__sample_weight=train_df["class_weight"],
    )
    return cv_pipeline


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    model_dir = config.resolve_path(config.data["model_dir"])
    metrics_dir = config.resolve_path(config.data["metrics_dir"])
    threshold = float(config.streaming["fraud_probability_threshold"])

    train_df, test_df = _build_train_test_features(config)
    if config.raw.get("feature_store", {}).get("enabled", False):
        store = DuckDBFeatureStore(config.resolve_path(config.raw["feature_store"]["path"]))
        store.write_features(pd.concat([train_df, test_df], ignore_index=True))

    training_mode = str(args.mode or config.model.get("training_mode", "individual"))
    model_name = str(args.model or config.model.get("components", {}).get("champion", "xgboost"))
    if training_mode == "stacked":
        model = build_stacked_model(
            train_df,
            seed=int(config.model["seed"]),
            max_depth=int(config.model["max_depth"]),
            num_trees=int(config.model["num_trees"]),
        )
        model_family = "stacked_anomaly_boosting_meta"
        best_cv_score = None
    else:
        cv_pipeline = _train_individual_model(config, train_df, model_name)
        model = cv_pipeline.best_estimator_
        model_family = model_name
        best_cv_score = float(cv_pipeline.best_score_)

    predictions = score_frame(model, test_df, threshold=threshold, label_column="isFraud")
    metrics = collect_metrics(predictions, label_column="isFraud")
    if best_cv_score is not None:
        metrics["best_cv_auc_pr"] = best_cv_score

    LOGGER.info("Model metrics: %s", metrics)
    if metrics["auc_pr"] < float(config.model.get("min_auc_pr", 0.20)) or metrics["recall"] < float(
        config.model.get("min_recall", 0.65)
    ):
        LOGGER.error(
            "Model did not pass quality gate (PR-AUC %.3f, recall %.3f). Aborting save.",
            metrics["auc_pr"],
            metrics["recall"],
        )
        return

    artifact_path = save_model(
        model,
        model_dir,
        metadata=_metadata(config, metrics, train_df, model_family, training_mode),
    )
    write_metrics(metrics, metrics_dir / "latest_metrics.json")
    LOGGER.info("Saved model artifact to %s", artifact_path)


if __name__ == "__main__":
    main()
