from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.config import load_config
from fraud_detection.feature_pipeline import FeaturePipeline
from fraud_detection.feature_store import DuckDBFeatureStore
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.logger import configure_logging
from fraud_detection.memory import (
    DEFAULT_MEMORY_LIMIT_GB,
    assert_memory_budget,
    bytes_from_gb,
    set_process_memory_limit,
)
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
from fraud_detection.schemas import SCHEMA_VERSION, iter_transactions, read_transactions

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rows per ingestion/feature/scoring batch. Defaults to model.batch_size or 25000.",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=DEFAULT_MEMORY_LIMIT_GB,
        help="Hard process memory ceiling in GiB. Default: 8.",
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


def _batch_size(config: Any, explicit_batch_size: int | None) -> int:
    value = (
        explicit_batch_size
        or config.model.get("batch_size")
        or config.data.get("batch_size", 25_000)
    )
    return max(1, int(value))


def _append_recent_history(
    history_df: pd.DataFrame,
    current_df: pd.DataFrame,
    retention_days: int = 30,
) -> pd.DataFrame:
    if current_df.empty:
        return history_df
    combined = (
        pd.concat([history_df, current_df], ignore_index=True)
        if not history_df.empty
        else current_df
    )
    timestamps = pd.to_datetime(combined["transactionDateTime"], errors="coerce")
    current_max = pd.to_datetime(current_df["transactionDateTime"], errors="coerce").max()
    if pd.isna(current_max):
        return combined.tail(len(current_df) + len(history_df)).copy()
    cutoff = current_max - pd.Timedelta(days=retention_days)
    return combined.loc[timestamps >= cutoff].copy()


def _discover_training_window(
    training_path: str,
    batch_size: int,
    test_window_days: int,
    train_ratio: float,
    memory_limit_bytes: int,
) -> tuple[pd.Timestamp | None, int]:
    max_timestamp: pd.Timestamp | None = None
    total_rows = 0
    for batch in iter_transactions(training_path, batch_size=batch_size):
        timestamp = pd.to_datetime(batch["transactionDateTime"], errors="coerce")
        batch_max = timestamp.max()
        if not pd.isna(batch_max):
            max_timestamp = batch_max if max_timestamp is None else max(max_timestamp, batch_max)
        total_rows += len(batch)
        assert_memory_budget(memory_limit_bytes, "training-window discovery")
    if max_timestamp is None:
        return None, max(1, int(total_rows * train_ratio))
    cutoff = max_timestamp - pd.Timedelta(days=test_window_days)
    train_rows = 0
    test_rows = 0
    for batch in iter_transactions(training_path, batch_size=batch_size):
        timestamp = pd.to_datetime(batch["transactionDateTime"], errors="coerce")
        train_rows += int((timestamp < cutoff).sum())
        test_rows += int((timestamp >= cutoff).sum())
        assert_memory_budget(memory_limit_bytes, "training-window sizing")
    if train_rows and test_rows:
        return cutoff, train_rows
    return None, max(1, min(total_rows - 1, int(total_rows * train_ratio)))


def _partition_batch(
    batch: pd.DataFrame,
    cutoff: pd.Timestamp | None,
    split_at: int,
    seen_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cutoff is not None:
        timestamp = pd.to_datetime(batch["transactionDateTime"], errors="coerce")
        return batch.loc[timestamp < cutoff].copy(), batch.loc[timestamp >= cutoff].copy()
    next_seen = seen_rows + len(batch)
    if next_seen <= split_at:
        return batch.copy(), batch.iloc[0:0].copy()
    if seen_rows >= split_at:
        return batch.iloc[0:0].copy(), batch.copy()
    local_split = split_at - seen_rows
    return batch.iloc[:local_split].copy(), batch.iloc[local_split:].copy()


def _fit_entity_encoders_from_batches(
    training_path: str,
    batch_size: int,
    cutoff: pd.Timestamp | None,
    split_at: int,
    memory_limit_bytes: int,
) -> tuple[dict[str, dict[str, float] | float], dict[int, int]]:
    from fraud_detection.feature_transforms.columns import ENTITY_ENCODING_SOURCE_COLUMNS

    label_sums: dict[str, dict[str, float]] = {
        column: {} for column in ENTITY_ENCODING_SOURCE_COLUMNS
    }
    label_counts: dict[str, dict[str, int]] = {
        column: {} for column in ENTITY_ENCODING_SOURCE_COLUMNS
    }
    class_counts = {0: 0, 1: 0}
    label_total = 0.0
    row_total = 0
    seen_rows = 0

    for batch in iter_transactions(training_path, batch_size=batch_size):
        train_raw, _ = _partition_batch(batch, cutoff, split_at, seen_rows)
        seen_rows += len(batch)
        if train_raw.empty:
            continue
        labels = train_raw["isFraud"].fillna(False).astype(bool).astype(int)
        class_counts[0] += int((labels == 0).sum())
        class_counts[1] += int((labels == 1).sum())
        label_total += float(labels.sum())
        row_total += len(labels)
        labeled = train_raw.assign(_label=labels)
        for column in ENTITY_ENCODING_SOURCE_COLUMNS:
            grouped = labeled.groupby(column, dropna=False)["_label"].agg(["sum", "count"])
            for key, values in grouped.iterrows():
                encoded_key = str(key)
                label_sums[column][encoded_key] = label_sums[column].get(encoded_key, 0.0) + float(
                    values["sum"]
                )
                label_counts[column][encoded_key] = label_counts[column].get(encoded_key, 0) + int(
                    values["count"]
                )
        assert_memory_budget(memory_limit_bytes, "entity encoder fitting")

    encoders: dict[str, dict[str, float] | float] = {
        "global_fraud_rate": float(label_total / row_total) if row_total else 0.0
    }
    for column in ENTITY_ENCODING_SOURCE_COLUMNS:
        encoders[column] = {
            key: float(label_sums[column][key] / count)
            for key, count in label_counts[column].items()
            if count
        }
    return encoders, class_counts


def _apply_class_weights(features: pd.DataFrame, class_counts: dict[int, int]) -> pd.DataFrame:
    weighted = features.copy()
    majority = max(class_counts.values()) if class_counts else 0
    fraud_count = class_counts.get(1, majority)
    fraud_weight = float(majority / fraud_count) if fraud_count else 1.0
    weighted["class_weight"] = weighted["isFraud"].astype(int).map({1: fraud_weight}).fillna(1.0)
    return weighted


def _build_train_test_features_batched(
    config: Any,
    batch_size: int,
    memory_limit_bytes: int,
) -> tuple[pd.DataFrame, list[Path], Path]:
    training_path = str(config.resolve_path(config.data["training_path"]))
    LOGGER.info("Reading training data from %s in batches of %s", training_path, batch_size)
    cutoff, split_at = _discover_training_window(
        training_path,
        batch_size=batch_size,
        test_window_days=int(config.model.get("test_window_days", 60)),
        train_ratio=float(config.model["train_ratio"]),
        memory_limit_bytes=memory_limit_bytes,
    )
    encoders, class_counts = _fit_entity_encoders_from_batches(
        training_path,
        batch_size=batch_size,
        cutoff=cutoff,
        split_at=split_at,
        memory_limit_bytes=memory_limit_bytes,
    )

    feature_pipeline = FeaturePipeline(entity_encoders=encoders)
    train_batches: list[pd.DataFrame] = []
    test_batch_paths: list[Path] = []
    test_batch_dir = Path(tempfile.mkdtemp(prefix="fraud-test-features-"))
    history_df = pd.DataFrame()
    seen_rows = 0
    for batch in iter_transactions(training_path, batch_size=batch_size):
        train_raw, test_raw = _partition_batch(batch, cutoff, split_at, seen_rows)
        seen_rows += len(batch)
        if not train_raw.empty:
            train_raw = train_raw.sort_values("transactionDateTime")
            train_features = feature_pipeline.transform(train_raw, history_df=history_df)
            train_features = _apply_class_weights(train_features, class_counts)
            train_batches.append(train_features)
            history_df = _append_recent_history(history_df, train_raw)
        if not test_raw.empty:
            test_raw = test_raw.sort_values("transactionDateTime")
            test_features = feature_pipeline.transform(test_raw, history_df=history_df)
            test_path = test_batch_dir / f"test-features-{len(test_batch_paths):06d}.csv"
            test_features.to_csv(test_path, index=False)
            test_batch_paths.append(test_path)
        assert_memory_budget(memory_limit_bytes, "batched feature generation")

    if not train_batches or not test_batch_paths:
        shutil.rmtree(test_batch_dir, ignore_errors=True)
        raise ValueError("Training split produced empty train or test features.")
    train_df = pd.concat(train_batches, ignore_index=True)
    train_df.attrs["entity_encoders"] = encoders
    assert_memory_budget(memory_limit_bytes, "training feature assembly")
    return train_df, test_batch_paths, test_batch_dir


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


def _score_test_batches(
    model: Any,
    test_batch_paths: list[Path],
    threshold: float,
    memory_limit_bytes: int,
) -> pd.DataFrame:
    metric_frames = []
    for path in test_batch_paths:
        features = pd.read_csv(path)
        scored = score_frame(model, features, threshold=threshold, label_column="isFraud")
        metric_frames.append(scored[["isFraud", "fraud_probability", "prediction"]].copy())
        assert_memory_budget(memory_limit_bytes, "batched test scoring")
    return pd.concat(metric_frames, ignore_index=True)


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
    memory_limit_bytes = bytes_from_gb(args.memory_limit_gb)
    set_process_memory_limit(memory_limit_bytes)
    np.seterr(all="ignore")
    config = load_config(args.config)

    model_dir = config.resolve_path(config.data["model_dir"])
    metrics_dir = config.resolve_path(config.data["metrics_dir"])
    threshold = float(config.streaming["fraud_probability_threshold"])

    batch_size = _batch_size(config, args.batch_size)
    train_df, test_batch_paths, test_batch_dir = _build_train_test_features_batched(
        config,
        batch_size=batch_size,
        memory_limit_bytes=memory_limit_bytes,
    )
    try:
        if config.raw.get("feature_store", {}).get("enabled", False):
            store = DuckDBFeatureStore(config.resolve_path(config.raw["feature_store"]["path"]))
            store.write_features(train_df)
            for path in test_batch_paths:
                store.append_features(pd.read_csv(path))
                assert_memory_budget(memory_limit_bytes, "feature-store write")

        training_mode = str(args.mode or config.model.get("training_mode", "individual"))
        model_name = str(
            args.model or config.model.get("components", {}).get("champion", "xgboost")
        )
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

        predictions = _score_test_batches(
            model,
            test_batch_paths,
            threshold=threshold,
            memory_limit_bytes=memory_limit_bytes,
        )
        metrics = collect_metrics(predictions, label_column="isFraud")
        if best_cv_score is not None:
            metrics["best_cv_auc_pr"] = best_cv_score

        LOGGER.info("Model metrics: %s", metrics)
        if metrics["auc_pr"] < float(config.model.get("min_auc_pr", 0.20)) or metrics[
            "recall"
        ] < float(config.model.get("min_recall", 0.65)):
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
    finally:
        shutil.rmtree(test_batch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
