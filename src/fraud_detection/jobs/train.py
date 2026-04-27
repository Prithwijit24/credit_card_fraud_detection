from __future__ import annotations

import argparse
import logging

from sklearn.model_selection import train_test_split

from fraud_detection.config import load_config
from fraud_detection.features import FEATURE_COLUMNS, add_derived_features, sanitize_input_columns
from fraud_detection.logger import configure_logging
from fraud_detection.pipeline.metrics import (
    collect_metrics,
    compute_class_weights,
    score_frame,
    write_metrics,
)
from fraud_detection.pipeline.modeling import build_cv_pipeline, save_model
from fraud_detection.pipeline.validation import validate_transactions
from fraud_detection.schemas import read_transactions_csv

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    training_path = str(config.resolve_path(config.data["training_path"]))
    model_dir = config.resolve_path(config.data["model_dir"])
    metrics_dir = config.resolve_path(config.data["metrics_dir"])
    threshold = float(config.streaming["fraud_probability_threshold"])

    LOGGER.info("Reading training data from %s", training_path)
    raw_df = read_transactions_csv(training_path)
    feature_df = add_derived_features(validate_transactions(sanitize_input_columns(raw_df)))
    feature_df["is_fraud"] = feature_df["is_fraud"].astype(int)
    feature_df = compute_class_weights(feature_df)

    stratify = feature_df["is_fraud"] if feature_df["is_fraud"].nunique() == 2 else None
    train_df, test_df = train_test_split(
        feature_df,
        train_size=float(config.model["train_ratio"]),
        random_state=int(config.model["seed"]),
        stratify=stratify,
    )

    cv_pipeline = build_cv_pipeline(
        seed=int(config.model["seed"]),
        max_bins=int(config.model.get("max_bins", 0)),
        max_depths=[max(2, int(config.model["max_depth"]) - 2), int(config.model["max_depth"])],
        num_trees_list=[int(config.model["num_trees"]), int(config.model["num_trees"]) + 20],
    )
    cv_pipeline.fit(
        train_df[FEATURE_COLUMNS],
        train_df["is_fraud"],
        classifier__sample_weight=train_df["class_weight"],
    )
    model = cv_pipeline.best_estimator_
    predictions = score_frame(model, test_df, threshold=threshold)
    metrics = collect_metrics(predictions)
    metrics["best_cv_auc_pr"] = float(cv_pipeline.best_score_)

    LOGGER.info("Model metrics: %s", metrics)
    if metrics["auc_pr"] < 0.20 or metrics["recall"] < 0.65:
        LOGGER.error(
            "Model did not pass quality gate (PR-AUC %.3f, recall %.3f). Aborting save.",
            metrics["auc_pr"],
            metrics["recall"],
        )
        return

    artifact_path = save_model(model, model_dir)
    write_metrics(metrics, metrics_dir / "latest_metrics.json")
    LOGGER.info("Saved model artifact to %s", artifact_path)


if __name__ == "__main__":
    main()
