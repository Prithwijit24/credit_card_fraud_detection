from __future__ import annotations

import argparse
import logging

from pyspark.sql import functions as F

from fraud_detection.config import load_config
from fraud_detection.features import add_derived_features, sanitize_input_columns
from fraud_detection.logger import configure_logging
from fraud_detection.pipeline.metrics import collect_metrics, compute_class_weights, write_metrics
from fraud_detection.pipeline.modeling import build_cv_pipeline
from fraud_detection.pipeline.validation import validate_transactions
from fraud_detection.schemas import TRANSACTION_SCHEMA
from fraud_detection.spark import build_spark_session

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    spark = build_spark_session(config, config.spark["app_name_train"])

    training_path = str(config.resolve_path(config.data["training_path"]))
    model_dir = config.resolve_path(config.data["model_dir"])
    metrics_dir = config.resolve_path(config.data["metrics_dir"])

    LOGGER.info("Reading training data from %s", training_path)
    raw_df = (
        spark.read.option("header", True)
        .schema(TRANSACTION_SCHEMA)
        .csv(training_path)
    )
    feature_df = add_derived_features(validate_transactions(sanitize_input_columns(raw_df)))
    feature_df = compute_class_weights(feature_df).withColumn(
        "is_fraud", F.col("is_fraud").cast("double")
    )

    train_df, test_df = feature_df.randomSplit(
        [config.model["train_ratio"], 1 - config.model["train_ratio"]],
        seed=config.model["seed"],
    )
    cv_pipeline = build_cv_pipeline(
        seed=config.model["seed"],
        max_bins=config.model["max_bins"],
        max_depths=[max(2, config.model["max_depth"] - 2), config.model["max_depth"]],
        num_trees_list=[config.model["num_trees"], config.model["num_trees"] + 20],
    )
    cv_model = cv_pipeline.fit(train_df)
    model = cv_model.bestModel
    predictions = model.transform(test_df).cache()
    metrics = collect_metrics(predictions)

    LOGGER.info("Model metrics: %s", metrics)
    if metrics["auc_roc"] < 0.85:
        LOGGER.error("Model did not pass quality gate (AUC-ROC %.3f < 0.85). Aborting save.", metrics["auc_roc"])
        spark.stop()
        return

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        LOGGER.info("Overwriting existing model at %s", model_dir)
    model.write().overwrite().save(str(model_dir))
    write_metrics(metrics, metrics_dir / "latest_metrics.json")

    spark.stop()


if __name__ == "__main__":
    main()
