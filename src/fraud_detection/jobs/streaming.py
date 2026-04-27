from __future__ import annotations

import argparse
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery

from fraud_detection.config import AppConfig, load_config
from fraud_detection.features import add_derived_features, sanitize_input_columns
from fraud_detection.pipeline.validation import validate_transactions
from fraud_detection.logger import configure_logging
from fraud_detection.pipeline.modeling import load_model
from fraud_detection.schemas import TRANSACTION_SCHEMA
from fraud_detection.spark import build_spark_session

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming fraud scoring.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def build_source_stream(config: AppConfig, spark_session: SparkSession) -> DataFrame:
    source = config.streaming["source"]
    if source == "kafka":
        return (
            spark_session.readStream.format("kafka")
            .option("kafka.bootstrap.servers", config.streaming["kafka_bootstrap_servers"])
            .option("subscribe", config.streaming["kafka_topic"])
            .option("startingOffsets", config.streaming["starting_offsets"])
            .option("maxOffsetsPerTrigger", config.streaming["max_offsets_per_trigger"])
            .load()
            .selectExpr("CAST(value AS STRING) AS payload")
            .select(F.from_json("payload", TRANSACTION_SCHEMA).alias("data"))
            .select("data.*")
        )

    landing_dir = str(config.resolve_path(config.data["landing_dir"]))
    return (
        spark_session.readStream.schema(TRANSACTION_SCHEMA)
        .option("header", True)
        .csv(landing_dir)
    )


def write_outputs(scored_df: DataFrame, config: AppConfig) -> list[StreamingQuery]:
    threshold = float(config.streaming["fraud_probability_threshold"])
    checkpoint_root = config.resolve_path(config.data["checkpoint_dir"])
    scored_dir = str(config.resolve_path(config.data["scored_dir"]))
    alerts_dir = str(config.resolve_path(config.data["alerts_dir"]))

    enriched = scored_df.withColumn("fraud_probability", F.col("probability")[1]).withColumn(
        "risk_band",
        F.when(F.col("probability")[1] >= threshold, F.lit("critical"))
        .when(F.col("probability")[1] >= 0.5, F.lit("elevated"))
        .otherwise(F.lit("normal")),
    )

    scored_query = (
        enriched.writeStream.outputMode(config.streaming["output_mode"])
        .format("parquet")
        .option("path", scored_dir)
        .option("checkpointLocation", str(checkpoint_root / "scored"))
        .trigger(processingTime=config.streaming["trigger_interval"])
        .start()
    )

    alerts_query = (
        enriched.filter(F.col("fraud_probability") >= threshold)
        .writeStream.outputMode(config.streaming["output_mode"])
        .format("json")
        .option("path", alerts_dir)
        .option("checkpointLocation", str(checkpoint_root / "alerts"))
        .trigger(processingTime=config.streaming["trigger_interval"])
        .start()
    )

    monitor_query = (
        enriched.withWatermark("event_ts", config.streaming["watermark_delay"])
        .groupBy(
            F.window(
                "event_ts",
                config.streaming["monitor_window"],
                config.streaming["monitor_slide"],
            ),
            "state",
            "risk_band",
        )
        .agg(
            F.count("*").alias("txn_count"),
            F.avg("fraud_probability").alias("avg_fraud_probability"),
        )
        .writeStream.outputMode("update")
        .format("console")
        .option("truncate", False)
        .option("checkpointLocation", str(checkpoint_root / "monitor"))
        .trigger(processingTime=config.streaming["trigger_interval"])
        .start()
    )

    return [scored_query, alerts_query, monitor_query]


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    spark = build_spark_session(config, config.spark["app_name_stream"])
    model_path = str(config.resolve_path(config.data["model_dir"]))

    LOGGER.info("Loading model from %s", model_path)
    model = load_model(model_path)
    source_stream = build_source_stream(config, spark)
    feature_stream = add_derived_features(validate_transactions(sanitize_input_columns(source_stream)))
    scored = model.transform(feature_stream)

    queries = write_outputs(scored, config)
    LOGGER.info("Started %s streaming queries", len(queries))
    for query in queries:
        query.awaitTermination()


if __name__ == "__main__":
    main()
