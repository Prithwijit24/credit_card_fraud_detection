from __future__ import annotations

from pyspark.sql import SparkSession

from fraud_detection.config import AppConfig


def build_spark_session(config: AppConfig, app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(config.spark["master"])
        .config("spark.sql.shuffle.partitions", str(config.spark["shuffle_partitions"]))
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

