from __future__ import annotations

import json
from pathlib import Path

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_class_weights(df: DataFrame) -> DataFrame:
    counts = {row["is_fraud"]: row["count"] for row in df.groupBy("is_fraud").count().collect()}
    majority = max(counts.values())
    fraud_weight = float(majority / counts.get(1, majority))
    return df.withColumn(
        "class_weight",
        F.when(F.col("is_fraud") == 1, fraud_weight).otherwise(1.0),
    )


def collect_metrics(predictions: DataFrame) -> dict[str, float]:
    auc = BinaryClassificationEvaluator(
        labelCol="is_fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)
    pr_auc = BinaryClassificationEvaluator(
        labelCol="is_fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="f1",
    ).evaluate(predictions)
    return {"auc_roc": auc, "auc_pr": pr_auc, "f1": f1}


def write_metrics(metrics: dict[str, float], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
