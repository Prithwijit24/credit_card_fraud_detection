from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_class_weights(df: pd.DataFrame, label_column: str = "isFraud") -> pd.DataFrame:
    weighted = df.copy()
    weighted[label_column] = weighted[label_column].astype(int)
    counts = weighted[label_column].value_counts(dropna=False)
    majority = counts.max()
    fraud_count = counts.get(1, majority)
    fraud_weight = float(majority / fraud_count) if fraud_count else 1.0
    weighted["class_weight"] = weighted[label_column].map({1: fraud_weight}).fillna(1.0)
    return weighted


def score_frame(
    model: Any,
    features: pd.DataFrame,
    threshold: float,
    label_column: str = "isFraud",
) -> pd.DataFrame:
    scored = features.copy()
    probabilities = model.predict_proba(features)[:, 1]
    scored["fraud_probability"] = probabilities
    scored["prediction"] = (probabilities >= threshold).astype(int)
    scored["risk_band"] = "normal"
    scored.loc[scored["fraud_probability"] >= 0.5, "risk_band"] = "elevated"
    scored.loc[scored["fraud_probability"] >= threshold, "risk_band"] = "critical"
    if label_column not in scored:
        scored[label_column] = 0
    return scored


def collect_metrics(predictions: pd.DataFrame, label_column: str = "isFraud") -> dict[str, float]:
    y_true = predictions[label_column].astype(int)
    y_score = predictions["fraud_probability"].astype(float)
    y_pred = predictions["prediction"].astype(int)
    has_two_classes = y_true.nunique() == 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)) if has_two_classes else 0.0,
        "auc_pr": float(average_precision_score(y_true, y_score)) if has_two_classes else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
    }


def write_metrics(metrics: dict[str, float], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
