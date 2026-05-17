from __future__ import annotations

import pandas as pd

from fraud_detection.features import FEATURE_COLUMNS, add_derived_features
from fraud_detection.pipeline.metrics import compute_class_weights
from fraud_detection.pipeline.modeling import build_training_pipeline
from tests.fixtures import notebook_transaction


def test_training_pipeline_fit_and_score() -> None:
    rows = [
        notebook_transaction(
            accountNumber=f"100{idx % 3}",
            customerId=f"cust-{idx % 3}",
            transactionDateTime=f"2016-08-{idx:02d}T0{idx % 9}:00:00",
            merchantName="merchant_fraud" if idx % 3 == 0 else "merchant_normal",
            merchantCategoryCode="online_retail" if idx % 3 == 0 else "grocery",
            transactionAmount=float(20 + idx * 10),
            cardCVV="123",
            enteredCVV="999" if idx % 3 == 0 else "123",
            isFraud=idx % 3 == 0,
        )
        for idx in range(1, 13)
    ]

    df = compute_class_weights(add_derived_features(pd.DataFrame(rows)))
    pipeline = build_training_pipeline(seed=42, max_bins=32, max_depth=4, num_trees=10)
    model = pipeline.fit(
        df[FEATURE_COLUMNS],
        df["isFraud"],
        classifier__sample_weight=df["class_weight"],
    )
    probabilities = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    predictions = model.predict(df[FEATURE_COLUMNS])

    assert len(probabilities) == len(rows)
    assert len(predictions) == len(rows)
