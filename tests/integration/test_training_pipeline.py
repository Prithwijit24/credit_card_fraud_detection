from __future__ import annotations

import pandas as pd

from fraud_detection.features import FEATURE_COLUMNS, add_derived_features
from fraud_detection.pipeline.metrics import compute_class_weights
from fraud_detection.pipeline.modeling import build_training_pipeline


def test_training_pipeline_fit_and_score() -> None:
    rows = [
        {
            "row_id": idx,
            "trans_date_trans_time": f"2019-01-01 0{idx % 9}:00:00",
            "cc_num": f"100{idx}",
            "merchant": "merchant_fraud" if idx % 2 else "merchant_normal",
            "category": "shopping_pos" if idx % 2 else "grocery_pos",
            "amt": float(20 + idx * 10),
            "first": "A",
            "last": "B",
            "gender": "F" if idx % 2 else "M",
            "street": "Street",
            "city": "City",
            "state": "CA" if idx % 2 else "NY",
            "zip": "10001",
            "lat": 34.05,
            "long": -118.24,
            "city_pop": 1000 + idx * 10,
            "job": "Engineer" if idx % 2 else "Teacher",
            "dob": "1985-06-15",
            "trans_num": f"txn-{idx}",
            "unix_time": 1546300800 + idx,
            "merch_lat": 34.15 + (idx * 0.01),
            "merch_long": -118.34 - (idx * 0.01),
            "is_fraud": 1 if idx % 3 == 0 else 0,
        }
        for idx in range(1, 13)
    ]

    df = compute_class_weights(add_derived_features(pd.DataFrame(rows)))
    pipeline = build_training_pipeline(seed=42, max_bins=32, max_depth=4, num_trees=10)
    model = pipeline.fit(
        df[FEATURE_COLUMNS],
        df["is_fraud"],
        classifier__sample_weight=df["class_weight"],
    )
    probabilities = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    predictions = model.predict(df[FEATURE_COLUMNS])

    assert len(probabilities) == len(rows)
    assert len(predictions) == len(rows)
