from __future__ import annotations

import pandas as pd

from fraud_detection.feature_pipeline import FeaturePipeline
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.pipeline.modeling import build_stacked_model
from tests.fixtures import notebook_transaction


def test_stacked_model_trains_in_logical_order() -> None:
    rows = [
        notebook_transaction(
            accountNumber=f"100{idx % 3}",
            customerId=f"cust-{idx % 3}",
            transactionDateTime=f"2016-08-{idx:02d}T0{idx % 9}:00:00",
            merchantName="merchant_fraud" if idx % 3 == 0 else "merchant_normal",
            transactionAmount=float(20 + idx * 10),
            isFraud=idx % 3 == 0,
        )
        for idx in range(1, 13)
    ]
    feature_pipeline = FeaturePipeline()
    features = feature_pipeline.fit_transform(pd.DataFrame(rows), with_class_weights=True)

    model = build_stacked_model(features, seed=42, max_depth=3, num_trees=5)
    probabilities = model.predict_proba(features[FEATURE_COLUMNS])

    assert model.anomaly_score_columns == [
        "isolation_forest_score",
        "one_class_svm_score",
        "autoencoder_score",
        "tabtransformer_score",
    ]
    assert model.booster_score_columns == ["xgboost_score", "lightgbm_score"]
    assert probabilities.shape == (len(rows), 2)
