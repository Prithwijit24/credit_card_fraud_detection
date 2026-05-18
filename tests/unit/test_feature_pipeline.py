from __future__ import annotations

import pandas as pd

from fraud_detection.feature_pipeline import FeaturePipeline
from fraud_detection.schemas import iter_transactions
from tests.fixtures import notebook_transaction


def test_feature_pipeline_fits_and_reuses_entity_encoders() -> None:
    train_df = pd.DataFrame(
        [
            notebook_transaction(merchantName="risky", isFraud=True),
            notebook_transaction(merchantName="normal", isFraud=False),
        ]
    )
    score_df = pd.DataFrame([notebook_transaction(merchantName="risky", isFraud=False)])

    pipeline = FeaturePipeline()
    train_features = pipeline.fit_transform(train_df, with_class_weights=True)
    score_features = pipeline.transform(score_df, history_df=train_df)

    assert "class_weight" in train_features
    assert pipeline.entity_encoders["global_fraud_rate"] == 0.5
    assert score_features.iloc[0]["merchantName_fraud_rate"] == 1.0


def test_iter_transactions_reads_csv_in_batches(tmp_path) -> None:
    path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            notebook_transaction(accountNumber=f"1000{idx}", isFraud=idx == 2)
            for idx in range(3)
        ]
    ).to_csv(path, index=False)

    batches = list(iter_transactions(path, batch_size=2))

    assert [len(batch) for batch in batches] == [2, 1]
    assert batches[0]["isFraud"].tolist() == [False, False]
    assert batches[1]["isFraud"].tolist() == [True]
