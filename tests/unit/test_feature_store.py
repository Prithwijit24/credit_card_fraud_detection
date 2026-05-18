from __future__ import annotations

import pandas as pd

from fraud_detection.feature_store import DuckDBFeatureStore


def test_feature_store_ignores_training_only_class_weight_on_append(tmp_path) -> None:
    store = DuckDBFeatureStore(tmp_path / "features.duckdb")
    train_features = pd.DataFrame(
        [
            {
                "accountNumber": "10001",
                "transactionDateTime": "2016-08-01T23:45:00",
                "isFraud": 1,
                "amount_to_limit_ratio": 0.02,
                "class_weight": 4.0,
            }
        ]
    )
    score_features = train_features.drop(columns=["class_weight"]).assign(
        accountNumber="10002",
        isFraud=0,
    )

    assert store.write_features(train_features) == 1
    assert store.append_features(score_features) == 1

    with store._connect() as connection:
        table_info = connection.execute("PRAGMA table_info('transaction_features')").fetchall()
        columns = [row[1] for row in table_info]
        rows = connection.execute("SELECT COUNT(*) FROM transaction_features").fetchone()[0]

    assert "class_weight" not in columns
    assert rows == 2
