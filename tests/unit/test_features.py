from __future__ import annotations

import pandas as pd

from fraud_detection.features import add_derived_features
from tests.fixtures import notebook_transaction


def test_add_derived_features() -> None:
    rows = [
        notebook_transaction(
            transactionDateTime="2016-08-01T23:45:00",
            cardCVV="123",
            enteredCVV="999",
            acqCountry="US",
            merchantCountryCode="CAN",
        )
    ]
    transformed = add_derived_features(pd.DataFrame(rows)).iloc[0]

    assert int(transformed["txn_hour"]) == 23
    assert int(transformed["is_night_txn"]) == 1
    assert int(transformed["cvv_mismatch"]) == 1
    assert int(transformed["cross_border"]) == 1
    assert transformed["account_age_days"] > 0
    assert transformed["utilization"] > 0


def test_history_features_use_only_prior_transactions() -> None:
    rows = [
        notebook_transaction(transactionDateTime="2016-08-01T10:00:00", transactionAmount=10.0),
        notebook_transaction(transactionDateTime="2016-08-01T11:00:00", transactionAmount=20.0),
    ]
    transformed = add_derived_features(pd.DataFrame(rows)).sort_values("transactionDateTime")

    assert transformed.iloc[0]["txn_count_24h"] == 0
    assert transformed.iloc[1]["txn_count_24h"] == 1
    assert transformed.iloc[1]["txn_amount_sum_24h"] == 10.0
