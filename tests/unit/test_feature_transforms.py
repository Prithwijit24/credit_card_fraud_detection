from __future__ import annotations

import pandas as pd

from fraud_detection.feature_transforms import (
    AccountAgeFeature,
    AmountRatioFeature,
    BaseFeature,
    SummaryFeaturePipeline,
    TemporalFeature,
    TransactionRiskFeature,
)
from tests.fixtures import notebook_transaction


def test_summary_feature_pipeline_composes_child_features() -> None:
    rows = [
        notebook_transaction(
            transactionDateTime="2016-08-01T23:45:00",
            cardCVV="123",
            enteredCVV="999",
            acqCountry="US",
            merchantCountryCode="CAN",
        )
    ]

    transformed = SummaryFeaturePipeline().transform(pd.DataFrame(rows)).iloc[0]

    assert int(transformed["txn_hour"]) == 23
    assert int(transformed["cvv_mismatch"]) == 1
    assert int(transformed["cross_border"]) == 1
    assert transformed["utilization"] > 0


def test_feature_children_share_base_class() -> None:
    children = [
        TemporalFeature(),
        AccountAgeFeature(),
        AmountRatioFeature(),
        TransactionRiskFeature(),
    ]

    assert all(isinstance(child, BaseFeature) for child in children)
