from __future__ import annotations

import pandas as pd

from fraud_detection.feature_transforms.account import AccountAgeFeature
from fraud_detection.feature_transforms.amount import AmountRatioFeature
from fraud_detection.feature_transforms.base import BaseFeature
from fraud_detection.feature_transforms.categorical import FeatureTypeNormalizer
from fraud_detection.feature_transforms.history import TransactionHistoryFeature
from fraud_detection.feature_transforms.risk import TransactionRiskFeature
from fraud_detection.feature_transforms.temporal import TemporalFeature


class SummaryFeaturePipeline(BaseFeature):
    name = "summary_feature_pipeline"

    def __init__(self, steps: list[BaseFeature] | None = None) -> None:
        self.steps = steps or [
            TransactionHistoryFeature(),
            TemporalFeature(),
            AccountAgeFeature(),
            AmountRatioFeature(),
            TransactionRiskFeature(),
            FeatureTypeNormalizer(),
        ]

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        transformed = df.copy()
        for step in self.steps:
            transformed = step.transform(transformed, history_df=history_df)
        return transformed
