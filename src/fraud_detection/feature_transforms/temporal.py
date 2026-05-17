from __future__ import annotations

import pandas as pd

from fraud_detection.feature_transforms.base import BaseFeature
from fraud_detection.feature_transforms.utils import safe_datetime


class TemporalFeature(BaseFeature):
    name = "temporal"
    output_columns = ("txn_hour", "txn_day_of_week", "txn_month", "is_weekend", "is_night_txn")

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del history_df
        transformed = df.copy()
        transaction_ts = safe_datetime(transformed["transactionDateTime"])
        transformed["event_ts"] = transaction_ts
        transformed["txn_hour"] = transaction_ts.dt.hour.fillna(-1).astype(int)
        transformed["txn_day_of_week"] = transaction_ts.dt.dayofweek.fillna(-1).astype(int)
        transformed["txn_month"] = transaction_ts.dt.month.fillna(0).astype(int)
        transformed["is_weekend"] = transformed["txn_day_of_week"].isin([5, 6]).astype(int)
        transformed["is_night_txn"] = transformed["txn_hour"].isin([0, 1, 2, 3, 4, 22, 23]).astype(
            int
        )
        return transformed
