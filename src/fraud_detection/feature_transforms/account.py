from __future__ import annotations

import pandas as pd

from fraud_detection.feature_transforms.base import BaseFeature
from fraud_detection.feature_transforms.utils import safe_datetime


class AccountAgeFeature(BaseFeature):
    name = "account_age"
    output_columns = ("account_age_days", "days_since_address_change", "months_to_expiry")

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del history_df
        transformed = df.copy()
        transaction_ts = safe_datetime(transformed["transactionDateTime"])
        account_open_ts = safe_datetime(transformed["accountOpenDate"])
        address_change_ts = safe_datetime(transformed["dateOfLastAddressChange"])
        expiry_ts = safe_datetime(transformed["currentExpDate"])

        transformed["account_age_days"] = (
            (transaction_ts - account_open_ts).dt.days.fillna(0).clip(lower=0)
        )
        transformed["days_since_address_change"] = (
            (transaction_ts - address_change_ts).dt.days.fillna(0).clip(lower=0)
        )
        transformed["months_to_expiry"] = ((expiry_ts - transaction_ts).dt.days // 30).fillna(0)
        return transformed
