from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.feature_transforms.base import BaseFeature
from fraud_detection.feature_transforms.utils import safe_numeric


class AmountRatioFeature(BaseFeature):
    name = "amount_ratio"
    output_columns = ("utilization", "amount_to_limit_ratio")

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del history_df
        transformed = df.copy()
        transformed["creditLimit"] = safe_numeric(transformed["creditLimit"]).fillna(0.0)
        transformed["availableMoney"] = safe_numeric(transformed["availableMoney"]).fillna(0.0)
        transformed["transactionAmount"] = safe_numeric(
            transformed["transactionAmount"]
        ).fillna(0.0)
        transformed["currentBalance"] = safe_numeric(transformed["currentBalance"]).fillna(0.0)
        credit_limit = transformed["creditLimit"].replace(0, np.nan)
        used_credit = (transformed["creditLimit"] - transformed["availableMoney"]).clip(lower=0.0)
        transformed["utilization"] = (
            (used_credit / credit_limit).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        transformed["amount_to_limit_ratio"] = (
            transformed["transactionAmount"] / credit_limit
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return transformed
