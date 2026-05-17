from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.feature_transforms.base import BaseFeature
from fraud_detection.feature_transforms.utils import bool_as_int, safe_numeric


class TransactionRiskFeature(BaseFeature):
    name = "transaction_risk"
    output_columns = (
        "cvv_mismatch",
        "expiry_key_mismatch",
        "card_present_flag",
        "cross_border",
        "pos_risk_tier",
    )

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del history_df
        transformed = df.copy()
        transformed["cvv_mismatch"] = (
            transformed["cardCVV"].astype("string") != transformed["enteredCVV"].astype("string")
        ).astype(int)
        transformed["expiry_key_mismatch"] = 1 - bool_as_int(
            transformed["expirationDateKeyInMatch"]
        )
        transformed["card_present_flag"] = bool_as_int(transformed["cardPresent"])
        transformed["cross_border"] = (
            transformed["acqCountry"].astype("string").str.upper().fillna("UNKNOWN")
            != transformed["merchantCountryCode"].astype("string").str.upper().fillna("UNKNOWN")
        ).astype(int)

        pos_entry = safe_numeric(transformed["posEntryMode"]).fillna(-1)
        pos_condition = safe_numeric(transformed["posConditionCode"]).fillna(-1)
        transformed["pos_risk_tier"] = np.select(
            [
                transformed["card_present_flag"].eq(0),
                pos_entry.isin([81, 90]) | pos_condition.isin([8, 59, 99]),
            ],
            [2, 3],
            default=1,
        )
        return transformed
