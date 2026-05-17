from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.feature_transforms.base import BaseFeature
from fraud_detection.feature_transforms.utils import safe_datetime, safe_numeric


class TransactionHistoryFeature(BaseFeature):
    name = "transaction_history"
    output_columns = (
        "event_ts",
        "txn_count_24h",
        "txn_count_48h",
        "txn_count_7d",
        "txn_count_30d",
        "txn_amount_sum_24h",
        "txn_amount_sum_7d",
        "txn_amount_mean_7d",
        "seconds_since_last_txn",
        "is_new_merchant",
    )

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        current = df.copy()
        current["_feature_row_id"] = np.arange(len(current))
        current["_is_current_row"] = True

        frames = [current]
        if history_df is not None and not history_df.empty:
            history = history_df.copy()
            history["_feature_row_id"] = -1
            history["_is_current_row"] = False
            frames.insert(0, history)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["event_ts"] = safe_datetime(combined["transactionDateTime"])
        combined["transactionAmount"] = safe_numeric(combined["transactionAmount"]).fillna(0.0)
        combined = combined.sort_values(
            ["accountNumber", "event_ts", "_is_current_row"]
        ).reset_index(drop=True)
        combined = combined.set_index("event_ts", drop=False)

        grouped = combined.groupby("accountNumber", dropna=False)
        for window, suffix in [("24h", "24h"), ("48h", "48h"), ("7d", "7d"), ("30d", "30d")]:
            combined[f"txn_count_{suffix}"] = (
                grouped["transactionAmount"]
                .rolling(window, closed="left")
                .count()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

        for window, suffix in [("24h", "24h"), ("7d", "7d")]:
            combined[f"txn_amount_sum_{suffix}"] = (
                grouped["transactionAmount"]
                .rolling(window, closed="left")
                .sum()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

        combined["txn_amount_mean_7d"] = (
            grouped["transactionAmount"]
            .rolling("7d", closed="left")
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        combined["seconds_since_last_txn"] = (
            grouped["event_ts"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
        )
        combined["is_new_merchant"] = (
            grouped["merchantName"].transform(lambda values: ~values.duplicated()).fillna(True)
        ).astype(int)

        result = (
            combined.loc[combined["_is_current_row"]]
            .sort_values("_feature_row_id")
            .reset_index(drop=True)
        )
        return result.drop(columns=["_feature_row_id", "_is_current_row"])
