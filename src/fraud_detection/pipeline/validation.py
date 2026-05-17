from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop records that cannot be scored under the notebook transaction schema.
    """
    initial_count = len(df)
    amount = (
        pd.to_numeric(df["transactionAmount"], errors="coerce")
        if "transactionAmount" in df
        else pd.Series(index=df.index, dtype="float64")
    )
    timestamp = (
        pd.to_datetime(df["transactionDateTime"], errors="coerce")
        if "transactionDateTime" in df
        else pd.Series(index=df.index, dtype="datetime64[ns]")
    )

    required_present = (
        df.get("accountNumber", pd.Series(index=df.index)).notna()
        & df.get("customerId", pd.Series(index=df.index)).notna()
        & df.get("merchantName", pd.Series(index=df.index)).notna()
        & timestamp.notna()
    )
    valid_mask = (amount > 0) & required_present
    valid_df = df.loc[valid_mask].copy()

    dropped_count = initial_count - len(valid_df)
    if dropped_count > 0:
        LOGGER.warning("Data validation dropped %s malformed records.", dropped_count)
    else:
        LOGGER.info("Data validation passed. No records dropped.")
    return valid_df
