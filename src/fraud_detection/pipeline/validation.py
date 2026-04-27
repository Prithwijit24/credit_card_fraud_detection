from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data quality checks to the transaction frame.
    Drops records that violate core assumptions.
    """
    initial_count = len(df)
    amount = pd.to_numeric(df["amt"], errors="coerce") if "amt" in df else pd.Series(index=df.index)
    timestamp = (
        pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        if "trans_date_trans_time" in df
        else pd.Series(index=df.index)
    )

    required_present = (
        df.get("cc_num", pd.Series(index=df.index)).notna()
        & df.get("merchant", pd.Series(index=df.index)).notna()
        & timestamp.notna()
    )
    valid_mask = (amount > 0) & required_present
    valid_df = df.loc[valid_mask].copy()

    final_count = len(valid_df)
    dropped_count = initial_count - final_count

    if dropped_count > 0:
        LOGGER.warning("Data validation dropped %s malformed records.", dropped_count)
    else:
        LOGGER.info("Data validation passed. No records dropped.")

    return valid_df
