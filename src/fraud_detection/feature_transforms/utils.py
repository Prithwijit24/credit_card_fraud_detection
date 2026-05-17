from __future__ import annotations

import pandas as pd


def safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bool_as_int(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(int)
