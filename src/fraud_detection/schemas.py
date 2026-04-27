from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

TRANSACTION_COLUMNS = [
    "row_id",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

STRING_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "job",
    "dob",
    "trans_num",
]
NUMERIC_COLUMNS = [
    "row_id",
    "amt",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


def coerce_transaction_schema(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    for column in TRANSACTION_COLUMNS:
        if column not in coerced:
            coerced[column] = pd.NA
    for column in STRING_COLUMNS:
        coerced[column] = coerced[column].astype("string")
    for column in NUMERIC_COLUMNS:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced[TRANSACTION_COLUMNS]


def frame_from_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return coerce_transaction_schema(pd.DataFrame.from_records(records))


def read_transactions_csv(path: str) -> pd.DataFrame:
    return coerce_transaction_schema(pd.read_csv(path))
