from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "capital-one-transactions-v1"

TRANSACTION_COLUMNS = [
    "accountNumber",
    "customerId",
    "creditLimit",
    "availableMoney",
    "transactionDateTime",
    "transactionAmount",
    "merchantName",
    "acqCountry",
    "merchantCountryCode",
    "posEntryMode",
    "posConditionCode",
    "merchantCategoryCode",
    "currentExpDate",
    "accountOpenDate",
    "dateOfLastAddressChange",
    "cardCVV",
    "enteredCVV",
    "cardLast4Digits",
    "transactionType",
    "echoBuffer",
    "currentBalance",
    "merchantCity",
    "merchantState",
    "merchantZip",
    "cardPresent",
    "posOnPremises",
    "recurringAuthInd",
    "expirationDateKeyInMatch",
    "isFraud",
]

STRING_COLUMNS = [
    "accountNumber",
    "customerId",
    "merchantName",
    "acqCountry",
    "merchantCountryCode",
    "posEntryMode",
    "posConditionCode",
    "merchantCategoryCode",
    "currentExpDate",
    "accountOpenDate",
    "dateOfLastAddressChange",
    "cardCVV",
    "enteredCVV",
    "cardLast4Digits",
    "transactionType",
    "echoBuffer",
    "merchantCity",
    "merchantState",
    "merchantZip",
    "posOnPremises",
    "recurringAuthInd",
]

NUMERIC_COLUMNS = [
    "creditLimit",
    "availableMoney",
    "transactionAmount",
    "currentBalance",
]

BOOLEAN_COLUMNS = ["cardPresent", "expirationDateKeyInMatch", "isFraud"]


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )


def coerce_transaction_schema(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    for column in TRANSACTION_COLUMNS:
        if column not in coerced:
            coerced[column] = pd.NA
    for column in STRING_COLUMNS:
        coerced[column] = coerced[column].astype("string")
    for column in NUMERIC_COLUMNS:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    for column in BOOLEAN_COLUMNS:
        coerced[column] = _coerce_bool(coerced[column])
    return coerced[TRANSACTION_COLUMNS]


def frame_from_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return coerce_transaction_schema(pd.DataFrame.from_records(records))


def read_transactions_csv(path: str | Path) -> pd.DataFrame:
    return coerce_transaction_schema(pd.read_csv(path))


def read_transactions_jsonl(path: str | Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return frame_from_records(records)


def read_transactions(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".jsonl", ".txt", ".ndjson"}:
        return read_transactions_jsonl(source)
    return read_transactions_csv(source)
