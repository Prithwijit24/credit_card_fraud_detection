from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.features import FEATURE_COLUMNS

TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DuckDBFeatureStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> Any:
        try:
            import duckdb
        except ImportError as exc:
            message = "Install DuckDB support with `pip install -e .`."
            raise RuntimeError(message) from exc
        return duckdb.connect(str(self.database_path))

    def _table(self, table: str) -> str:
        if not TABLE_NAME_PATTERN.fullmatch(table):
            raise ValueError(f"Invalid feature-store table name: {table}")
        return table

    def write_features(self, features: pd.DataFrame, table: str = "transaction_features") -> int:
        if features.empty:
            return 0
        table = self._table(table)
        payload = features.copy()
        with self._connect() as connection:
            connection.register("feature_payload", payload)
            sql = f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM feature_payload"  # noqa: S608
            connection.execute(sql)
        return len(payload)

    def append_features(self, features: pd.DataFrame, table: str = "transaction_features") -> int:
        if features.empty:
            return 0
        table = self._table(table)
        payload = features.copy()
        with self._connect() as connection:
            connection.register("feature_payload", payload)
            exists = connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table],
            ).fetchone()[0]
            if exists:
                connection.execute(f"INSERT INTO {table} SELECT * FROM feature_payload")  # noqa: S608
            else:
                sql = f"CREATE TABLE {table} AS SELECT * FROM feature_payload"  # noqa: S608
                connection.execute(sql)
        return len(payload)

    def load_recent_history(
        self,
        account_number: str,
        limit: int = 500,
        table: str = "transaction_features",
    ) -> pd.DataFrame:
        table = self._table(table)
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table],
            ).fetchone()[0]
            if not exists:
                return pd.DataFrame()
            sql = f"""
                SELECT *
                FROM {table}
                WHERE accountNumber = ?
                ORDER BY transactionDateTime DESC
                LIMIT ?
                """  # noqa: S608
            return connection.execute(sql, [account_number, limit]).fetchdf()

    def feature_columns(self) -> list[str]:
        return list(FEATURE_COLUMNS)
