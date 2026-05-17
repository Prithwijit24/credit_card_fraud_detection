from __future__ import annotations

import pandas as pd

from fraud_detection.feature_transforms.base import BaseFeature
from fraud_detection.feature_transforms.columns import ENTITY_ENCODING_SOURCE_COLUMNS
from fraud_detection.feature_transforms.utils import safe_numeric

EntityEncoders = dict[str, dict[str, float] | float]


class EntityEncodingFeature(BaseFeature):
    name = "entity_encoding"

    def __init__(self, encoders: EntityEncoders | None = None) -> None:
        self.encoders = encoders or {}

    def fit(self, df: pd.DataFrame, label_column: str = "isFraud") -> EntityEncoders:
        encoded: EntityEncoders = {}
        labels = safe_numeric(df[label_column]).fillna(0.0)
        global_rate = float(labels.mean()) if len(labels) else 0.0
        encoded["global_fraud_rate"] = global_rate
        for column in ENTITY_ENCODING_SOURCE_COLUMNS:
            rates = df.assign(_label=labels).groupby(column, dropna=False)["_label"].mean()
            encoded[column] = {str(key): float(value) for key, value in rates.items()}
        self.encoders = encoded
        return encoded

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del history_df
        transformed = df.copy()
        raw_default = self.encoders.get("global_fraud_rate", 0.0)
        default = raw_default if isinstance(raw_default, float) else 0.0
        for column in ENTITY_ENCODING_SOURCE_COLUMNS:
            mapping = self.encoders.get(column, {})
            if not isinstance(mapping, dict):
                mapping = {}
            transformed[f"{column}_fraud_rate"] = (
                transformed[column].astype("string").map(mapping).astype(float).fillna(default)
            )
        return transformed
