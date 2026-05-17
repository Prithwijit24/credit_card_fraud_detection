from __future__ import annotations

import pandas as pd

from fraud_detection.feature_transforms import (
    CATEGORICAL_COLUMNS,
    ENTITY_ENCODING_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    EntityEncoders,
    EntityEncodingFeature,
    SummaryFeaturePipeline,
)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "ENTITY_ENCODING_COLUMNS",
    "FEATURE_COLUMNS",
    "NUMERIC_COLUMNS",
    "add_derived_features",
    "apply_entity_encoders",
    "fit_entity_encoders",
    "sanitize_input_columns",
]


def sanitize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {
        column: "row_id"
        for column in renamed.columns
        if str(column).startswith("Unnamed") or column == ""
    }
    return renamed.rename(columns=rename_map)


def add_derived_features(df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return SummaryFeaturePipeline().transform(df, history_df=history_df)


def fit_entity_encoders(
    df: pd.DataFrame,
    label_column: str = "isFraud",
) -> EntityEncoders:
    return EntityEncodingFeature().fit(df, label_column=label_column)


def apply_entity_encoders(
    df: pd.DataFrame,
    encoders: EntityEncoders,
) -> pd.DataFrame:
    return EntityEncodingFeature(encoders=encoders).transform(df)
