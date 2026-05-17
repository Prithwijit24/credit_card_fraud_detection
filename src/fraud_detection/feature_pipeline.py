from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fraud_detection.features import (
    FEATURE_COLUMNS,
    add_derived_features,
    apply_entity_encoders,
    fit_entity_encoders,
    sanitize_input_columns,
)
from fraud_detection.pipeline.metrics import compute_class_weights
from fraud_detection.pipeline.validation import validate_transactions


@dataclass
class FeaturePipeline:
    label_column: str = "isFraud"
    entity_encoders: dict[str, dict[str, float] | float] = field(default_factory=dict)

    def validate_raw(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        valid_df = validate_transactions(sanitize_input_columns(raw_df))
        if self.label_column in valid_df:
            valid_df[self.label_column] = (
                valid_df[self.label_column].fillna(False).astype(bool).astype(int)
            )
        return valid_df

    def fit_transform(
        self,
        raw_df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
        with_class_weights: bool = False,
    ) -> pd.DataFrame:
        valid_df = self.validate_raw(raw_df)
        features = add_derived_features(valid_df, history_df=history_df)
        self.entity_encoders = fit_entity_encoders(features, label_column=self.label_column)
        features = apply_entity_encoders(features, self.entity_encoders)
        features.attrs["entity_encoders"] = self.entity_encoders
        if with_class_weights:
            features = compute_class_weights(features, label_column=self.label_column)
            features.attrs["entity_encoders"] = self.entity_encoders
        return features

    def transform(
        self,
        raw_df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        valid_df = self.validate_raw(raw_df)
        features = add_derived_features(valid_df, history_df=history_df)
        return apply_entity_encoders(features, self.entity_encoders)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> FeaturePipeline:
        metadata = metadata or {}
        encoders = dict(metadata.get("entity_encoders", {}))
        return cls(entity_encoders=encoders)


def required_feature_columns() -> list[str]:
    return list(FEATURE_COLUMNS)
