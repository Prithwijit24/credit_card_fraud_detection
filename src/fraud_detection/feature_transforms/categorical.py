from __future__ import annotations

import pandas as pd

from fraud_detection.feature_transforms.base import BaseFeature
from fraud_detection.feature_transforms.columns import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from fraud_detection.feature_transforms.utils import safe_numeric


class FeatureTypeNormalizer(BaseFeature):
    name = "feature_type_normalizer"

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del history_df
        transformed = df.copy()
        categorical_defaults = {column: "unknown" for column in CATEGORICAL_COLUMNS}
        transformed = transformed.fillna(categorical_defaults)
        for column in CATEGORICAL_COLUMNS:
            transformed[column] = transformed[column].astype("string").fillna("unknown")
        for column in NUMERIC_COLUMNS:
            if column not in transformed:
                transformed[column] = 0.0
            transformed[column] = safe_numeric(transformed[column]).fillna(0.0)
        return transformed
