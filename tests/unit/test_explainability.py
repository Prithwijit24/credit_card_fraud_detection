from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.pipeline.explainability import (
    EXPLAINABILITY_METADATA_KEY,
    ExplainabilityService,
    build_explainability_metadata,
)


def test_build_explainability_metadata_samples_training_background():
    train_df = pd.DataFrame(
        [{column: index for column in FEATURE_COLUMNS} for index in range(5)]
    )

    metadata = build_explainability_metadata(
        train_df,
        seed=42,
        max_background_rows=2,
    )

    assert metadata["enabled"] is True
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["background_rows"] == 2
    assert len(metadata["background"]) == 2
    assert metadata["methods"] == ["shap", "lime"]


def test_explainability_service_falls_back_to_heuristic_reasons_without_background():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.1, 0.9]])
    features = pd.DataFrame(
        [
            {
                **{column: 0 for column in FEATURE_COLUMNS},
                "cvv_mismatch": 1,
            }
        ]
    )

    explanation = ExplainabilityService.from_metadata(model, {}).explain(features, 0.9)

    assert explanation["top_reasons"] == ["CVV mismatch"]
    assert explanation["explanation_methods"] == ["heuristic"]
    assert explanation["feature_contributions"] == []


def test_explainability_service_reads_background_from_artifact_metadata():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.4, 0.6]])
    background = pd.DataFrame([{column: 0 for column in FEATURE_COLUMNS}])
    metadata = {
        EXPLAINABILITY_METADATA_KEY: build_explainability_metadata(
            background,
            seed=42,
        )
    }

    service = ExplainabilityService.from_metadata(model, metadata)

    assert service.background.shape == (1, len(FEATURE_COLUMNS))
