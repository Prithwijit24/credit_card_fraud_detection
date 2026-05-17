from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from fraud_detection.jobs.api import score_payload
from tests.fixtures import notebook_transaction


@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.2, 0.8]])
    return mock


def test_score_transaction_success(mock_model):
    data = score_payload(
        notebook_transaction(cardCVV="123", enteredCVV="999"),
        mock_model,
        threshold=0.75,
        schema_version="capital-one-transactions-v1",
    )
    assert data["fraud_probability"] == 0.8
    assert data["risk_band"] == "critical"
    assert data["schema_version"] == "capital-one-transactions-v1"
    assert "CVV mismatch" in data["top_reasons"]
    assert data["explanation_methods"] == ["heuristic"]
    assert data["feature_contributions"] == []


def test_score_transaction_validation_failure(mock_model):
    with pytest.raises(ValueError, match="Transaction failed data quality validation"):
        score_payload({"merchantName": "fraud_inc"}, mock_model, threshold=0.75)
