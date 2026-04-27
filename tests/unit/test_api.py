from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from fraud_detection.jobs.api import score_payload


@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.2, 0.8]])
    return mock


def _valid_payload() -> dict:
    return {
        "row_id": 1,
        "trans_date_trans_time": "2020-01-01 00:00:00",
        "cc_num": "123456789",
        "merchant": "fraud_inc",
        "category": "shopping",
        "amt": 500.0,
        "first": "John",
        "last": "Doe",
        "gender": "M",
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA",
        "zip": "12345",
        "lat": 34.0,
        "long": -118.0,
        "city_pop": 1000,
        "job": "Teacher",
        "dob": "1980-01-01",
        "trans_num": "txn123",
        "unix_time": 1577836800,
        "merch_lat": 34.1,
        "merch_long": -118.1,
        "is_fraud": 0,
    }


def test_score_transaction_success(mock_model):
    data = score_payload(_valid_payload(), mock_model, threshold=0.75)
    assert data["fraud_probability"] == 0.8
    assert data["risk_band"] == "critical"


def test_score_transaction_validation_failure(mock_model):
    with pytest.raises(ValueError, match="Transaction failed data quality validation"):
        score_payload({"row_id": 1, "merchant": "fraud_inc"}, mock_model, threshold=0.75)
