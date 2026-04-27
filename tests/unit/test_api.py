from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fraud_detection.jobs.api import get_app


@pytest.fixture
def mock_model(spark):
    mock = MagicMock()
    
    def mock_transform(df):
        # Return a dataframe with a probability column that simulates a Spark ML DenseVector
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.functions import lit, udf
        from pyspark.ml.linalg import VectorUDT
        
        # We need a udf to create Vectors because lit() doesn't support Vectors directly
        vector_udf = udf(lambda: Vectors.dense([0.1, 0.8]), VectorUDT())
        return df.withColumn("probability", vector_udf())
        
    mock.transform.side_effect = mock_transform
    return mock


@pytest.fixture
def client(project_root, mock_model):
    os.environ["FRAUD_CONFIG"] = str(project_root / "configs" / "base.yaml")
    
    with patch("fraud_detection.jobs.api.load_model", return_value=mock_model):
        app = get_app()
        with TestClient(app) as test_client:
            yield test_client


def test_score_transaction_success(client):
    payload = {
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
        "is_fraud": 0
    }
    
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert data["fraud_probability"] == 0.8
    assert data["risk_band"] == "critical"  # threshold is likely < 0.8 in base.yaml


def test_score_transaction_invalid_json(client):
    response = client.post("/score", data="this is not json")
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid JSON"}


def test_score_transaction_validation_failure(client):
    # Missing required fields like cc_num and amt
    payload = {
        "row_id": 1,
        "merchant": "fraud_inc"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 400
    assert response.json() == {"error": "Transaction failed data quality validation"}
