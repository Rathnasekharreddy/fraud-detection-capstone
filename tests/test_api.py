
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_module


class MockModel:
    """A lightweight model used only for API testing."""

    feature_names_in_ = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]

    def predict_proba(self, features):
        return np.array([[0.8, 0.2]])


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    monkeypatch.setattr(api_module, "model", MockModel())


client = TestClient(api_module.app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Fraud Detection API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_valid_transaction():
    transaction = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000,
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 5000,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    result = response.json()
    assert result["isFraud"] in [0, 1]
    assert 0 <= result["fraud_probability"] <= 1
    assert result["threshold"] == 0.5


def test_predict_invalid_transaction_type():
    transaction = {
        "step": 1,
        "type": "INVALID_TYPE",
        "amount": 5000,
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 5000,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 400


def test_predict_missing_field():
    transaction = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000,
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_predict_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(api_module, "model", None)

    transaction = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000,
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 5000,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 503
