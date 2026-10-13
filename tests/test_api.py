
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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
