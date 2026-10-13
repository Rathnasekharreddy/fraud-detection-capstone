
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Online Fraud Transaction Detection API",
    description="Predicts whether a transaction may be fraudulent.",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_detection_model.pkl"
model = joblib.load(MODEL_PATH)


class Transaction(BaseModel):
    step: int = Field(ge=1)
    type: str
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(transaction: Transaction):
    allowed_types = {
        "PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"
    }

    if transaction.type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported transaction type",
        )

    row = {
        "step": transaction.step,
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "oldbalanceDest": transaction.oldbalanceDest,
        "newbalanceDest": transaction.newbalanceDest,
    }

    for transaction_type in allowed_types:
        row[f"type_{transaction_type}"] = int(
            transaction.type == transaction_type
        )

    features = pd.DataFrame([row])

    if hasattr(model, "feature_names_in_"):
        features = features.reindex(
            columns=model.feature_names_in_,
            fill_value=0,
        )

    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= 0.5)

    return {
        "isFraud": prediction,
        "fraud_probability": round(probability, 4),
        "interpretation": (
            "Potential fraud" if prediction == 1 else "Likely legitimate"
        ),
        "threshold": 0.5,
    }
