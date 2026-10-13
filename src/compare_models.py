
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Load the original dataset
df = pd.read_csv("data/raw/onlinefraud.csv")
df = df.drop(columns=["nameOrig", "nameDest"])

X = df.drop(columns=["isFraud", "isFlaggedFraud"])
X = pd.get_dummies(X, columns=["type"], dtype=int)
y = df["isFraud"]

# Use a smaller stratified sample for a quick experiment
X_sample, _, y_sample, _ = train_test_split(
    X,
    y,
    train_size=200_000,
    random_state=42,
    stratify=y
)

# Split sample into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42,
    stratify=y_sample
)

print("Sample training rows:", len(X_train))
print("Sample testing rows:", len(X_test))
print("Training Random Forest...")

random_forest = RandomForestClassifier(
    n_estimators=30,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

# Evaluate Random Forest on the sample test set
predictions = random_forest.predict(X_test)
probabilities = random_forest.predict_proba(X_test)[:, 1]

print("\nRandom Forest results on sample test data:")
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1-score:", f1_score(y_test, predictions, zero_division=0))
print("Average precision:", average_precision_score(y_test, probabilities))

# Save this experimental model separately
joblib.dump(
    random_forest,
    "models/fraud_detection_random_forest_sample.pkl"
)

print("\nSample Random Forest model saved.")
