
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# Load dataset
df = pd.read_csv("data/raw/onlinefraud.csv")

# Prepare features and target
df = df.drop(columns=["nameOrig", "nameDest"])

X = df.drop(columns=["isFraud", "isFlaggedFraud"])
X = pd.get_dummies(X, columns=["type"], dtype=int)
y = df["isFraud"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train a baseline model
model = DecisionTreeClassifier(
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluate the model
print("\nGenerating predictions...")
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print(
    "Average precision score:",
    round(average_precision_score(y_test, y_scores), 4)
)

# Save the trained model
import joblib

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/fraud_detection_model.pkl")

print("\nModel saved to models/fraud_detection_model.pkl")
