
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# Load processed test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Load trained model
model = joblib.load("models/fraud_detection_model.pkl")

print("Model loaded successfully!")

# Generate predictions
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print(
    "\nAverage Precision:",
    round(average_precision_score(y_test, y_scores), 4)
)

# Save evaluation report
report = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("models/evaluation_report.csv")

print("\nEvaluation report saved successfully!")
