
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "fraud_detection_model.pkl"
X_TEST_PATH = ROOT_DIR / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = ROOT_DIR / "data" / "processed" / "y_test.csv"

# Load the existing trained model and test data
model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

# Generate predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Calculate metrics from the existing test set
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "fraud_precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "fraud_recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "fraud_f1_score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "average_precision": average_precision_score(
        y_test, y_probability
    ),
}

# Store experiment results locally
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("online-fraud-detection")

with mlflow.start_run(run_name="decision-tree-baseline"):
    mlflow.log_param("model_type", type(model).__name__)
    mlflow.log_param("max_depth", model.max_depth)
    mlflow.log_param("class_weight", str(model.class_weight))
    mlflow.log_param("random_state", model.random_state)
    mlflow.log_param("test_rows", len(X_test))
    mlflow.log_param("decision_threshold", 0.5)

    mlflow.log_metrics(metrics)

    mlflow.log_text(
        str(confusion_matrix(y_test, y_pred)),
        "confusion_matrix.txt",
    )
    mlflow.log_text(
        classification_report(y_test, y_pred),
        "classification_report.txt",
    )

print("\nMLflow experiment recorded successfully.")
print("\nMetrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print("\nTracking directory:", ROOT_DIR / "mlruns")
