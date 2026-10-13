
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/raw/onlinefraud.csv")

print("Original dataset shape:", df.shape)

# Remove identifier columns
df = df.drop(columns=["nameOrig", "nameDest"])

# Separate features and target
X = df.drop(columns=["isFraud", "isFlaggedFraud"])
y = df["isFraud"]

# Encode transaction type
X = pd.get_dummies(X, columns=["type"], dtype=int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create output directory
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save datasets
X_train.to_csv(output_dir / "X_train.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)
y_train.to_csv(output_dir / "y_train.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)

print("\nPrepared datasets saved successfully!")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
