print("Script started")

import pandas as pd
from pathlib import Path

file_path = Path("data/raw/onlinefraud.csv")

print("File exists:", file_path.exists())

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print("\nAll column names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nFirst 5 rows:")
print(df.head())


print("\nFraud distribution:")
print(df["isFraud"].value_counts())

print("\nFraud percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)

print("\nDataset data types:")
print(df.dtypes)



print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTotal duplicate rows:")
print(df.duplicated().sum())

print("\nTarget column:")
print(df["isFraud"].value_counts())
