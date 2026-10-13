
import pandas as pd
from pathlib import Path

results = [
    {
        "Model": "Decision Tree",
        "Evaluation": "Full test set",
        "Precision": 0.2223,
        "Recall": 0.9763,
        "F1 Score": 0.3621,
        "Average Precision": 0.8544
    },
    {
        "Model": "Random Forest",
        "Evaluation": "Sample test set (40000 rows)",
        "Precision": 0.2529411765,
        "Recall": 0.8269230769,
        "F1 Score": 0.3873873874,
        "Average Precision": 0.76017371399
    }
]

output = Path("models/model_comparison_initial.csv")
pd.DataFrame(results).to_csv(output, index=False)

print(pd.DataFrame(results).to_string(index=False))
print(f"\nComparison saved to {output}")
print("Note: evaluation datasets differ; do not treat this as a final ranking.")
