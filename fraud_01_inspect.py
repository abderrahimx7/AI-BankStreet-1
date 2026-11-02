from pathlib import Path

import numpy as np
import pandas as pd


project_folder = Path(__file__).resolve().parent
data_path = project_folder / "data" / "fraud" / "creditcard.csv"

data = pd.read_csv(data_path)

expected_columns = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)

if set(data.columns) != set(expected_columns):
    raise ValueError("Unexpected dataset columns.")

if not all(pd.api.types.is_numeric_dtype(data[c]) for c in data):
    raise ValueError("Expected numeric columns.")

if not np.isfinite(data.to_numpy(dtype=float)).all():
    raise ValueError("Missing or infinite values found.")

if not data["Class"].isin([0, 1]).all():
    raise ValueError("Class must contain only 0 and 1.")

if (data["Time"] < 0).any() or (data["Amount"] < 0).any():
    raise ValueError("Negative time or amount found.")

counts = (
    data["Class"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

print("Rows:", len(data))
print("Columns:", len(data.columns))
print("Normal transactions:", counts.loc[0])
print("Fraud transactions:", counts.loc[1])
print("Fraud rate (%):", round(data["Class"].mean() * 100, 4))
print("Identical extra rows:", data.duplicated().sum())
print("Time already sorted:", data["Time"].is_monotonic_increasing)
print("Time range (seconds):", data["Time"].min(), data["Time"].max())

# Proposed chronological boundaries, independent of Class.
ordered_times = data["Time"].sort_values().reset_index(drop=True)

validation_start = ordered_times.iloc[int(len(data) * 0.60)]
test_start = ordered_times.iloc[int(len(data) * 0.80)]

if validation_start >= test_start:
    raise ValueError("Chronological boundaries are not distinct.")

parts = {
    "Train": data.loc[data["Time"] < validation_start],
    "Validation": data.loc[
        (data["Time"] >= validation_start)
        & (data["Time"] < test_start)
    ],
    "Test": data.loc[data["Time"] >= test_start],
}

print("\nProposed chronological split:")
print("Validation starts at:", validation_start)
print("Test starts at:", test_start)

for name, part in parts.items():
    print(f"{name}: {len(part)} transactions")

print("\nDataset inspection completed.")