from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


project_folder = Path(__file__).resolve().parent
report_folder = project_folder / "results" / "fraud" / "logistic_v1"

data = pd.read_csv(report_folder / "validation_predictions.csv")

if data.empty or data["row_id"].duplicated().any():
    raise ValueError("Expected non-empty predictions with unique row IDs.")

if not data["target"].isin([0, 1]).all():
    raise ValueError("Invalid targets.")

if not (
    (data["Time"] >= 120396.0)
    & (data["Time"] < 145248.0)
).all():
    raise ValueError("Expected validation observations only.")

y = data["target"].to_numpy(dtype=int)
p = data["probability"].to_numpy(dtype=float)

if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
    raise ValueError("Invalid probabilities.")

if np.unique(y).size != 2:
    raise ValueError("Both classes are required.")

thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
rows = []

for threshold in thresholds:
    alerts = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, alerts, labels=[0, 1]
    ).ravel()

    alert_count = int(tp + fp)

    rows.append({
        "threshold": threshold,
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "precision": tp / alert_count if alert_count else np.nan,
        "recall": tp / (tp + fn),
        "alerts": alert_count,
        "alerts_per_1000": 1000 * alert_count / len(y),
    })

summary = pd.DataFrame(rows)

print("Validation transactions:", len(y))
print("Actual frauds:", int(y.sum()))
print(summary.round(4).to_string(index=False))

summary.to_csv(
    report_folder / "validation_thresholds.csv",
    index=False,
)

print("\nValidation threshold report saved.")