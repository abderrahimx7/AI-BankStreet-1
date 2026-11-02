import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


project_folder = Path(__file__).resolve().parent
report_folder = project_folder / "results" / "fraud" / "xgboost_v1"

model_path = project_folder / "models" / "fraud" / "xgboost_v1.json"

with (report_folder / "metadata.json").open(
    encoding="utf-8"
) as file:
    metadata = json.load(file)

predictions_path = report_folder / "validation_predictions.csv"
data = pd.read_csv(predictions_path)

budget = 94

if len(data) < budget or data["row_id"].duplicated().any():
    raise ValueError("Invalid validation rows.")

if not (
    (data["Time"] >= metadata["validation_start"])
    & (data["Time"] < metadata["test_start"])
).all():
    raise ValueError("Expected validation observations only.")

p = data["probability"].to_numpy(dtype=float)

if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
    raise ValueError("Invalid probabilities.")

# Examine complete groups of equal scores.
score_groups = (
    data.groupby("probability")
    .size()
    .sort_index(ascending=False)
)

cumulative_alerts = score_groups.cumsum()
eligible = cumulative_alerts[cumulative_alerts <= budget]

if eligible.empty:
    raise ValueError("No observed threshold satisfies this budget.")

# Lowest observed threshold that stays within the review budget.
threshold = float(eligible.index[-1])
alerts = p >= threshold

# Labels are used only to report performance after choosing the threshold.
if not data["target"].isin([0, 1]).all():
    raise ValueError("Invalid targets.")

y = data["target"].to_numpy(dtype=int)

tn, fp, fn, tp = confusion_matrix(
    y, alerts, labels=[0, 1]
).ravel()

alert_count = int(alerts.sum())

policy = {
    "model": "xgboost_v1",
    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "data_sha256": metadata["data_sha256"],
    "validation_predictions_sha256": hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest(),
    "selected_on": "validation",
    "rule": "probability >= threshold",
    "threshold": threshold,
    "selection_rule": "largest alert count not exceeding validation budget",
    "validation_review_budget": budget,
    "validation_rows": len(data),
    "validation_alerts": alert_count,
    "validation_alerts_per_1000": 1000 * alert_count / len(data),
    "test_start": metadata["test_start"],
    "action": "manual_review",
}

policy_path = report_folder / "decision_policy.json"

# Avoid silently replacing an already frozen policy.
with policy_path.open("x", encoding="utf-8") as file:
    json.dump(policy, file, indent=2)

print("Frozen threshold:", repr(threshold))
print("Validation alerts:", alert_count)
print("TP:", int(tp), "| FP:", int(fp), "| FN:", int(fn))
print("Precision:", round(tp / alert_count, 4))
print("Recall:", round(tp / (tp + fn), 4))
print("Policy saved:", policy_path)