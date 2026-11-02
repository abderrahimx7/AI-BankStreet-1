import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


project_folder = Path(__file__).resolve().parent
report_folder = project_folder / "results" / "fraud" / "xgboost_v1"

model_path = project_folder / "models" / "fraud" / "xgboost_v1.json"
data_path = project_folder / "data" / "fraud" / "creditcard.csv"

with (report_folder / "metadata.json").open(
    encoding="utf-8"
) as file:
    metadata = json.load(file)

with (report_folder / "decision_policy.json").open(
    encoding="utf-8"
) as file:
    policy = json.load(file)

if hashlib.sha256(model_path.read_bytes()).hexdigest() != policy["model_sha256"]:
    raise ValueError("Model differs from the frozen policy.")

data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

if (
    data_hash != policy["data_sha256"]
    or data_hash != metadata["data_sha256"]
):
    raise ValueError("Dataset differs from the saved experiment.")

if policy["test_start"] != metadata["test_start"]:
    raise ValueError("Inconsistent test boundary.")

data = pd.read_csv(data_path).sort_values("Time", kind="stable")

train = data.loc[data["Time"] < metadata["validation_start"]]
test = data.loc[data["Time"] >= policy["test_start"]].copy()

if test.empty or test["Class"].nunique() != 2:
    raise ValueError("Test must contain both classes.")

features = metadata["features"]
threshold = policy["threshold"]

model = xgb.Booster()
model.load_model(model_path)

# Inference only: no training or model updates.
model.set_param({"device": "cpu"})

probabilities = model.predict(
    xgb.DMatrix(test[features])
)

if not (
    np.isfinite(probabilities).all()
    and ((probabilities >= 0) & (probabilities <= 1)).all()
):
    raise ValueError("Invalid probabilities.")

actual = test["Class"].to_numpy(dtype=int)
alerts = probabilities >= threshold

# Baseline probability comes from training, never from Test.
baseline = np.full(len(test), float(train["Class"].mean()))

rows = []

for name, predictions in [
    ("Baseline", baseline),
    ("XGBoost", probabilities),
]:
    rows.append({
        "model": name,
        "brier": brier_score_loss(actual, predictions),
        "roc_auc": roc_auc_score(actual, predictions),
        "average_precision": average_precision_score(
            actual, predictions
        ),
    })

metrics = pd.DataFrame(rows).set_index("model")

tn, fp, fn, tp = confusion_matrix(
    actual, alerts, labels=[0, 1]
).ravel()

alert_count = int(alerts.sum())

decision_results = {
    "threshold": threshold,
    "test_rows": len(test),
    "actual_frauds": int(actual.sum()),
    "TP": int(tp),
    "FP": int(fp),
    "FN": int(fn),
    "TN": int(tn),
    "alerts": alert_count,
    "precision": float(tp / alert_count) if alert_count else None,
    "recall": float(tp / (tp + fn)),
    "alerts_per_1000": 1000 * alert_count / len(test),
    "model_sha256": policy["model_sha256"],
}

metrics.to_csv(report_folder / "test_metrics.csv")

with (report_folder / "test_decision_results.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(decision_results, file, indent=2)

pd.DataFrame({
    "row_id": test.index,
    "Time": test["Time"].to_numpy(),
    "target": actual,
    "probability": probabilities,
    "alert": alerts.astype(int),
}).to_csv(report_folder / "test_predictions.csv", index=False)

print("Test transactions:", len(test))
print("Actual frauds:", int(actual.sum()))
print("Frozen threshold:", threshold)
print("\n", metrics.round(6))
print("\n", json.dumps(decision_results, indent=2))
print("\nFrozen-policy test evaluation saved.")