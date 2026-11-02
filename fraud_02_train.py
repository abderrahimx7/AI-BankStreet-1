import hashlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


project_folder = Path(__file__).resolve().parent
data_path = project_folder / "data" / "fraud" / "creditcard.csv"

data = pd.read_csv(data_path).sort_values("Time", kind="stable")

validation_start = 120396.0
test_start = 145248.0

train = data.loc[data["Time"] < validation_start].copy()

validation = data.loc[
    (data["Time"] >= validation_start)
    & (data["Time"] < test_start)
].copy()

feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]

for name, part in [("Train", train), ("Validation", validation)]:
    if part.empty or part["Class"].nunique() != 2:
        raise ValueError(f"{name} must contain both classes.")

    print(
        f"{name}: {len(part)} rows, "
        f"{int(part['Class'].sum())} frauds"
    )

assert train["Time"].max() < validation["Time"].min()

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)

# Stop if training does not converge.
with warnings.catch_warnings():
    warnings.simplefilter("error", ConvergenceWarning)
    model.fit(train[feature_columns], train["Class"])

positive_column = list(model.classes_).index(1)

probabilities = model.predict_proba(
    validation[feature_columns]
)[:, positive_column]

baseline_probability = float(train["Class"].mean())
baseline = np.full(len(validation), baseline_probability)
actual = validation["Class"].astype(int)

rows = []

for name, predictions in [
    ("Baseline", baseline),
    ("Logistic", probabilities),
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

print("\nValidation positive rate:", round(float(actual.mean()), 6))
print(metrics.round(6))

report_folder = project_folder / "results" / "fraud" / "logistic_v1"
model_folder = project_folder / "models" / "fraud"

report_folder.mkdir(parents=True, exist_ok=True)
model_folder.mkdir(parents=True, exist_ok=True)

metrics.to_csv(report_folder / "validation_metrics.csv")

pd.DataFrame({
    "row_id": validation.index,
    "Time": validation["Time"].to_numpy(),
    "target": actual.to_numpy(),
    "probability": probabilities,
}).to_csv(report_folder / "validation_predictions.csv", index=False)

joblib.dump(
    {
        "pipeline": model,
        "features": feature_columns,
        "baseline_probability": baseline_probability,
        "validation_start": validation_start,
        "test_start": test_start,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    },
    model_folder / "logistic_v1.joblib",
)

print("\nFraud model and validation results saved.")