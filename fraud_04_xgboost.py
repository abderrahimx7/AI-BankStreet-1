import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


project_folder = Path(__file__).resolve().parent
data_path = project_folder / "data" / "fraud" / "creditcard.csv"

bundle = joblib.load(
    project_folder / "models" / "fraud" / "logistic_v1.joblib"
)

data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

if data_hash != bundle["data_sha256"]:
    raise ValueError("Dataset differs from the Logistic experiment.")

data = pd.read_csv(data_path).sort_values("Time", kind="stable")

features = bundle["features"]

train = data.loc[
    data["Time"] < bundle["validation_start"]
].copy()

validation = data.loc[
    (data["Time"] >= bundle["validation_start"])
    & (data["Time"] < bundle["test_start"])
].copy()

assert train["Time"].max() < validation["Time"].min()

dtrain = xgb.DMatrix(
    train[features],
    label=train["Class"],
)

dvalidation = xgb.DMatrix(
    validation[features],
    label=validation["Class"],
)

parameters = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cuda:0",
    "max_depth": 3,
    "eta": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "scale_pos_weight": 1.0,
    "eval_metric": "logloss",
    "seed": 42,
    "nthread": 4,
}

start = perf_counter()

model = xgb.train(
    parameters,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dvalidation, "validation")],
    verbose_eval=50,
)

training_seconds = perf_counter() - start

configuration = json.loads(model.save_config())
device = configuration["learner"]["generic_param"]["device"]

if not device.startswith("cuda"):
    raise RuntimeError("Training did not use CUDA.")

xgb_probabilities = model.predict(dvalidation)

logistic = bundle["pipeline"]
positive_column = list(logistic.classes_).index(1)

logistic_probabilities = logistic.predict_proba(
    validation[features]
)[:, positive_column]

actual = validation["Class"].to_numpy(dtype=int)
baseline = np.full(len(actual), bundle["baseline_probability"])

rows = []

for name, probabilities in [
    ("Baseline", baseline),
    ("Logistic", logistic_probabilities),
    ("XGBoost", xgb_probabilities),
]:
    rows.append({
        "model": name,
        "brier": brier_score_loss(actual, probabilities),
        "roc_auc": roc_auc_score(actual, probabilities),
        "average_precision": average_precision_score(
            actual, probabilities
        ),
    })

metrics = pd.DataFrame(rows).set_index("model")

report_folder = project_folder / "results" / "fraud" / "xgboost_v1"
report_folder.mkdir(parents=True, exist_ok=True)

model_path = project_folder / "models" / "fraud" / "xgboost_v1.json"
model.save_model(model_path)

metrics.to_csv(report_folder / "validation_metrics.csv")

pd.DataFrame({
    "row_id": validation.index,
    "Time": validation["Time"].to_numpy(),
    "target": actual,
    "probability": xgb_probabilities,
}).to_csv(report_folder / "validation_predictions.csv", index=False)

metadata = {
    "features": features,
    "parameters": parameters,
    "boost_rounds": 300,
    "xgboost_version": xgb.__version__,
    "training_device": device,
    "training_seconds": training_seconds,
    "data_sha256": data_hash,
    "validation_start": bundle["validation_start"],
    "test_start": bundle["test_start"],
}

with (report_folder / "metadata.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=2)

print("\nTraining device:", device)
print("Training seconds:", round(training_seconds, 2))
print("Validation transactions:", len(validation))
print(metrics.round(6))
print("\nXGBoost model and validation results saved.")