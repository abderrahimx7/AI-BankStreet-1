import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from bankstreet.labels import build_labels


project_folder = Path(__file__).resolve().parent
report_folder = project_folder / "results" / "replay"

metadata_path = (
    project_folder / "results" / "experiment_v1" / "metadata.json"
)

with metadata_path.open(encoding="utf-8") as file:
    metadata = json.load(file)

predictions = pd.read_csv(
    report_folder / "kbe_predictions.csv",
    index_col="date",
    parse_dates=["date"],
).sort_index()

if predictions.empty:
    raise ValueError("No predictions found.")

if not (
    (predictions.index >= pd.Timestamp("2023-01-01"))
    & (predictions.index < pd.Timestamp("2024-01-01"))
).all():
    raise ValueError("Expected predictions from 2023 only.")

prices = pd.read_csv(
    project_folder / "data" / metadata["prices_file"],
    index_col="Date",
    parse_dates=["Date"],
).sort_index()

labels = build_labels(
    prices,
    horizon=metadata["horizon_sessions"],
    threshold=metadata["target_threshold"],
)

evaluated = predictions.join(labels, validate="one_to_one")

evaluated = evaluated.loc[
    evaluated["target"].notna()
    & (evaluated["label_end_date"] < pd.Timestamp("2024-01-01"))
].copy()

if evaluated.empty:
    raise ValueError("No outcomes available for evaluation.")

y = evaluated["target"].astype(int)
p = evaluated["p_target_1"].to_numpy(dtype=float)

if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
    raise ValueError("Invalid predicted probabilities.")

baseline = np.full(len(y), metadata["baseline_probability"])

rows = []

for name, probabilities in [
    ("Baseline", baseline),
    ("Logistic", p),
]:
    rows.append({
        "model": name,
        "brier": brier_score_loss(y, probabilities),
        "roc_auc": (
            roc_auc_score(y, probabilities)
            if y.nunique() == 2 else np.nan
        ),
        "average_precision": (
            average_precision_score(y, probabilities)
            if y.nunique() == 2 else np.nan
        ),
    })

metrics = pd.DataFrame(rows).set_index("model")

print("Prediction rows:", len(predictions))
print("Evaluated rows:", len(evaluated))
print("Excluded rows:", len(predictions) - len(evaluated))
print("Positive rate:", round(float(y.mean()), 6))
print(metrics.round(6))

metrics.to_csv(report_folder / "metrics_2023.csv")
evaluated.to_csv(report_folder / "evaluated_predictions_2023.csv")

print("Replay evaluation saved.")