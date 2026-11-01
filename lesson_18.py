from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bankstreet.features import build_features


project_folder = Path(__file__).resolve().parent

bundle = joblib.load(
    project_folder / "results" / "experiment_v1" / "kbe_model.joblib"
)

model = bundle["pipeline"]
metadata = bundle["metadata"]
feature_columns = metadata["features"]

report_folder = project_folder / "results" / "replay"

data = pd.read_csv(
    report_folder / "market_risk.csv",
    parse_dates=["date"],
)

missing = set(feature_columns) - set(data.columns)

if missing:
    raise ValueError(f"Missing features: {sorted(missing)}")

kbe = (
    data.loc[data["ticker"] == metadata["ticker"]]
    .sort_values("date")
    .set_index("date")
)

if kbe.empty:
    raise ValueError("No KBE rows found.")

# Independently verify all model inputs against the training feature code.
prices = pd.read_csv(
    project_folder / "data" / metadata["prices_file"],
    index_col="Date",
    parse_dates=["Date"],
).sort_index()

reference = build_features(prices).reindex(kbe.index)

for column in feature_columns:
    np.testing.assert_allclose(
        kbe[column].to_numpy(dtype=float),
        reference[column].to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-10,
        equal_nan=True,
        err_msg=f"Feature mismatch: {column}",
    )

score_start = pd.Timestamp(metadata["test_start"])
score_end = pd.Timestamp("2024-01-01")

ready = (
    kbe.loc[
        (kbe.index >= score_start)
        & (kbe.index < score_end)
    ]
    .dropna(subset=feature_columns)
    .copy()
)

if ready.empty:
    raise ValueError("No rows with complete model features.")

X = ready[feature_columns]

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Model inputs contain infinite values.")

positive_column = list(model.classes_).index(1)

ready["p_target_1"] = model.predict_proba(X)[:, positive_column]
ready["usage"] = "test_period_replay"
output_path = report_folder / "kbe_predictions.csv"
ready.reset_index().to_csv(output_path, index=False)

print("All model features match build_features.")
print("Prediction rows:", len(ready))
print(
    ready[feature_columns + ["p_target_1"]]
    .tail(5)
    .to_string()
)
print("Replay predictions saved:", output_path)