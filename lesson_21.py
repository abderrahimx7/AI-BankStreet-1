from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from bankstreet.features import build_features
from bankstreet.labels import build_labels
from bankstreet.modeling import train_model
from bankstreet.splits import split_by_time


project_folder = Path(__file__).resolve().parent

bundle = joblib.load(
    project_folder / "results" / "experiment_v1" / "kbe_model.joblib"
)

metadata = bundle["metadata"]
original_features = metadata["features"]

prices = pd.read_csv(
    project_folder / "data" / metadata["prices_file"],
    index_col="Date",
    parse_dates=["Date"],
).sort_index()

dataset = build_features(prices).join(
    build_labels(
        prices,
        horizon=metadata["horizon_sessions"],
        threshold=metadata["target_threshold"],
    )
)

# Use the same eligible rows for both models.
dataset = dataset.dropna(
    subset=original_features + ["target", "label_end_date"]
)

train, validation, _ = split_by_time(
    dataset,
    validation_start=metadata["validation_start"],
    test_start=metadata["test_start"],
)

volatility_features = ["volatility_20d"]

volatility_model, baseline_probability = train_model(
    train,
    volatility_features,
)

# Confirm that the training baseline matches the saved experiment.
if abs(
    baseline_probability - metadata["baseline_probability"]
) > 1e-12:
    raise ValueError("Training baseline differs from experiment_v1.")

models = {
    "Three features": (bundle["pipeline"], original_features),
    "Volatility only": (volatility_model, volatility_features),
}

predictions = pd.DataFrame(index=validation.index)
predictions["Baseline"] = baseline_probability

for name, (model, columns) in models.items():
    positive_column = list(model.classes_).index(1)
    predictions[name] = model.predict_proba(
        validation[columns]
    )[:, positive_column]

rows = []

periods = [("All validation", validation)]

periods.extend(
    (str(year), part)
    for year, part in validation.groupby(validation.index.year)
)

for period, part in periods:
    y = part["target"].astype(int)

    for name in predictions.columns:
        p = predictions.loc[part.index, name]

        rows.append({
            "period": period,
            "model": name,
            "examples": len(part),
            "positive_rate": y.mean(),
            "mean_prediction": p.mean(),
            "brier": brier_score_loss(y, p),
            "roc_auc": (
                roc_auc_score(y, p)
                if y.nunique() == 2 else float("nan")
            ),
            "average_precision": (
                average_precision_score(y, p)
                if y.nunique() == 2 else float("nan")
            ),
        })

comparison = pd.DataFrame(rows)

print(comparison.round(6).to_string(index=False))

output_folder = (
    project_folder / "results" / "volatility_only_experiment"
)
output_folder.mkdir(parents=True, exist_ok=True)

comparison.to_csv(output_folder / "validation_comparison.csv", index=False)
predictions["target"] = validation["target"].astype(int)

predictions.to_csv(
    output_folder / "validation_predictions.csv",
    index_label="date",
)
print("\nValidation comparison saved.")