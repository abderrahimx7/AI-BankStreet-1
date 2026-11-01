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
feature_columns = metadata["features"]
fixed_model = bundle["pipeline"]

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

dataset = dataset.dropna(
    subset=feature_columns + ["target", "label_end_date"]
)

_, validation, _ = split_by_time(
    dataset,
    validation_start=metadata["validation_start"],
    test_start=metadata["test_start"],
)

fold_predictions = []

for year, part in validation.groupby(validation.index.year):
    cutoff = pd.Timestamp(f"{year}-01-01")

    fold_train = dataset.loc[
        (dataset.index < cutoff)
        & (dataset["label_end_date"] < cutoff)
    ].copy()

    if fold_train.empty:
        raise ValueError(f"No training rows for {year}.")

    assert fold_train["label_end_date"].max() < part.index.min()

    annual_model, annual_baseline = train_model(
        fold_train,
        feature_columns,
    )

    fixed_positive = list(fixed_model.classes_).index(1)
    annual_positive = list(annual_model.classes_).index(1)

    fold = pd.DataFrame(index=part.index)
    fold["target"] = part["target"].astype(int)

    fold["Fixed baseline"] = metadata["baseline_probability"]
    fold["Annual baseline"] = annual_baseline

    fold["Fixed model"] = fixed_model.predict_proba(
        part[feature_columns]
    )[:, fixed_positive]

    fold["Annual model"] = annual_model.predict_proba(
        part[feature_columns]
    )[:, annual_positive]

    fold_predictions.append(fold)

    print(
        f"{year}: train={len(fold_train)}, "
        f"evaluation={len(part)}, "
        f"last training label={fold_train['label_end_date'].max().date()}"
    )

predictions = pd.concat(fold_predictions).sort_index()

assert predictions.index.equals(validation.index)
assert not predictions.isna().any().any()

model_columns = [
    "Fixed baseline",
    "Annual baseline",
    "Fixed model",
    "Annual model",
]

periods = [("All validation", predictions)]
periods.extend(
    (str(year), part)
    for year, part in predictions.groupby(predictions.index.year)
)

rows = []

for period, part in periods:
    y = part["target"]

    for name in model_columns:
        p = part[name]

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

output_folder = (
    project_folder / "results" / "annual_retraining_experiment"
)
output_folder.mkdir(parents=True, exist_ok=True)

predictions.to_csv(
    output_folder / "validation_predictions.csv",
    index_label="date",
)

comparison.to_csv(
    output_folder / "validation_comparison.csv",
    index=False,
)

print("\n", comparison.round(6).to_string(index=False))
print("\nAnnual retraining comparison saved.")