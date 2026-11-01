from pathlib import Path

import joblib
import pandas as pd

from bankstreet.features import build_features
from bankstreet.labels import build_labels
from bankstreet.splits import split_by_time
from bankstreet.evaluation import evaluate_model


project_folder = Path(__file__).resolve().parent
experiment_folder = project_folder / "results" / "experiment_v1"

bundle = joblib.load(experiment_folder / "kbe_model.joblib")

model = bundle["pipeline"]
metadata = bundle["metadata"]
feature_columns = metadata["features"]

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

rows = []

for year, part in validation.groupby(validation.index.year):
    metrics = evaluate_model(
        model,
        part,
        feature_columns,
        metadata["baseline_probability"],
    )

    probabilities = model.predict_proba(
        part[feature_columns]
    )[:, list(model.classes_).index(1)]

    rows.append({
        "year": year,
        "examples": len(part),
        "positive_rate": part["target"].astype(int).mean(),
        "mean_prediction": probabilities.mean(),
        "baseline_brier": metrics.loc["Baseline", "brier"],
        "model_brier": metrics.loc["Logistic", "brier"],
        "brier_gain": (
            metrics.loc["Baseline", "brier"]
            - metrics.loc["Logistic", "brier"]
        ),
        "roc_auc": metrics.loc["Logistic", "roc_auc"],
        "average_precision": metrics.loc[
            "Logistic", "average_precision"
        ],
    })

annual = pd.DataFrame(rows).set_index("year")

classifier = model.named_steps["logisticregression"]

weights = pd.DataFrame({
    "feature": feature_columns,
    "standardized_coefficient": classifier.coef_[0],
})

print("\nValidation — fixed saved model:")
print(annual.round(6).to_string())

print("\nModel coefficients:")
print(weights.to_string(index=False))

output_folder = project_folder / "results" / "diagnostics_v1"
output_folder.mkdir(parents=True, exist_ok=True)

annual.to_csv(output_folder / "validation_by_year.csv")
weights.to_csv(output_folder / "coefficients.csv", index=False)

print("\nValidation diagnostics saved.")