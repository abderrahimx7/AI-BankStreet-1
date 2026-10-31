from pathlib import Path
import pandas as pd

from bankstreet.features import build_features
from bankstreet.labels import build_labels
from bankstreet.splits import split_by_time
from bankstreet.modeling import train_model
from bankstreet.evaluation import evaluate_model


# 1. Load saved prices
project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

kbe = pd.read_csv(
    data_folder / "kbe_prices_v1.csv",
    index_col="Date",
    parse_dates=["Date"]
).sort_index()

spy = pd.read_csv(
    data_folder / "spy_prices_v1.csv",
    index_col="Date",
    parse_dates=["Date"]
).sort_index()

assert kbe.index.is_unique
assert spy.index.is_unique

print("Same dates and order:", kbe.index.equals(spy.index))

aligned_prices = kbe[["Close"]].rename(
    columns={"Close": "kbe_close"}
).join(
    spy[["Close"]].rename(columns={"Close": "spy_close"}),
    how="left"
)

print("\nRows:", len(aligned_prices))
print("\nMissing prices:")
print(aligned_prices.isna().sum())


# 2. Build features and labels
features = build_features(kbe)

features["market_return"] = (
    aligned_prices["spy_close"].pct_change(fill_method=None)
)

labels = build_labels(
    kbe,
    horizon=5,
    threshold=-0.03
)

base_features = [
    "daily_return",
    "momentum",
    "volatility_20d"
]

market_features = base_features + ["market_return"]

dataset = features.join(labels)

complete_dataset = dataset.dropna(
    subset=market_features + ["target", "label_end_date"]
).copy()


# 3. Fixed training-period comparison
train, validation, test = split_by_time(complete_dataset)

versions = [
    ("KBE only", base_features),
    ("KBE + market", market_features)
]

print("\nSame rows — adding market return:")

for name, columns in versions:
    experiment_model, baseline = train_model(
        train,
        columns
    )

    metrics = evaluate_model(
        experiment_model,
        validation,
        columns,
        baseline
    )

    print(f"\n{name}")
    print(metrics.round(6))


# 4. Annual retraining and prediction collection
annual_results = []
prediction_rows = []

for year in [2020, 2021, 2022]:
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year + 1}-01-01")

    fold_train = complete_dataset.loc[
        (complete_dataset.index < year_start)
        & (complete_dataset["label_end_date"] < year_start)
    ].copy()

    fold_validation = complete_dataset.loc[
        (complete_dataset.index >= year_start)
        & (complete_dataset.index < year_end)
        & (
            complete_dataset["label_end_date"]
            < pd.Timestamp("2023-01-01")
        )
    ].copy()

    for name, columns in versions:
        fold_model, baseline = train_model(
            fold_train,
            columns
        )

        metrics = evaluate_model(
            fold_model,
            fold_validation,
            columns,
            baseline
        )

        annual_results.append({
            "year": year,
            "version": name,
            "examples": len(fold_validation),
            "baseline_brier": metrics.loc["Baseline", "brier"],
            "model_brier": metrics.loc["Logistic", "brier"],
            "roc_auc": metrics.loc["Logistic", "roc_auc"],
            "ap": metrics.loc["Logistic", "average_precision"]
        })

        fold_predictions = pd.DataFrame(
            index=fold_validation.index
        )

        fold_predictions["version"] = name
        fold_predictions["year"] = year
        fold_predictions["actual"] = (
            fold_validation["target"].astype(int)
        )
        fold_predictions["probability"] = (
            fold_model.predict_proba(
                fold_validation[columns]
            )[:, 1]
        )
        fold_predictions["baseline_probability"] = baseline

        prediction_rows.append(fold_predictions)


# 5. Display annual results
annual_results = pd.DataFrame(annual_results)

print("\nAnnual comparison:")
print(annual_results.round(6).to_string(index=False))


# 6. Calculate weighted Brier scores
summary_rows = []

for version, part in annual_results.groupby("version"):
    total_examples = part["examples"].sum()

    summary_rows.append({
        "version": version,
        "examples": total_examples,
        "baseline_brier": (
            part["baseline_brier"] * part["examples"]
        ).sum() / total_examples,
        "model_brier": (
            part["model_brier"] * part["examples"]
        ).sum() / total_examples
    })

summary = pd.DataFrame(summary_rows).set_index("version")

print("\nOverall walk-forward Brier:")
print(summary.round(6))


# 7. Combine and check individual predictions
all_predictions = pd.concat(prediction_rows)
all_predictions.index.name = "Date"

assert len(annual_results) == 6
assert len(all_predictions) == 1502

assert not (
    all_predictions.reset_index()
    .duplicated(["Date", "version"])
    .any()
)


# 8. Save results
results_folder = (
    project_folder
    / "results"
    / "market_return_experiment"
)

results_folder.mkdir(parents=True, exist_ok=True)

annual_results.to_csv(
    results_folder / "annual_comparison.csv",
    index=False
)

summary.to_csv(
    results_folder / "walk_forward_summary.csv"
)

all_predictions.to_csv(
    results_folder / "annual_predictions.csv"
)

print("\nSaved:", results_folder)
print("Prediction rows saved:", len(all_predictions))