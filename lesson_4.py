from pathlib import Path
import pandas as pd
from bankstreet.features import build_features
from bankstreet.labels import build_labels
from bankstreet.splits import split_by_time
from bankstreet.modeling import train_model
from bankstreet.evaluation import evaluate_model
import json 
import hashlib


prices_path = (
    Path(__file__).resolve().parent
    / "data"
    / "kbe_prices_v1.csv"
)

prices = pd.read_csv(
    prices_path,
    index_col="Date",
    parse_dates=["Date"]
).sort_index()

feature_columns = [
    "daily_return",
    "momentum",
    "volatility_20d"
]

features = build_features(prices)

labels = build_labels(
    prices,
    horizon=5,
    threshold=-0.03
)

dataset = features.join(labels)

complete_dataset = dataset.dropna(
    subset=feature_columns + ["target", "label_end_date"]
).copy()

train, validation, test = split_by_time(complete_dataset)

for name, part in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(f"{name}: {len(part)} examples")

assert train["label_end_date"].max() < validation.index.min()
assert validation["label_end_date"].max() < test.index.min()

model, baseline_probability = train_model(
    train,
    feature_columns
)

validation_metrics = evaluate_model(
    model,
    validation,
    feature_columns,
    baseline_probability
)

test_metrics = evaluate_model(
    model,
    test,
    feature_columns,
    baseline_probability
)

print("\nValidation:")
print(validation_metrics.round(6))

print("\nTest:")
print(test_metrics.round(6))



results_folder = (
    Path(__file__).resolve().parent
    / "results"
    / "experiment_v1"
)

results_folder.mkdir(parents=True, exist_ok=True)

validation_metrics.to_csv(
    results_folder / "validation_metrics.csv"
)

test_metrics.to_csv(
    results_folder / "test_metrics.csv"
)

metadata = {
    "ticker": "KBE",
    "prices_file": prices_path.name,
    "prices_sha256": hashlib.sha256(
        prices_path.read_bytes()
    ).hexdigest(),
    "features": feature_columns,
    "horizon_sessions": 5,
    "target_threshold": -0.03,
    "validation_start": "2020-01-01",
    "test_start": "2023-01-01",
    "model": "StandardScaler + LogisticRegression",
    "max_iter": 1000,
    "train_examples": len(train),
    "validation_examples": len(validation),
    "test_examples": len(test),
    "baseline_probability": float(baseline_probability)
}

with (results_folder / "metadata.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=2)

print("\nExperiment saved:", results_folder.resolve())


print("\nWalk-forward — annual retraining:")

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

    fold_model, fold_baseline = train_model(
        fold_train,
        feature_columns
    )

    fold_metrics = evaluate_model(
        fold_model,
        fold_validation,
        feature_columns,
        fold_baseline
    )

    print(f"\nYear: {year}")
    print("Training examples:", len(fold_train))
    print("Evaluation examples:", len(fold_validation))
    print(fold_metrics.round(6))