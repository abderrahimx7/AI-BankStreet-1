import yfinance as yf 
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
print("RUNNING FILE:", __file__)
print("VERSION: baseline-fixed")
df = yf.download(
    "KBE",
    start = "2010-01-01",

    end = "2026-01-01",
    interval = "1d",
    auto_adjust = True,
    multi_level_index = False,
    threads = False,
    timeout = 30,
    progress = False

)
if df is None or df.empty:
    raise RuntimeError(
        "KBE download failed. No data recieved"
    )
df = df.sort_index()

print("First rows :")
print(df.head())

print("\nLast rows :")
print(df.tail())

print("\nNumber of sessions :", len(df))

print("\nMissing values :")
print(df.isna().sum())
df ["daily_return"] = df["Close"].pct_change(fill_method=None)

df["momentum"] = (
    df["Close"] / df["Close"].shift(3) - 1

)
df["future_return_5d"] = (
    df["Close"].shift(-5) / df["Close"] - 1
)
df["target"]= pd.Series(

    pd.NA, index=df .index, dtype="Int64"
)
known_future = df["future_return_5d"].notna()

df.loc[known_future, "target"] = (
    df.loc[known_future, "future_return_5d"] <= -0.03
).astype(int)

df["volatility_20d"] = (
    df["daily_return"]
    .rolling(window=20, min_periods=20)
    .std(ddof=1)

)

feature_columns = ["daily_return",
    "momentum",
    "volatility_20d"]

training_data = df.dropna(
    subset=feature_columns + ["target"]
).copy()

X = training_data[feature_columns]
y = training_data["target"].astype(int)

print("\nComplete examples:", len(training_data))

print("\nTarget counts:")
print(y.value_counts().sort_index())

print(f"\nPositive rate: {y.mean():.2%}")

label_end_date = pd.Series(
    df.index, index=df.index
).shift(-5)

training_data["label_end_date"] = label_end_date.reindex(
    training_data.index
)
validation_start = pd.Timestamp("2020-01-01")
test_start = pd.Timestamp("2023-01-01")

train = training_data.loc[
    (training_data.index < validation_start)
    & (training_data["label_end_date"] < validation_start)
].copy()

validation = training_data.loc[
    (training_data.index >= validation_start)
    & (training_data.index < test_start)
    & (training_data["label_end_date"] < test_start)
].copy()
test = training_data.loc[
    training_data.index >= test_start
].copy()



for name, part in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(
       f"{name}: {len(part)} examples | "
       f"{part.index.min().date()} ->  "
       f"{part.index.max().date()}"
    )

    assert train["label_end_date"].max() < validation.index.min()
    assert validation["label_end_date"].max() < test.index.min()

baseline_probability = train["target"].astype(int).mean()

y_validation = validation["target"].astype(int)

baseline_predictions = pd.Series(
    baseline_probability,
    index=validation.index
)

brier_baseline = (
    (baseline_predictions - y_validation) ** 2
).mean()

print(f"\nBaseline probability: {baseline_probability:.2%}")
print(f"Validation positive rate: {y_validation.mean():.2%}")
print(f"Baseline Brier score: {brier_baseline:.6f}")


X_train = train[feature_columns]
y_train = train["target"].astype(int)


X_validation = validation[feature_columns]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)

)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_validation)

print("\nClass order :", model.classes_)

model_predictions = pd.Series(
    probabilities[:, 1],
    index=validation.index

)

brier_model = (
    (model_predictions - y_validation) ** 2
).mean()

print(f"\nBaseline Brier : {brier_baseline:.6f}")
print(f"Logistic Brier : {brier_model:.6f}")

print("\nFirst predicted probabilities :")
print(model_predictions.head())


train_predictions = pd.Series(
    model.predict_proba(X_train)[:, 1],
    index=train.index
)

train_brier_baseline = (
    (baseline_probability - y_train) ** 2
).mean()

train_brier_model = (
    (train_predictions - y_train) ** 2
).mean()

print("\nTrain comparison:")
print(f"Baseline: {train_brier_baseline:.6f}")
print(f"Logistic: {train_brier_model:.6f}")

print(
    "\nMean predicted probability on Validation:",
    f"{model_predictions.mean():.2%}"
)
expected_y_train = train["target"].astype(int)
p_train = expected_y_train.mean()

print("\nCHECK:")
print("Train rows:", len(train))
print("X_train rows:", len(X_train))
print("y_train rows:", len(y_train))

print("Mean from train:", p_train)
print("Mean from y_train:", y_train.mean())
print("Stored baseline:", baseline_probability)

print(
    "X index matches:",
    X_train.index.equals(train.index)
)

print(
    "y matches:",
    y_train.equals(expected_y_train)
)

print(
    "Recomputed Train baseline Brier:",
    ((p_train - expected_y_train) ** 2).mean()
)

print("Expected p * (1-p):", p_train * (1 - p_train))


features_without_volatility = ["daily_return", "momentum"]

model_without_volatility = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model_without_volatility.fit(
    train[features_without_volatility],
    y_train
)

predictions_without_volatility = pd.Series(
    model_without_volatility.predict_proba(
        validation[features_without_volatility]
    )[:, 1],
    index=validation.index
)

brier_without_volatility = (
    (predictions_without_volatility - y_validation) ** 2
).mean()

print("\nSame rows — Validation comparison:")
print(f"Baseline:           {brier_baseline:.6f}")
print(f"Without volatility: {brier_without_volatility:.6f}")
print(f"With volatility:    {brier_model:.6f}")

print("\nValidation ranking metrics:")

for name, predictions in [
    ("Baseline", baseline_predictions),
    ("Without volatility", predictions_without_volatility),
    ("With volatility", model_predictions)
]:
    auc = roc_auc_score(y_validation, predictions)
    ap = average_precision_score(y_validation, predictions)

    print(f"{name}: ROC-AUC={auc:.4f} | AP={ap:.4f}")

    validation_results = pd.DataFrame({
    "actual": y_validation,
    "probability": model_predictions,
    "baseline": baseline_predictions
})

print("\nValidation — year by year:")

for year, part in validation_results.groupby(
    validation_results.index.year
):
    actual = part["actual"]
    predictions = part["probability"]

    model_brier = ((predictions - actual) ** 2).mean()
    baseline_brier = ((part["baseline"] - actual) ** 2).mean()

    print(f"\nYear: {year} | Examples: {len(part)}")
    print(f"Actual positive rate: {actual.mean():.2%}")
    print(f"Mean prediction:      {predictions.mean():.2%}")
    print(f"Baseline Brier:       {baseline_brier:.6f}")
    print(f"Model Brier:          {model_brier:.6f}")

    if actual.nunique() == 2:
        auc = roc_auc_score(actual, predictions)
        ap = average_precision_score(actual, predictions)

        print(f"ROC-AUC: {auc:.4f}")
        print(f"AP: {ap:.4f}")


# New feature: return over the previous 20 sessions
df["momentum_20d"] = (
    df["Close"] / df["Close"].shift(20) - 1
)

# Same rows as the existing experiment
train_extended = train.join(df[["momentum_20d"]])
validation_extended = validation.join(df[["momentum_20d"]])

extended_features = feature_columns + ["momentum_20d"]

assert train_extended[extended_features].notna().all().all()
assert validation_extended[extended_features].notna().all().all()

model_extended = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model_extended.fit(
    train_extended[extended_features],
    y_train
)

extended_predictions = pd.Series(
    model_extended.predict_proba(
        validation_extended[extended_features]
    )[:, 1],
    index=validation.index
)

print("\nValidation — adding 20-session momentum:")

for name, predictions in [
    ("Current model", model_predictions),
    ("Extended model", extended_predictions)
]:
    brier = ((predictions - y_validation) ** 2).mean()
    auc = roc_auc_score(y_validation, predictions)
    ap = average_precision_score(y_validation, predictions)

    print(f"{name}: Brier={brier:.6f} | AUC={auc:.4f} | AP={ap:.4f}")    

    tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=100,
    random_state=42
)

# Same three features and same training rows
tree_model.fit(X_train, y_train)

tree_predictions = pd.Series(
    tree_model.predict_proba(X_validation)[:, 1],
    index=validation.index
)

print("\nValidation — model comparison:")

for name, predictions in [
    ("Baseline", baseline_predictions),
    ("Logistic", model_predictions),
    ("Decision tree", tree_predictions)
]:
    brier = ((predictions - y_validation) ** 2).mean()
    auc = roc_auc_score(y_validation, predictions)
    ap = average_precision_score(y_validation, predictions)

    print(
        f"{name}: Brier={brier:.6f} | "
        f"AUC={auc:.4f} | AP={ap:.4f}"
    )



# Evaluate the selected, already-trained model
selected_features = [
    "daily_return",
    "momentum",
    "volatility_20d"
]

assert feature_columns == selected_features

X_test = test[selected_features]
y_test = test["target"].astype(int)

test_predictions = pd.Series(
    model.predict_proba(X_test)[:, 1],
    index=test.index
)

# Keep the probability learned from Train
test_baseline_predictions = pd.Series(
    baseline_probability,
    index=test.index
)

print("\nFINAL TEST — selected model:")
print(f"Examples: {len(test)}")
print(f"Actual positive rate: {y_test.mean():.2%}")
print(f"Mean model prediction: {test_predictions.mean():.2%}")

for name, predictions in [
    ("Baseline", test_baseline_predictions),
    ("Selected Logistic", test_predictions)
]:
    brier = ((predictions - y_test) ** 2).mean()
    auc = roc_auc_score(y_test, predictions)
    ap = average_precision_score(y_test, predictions)

    print(
        f"{name}: Brier={brier:.6f} | "
        f"AUC={auc:.4f} | AP={ap:.4f}"
    )


scaler = model.named_steps["standardscaler"]
logistic = model.named_steps["logisticregression"]

learned_parameters = pd.DataFrame({
    "feature": feature_columns,
    "train_mean": scaler.mean_,
    "train_std": scaler.scale_,
    "coefficient": logistic.coef_[0]
})

print("\nWhat the model learned:")
print(learned_parameters.to_string(index=False))

print("\nIntercept:", logistic.intercept_[0])




example = X_validation.iloc[[0]]

scaled = scaler.transform(example)
contributions = scaled[0] * logistic.coef_[0]

z = logistic.intercept_[0] + contributions.sum()
manual_probability = 1 / (1 + np.exp(-z))

details = pd.DataFrame({
    "feature": feature_columns,
    "raw_value": example.iloc[0].to_numpy(),
    "scaled_value": scaled[0],
    "contribution_to_z": contributions
})

print("\nPrediction explained:", example.index[0].date())
print(details.to_string(index=False))
print("z:", z)
print(f"Manual probability: {manual_probability:.6%}")
print(f"Pipeline probability: {model.predict_proba(example)[0, 1]:.6%}")



# Inspect one training example; do not change the fitted model
training_example = X_train.iloc[[0]]
actual = int(y_train.iloc[0])

scaled_example = scaler.transform(training_example)[0]
probability = model.predict_proba(training_example)[0, 1]

error_signal = probability - actual
weight_gradients = error_signal * scaled_example

print("\nOne training example:")
print("Actual answer:", actual)
print(f"Predicted probability: {probability:.6f}")
print(f"Intercept gradient: {error_signal:.6f}")

for feature, gradient in zip(feature_columns, weight_gradients):
    print(f"{feature}: gradient={gradient:.6f}")


    learning_rate = 0.01

demo_weights = logistic.coef_[0].copy()
demo_intercept = float(logistic.intercept_[0])

demo_weights -= learning_rate * weight_gradients
demo_intercept -= learning_rate * error_signal

new_z = demo_intercept + np.dot(
    demo_weights, scaled_example
)

new_probability = 1 / (1 + np.exp(-new_z))


assert actual == 1

loss_before = -np.log(probability)
loss_after = -np.log(new_probability)

print("\nOne educational update:")
print(f"Probability before: {probability:.6%}")
print(f"Probability after:  {new_probability:.6%}")
print(f"Loss before: {loss_before:.6f}")
print(f"Loss after:  {loss_after:.6f}")


from pathlib import Path

data_folder = Path(__file__).resolve().parent / "data"
data_folder.mkdir(exist_ok=True)

prices_path = data_folder / "kbe_prices_v1.csv"

price_columns = ["Open", "High", "Low", "Close", "Volume"]

if not prices_path.exists():
    df[price_columns].to_csv(
        prices_path,
        index=True,
        index_label="Date"
    )
    print("\nSaved prices:", prices_path)
else:
    print("\nExisting snapshot preserved:", prices_path)