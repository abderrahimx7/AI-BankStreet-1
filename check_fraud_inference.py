from pathlib import Path

import numpy as np
import pandas as pd

from bankstreet.fraud.inference import score_transactions


project_folder = Path(__file__).resolve().parent

prices_path = (
    project_folder / "data" / "fraud" / "creditcard.csv"
)

reference_path = (
    project_folder
    / "results"
    / "fraud"
    / "xgboost_v1"
    / "test_predictions.csv"
)

data = pd.read_csv(prices_path)
reference = pd.read_csv(reference_path).head(1000)

# Recover exactly the transactions used in the saved evaluation.
sample = (
    data.loc[reference["row_id"].to_numpy()]
    .drop(columns=["Class"])
    .reset_index(drop=True)
)

scored, threshold = score_transactions(sample, project_folder)

np.testing.assert_allclose(
    scored["model_score"].to_numpy(),
    reference["probability"].to_numpy(),
    rtol=1e-6,
    atol=1e-8,
)

np.testing.assert_array_equal(
    scored["review_required"].to_numpy(),
    reference["alert"].to_numpy(dtype=bool),
)

print("Scores and alerts match the saved test evaluation.")


def expect_rejection(invalid_data, message):
    try:
        score_transactions(invalid_data, project_folder)
    except ValueError:
        print(message)
    else:
        raise AssertionError("Invalid data was accepted.")


expect_rejection(
    sample.drop(columns=["Amount"]),
    "Missing Amount column rejected.",
)

missing_value = sample.head(1).copy()
missing_value.loc[0, "V1"] = np.nan

expect_rejection(
    missing_value,
    "Missing feature value rejected.",
)

negative_amount = sample.head(1).copy()
negative_amount.loc[0, "Amount"] = -1

expect_rejection(
    negative_amount,
    "Negative Amount rejected.",
)

print("Fraud inference checks passed.")