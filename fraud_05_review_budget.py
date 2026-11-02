from pathlib import Path

import numpy as np
import pandas as pd


project_folder = Path(__file__).resolve().parent
fraud_results = project_folder / "results" / "fraud"

files = {
    "Logistic": fraud_results / "logistic_v1" / "validation_predictions.csv",
    "MLP": fraud_results / "mlp_v1" / "validation_predictions.csv",
    "XGBoost": fraud_results / "xgboost_v1" / "validation_predictions.csv",
}

budget = 94
reference = None
rows = []

output_folder = fraud_results / "review_budget_comparison"
output_folder.mkdir(parents=True, exist_ok=True)

for name, path in files.items():
    data = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)

    if len(data) < budget or data["row_id"].duplicated().any():
        raise ValueError(f"{name}: invalid rows or review budget.")

    if not data["target"].isin([0, 1]).all():
        raise ValueError(f"{name}: invalid targets.")

    probabilities = data["probability"].to_numpy(dtype=float)

    if not (
        np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError(f"{name}: invalid probabilities.")

    identity = data[["row_id", "Time", "target"]]

    if reference is None:
        reference = identity.copy()
    else:
        pd.testing.assert_frame_equal(reference, identity)

    # Deterministic tie-break: original row order.
    ranked = data.sort_values(
        ["probability", "row_id"],
        ascending=[False, True],
    )

    selected = ranked.head(budget)

    total_frauds = int(data["target"].sum())

    if total_frauds == 0:
        raise ValueError("No fraud cases available.")

    tp = int(selected["target"].sum())
    fp = budget - tp
    fn = total_frauds - tp

    cutoff = float(selected["probability"].iloc[-1])
    above_cutoff = int((data["probability"] > cutoff).sum())
    at_cutoff = int((data["probability"] == cutoff).sum())

    rows.append({
        "model": name,
        "reviews": budget,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision_at_94": tp / budget,
        "recall_at_94": tp / total_frauds,
        "alerts_per_1000": 1000 * budget / len(data),
        "boundary_tie": above_cutoff + at_cutoff > budget,
    })

    selected.to_csv(
        output_folder / f"{name.lower()}_top94.csv",
        index=False,
    )

summary = pd.DataFrame(rows).set_index("model")

print("Validation transactions:", len(reference))
print("Actual frauds:", int(reference["target"].sum()))
print(summary.round(4).to_string())

summary.to_csv(output_folder / "validation_top94_comparison.csv")

print("\nReview budget comparison saved.")