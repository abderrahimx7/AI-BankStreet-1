from pathlib import Path
import numpy as np
import pandas as pd

results_folder = (
    Path(__file__).resolve().parent
    / "results"
    / "market_return_experiment"
)

predictions = pd.read_csv(
    results_folder / "annual_predictions.csv",
    index_col="Date",
    parse_dates=["Date"]
)

selected = predictions.loc[
    predictions["version"] == "KBE only"
].sort_index().copy()

assert len(selected) == 751
assert selected.index.is_unique

selected["gain"] = (
    (selected["baseline_probability"] - selected["actual"]) ** 2
    - (selected["probability"] - selected["actual"]) ** 2
)

# Positive gain means the model has lower squared error.
observed_gain = selected["gain"].mean()

year_groups = [
    part["gain"].to_numpy()
    for _, part in selected.groupby("year")
]


def bootstrap_interval(groups, block_length, repetitions=2000):
    rng = np.random.default_rng(42)
    total_examples = sum(len(group) for group in groups)
    bootstrap_means = np.empty(repetitions)

    for repetition in range(repetitions):
        sampled_sum = 0.0

        for values in groups:
            n = len(values)
            number_of_blocks = (
                n + block_length - 1
            ) // block_length

            starts = rng.integers(
                0, n, size=number_of_blocks
            )

            indices = (
                starts[:, None] + np.arange(block_length)
            ) % n

            sampled_indices = indices.ravel()[:n]
            sampled_sum += values[sampled_indices].sum()

        bootstrap_means[repetition] = (
            sampled_sum / total_examples
        )

    return np.quantile(bootstrap_means, [0.025, 0.975])


rows = []

for block_length in [5, 20, 60]:
    lower, upper = bootstrap_interval(
        year_groups,
        block_length
    )

    rows.append({
        "block_sessions": block_length,
        "observed_gain": observed_gain,
        "lower_95": lower,
        "upper_95": upper
    })

intervals = pd.DataFrame(rows)

print("\nApproximate bootstrap intervals:")
print(intervals.round(6).to_string(index=False))

intervals.to_csv(
    results_folder / "brier_gain_intervals.csv",
    index=False
)



conclusion = (
    "The annual walk-forward model showed a mean Brier gain "
    "of approximately 0.003796 over its baseline on 2020-2022. "
    "All approximate 95% block-bootstrap intervals included zero. "
    "The evidence does not establish a reliable improvement. "
    "These intervals are conditional on the saved predictions "
    "and do not account for model-selection uncertainty. "
    "The original frozen model also underperformed its baseline "
    "on the 2023-2025 test period."
)

(results_folder / "conclusion.txt").write_text(
    conclusion,
    encoding="utf-8"
)

print("\nConclusion saved.")