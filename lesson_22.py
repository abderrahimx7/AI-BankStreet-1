from pathlib import Path

import numpy as np
import pandas as pd


project_folder = Path(__file__).resolve().parent
experiment_folder = (
    project_folder / "results" / "volatility_only_experiment"
)

data = pd.read_csv(
    experiment_folder / "validation_predictions.csv",
    parse_dates=["date"],
).sort_values("date")

if data.empty or data["date"].duplicated().any():
    raise ValueError("Expected non-empty data with unique dates.")

if not (
    (data["date"] >= pd.Timestamp("2020-01-01"))
    & (data["date"] < pd.Timestamp("2023-01-01"))
).all():
    raise ValueError("Expected validation dates only.")

y = data["target"].to_numpy(dtype=float)
p_three = data["Three features"].to_numpy(dtype=float)
p_volatility = data["Volatility only"].to_numpy(dtype=float)

if not np.isin(y, [0, 1]).all():
    raise ValueError("Invalid targets.")

for probabilities in [p_three, p_volatility]:
    if not (
        np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Invalid probabilities.")

# Paired loss difference: both models evaluated on the same session.
gain_per_session = (
    (p_three - y) ** 2
    - (p_volatility - y) ** 2
)

observed_gain = gain_per_session.mean()
n = len(data)
repetitions = 5000
results = []

for block_size in [5, 20, 60]:
    rng = np.random.default_rng(42 + block_size)

    blocks_needed = (n + block_size - 1) // block_size
    bootstrap_gains = np.empty(repetitions)

    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)

        indices = (
            starts[:, None] + np.arange(block_size)
        ) % n

        indices = indices.ravel()[:n]

        bootstrap_gains[repetition] = (
            gain_per_session[indices].mean()
        )

    lower, upper = np.quantile(
        bootstrap_gains, [0.025, 0.975]
    )

    results.append({
        "block_sessions": block_size,
        "observed_gain": observed_gain,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "includes_zero": bool(lower <= 0 <= upper),
    })

summary = pd.DataFrame(results)

print("Validation sessions:", n)
print("Bootstrap repetitions:", repetitions)
print(summary.round(6).to_string(index=False))

summary.to_csv(
    experiment_folder / "brier_gain_bootstrap.csv",
    index=False,
)

print("Block bootstrap comparison saved.")