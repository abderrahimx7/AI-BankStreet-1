from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data_folder = Path(__file__).resolve().parent / "data"

kbe = pd.read_csv(
    data_folder / "kbe_prices_v1.csv",
    index_col="Date",
    parse_dates=["Date"]
).sort_index()

analysis = pd.DataFrame(index=kbe.index)

analysis["close"] = kbe["Close"]

analysis["running_peak"] = (
    analysis["close"].cummax()
)

analysis["drawdown"] = (
    analysis["close"] / analysis["running_peak"] - 1
)

print(analysis.head(10))

maximum_drawdown = analysis["drawdown"].min()
trough_date = analysis["drawdown"].idxmin()

print(f"\nMaximum drawdown: {maximum_drawdown:.2%}")
print("Trough date:", trough_date.date())


# Peak associated with the deepest drawdown
peak_value = analysis.loc[trough_date, "running_peak"]

history_to_trough = analysis.loc[:trough_date, "close"]

# If the same peak occurred several times, use the latest occurrence
peak_date = history_to_trough.loc[
    history_to_trough == peak_value
].index[-1]

trough_value = analysis.loc[trough_date, "close"]

# First close at or above the old peak after the trough
after_trough = analysis.loc[
    analysis.index > trough_date,
    "close"
]

recovered = after_trough.loc[after_trough >= peak_value]

print("\nDeepest drawdown episode:")
print("Peak date:", peak_date.date())
print("Trough date:", trough_date.date())
print(f"Peak value: {peak_value:.4f}")
print(f"Trough value: {trough_value:.4f}")

print(
    "Peak to trough — calendar days:",
    (trough_date - peak_date).days
)

if recovered.empty:
    print("Recovery: not reached by the end of the dataset.")
else:
    recovery_date = recovered.index[0]

    print("Recovery date:", recovery_date.date())
    print(
        "Peak to recovery — calendar days:",
        (recovery_date - peak_date).days
    )
    print(
        "Trough to recovery — calendar days:",
        (recovery_date - trough_date).days
    )


    if not recovered.empty:
        episode = analysis.loc[peak_date:recovery_date]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True
    )

    axes[0].plot(
        episode.index,
        episode["close"],
        color="blue",
        label="Adjusted close"
    )

    axes[0].axhline(
        peak_value,
        color="gray",
        linestyle="--",
        label="Previous peak"
    )

    for date, label, color in [
        (peak_date, "Peak", "green"),
        (trough_date, "Trough", "red"),
        (recovery_date, "Recovery", "purple")
    ]:
        axes[0].scatter(
            date,
            analysis.loc[date, "close"],
            color=color,
            label=label,
            zorder=3
        )

    axes[0].set_title("KBE — deepest drawdown episode")
    axes[0].set_ylabel("Adjusted price")
    axes[0].legend()

    drawdown_percent = episode["drawdown"] * 100

    axes[1].fill_between(
        episode.index,
        drawdown_percent.to_numpy(),
        0,
        color="red",
        alpha=0.3
    )

    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")

    for axis in axes:
        axis.grid(alpha=0.25)

    plt.tight_layout()
charts_folder = (
        Path(__file__).resolve().parent
        / "results"
        / "risk_analysis"
    )
charts_folder.mkdir(parents=True, exist_ok=True)

fig.savefig(
        charts_folder / "kbe_deepest_drawdown.png",
        dpi=200,
        bbox_inches="tight"
    )
plt.show()