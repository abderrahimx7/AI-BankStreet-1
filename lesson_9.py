from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from bankstreet.risk import calculate_rolling_volatility

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

prices = {}

for ticker in ["KBE", "SPY"]:
    table = pd.read_csv(
        data_folder / f"{ticker.lower()}_prices_v1.csv",
        index_col="Date",
        parse_dates=["Date"]
    ).sort_index()

    assert table.index.is_unique
    prices[ticker] = table["Close"]

assert prices["KBE"].index.equals(prices["SPY"].index)

close = pd.DataFrame(prices)

rolling_volatility = calculate_rolling_volatility(
    close,
    window=20
)

# Display the same period as the drawdown comparison
period_volatility = rolling_volatility.loc[
    "2018-03-09":"2021-02-16"
]

summary = pd.DataFrame({
    "max_daily_volatility_pct": period_volatility.max() * 100,
    "date_of_max": period_volatility.idxmax()
})

print("\n20-session rolling daily volatility:")
print(
    summary.round({
        "max_daily_volatility_pct": 2
    }).to_string()
)
valid_volatility = period_volatility.dropna()

higher_volatility_share = (
    valid_volatility["KBE"] > valid_volatility["SPY"]
).mean()

print(
    "\nShare of sessions where KBE volatility exceeded SPY:",
    f"{higher_volatility_share:.2%}"
)

print("\nExact maximum daily volatility (%):")
print((valid_volatility.max() * 100).round(2))

fig, ax = plt.subplots(figsize=(11, 5))

(period_volatility * 100).plot(ax=ax)

ax.set_title("KBE vs SPY — 20-session rolling volatility")
ax.set_ylabel("Daily volatility (%)")
ax.set_xlabel("Date")
ax.grid(alpha=0.25)

plt.tight_layout()

output_folder = project_folder / "results" / "risk_analysis"
output_folder.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_folder / "rolling_volatility_20d.png",
    dpi=200,
    bbox_inches="tight"
)

summary.to_csv(
    output_folder / "volatility_comparison.csv",
    date_format="%Y-%m-%d"
)

plt.show()

comparison = pd.DataFrame([{
    "period_start": valid_volatility.index.min().date(),
    "period_end": valid_volatility.index.max().date(),
    "window_sessions": 20,
    "kbe_higher_volatility_share_pct": (
        higher_volatility_share * 100
    ),
    "kbe_max_daily_volatility_pct": (
        valid_volatility["KBE"].max() * 100
    ),
    "spy_max_daily_volatility_pct": (
        valid_volatility["SPY"].max() * 100
    )
}])

comparison.to_csv(
    output_folder / "volatility_period_summary.csv",
    index=False
)

print("\nVolatility comparison saved.")