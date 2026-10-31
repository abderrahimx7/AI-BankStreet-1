from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from bankstreet.risk import calculate_drawdown
from bankstreet.risk import summarize_drawdown


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

risk_summary = pd.DataFrame({
    ticker: summarize_drawdown(
        close[ticker],
        start="2018-03-09",
        end="2021-02-16"
    )
    for ticker in close.columns
}).T

risk_summary.index.name = "ticker"

print("\nRecovery comparison:")
print(
    risk_summary.to_string(
        formatters={
            "drawdown_pct": lambda value: f"{value:.2f}"
        }
    )
)

# Calculate historical drawdowns before selecting the display period
drawdowns = calculate_drawdown(close)

start = "2018-03-09"
end = "2021-02-16"

period_prices = close.loc[start:end]
period_drawdowns = drawdowns.loc[start:end]

normalized = period_prices / period_prices.iloc[0] * 100

print("\nLowest historical drawdown within this period:")
print((period_drawdowns.min() * 100).round(2))

print("\nChange from the selected starting date (%):")
print(
    ((normalized.iloc[-1] / 100 - 1) * 100).round(2)
)



fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 7),
    sharex=True
)

normalized.plot(ax=axes[0])
axes[0].axhline(100, color="gray", linestyle="--")
axes[0].set_title("KBE vs SPY — same period")
axes[0].set_ylabel("Starting value = 100")

(period_drawdowns * 100).plot(ax=axes[1])
axes[1].set_ylabel("Historical drawdown (%)")
axes[1].set_xlabel("Date")

for axis in axes:
    axis.grid(alpha=0.25)

plt.tight_layout()

charts_folder = project_folder / "results" / "risk_analysis"
charts_folder.mkdir(parents=True, exist_ok=True)

fig.savefig(
    charts_folder / "kbe_vs_spy.png",
    dpi=200,
    bbox_inches="tight"
)


risk_summary.to_csv(
    charts_folder / "risk_summary.csv",
    date_format="%Y-%m-%d"
)

plt.show()