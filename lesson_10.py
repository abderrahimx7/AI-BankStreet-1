from pathlib import Path
import pandas as pd

from bankstreet.risk import (
    summarize_drawdown,
    calculate_rolling_volatility
)

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

start = "2018-03-09"
end = "2021-02-16"

rows = []

for ticker in ["KBE", "SPY"]:
    prices = pd.read_csv(
        data_folder / f"{ticker.lower()}_prices_v1.csv",
        index_col="Date",
        parse_dates=["Date"]
    ).sort_index()

    close = prices["Close"]

    # Drawdown and recovery
    metrics = summarize_drawdown(close, start, end)

    # Return over the selected period
    period_close = close.loc[start:end]

    metrics["period_return_pct"] = (
        period_close.iloc[-1] / period_close.iloc[0] - 1
    ) * 100

    # Calculate on full history, then select the period
    volatility = calculate_rolling_volatility(close)
    period_volatility = volatility.loc[start:end]

    metrics["max_daily_volatility_pct"] = (
        period_volatility.max() * 100
    )

    metrics["ticker"] = ticker
    rows.append(metrics)

risk_report = pd.DataFrame(rows).set_index("ticker")

print(f"\nRisk report: {start} -> {end}")
print(
    risk_report.round({
        "drawdown_pct": 2,
        "period_return_pct": 2,
        "max_daily_volatility_pct": 2
    }).to_string()
)



results_folder = (
    project_folder / "results" / "risk_analysis"
)
results_folder.mkdir(parents=True, exist_ok=True)


risk_report["period_start"] = start
risk_report["period_end"] = end 
risk_report["volatility_window_sessions"] = 20

risk_report.to_csv(
    results_folder / "risk_report.csv",
    date_format="%Y-%m-%d"
)

print("\nRisk report saved.")


saved_report = pd.read_csv(
    results_folder / "risk_report.csv",
    index_col="ticker"
)

print("\nSaved report context:")
print(saved_report[[
    "period_start",
    "period_end",
    "volatility_window_sessions"
]])