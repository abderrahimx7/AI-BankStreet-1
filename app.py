from pathlib import Path
import pandas as pd
import streamlit as st

from bankstreet.risk import (
    calculate_drawdown,
    calculate_rolling_volatility,
    summarize_drawdown
)


st.set_page_config(
    page_title="BankStreet | Risk Analytics",
    layout="wide"
)

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

st.title("BankStreet — Risk Analytics")
st.caption("Historical analysis of KBE and SPY")


# 1. Load the full price history
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

if close.empty or close.isna().any().any():
    st.error("Prices are empty or contain missing values.")
    st.stop()


# 2. Select the analysis period
st.sidebar.header("Analysis period")

min_date = close.index.min().date()
max_date = close.index.max().date()

start_date = st.sidebar.date_input(
    "Start date",
    value=pd.Timestamp("2018-03-09").date(),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End date",
    value=pd.Timestamp("2021-02-16").date(),
    min_value=min_date,
    max_value=max_date
)

if start_date is None or end_date is None:
    st.info("Select both dates.")
    st.stop()

if start_date >= end_date:
    st.warning("Start date must be before end date.")
    st.stop()

start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date)

period_prices = close.loc[start:end]

if len(period_prices) < 2:
    st.warning("Select a period with at least two trading sessions.")
    st.stop()

st.write(
    "Trading sessions used: "
    f"{period_prices.index.min().date()} → "
    f"{period_prices.index.max().date()}"
)


# 3. Calculate on full history, then select the period
drawdowns = calculate_drawdown(close)
volatility = calculate_rolling_volatility(close, window=20)

period_drawdowns = drawdowns.loc[start:end] * 100
period_volatility = volatility.loc[start:end] * 100

normalized = period_prices / period_prices.iloc[0] * 100


# 4. Build the report for the selected period
rows = []

for ticker in close.columns:
    metrics = summarize_drawdown(
        close[ticker],
        start=start,
        end=end
    )

    metrics["ticker"] = ticker

    metrics["period_return_pct"] = (
        period_prices[ticker].iloc[-1]
        / period_prices[ticker].iloc[0]
        - 1
    ) * 100

    metrics["max_daily_volatility_pct"] = (
        period_volatility[ticker].max()
    )

    rows.append(metrics)

report = pd.DataFrame(rows).set_index("ticker")

report["trough_to_recovery_days"] = pd.to_numeric(
    report["trough_to_recovery_days"],
    errors="coerce"
)

display_report = report[[
    "drawdown_pct",
    "period_return_pct",
    "max_daily_volatility_pct",
    "trough_to_recovery_days"
]].rename(columns={
    "drawdown_pct": "Deepest drawdown (%)",
    "period_return_pct": "Period return (%)",
    "max_daily_volatility_pct": "Max daily volatility (%)",
    "trough_to_recovery_days": "Recovery from trough (days)"
})

st.dataframe(
    display_report.style.format(
        {
            "Deepest drawdown (%)": "{:.2f}",
            "Period return (%)": "{:.2f}",
            "Max daily volatility (%)": "{:.2f}",
            "Recovery from trough (days)": "{:.0f}"
        },
        na_rep="Unavailable"
    )
)

st.caption(
    "Recovery is measured in calendar days from the deepest "
    "trough in the selected period to the previous peak. "
    f"The search includes later data through {max_date}. "
    "Unavailable means recovery was not observed, or there "
    "was insufficient history to calculate volatility."
)


# 5. Display charts
st.subheader("Performance from the same starting date")
st.line_chart(normalized)

st.caption(
    "Both series start at 100 on the first selected trading "
    "session. Values use adjusted closing prices."
)

st.subheader("Historical drawdown (%)")
st.line_chart(period_drawdowns)

st.caption(
    "Decline from each asset's previous peak, using all "
    "available history before and within the selected period."
)

st.subheader("20-session rolling volatility (%)")
if period_volatility.dropna(how="all").empty:
    st.info(
        "Not enough price history to calculate "
        "20-session volatility for this period."
    )
else:
    st.line_chart(period_volatility)

st.caption(
    "Daily return standard deviation over the latest "
    "20 trading sessions, including upward and downward moves."
)





st.subheader("Download report")

export_report = report.copy()
export_report["requested_start"] = start.date()
export_report["requested_end"] = end.date()
export_report["first_session"] = period_prices.index.min().date()
export_report["last_session"] = period_prices.index.max().date()
export_report["volatility_window_sessions"] = 20
export_report["recovery_search_end"] = max_date

csv_data = export_report.to_csv(
    sep=";",
    decimal=",",
    date_format="%Y-%m-%d",
    index_label="ticker"
).encode("utf-8-sig")

st.download_button(
    label="Download risk report (CSV)",
    data=csv_data,
    file_name=f"bankstreet_risk_{start.date()}_{end.date()}.csv",
    mime="text/csv"
)



st.subheader("ML experiment — held-out test")

st.write(
    "Target: a KBE adjusted-close return of -3% or worse "
    "after 5 trading sessions."
)

test_results = pd.read_csv(
    project_folder
    / "results"
    / "experiment_v1"
    / "test_metrics.csv",
    index_col="model"
).rename(
    index={
        "Baseline": "Constant baseline",
        "Logistic": "Logistic regression"
    },
    columns={
        "brier": "Brier score",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average precision"
    }
)

st.dataframe(
    test_results.style.format("{:.6f}")
)

st.write(
    "The logistic model underperformed the constant baseline "
    "on the 2023–2025 test period. These results do not "
    "establish useful predictive performance."
)

st.caption(
    "Archived experiment: trained on data before 2020, "
    "selected using 2020–2022 validation data. "
    "These fixed test results do not change with "
    "the sidebar analysis dates."
)



st.caption(
    "When deepest drawdown is zero, recovery is not applicable. "
    "For a negative drawdown, an unavailable recovery means "
    "the previous peak was not regained in the available data."
)