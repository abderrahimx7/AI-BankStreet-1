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

st.subheader("Annual risk — Spark batch results")

annual_path = (
    project_folder
    / "results"
    / "spark"
    / "annual_risk_v1"
)

annual_report = pd.read_parquet(annual_path)

annual_report["deepest_drawdown_pct"] = (
    annual_report["deepest_drawdown"] * 100
)

annual_report["max_daily_volatility_pct"] = (
    annual_report["max_daily_volatility"] * 100
)

st.dataframe(
    annual_report[[
        "ticker",
        "year",
        "sessions",
        "deepest_drawdown_pct",
        "max_daily_volatility_pct"
    ]]
    .sort_values(["year", "ticker"])
    .style.format({
        "year": "{:.0f}",
        "sessions": "{:.0f}",
        "deepest_drawdown_pct": "{:.2f}",
        "max_daily_volatility_pct": "{:.2f}"
    })
)

st.caption(
    "Full-year results generated by the Spark batch pipeline. "
    "Independent of the sidebar dates. Drawdown uses historical "
    "peaks, including those before the year."
)




from pathlib import Path
import pandas as pd
import streamlit as st


st.divider()
st.header("Historical Replay — Market Risk")
st.caption(
    "Historical prices replayed through Kafka and Spark. "
    "Results reflect the last completed processing run."
)

replay_path = (
    Path(__file__).resolve().parent
    / "results"
    / "replay"
    / "market_risk.csv"
)

if not replay_path.exists():
    st.info("Run lesson_17.py to generate the replay report.")
else:
    st.button("Refresh replay report", key="refresh_replay")

    replay_data = pd.read_csv(replay_path, parse_dates=["date"])

    replay_ticker = st.selectbox(
        "Asset",
        sorted(replay_data["ticker"].unique()),
        key="replay_asset",
    )

    selected = (
        replay_data.loc[replay_data["ticker"] == replay_ticker]
        .sort_values("date")
        .copy()
    )

    latest = selected.iloc[-1]

    st.caption(
        f"{selected['date'].min():%Y-%m-%d} → "
        f"{selected['date'].max():%Y-%m-%d}"
    )

    left, right = st.columns(2)

    latest_return = latest["daily_return"]
    latest_volatility = latest["volatility_20d"]

    left.metric(
        "Latest session return",
        f"{latest_return * 100:.2f}%"
        if pd.notna(latest_return) else "Not available",
    )
    right.metric(
        "Daily volatility · 20-session window",
        f"{latest_volatility * 100:.2f}%"
        if pd.notna(latest_volatility) else "Not enough history",
    )

    selected["volatility_20d_pct"] = (
        selected["volatility_20d"] * 100
    )

    st.line_chart(
        selected,
        x="date",
        y="volatility_20d_pct",
        y_label="Daily volatility (%)",
    )

    with st.expander("View replay data"):
        st.dataframe(selected, hide_index=True)

st.divider()
st.header("KBE — Experimental ML Probability")
st.caption(
    "Historical replay of the previously evaluated 2023 test period. "
    "The saved model was trained before 2020. "
    "It underperformed the baseline on the held-out test set."
)

prediction_path = (
    Path(__file__).resolve().parent
    / "results"
    / "replay"
    / "kbe_predictions.csv"
)

if not prediction_path.exists():
    st.info("Run the replay pipeline to generate predictions.")
else:
    st.button("Refresh ML report", key="refresh_ml_report")

    predictions = pd.read_csv(
        prediction_path,
        parse_dates=["date"],
    ).sort_values("date")

    if predictions.empty:
        st.info("No predictions available.")
    else:
        latest_prediction = predictions.iloc[-1]

        st.caption(
            f"Latest replay session: "
            f"{latest_prediction['date']:%Y-%m-%d}"
        )

        st.metric(
            "Model probability · return ≤ −3% after 5 sessions",
            f"{latest_prediction['p_target_1'] * 100:.2f}%",
        )

        predictions["probability_pct"] = (
            predictions["p_target_1"] * 100
        )

        st.line_chart(
            predictions,
            x="date",
            y="probability_pct",
            y_label="Model probability (%)",
        )


st.divider()
st.subheader("Model evaluation — 2023")

evaluation_folder = (
    Path(__file__).resolve().parent / "results" / "replay"
)

metrics_path = evaluation_folder / "metrics_2023.csv"
evaluated_path = (
    evaluation_folder / "evaluated_predictions_2023.csv"
)

if not metrics_path.exists() or not evaluated_path.exists():
    st.info("Run lesson_19.py to generate the evaluation report.")
else:
    evaluation_metrics = pd.read_csv(
        metrics_path,
        index_col="model",
    )

    evaluated_predictions = pd.read_csv(
        evaluated_path,
        parse_dates=["date", "label_end_date"],
    )

    st.caption(
        "Previously evaluated test period: prediction dates and "
        "five-session outcomes both fall within 2023. "
        "The baseline uses the positive rate from training."
    )

    count_col, rate_col = st.columns(2)

    count_col.metric(
        "Evaluated sessions",
        len(evaluated_predictions),
    )

    rate_col.metric(
        "Observed event rate",
        f"{evaluated_predictions['target'].mean() * 100:.2f}%",
    )

    st.dataframe(
        evaluation_metrics.style.format("{:.6f}"),
    )

    st.caption(
        "Brier: lower is better. "
        "ROC AUC and Average Precision: higher is better."
    )

    baseline = evaluation_metrics.loc["Baseline"]
    logistic = evaluation_metrics.loc["Logistic"]

    if (
        logistic["brier"] > baseline["brier"]
        and logistic["roc_auc"] < baseline["roc_auc"]
        and logistic["average_precision"]
        < baseline["average_precision"]
    ):
        st.warning(
            "In this evaluation period, Logistic Regression "
            "underperformed the baseline on all three metrics."
        )

from bankstreet.fraud.dashboard import render_fraud_dashboard

render_fraud_dashboard(Path(__file__).resolve().parent)

from bankstreet.fraud.inference import score_transactions
import xgboost as xgb


st.divider()
st.header("Score transaction CSV")

st.caption(
    "ULB-format demo: requires V1–V28 and Amount with the same "
    "feature transformation as the training dataset. "
    "Class labels are not used for scoring."
)

fraud_upload = st.file_uploader(
    "Choose a transaction CSV",
    type=["csv"],
    key="fraud_transaction_upload",
    max_upload_size=50,
)

if fraud_upload is not None:
    try:
        uploaded_transactions = pd.read_csv(fraud_upload)

        scored_transactions, frozen_threshold = score_transactions(
            uploaded_transactions,
            Path(__file__).resolve().parent,
        )

        left, right = st.columns(2)

        left.metric("Transactions scored", len(scored_transactions))
        right.metric(
            "Flagged for review",
            int(scored_transactions["review_required"].sum()),
        )

        st.caption(
            f"Frozen threshold: {frozen_threshold:.9f}. "
            "No alert does not guarantee a transaction is legitimate."
        )

        display_results = scored_transactions.copy()
        display_results["decision"] = (
            display_results["review_required"].map({
                True: "Manual review",
                False: "No alert",
            })
        )



        only_alerts = st.checkbox(
            "Show flagged transactions only",
            key="fraud_only_alerts",
        )

        if only_alerts:
            display_results = display_results.loc[
                display_results["review_required"]
            ]

        st.dataframe(
            display_results.sort_values(
                ["model_score", "input_row"],
                ascending=[False, True],
            ),
            hide_index=True,
        )


        st.download_button(
            "Download scoring results",
            data=scored_transactions.to_csv(index=False).encode("utf-8"),
            file_name="fraud_scoring_results.csv",
            mime="text/csv",
            key="download_fraud_scores",
        )

    except (
        ValueError,
        FileNotFoundError,
        UnicodeDecodeError,
        xgb.core.XGBoostError,
    ) as error:
        st.error(f"Could not score the file: {error}")