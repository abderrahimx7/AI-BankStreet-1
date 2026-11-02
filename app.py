"""BankStreet AI dashboard. Existing risk and inference functions are reused."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bankstreet.risk import (
    calculate_drawdown,
    calculate_rolling_volatility,
    summarize_drawdown,
)
from bankstreet.ui import apply_theme, callout, footer, line_chart, metric_row, page_header, section
from bankstreet.fraud.dashboard import load_fraud_summary, render_fraud_dashboard


st.set_page_config(page_title="BankStreet AI | Analytics Workspace", page_icon="B", layout="wide")
apply_theme(dark=bool(st.session_state.get("night_mode", False)))
ROOT = Path(__file__).resolve().parent


def read_prices():
    prices = {}
    for ticker in ["KBE", "SPY"]:
        table = pd.read_csv(
            ROOT / "data" / f"{ticker.lower()}_prices_v1.csv",
            index_col="Date", parse_dates=["Date"],
        ).sort_index()
        if not table.index.is_unique:
            raise ValueError(f"{ticker}: price dates must be unique.")
        prices[ticker] = table["Close"]
    if not prices["KBE"].index.equals(prices["SPY"].index):
        raise ValueError("KBE and SPY dates are not aligned.")
    close = pd.DataFrame(prices)
    if close.empty or close.isna().any().any():
        raise ValueError("Prices are empty or contain missing values.")
    return close


def go_to(page):
    st.session_state["workspace_page"] = page


with st.sidebar:
    st.markdown('<div class="bs-brand"><div class="bs-monogram">B</div><div>'
                '<strong>BankStreet<span style="color:#92C9B0"> AI</span></strong>'
                '<small>ANALYTICS WORKSPACE</small></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="bs-navlabel">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio("Workspace navigation", ["Overview", "Market Risk", "Fraud Detection", "Score CSV", "Experiments"],
                    key="workspace_page", label_visibility="collapsed")
    st.markdown("---")
    st.toggle("Night mode", key="night_mode", help="Switch between light and dark appearance for this session.")
    st.markdown("---")
    st.caption("DATA MODE")
    st.markdown("**Historical research**")
    st.caption("Local price snapshots and an anonymized transaction benchmark.")
    st.button("Refresh saved reports", key="refresh_all_reports", width="stretch")
    st.markdown('<div class="bs-side-note"><b>Built for understanding.</b><br>'
                'Measured risk. Verified calculations.<br>Human-reviewed fraud alerts.</div>', unsafe_allow_html=True)


def render_overview():
    page_header("WORKSPACE / OVERVIEW", "A clearer view of financial risk.",
                "Explore market history, inspect fraud signals, and understand the evidence behind every model.",
                "RESEARCH WORKSPACE")
    close = None
    summary = None
    try:
        close = read_prices()
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.info(f"Market snapshots unavailable: {error}")
    try:
        summary, _ = load_fraud_summary(ROOT)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
        st.info(f"Fraud evaluation unavailable: {error}")

    metric_row([
        ("Market assets", "KBE / SPY" if close is not None else "Unavailable"),
        ("Price sessions per asset", f"{len(close):,}" if close is not None else "—"),
        ("Fraud test recall", f"{summary['recall']:.2%}" if summary else "—"),
        ("Test alerts per 1,000", f"{summary['alerts_per_1000']:.2f}" if summary else "—"),
    ])

    primary, secondary = st.columns([2.2, 1], gap="large")
    with primary:
        section("Market pulse", "ADJUSTED CLOSE · INDEXED TO 100")
        with st.container(border=True):
            if close is not None:
                recent = close.tail(252)
                line_chart(recent / recent.iloc[0] * 100, "Indexed value", height=310)
                st.caption(f"{recent.index.min():%d %b %Y} — {recent.index.max():%d %b %Y} · "
                           f"Latest {len(recent)} available trading sessions. Historical prices.")
            else:
                st.info("Add the KBE and SPY snapshots to display this chart.")
    with secondary:
        section("Review desk", "FROZEN TEST POLICY")
        with st.container(border=True):
            if summary:
                st.metric("Fraud cases detected", f"{summary['TP']} / {summary['actual_frauds']}")
                st.markdown(f"**{summary['alerts']}** transactions flagged for review  \n"
                            f"**{summary['FP']}** false alerts · **{summary['FN']}** fraud cases missed")
                st.caption("Chronological ULB test split. The model and threshold were selected on validation.")
            else:
                st.info("Saved evaluation reports are not available.")
            st.button("Open fraud evaluation →", on_click=go_to, args=("Fraud Detection",), width="stretch")

    section("Your research workspace", "TWO DOMAINS · ONE WORKSPACE")
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("### Market risk")
        st.write("Compare banking-sector and broad-market performance, drawdowns, and daily volatility across history.")
        st.caption("KBE · SPY · Kafka / Spark historical replay")
        st.button("Explore market history →", on_click=go_to, args=("Market Risk",))
    with right, st.container(border=True):
        st.markdown("### Transaction screening")
        st.write("Upload transactions in the benchmark feature format and inspect the review queue from the saved XGBoost model.")
        st.caption("ULB FORMAT · FIXED THRESHOLD · MANUAL REVIEW")
        st.button("Score a CSV file →", on_click=go_to, args=("Score CSV",))
    callout("Different evidence, different conclusions: the fraud benchmark is promising; the KBE market prediction model did not outperform its full-test baseline. See Experiments for both results.")


def render_selected_market(close):
    minimum, maximum = close.index.min().date(), close.index.max().date()
    clamp = lambda value: min(max(pd.Timestamp(value).date(), minimum), maximum)
    with st.container(border=True):
        a, b, c = st.columns([1, 1, 1.35], vertical_alignment="bottom")
        start_date = a.date_input("Start date", value=clamp("2018-03-09"), min_value=minimum, max_value=maximum, key="market_start")
        end_date = b.date_input("End date", value=clamp("2021-02-16"), min_value=minimum, max_value=maximum, key="market_end")
        c.caption("Both assets use the same period. Calculations include earlier history where required.")
    if start_date is None or end_date is None:
        st.info("Select both dates.")
        return
    if start_date >= end_date:
        st.warning("Start date must be before end date.")
        return
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    period_prices = close.loc[start:end]
    if len(period_prices) < 2:
        st.warning("Select at least two trading sessions.")
        return

    # Same calculations and slicing semantics as the supplied app.py.
    drawdowns = calculate_drawdown(close)
    volatility = calculate_rolling_volatility(close, window=20)
    period_drawdowns = drawdowns.loc[start:end] * 100
    period_volatility = volatility.loc[start:end] * 100
    normalized = period_prices / period_prices.iloc[0] * 100
    rows = []
    for ticker in close.columns:
        metrics = summarize_drawdown(close[ticker], start=start, end=end)
        metrics["ticker"] = ticker
        metrics["period_return_pct"] = (period_prices[ticker].iloc[-1] / period_prices[ticker].iloc[0] - 1) * 100
        metrics["max_daily_volatility_pct"] = period_volatility[ticker].max()
        rows.append(metrics)
    report = pd.DataFrame(rows).set_index("ticker")
    report["trough_to_recovery_days"] = pd.to_numeric(report["trough_to_recovery_days"], errors="coerce")
    metric_row([
        ("KBE · period return", f"{report.loc['KBE', 'period_return_pct']:+.2f}%"),
        ("SPY · period return", f"{report.loc['SPY', 'period_return_pct']:+.2f}%"),
        ("KBE · deepest drawdown", f"{report.loc['KBE', 'drawdown_pct']:.2f}%"),
        ("SPY · deepest drawdown", f"{report.loc['SPY', 'drawdown_pct']:.2f}%"),
    ])
    section("Performance comparison", "COMMON START = 100")
    with st.container(border=True):
        line_chart(normalized, "Indexed value", height=330)
        st.caption(f"{period_prices.index.min():%d %b %Y} — {period_prices.index.max():%d %b %Y} · Adjusted closing prices.")
    left, right = st.columns(2, gap="large")
    with left:
        section("Drawdown", "HISTORICAL PEAKS")
        with st.container(border=True):
            line_chart(period_drawdowns, "Drawdown (%)", height=240)
    with right:
        section("Rolling daily volatility", "20 RETURNS · NOT ANNUALIZED")
        with st.container(border=True):
            line_chart(period_volatility, "Daily volatility (%)", height=240)
    section("Period risk report")
    display = report[["drawdown_pct", "period_return_pct", "max_daily_volatility_pct", "trough_to_recovery_days"]].rename(columns={
        "drawdown_pct": "Deepest drawdown (%)", "period_return_pct": "Period return (%)",
        "max_daily_volatility_pct": "Max daily volatility (%)", "trough_to_recovery_days": "Recovery from trough (days)",
    })
    st.dataframe(display.style.format({
        "Deepest drawdown (%)": "{:.2f}", "Period return (%)": "{:.2f}",
        "Max daily volatility (%)": "{:.2f}", "Recovery from trough (days)": "{:.0f}",
    }, na_rep="Unavailable"), width="stretch")
    st.caption(f"Drawdown uses full-history peaks. Recovery is measured in calendar days; the search can extend through {maximum}. "
               "A zero drawdown has no applicable recovery. Unavailable recovery means the previous peak was not regained; volatility can be unavailable with insufficient history.")
    export = report.copy()
    export["requested_start"], export["requested_end"] = start.date(), end.date()
    export["first_session"], export["last_session"] = period_prices.index.min().date(), period_prices.index.max().date()
    export["volatility_window_sessions"], export["recovery_search_end"] = 20, maximum
    st.download_button("Download period report", data=export.to_csv(sep=";", decimal=",", date_format="%Y-%m-%d", index_label="ticker").encode("utf-8-sig"),
                       file_name=f"bankstreet_risk_{start.date()}_{end.date()}.csv", mime="text/csv", key="market_report_download")


def render_replay():
    path = ROOT / "results" / "replay" / "market_risk.csv"
    if not path.exists():
        st.info("Replay report unavailable. Run the replay pipeline to generate it.")
        return
    data = pd.read_csv(path, parse_dates=["date"])
    if data.empty:
        st.info("The replay report contains no observations.")
        return
    asset = st.selectbox("Replay asset", sorted(data["ticker"].unique()), key="replay_asset")
    selected = data.loc[data["ticker"] == asset].sort_values("date").copy()
    latest = selected.iloc[-1]
    metric_row([
        ("Latest replay session", f"{latest['date']:%d %b %Y}"),
        ("Session return", f"{latest['daily_return'] * 100:+.2f}%" if pd.notna(latest["daily_return"]) else "N/A"),
        ("Daily volatility · 20 sessions", f"{latest['volatility_20d'] * 100:.2f}%" if pd.notna(latest["volatility_20d"]) else "N/A"),
    ])
    section("Historical replay volatility", "LATEST COMPLETED PROCESSING RUN")
    with st.container(border=True):
        line_chart((selected.set_index("date")[["volatility_20d"]] * 100).rename(columns={"volatility_20d": asset}),
                   "Daily volatility (%)", area=True)
    st.caption(f"{selected['date'].min():%Y-%m-%d} — {selected['date'].max():%Y-%m-%d}. Historical replay, not live market prices.")
    with st.expander("Inspect replay observations"):
        st.dataframe(selected, hide_index=True, width="stretch")


def render_annual():
    path = ROOT / "results" / "spark" / "annual_risk_v1"
    if not path.exists():
        st.info("Annual Spark report is not available. Generate it with the existing batch workflow.")
        return
    annual = pd.read_parquet(path)
    annual["deepest_drawdown_pct"] = annual["deepest_drawdown"] * 100
    annual["max_daily_volatility_pct"] = annual["max_daily_volatility"] * 100
    st.dataframe(annual[["ticker", "year", "sessions", "deepest_drawdown_pct", "max_daily_volatility_pct"]]
                 .sort_values(["year", "ticker"]).style.format({"year": "{:.0f}", "sessions": "{:.0f}",
                   "deepest_drawdown_pct": "{:.2f}", "max_daily_volatility_pct": "{:.2f}"}), width="stretch", hide_index=True)
    st.caption("Full-year Spark batch results. Independent of the selected analysis period. Historical peaks may precede the year.")


def render_market():
    page_header("WORKSPACE / MARKET RISK", "Understand the path, not just the return.",
                "Banking-sector and broad-market history, with consistent drawdown and volatility calculations.")
    selected, replay, annual = st.tabs(["Period analysis", "Historical replay", "Annual Spark report"])
    with selected:
        try:
            render_selected_market(read_prices())
        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(f"Cannot load the market analysis: {error}")
    with replay:
        render_replay()
    with annual:
        render_annual()


def render_scoring():
    page_header("WORKSPACE / TRANSACTION SCREENING", "From transactions to a review queue.",
                "Score a compatible CSV with the saved XGBoost model and the frozen validation-selected threshold.", "SAVED MODEL · INFERENCE")
    with st.container(border=True):
        st.markdown("### Upload transaction data")
        st.caption("Required: V1–V28 and Amount in the ULB feature representation. Class labels and extra columns are ignored. Maximum 50 MB.")
        uploaded = st.file_uploader("Choose a transaction CSV", type=["csv"], key="fraud_transaction_upload", max_upload_size=50)
    if uploaded is None:
        callout("Use data/fraud/demo_transactions.csv to try the interface. Arbitrary bank exports are not supported: the model expects the benchmark's pre-transformed PCA features.")
        st.caption("An alert requests manual review. No alert does not guarantee that a transaction is legitimate.")
        return
    import xgboost as xgb
    import requests
    try:
        raw = uploaded.getvalue()
        # Invalidate saved UI results whenever the file, model, policy, or feature metadata changes.
        dependencies = [ROOT / "models" / "fraud" / "xgboost_v1.json",
                        ROOT / "results" / "fraud" / "xgboost_v1" / "decision_policy.json",
                        ROOT / "results" / "fraud" / "xgboost_v1" / "metadata.json"]
        import os

        api_url = os.environ.get("FRAUD_API_URL", "").strip().rstrip("/")
        backend = api_url or "direct-model"
        signature = hashlib.sha256(      
            backend.encode("utf-8")
            + raw
            + b"".join(path.read_bytes() for path in dependencies)
        ).hexdigest()
        cached = st.session_state.get("fraud_score_result")
        if cached is None or cached[0] != signature:
            with st.spinner("Scoring transactions…"):
                if not api_url:
                    from io import BytesIO
                    from bankstreet.fraud.inference import score_transactions

                    scored, threshold = score_transactions(
                        pd.read_csv(BytesIO(raw)),
                        ROOT,
                    )

                else:
                    try:
                        response = requests.post(
                            f"{api_url}/score-csv",
                            files={
                                "file": (
                                    uploaded.name,
                                    raw,
                                    "text/csv",
                                )
                            },
                            timeout=(5, 120),
                        )
                    except requests.exceptions.RequestException:
                        st.error(
                            "Could not reach the configured Fraud API."
                        )
                        return

                    try:
                        payload = response.json()
                    except ValueError:
                        st.error("The Fraud API returned an invalid response.")
                        return

                    if not response.ok:
                        detail = (
                            payload.get("detail", "Scoring failed.")
                            if isinstance(payload, dict)
                            else "Scoring failed."
                        )
                        st.error(f"Fraud API: {detail}")
                        return

                    try:
                        scored = pd.DataFrame(payload["results"])
                        threshold = float(payload["threshold"])

                        required = {
                            "input_row",
                            "Amount",
                            "model_score",
                            "review_required",
                        }

                        if scored.empty or not required.issubset(scored.columns):
                            raise ValueError("Missing scoring results.")

                    except (KeyError, TypeError, ValueError):
                        st.error("The Fraud API returned invalid scoring results.")
                        return

            st.session_state["fraud_score_result"] = (
                signature,
                scored,
                threshold,
            )
        _, scored, threshold = st.session_state["fraud_score_result"]
        total, alerts = len(scored), int(scored["review_required"].sum())
        metric_row([("Transactions scored", f"{total:,}"), ("Flagged for review", f"{alerts:,}"),
                    ("Share flagged", f"{alerts / total:.2%}" if total else "N/A")])
        section("Screening results", f"FROZEN THRESHOLD · {threshold:.9f}")
        a, b = st.columns([2, 1], vertical_alignment="center")
        only_alerts = a.checkbox("Show flagged transactions only", key="fraud_only_alerts")
        b.download_button("Download all scoring results", scored.to_csv(index=False).encode("utf-8"),
                          "fraud_scoring_results.csv", "text/csv", key="download_fraud_scores", width="stretch")
        view = scored.copy()
        view["decision"] = view["review_required"].map({True: "Manual review", False: "No alert"})
        if only_alerts:
            view = view.loc[view["review_required"]]
        view = view.sort_values(["model_score", "input_row"], ascending=[False, True])
        st.dataframe(view[["input_row", "Amount", "model_score", "decision"]],
            hide_index=True, width="stretch", column_config={
                "input_row": st.column_config.NumberColumn("Input row", format="%d"),
                "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                "model_score": st.column_config.ProgressColumn("Model score", min_value=0, max_value=1, format="%.4f"),
                "decision": st.column_config.TextColumn("Decision"),
            })
        st.caption(f"Showing {len(view):,} of {total:,} input rows. Scores are model estimates, not confirmed outcomes. The download includes all rows, regardless of the display filter.")
        callout("The threshold is fixed. Review alerts may vary between files; do not change the threshold using the outcomes of a new test sample.")
    except (ValueError, FileNotFoundError, UnicodeDecodeError, xgb.core.XGBoostError) as error:
        st.error(f"Could not score the file: {error}")


def show_metrics(path, index="model"):
    if not path.exists():
        st.info(f"Saved report not available: {path.relative_to(ROOT)}")
        return None
    table = pd.read_csv(path, index_col=index)
    st.dataframe(table.style.format("{:.6f}"), width="stretch")
    return table


def render_experiments():
    page_header("WORKSPACE / RESEARCH & EVALUATION", "The evidence behind the models.",
                "Keep model quality separate from pipeline correctness. Compare saved experiments and their limitations.", "REPORTED RESULTS")
    market, fraud, validation = st.tabs(["KBE market model", "Fraud candidates", "Verification & scope"])
    with market:
        callout("The KBE Logistic Regression model underperformed its constant baseline on the full held-out test. Its probabilities remain an experimental demonstration.")
        section("Original experiment", "2023–2025 HELD-OUT TEST")
        show_metrics(ROOT / "results" / "experiment_v1" / "test_metrics.csv")
        st.caption("Target: KBE closing return ≤ −3% after five trading sessions, not any dip during the window. Trained before 2020; validation 2020–2022. These reports do not depend on market date filters.")
        section("2023 replay probabilities", "PREVIOUSLY EVALUATED TEST PERIOD")
        path = ROOT / "results" / "replay" / "kbe_predictions.csv"
        if path.exists():
            data = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            if not data.empty:
                latest = data.iloc[-1]
                metric_row([("Latest replay session", f"{latest['date']:%d %b %Y}"),
                            ("Estimated event probability", f"{latest['p_target_1']:.2%}")])
                with st.container(border=True):
                    line_chart((data.set_index("date")[["p_target_1"]] * 100).rename(columns={"p_target_1": "KBE model"}), "Probability (%)", height=240)
        else:
            st.info("Run the replay pipeline to generate KBE predictions.")
        section("2023 replay evaluation")
        evaluation = ROOT / "results" / "replay"
        outcomes_path = evaluation / "evaluated_predictions_2023.csv"
        if outcomes_path.exists():
            outcomes = pd.read_csv(outcomes_path)
            st.caption(f"{len(outcomes):,} evaluated sessions · observed event rate {outcomes['target'].mean():.2%}. Five-session outcomes must end within 2023.")
        table = show_metrics(evaluation / "metrics_2023.csv")
        if table is not None and {"Baseline", "Logistic"}.issubset(table.index):
            baseline, logistic = table.loc["Baseline"], table.loc["Logistic"]
            if (logistic["brier"] > baseline["brier"] and logistic["roc_auc"] < baseline["roc_auc"]
                    and logistic["average_precision"] < baseline["average_precision"]):
                st.warning("KBE market model — 2023: Logistic Regression underperformed the baseline on all three metrics.")
        with st.expander("Validation diagnostics and alternative models"):
            for title, relative in [
                ("Fixed-model annual validation", "diagnostics_v1/validation_by_year.csv"),
                ("Volatility-only comparison", "volatility_only_experiment/validation_comparison.csv"),
                ("Paired Brier block bootstrap", "volatility_only_experiment/brier_gain_bootstrap.csv"),
                ("Annual retraining", "annual_retraining_experiment/validation_comparison.csv"),
            ]:
                path = ROOT / "results" / relative
                if path.exists():
                    st.markdown(f"**{title}**")
                    st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")
    with fraud:
        section("Candidate comparison", "CHRONOLOGICAL VALIDATION ONLY")
        parts = []
        for name, directory in [("Logistic", "logistic_v1"), ("XGBoost", "xgboost_v1"), ("MLP", "mlp_v1")]:
            path = ROOT / "results" / "fraud" / directory / "validation_metrics.csv"
            if path.exists():
                table = pd.read_csv(path, index_col="model")
                if name in table.index:
                    parts.append(table.loc[[name]])
        if parts:
            st.dataframe(pd.concat(parts).style.format("{:.6f}"), width="stretch")
        else:
            st.info("Candidate evaluation reports are not available.")
        st.caption("XGBoost and MLP were trained on the RTX 3070. MLP: 29 → 64 → 32 → 1, 20 fixed epochs, one seed. Model size and GPU use alone do not establish better performance.")
        section("Equal review budget", "94 VALIDATION TRANSACTIONS")
        path = ROOT / "results" / "fraud" / "review_budget_comparison" / "validation_top94_comparison.csv"
        if path.exists():
            st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")
        section("MLP training history")
        path = ROOT / "results" / "fraud" / "mlp_v1" / "training_history.csv"
        if path.exists():
            history = pd.read_csv(path)
            line_chart(history.set_index("epoch")[["train_loss", "validation_loss"]], "Binary cross-entropy", numeric_index=True)
        else:
            st.info("MLP history is not available.")
        callout("The fraud model and threshold were selected using validation. Test outcomes are reported on the Fraud Detection page; they were not used to choose the threshold.")
    with validation:
        left, right = st.columns(2, gap="large")
        with left, st.container(border=True):
            st.markdown("### Verified calculation paths")
            st.markdown("- Drawdown and recovery edge cases.\n- Spark features compared with pandas.\n- CSV inference compared with saved test predictions.\n- Missing features, missing values, and negative amounts rejected.")
            st.caption("These checks were run during development. This page lists their scope; it does not run them or claim a new test pass.")
        with right, st.container(border=True):
            st.markdown("### Practical limits")
            st.markdown("- Fraud benchmark covers two days.\n- Precomputed PCA representation is required.\n- No live bank integration or automatic blocking.\n- Market replay is historical; downstream reports are batch generated.\n- Credit-default modeling is not implemented.")
        st.caption("Brier: lower is better. ROC AUC and Average Precision: higher is better. Average Precision is not recall, precision at a chosen threshold, or classification accuracy.")


if page == "Overview":
    render_overview()
elif page == "Market Risk":
    render_market()
elif page == "Fraud Detection":
    render_fraud_dashboard(ROOT)
elif page == "Score CSV":
    render_scoring()
else:
    render_experiments()

footer()
