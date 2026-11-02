"""Presentation of the saved, frozen-policy fraud evaluation."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from bankstreet.ui import callout, chart_colors, metric_row, page_header, section


def load_fraud_summary(project_folder):
    folder = Path(project_folder) / "results" / "fraud" / "xgboost_v1"
    with (folder / "decision_policy.json").open(encoding="utf-8") as file:
        policy = json.load(file)
    with (folder / "test_decision_results.json").open(encoding="utf-8") as file:
        summary = json.load(file)
    if (summary["model_sha256"] != policy["model_sha256"]
            or summary["threshold"] != policy["threshold"]):
        raise ValueError("The saved test report does not match the frozen decision policy.")
    return summary, policy


def render_fraud_dashboard(project_folder):
    folder = Path(project_folder) / "results" / "fraud" / "xgboost_v1"
    page_header("WORKSPACE / FRAUD DETECTION", "Signals that deserve a second look.",
                "Inspect the saved XGBoost evaluation and the trade-off between detected fraud and review workload.",
                "FROZEN POLICY · HELD-OUT TEST")
    try:
        summary, policy = load_fraud_summary(project_folder)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Cannot display the fraud evaluation: {error}")
        return

    metric_row([
        ("Fraud recall", f"{summary['recall']:.2%}"),
        ("Alert precision", f"{summary['precision']:.2%}" if summary["precision"] is not None else "N/A"),
        ("Transactions flagged", f"{summary['alerts']:,}"),
        ("Reviews per 1,000", f"{summary['alerts_per_1000']:.2f}"),
    ])
    st.caption(f"{summary['test_rows']:,} test transactions · {summary['actual_frauds']} actual fraud cases · "
               "Chronological ULB test split. Labels are historical evaluation outcomes.")

    outcomes_tab, queue_tab, metrics_tab = st.tabs(["Decision outcomes", "Historical review queue", "Metrics & policy"])
    with outcomes_tab:
        section("Every decision has a trade-off", "THRESHOLD FIXED BEFORE TEST")
        left, right = st.columns([1.4, 1], gap="large")
        outcomes = pd.DataFrame({
            "Outcome": ["Fraud detected", "Fraud missed", "Normal transaction flagged"],
            "Count": [summary["TP"], summary["FN"], summary["FP"]],
        })
        with left, st.container(border=True):
            colors = chart_colors()
            chart = alt.Chart(outcomes).mark_bar(cornerRadiusEnd=4, size=36).encode(
                y=alt.Y("Outcome:N", sort=list(outcomes["Outcome"]), title=None,
                        axis=alt.Axis(labelLimit=180, labelPadding=12, ticks=False, domain=False)),
                x=alt.X("Count:Q", title="Transactions", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Outcome:N", legend=None, scale=alt.Scale(
                    domain=list(outcomes["Outcome"]), range=[colors["green"], colors["amber"], colors["blue"]])),
                tooltip=["Outcome:N", "Count:Q"],
            ).properties(height=240).configure(background=colors["background"]).configure_view(stroke=None).configure_axis(
                gridColor=colors["grid"], labelColor=colors["text"], titleColor=colors["text"], labelFontSize=11,
            )
            st.altair_chart(chart, width="stretch", theme=None)
        with right, st.container(border=True):
            st.markdown("### What the result means")
            st.write(f"**{summary['TP']} of {summary['actual_frauds']}** fraud cases were flagged. "
                     f"**{summary['FN']}** were missed.")
            st.write(f"Reviewing **{summary['alerts']}** transactions included **{summary['FP']}** normal transactions.")
            st.caption(f"Another {summary['TN']:,} normal transactions were not flagged. A fixed threshold does not fix the future number of alerts.")
        callout("An alert requests human review; it is not a confirmed fraud finding. The source dataset spans only two days and does not establish live-bank performance.")

    with queue_tab:
        section("Prioritized review queue", "RETROSPECTIVE TEST OUTCOMES")
        predictions_path = folder / "test_predictions.csv"
        if not predictions_path.exists():
            st.info("Saved test predictions are unavailable.")
        else:
            predictions = pd.read_csv(predictions_path)
            queue = predictions.loc[predictions["alert"] == 1].copy()
            outcome = st.selectbox("Actual outcome", ["All flagged transactions", "Confirmed fraud", "False alerts"], key="historical_queue_outcome")
            if outcome == "Confirmed fraud":
                queue = queue.loc[queue["target"] == 1]
            elif outcome == "False alerts":
                queue = queue.loc[queue["target"] == 0]
            queue = queue.sort_values(["probability", "row_id"], ascending=[False, True])
            queue["actual_outcome"] = queue["target"].map({0: "Normal", 1: "Fraud"})
            st.caption("Actual outcomes are shown only for retrospective evaluation; they are not inputs to the model.")
            st.dataframe(queue[["row_id", "Time", "probability", "actual_outcome"]],
                hide_index=True, width="stretch", column_config={
                    "row_id": st.column_config.NumberColumn("Dataset row ID", format="%d"),
                    "Time": st.column_config.NumberColumn("Elapsed seconds", format="%.0f"),
                    "probability": st.column_config.ProgressColumn("Model score", min_value=0, max_value=1, format="%.4f"),
                    "actual_outcome": st.column_config.TextColumn("Actual outcome"),
                })
            st.download_button("Download displayed review queue", queue.to_csv(index=False).encode("utf-8"),
                               "historical_review_queue.csv", "text/csv", key="download_historical_queue")
            st.caption(f"{len(queue):,} transactions displayed. This is a saved benchmark queue, not a live transaction feed.")

    with metrics_tab:
        section("Probability evaluation", "BASELINE VS XGBOOST")
        path = folder / "test_metrics.csv"
        if path.exists():
            metrics = pd.read_csv(path, index_col="model")
            st.dataframe(metrics.style.format("{:.6f}"), width="stretch")
        else:
            st.info("Saved test metrics are unavailable.")
        st.caption("Brier: lower is better. ROC AUC and Average Precision: higher is better. Average Precision is not the fraction of fraud detected at the threshold.")
        section("Decision policy")
        with st.container(border=True):
            metric_row([("Model", "XGBoost v1"), ("Frozen threshold", f"{policy['threshold']:.9f}"),
                        ("Action", "Manual review")])
            st.caption("Rule: model probability ≥ threshold. The model and threshold were selected using validation, before evaluating Test.")
            with st.expander("Model fingerprint"):
                st.code(policy["model_sha256"], language=None)
        st.caption("Inputs require V1–V28 and Amount with the ULB PCA feature representation. Arbitrary bank transaction exports are not compatible.")
