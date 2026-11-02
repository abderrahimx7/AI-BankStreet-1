import json
from pathlib import Path

import pandas as pd
import streamlit as st


def render_fraud_dashboard(project_folder):
    folder = (
        Path(project_folder)
        / "results"
        / "fraud"
        / "xgboost_v1"
    )

    paths = {
        "policy": folder / "decision_policy.json",
        "summary": folder / "test_decision_results.json",
        "metrics": folder / "test_metrics.csv",
        "predictions": folder / "test_predictions.csv",
    }

    st.divider()
    st.header("Fraud Detection — XGBoost")

    st.caption(
        "Historical benchmark on the chronological test split. "
        "Model and decision threshold were selected using validation."
    )

    if not all(path.exists() for path in paths.values()):
        st.info("Run the fraud evaluation to generate these reports.")
        return

    with paths["policy"].open(encoding="utf-8") as file:
        policy = json.load(file)

    with paths["summary"].open(encoding="utf-8") as file:
        summary = json.load(file)

    if (
        summary["model_sha256"] != policy["model_sha256"]
        or summary["threshold"] != policy["threshold"]
    ):
        st.error("The evaluation report does not match the decision policy.")
        return

    metrics = pd.read_csv(paths["metrics"], index_col="model")
    predictions = pd.read_csv(paths["predictions"])

    first, second, third = st.columns(3)

    first.metric("Test transactions", f"{summary['test_rows']:,}")
    second.metric("Actual fraud cases", summary["actual_frauds"])
    third.metric("Flagged for review", summary["alerts"])

    first, second, third = st.columns(3)

    first.metric("Fraud recall", f"{summary['recall']:.2%}")

    precision = summary["precision"]
    second.metric(
        "Alert precision",
        f"{precision:.2%}" if precision is not None else "N/A",
    )

    third.metric(
        "Reviews per 1,000 transactions",
        f"{summary['alerts_per_1000']:.2f}",
    )

    st.subheader("Decision outcomes")

    outcomes = pd.DataFrame({
        "Outcome": [
            "Fraud detected",
            "Fraud missed",
            "Normal transaction flagged",
            "Normal transaction not flagged",
        ],
        "Count": [
            summary["TP"],
            summary["FN"],
            summary["FP"],
            summary["TN"],
        ],
    })

    st.dataframe(outcomes, hide_index=True)

    st.subheader("Probability evaluation")
    st.dataframe(metrics.style.format("{:.6f}"))

    st.caption(
        "Brier: lower is better. "
        "ROC AUC and Average Precision: higher is better."
    )

    st.subheader("Historical review queue")

    st.caption(
        f"Frozen rule: model probability ≥ {policy['threshold']:.9f}. "
        "An alert requests manual review; it does not confirm fraud. "
        "Actual labels below are shown for retrospective evaluation."
    )

    queue = (
        predictions.loc[predictions["alert"] == 1]
        .sort_values(
            ["probability", "row_id"],
            ascending=[False, True],
        )
        .copy()
    )

    queue["model_score_pct"] = queue["probability"] * 100
    queue["actual_outcome"] = queue["target"].map({
        0: "Normal",
        1: "Fraud",
    })

    st.dataframe(
        queue[
            ["row_id", "Time", "model_score_pct", "actual_outcome"]
        ].rename(columns={
            "row_id": "Dataset row ID",
            "Time": "Elapsed seconds",
            "model_score_pct": "Model score (%)",
            "actual_outcome": "Actual outcome",
        }),
        hide_index=True,
    )

    st.caption(
        "The source dataset spans only two days. "
        "These results do not establish performance in a live bank."
    )