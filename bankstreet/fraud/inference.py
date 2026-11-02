import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def score_transactions(data, project_folder):
    root = Path(project_folder)
    report_folder = root / "results" / "fraud" / "xgboost_v1"
    model_path = root / "models" / "fraud" / "xgboost_v1.json"

    with (report_folder / "metadata.json").open(
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    with (report_folder / "decision_policy.json").open(
        encoding="utf-8"
    ) as file:
        policy = json.load(file)

    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    if model_hash != policy["model_sha256"]:
        raise ValueError("Model does not match the frozen policy.")

    features = metadata["features"]
    missing = sorted(set(features) - set(data.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data.empty:
        raise ValueError("The CSV contains no transactions.")

    # Select only model features: Class and other extra columns are ignored.
    X = data[features].apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or infinite values.")

    if (X["Amount"] < 0).any():
        raise ValueError("Amount must not be negative.")

    model = xgb.Booster()
    model.load_model(model_path)
    model.set_param({"device": "cpu"})

    probabilities = model.predict(xgb.DMatrix(X))

    if not (
        np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Invalid model scores.")

    result = pd.DataFrame({
        "input_row": np.arange(1, len(data) + 1),
        "Amount": X["Amount"].to_numpy(),
        "model_score": probabilities,
        "review_required": probabilities >= policy["threshold"],
    })

    return result, policy["threshold"]