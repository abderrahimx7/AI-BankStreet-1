import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import hashlib

def evaluate_model(model, part, feature_columns, baseline_probability):
    actual = part["target"].astype(int)

    model_predictions = pd.Series(
        model.predict_proba(part[feature_columns])[:, 1],
        index=part.index
    )

    baseline_predictions = pd.Series(
        baseline_probability,
        index=part.index
    )

    results = []

    for name, predictions in [
        ("Baseline", baseline_predictions),
        ("Logistic", model_predictions)
    ]:
        results.append({
            "model": name,
            "brier": ((predictions - actual) ** 2).mean(),
            "roc_auc": roc_auc_score(actual, predictions),
            "average_precision": average_precision_score(
                actual, predictions
            )
        })

    return pd.DataFrame(results).set_index("model")
