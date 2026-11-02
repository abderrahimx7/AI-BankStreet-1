import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn


project_folder = Path(__file__).resolve().parent
fraud_results = project_folder / "results" / "fraud"

with (
    fraud_results / "xgboost_v1" / "metadata.json"
).open(encoding="utf-8") as file:
    settings = json.load(file)

data_path = project_folder / "data" / "fraud" / "creditcard.csv"
data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

if data_hash != settings["data_sha256"]:
    raise ValueError("Dataset differs from previous experiments.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda:0")
features = settings["features"]

data = pd.read_csv(data_path).sort_values("Time", kind="stable")

train = data.loc[
    data["Time"] < settings["validation_start"]
].copy()

validation = data.loc[
    (data["Time"] >= settings["validation_start"])
    & (data["Time"] < settings["test_start"])
].copy()

# Check that validation matches the previous experiment.
reference = pd.read_csv(
    fraud_results / "logistic_v1" / "validation_predictions.csv"
)

identity = pd.DataFrame({
    "row_id": validation.index,
    "Time": validation["Time"].to_numpy(),
    "target": validation["Class"].to_numpy(),
})

pd.testing.assert_frame_equal(
    identity.sort_values("row_id").reset_index(drop=True),
    reference[["row_id", "Time", "target"]]
    .sort_values("row_id")
    .reset_index(drop=True),
)

# Learn scaling from training data only.
scaler = StandardScaler()

X_train = torch.from_numpy(
    scaler.fit_transform(train[features]).astype(np.float32)
).to(device)

X_validation = torch.from_numpy(
    scaler.transform(validation[features]).astype(np.float32)
).to(device)

y_train = torch.tensor(
    train["Class"].to_numpy(),
    dtype=torch.float32,
    device=device,
).reshape(-1, 1)

y_validation = torch.tensor(
    validation["Class"].to_numpy(),
    dtype=torch.float32,
    device=device,
).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(len(features), 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(device)

prior = float(train["Class"].mean())

with torch.no_grad():
    model[-1].bias.fill_(np.log(prior / (1 - prior)))

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001,
)

epochs = 20
batch_size = 1024
history = []

print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)
print("Train rows:", len(train))
print("Validation rows:", len(validation))

torch.cuda.synchronize()
start = perf_counter()

for epoch in range(1, epochs + 1):
    model.train()
    order = torch.randperm(len(train), device=device)
    total_loss = torch.zeros((), device=device)

    for indices in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)

        logits = model(X_train[indices])
        loss = criterion(logits, y_train[indices])

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")

        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * len(indices)

    model.eval()

    with torch.inference_mode():
        validation_loss = criterion(
            model(X_validation),
            y_validation,
        ).item()

    training_loss = total_loss.item() / len(train)

    history.append({
        "epoch": epoch,
        "train_loss": training_loss,
        "validation_loss": validation_loss,
    })

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"train_loss={training_loss:.6f} | "
        f"validation_loss={validation_loss:.6f}"
    )

torch.cuda.synchronize()
training_seconds = perf_counter() - start

model.eval()

with torch.inference_mode():
    probabilities = (
        torch.sigmoid(model(X_validation))
        .squeeze(1)
        .cpu()
        .numpy()
    )

actual = validation["Class"].to_numpy(dtype=int)

metrics = pd.DataFrame([{
    "model": "MLP",
    "brier": brier_score_loss(actual, probabilities),
    "roc_auc": roc_auc_score(actual, probabilities),
    "average_precision": average_precision_score(
        actual, probabilities
    ),
}]).set_index("model")

report_folder = fraud_results / "mlp_v1"
report_folder.mkdir(parents=True, exist_ok=True)

metrics.to_csv(report_folder / "validation_metrics.csv")

pd.DataFrame(history).to_csv(
    report_folder / "training_history.csv",
    index=False,
)

identity["probability"] = probabilities
identity.to_csv(
    report_folder / "validation_predictions.csv",
    index=False,
)

model_folder = project_folder / "models" / "fraud"
model_folder.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "features": features,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "hidden_layers": [64, 32],
        "dropout": 0.1,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seed": 42,
        "data_sha256": data_hash,
        "validation_start": settings["validation_start"],
        "test_start": settings["test_start"],
        "torch_version": str(torch.__version__),
        "sklearn_version": sklearn.__version__,
    },
    model_folder / "mlp_v1.pt",
)

comparison = pd.concat([
    pd.read_csv(
        fraud_results / "logistic_v1" / "validation_metrics.csv",
        index_col="model",
    ),
    pd.read_csv(
        fraud_results / "xgboost_v1" / "validation_metrics.csv",
        index_col="model",
    ).loc[["XGBoost"]],
    metrics,
])

comparison.to_csv(report_folder / "validation_comparison.csv")

print("\nTraining seconds:", round(training_seconds, 2))
print(comparison.round(6))
print("\nMLP model and validation results saved.")