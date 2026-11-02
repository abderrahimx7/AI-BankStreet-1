import json

import numpy as np
import xgboost as xgb


print("XGBoost version:", xgb.__version__)
print("Build information:", xgb.build_info())

# Synthetic data for checking GPU execution only.
rng = np.random.default_rng(42)

X = rng.normal(size=(20000, 20)).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

training_data = xgb.DMatrix(X, label=y)

model = xgb.train(
    {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda:0",
        "max_depth": 3,
        "seed": 42,
    },
    training_data,
    num_boost_round=10,
)

configuration = json.loads(model.save_config())
device = configuration["learner"]["generic_param"]["device"]

print("Training device:", device)

if not device.startswith("cuda"):
    raise RuntimeError("Training did not use CUDA.")

print("XGBoost GPU training check passed.")