# BankStreet AI

Historical Market Risk · Kafka/Spark Replay · Fraud Detection

BankStreet AI is a Python prototype combining historical KBE/SPY risk analytics, a reproducible historical-price replay pipeline, and a transaction fraud detection benchmark with CSV scoring.

Author: Abderrahim amazal

## Implemented features

- Compare KBE and SPY over a selected analysis period.
- Compare adjusted-price performance from a common starting value.
- Calculate historical drawdown, recovery duration, and rolling daily volatility.
- Download market risk reports as CSV.
- Replay historical prices through Kafka and ingest events with Spark Structured Streaming.
- Clean duplicate events and reject conflicting business events.
- Generate experimental KBE probabilities with a saved scikit-learn pipeline.
- Compare Logistic Regression, GPU-trained XGBoost, and a GPU-trained PyTorch MLP for fraud detection.
- Display held-out fraud evaluation, score compatible CSV files, filter review alerts, and download scoring results.

The market replay is historical, not a live market feed. The dashboard displays reports from completed processing runs. Fraud CSV scoring invokes the saved model when a file is uploaded.

## Data

### Market prices

Source: Yahoo Finance, downloaded with yfinance using `auto_adjust=True`.

The local snapshots cover 2010-01-04 through 2025-12-31:

- `data/kbe_prices_v1.csv`
- `data/spy_prices_v1.csv`

The dashboard reads these snapshots without downloading fresh prices.

### Fraud transactions

Source: [ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Place the downloaded `creditcard.csv` in `data/fraud/`. The raw dataset is excluded from Git.

- 284,807 transactions, including 492 labeled fraud cases.
- Two days of transactions from September 2013.
- `V1`–`V28` are anonymized PCA components; `Amount` is the transaction amount.
- `Time` is elapsed time in seconds; `Class` is the target (0: normal, 1: fraud).

Exact duplicate rows were retained because the dataset does not provide a unique transaction identifier. Time-based boundaries keep identical full rows in the same split.

## Environment and installation

Developed on Windows with Python 3.12.10 and an NVIDIA RTX 3070 Laptop GPU (8 GB VRAM).

From the project folder, create an environment if needed:

```powershell
py -3.12 -m venv .venv-check
.\.venv-check\Scripts\python.exe -m pip install -r requirements.txt
```

For the optional PyTorch MLP training experiment:

```powershell
.\.venv-check\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

The recorded GPU build is `torch==2.11.0+cu128`; XGBoost is `3.4.1`. GPU training requires a compatible NVIDIA GPU and driver. The dashboard does not require PyTorch, GPU training, or a running Kafka service.

Verify installed package compatibility:

```powershell
.\.venv-check\Scripts\python.exe -m pip check
```

## Run the dashboard

```powershell
.\run_app.ps1
```

Alternatively:

```powershell
.\.venv-check\Scripts\python.exe -m streamlit run app.py
```

Keep the market snapshots and generated reports available. The original market experiment panel uses `results/experiment_v1/test_metrics.csv`; replay panels use the reports in `results/replay/`.

The fraud evaluation panel uses these files in `results/fraud/xgboost_v1/`:

- `decision_policy.json`
- `test_decision_results.json`
- `test_metrics.csv`
- `test_predictions.csv`

Fraud CSV scoring additionally requires:

- `models/fraud/xgboost_v1.json`
- `results/fraud/xgboost_v1/metadata.json`
- `results/fraud/xgboost_v1/decision_policy.json`

The policy records the model's SHA-256 fingerprint. Preserve the evaluated model and its matching policy together. Missing artifacts must be restored or regenerated before the corresponding functionality can work.

### Score a CSV

Use the **Score transaction CSV** section and upload a compatible file containing `V1`–`V28` and `Amount`. The current upload limit is 50 MB.

Features must have the same representation as the ULB dataset. Arbitrary bank CSV files containing only amounts, merchant names, or customer details are not supported; the original PCA transformation is unavailable.

Extra columns, including `Class`, are ignored for scoring. Alerts request manual review, not automatic blocking. No alert does not guarantee that a transaction is legitimate.

The optional `data/fraud/demo_transactions.csv` contains 1,000 test transactions without labels and is used to demonstrate the interface, not as an additional holdout evaluation.

## Market risk definitions

- **Period return:** change between the first and last selected trading-session adjusted closes.
- **Historical drawdown:** decline from the highest adjusted close observed since the beginning of the available history.
- **Rolling volatility:** sample standard deviation of 20 daily returns (`ddof=1`), without annualization. A complete window requires 21 prices.
- **Recovery duration:** calendar days from the deepest selected trough to the first subsequent close reaching its previous peak.

For the historical risk view, drawdown and volatility are calculated before slicing the selected period. Recovery searches can extend beyond the selected end date through the end of the available snapshot.

## Historical replay pipeline

1. A Python producer sends price events to Kafka.
2. Spark Structured Streaming ingests new messages into raw Parquet files.
3. Batch cleaning removes identical business events and rejects conflicting values for the same event ID.
4. Batch Spark jobs calculate daily returns, three-session momentum, and 20-return rolling volatility.
5. The saved KBE model generates probabilities; reports are exported for Streamlit.

Checkpointing records Kafka reading progress. It does not deduplicate repeated business events; the cleaning stage handles those separately.

### Update replay reports

Start Docker Desktop, then run:

```powershell
docker compose up -d kafka
.\.venv-check\Scripts\python.exe replay_prices.py
.\run_replay_pipeline.ps1
```

Run each command after the previous one finishes. Re-running the producer resends historical events. The current replay sends prices from the snapshot's beginning through the end of 2023, preserving chronological history for feature calculation.

The replay pipeline requires a compatible Java/Hadoop setup for the local Spark installation. Update the author's `HADOOP_HOME` path in `run_replay_pipeline.ps1` to match your installation.

It also requires `results/experiment_v1/kbe_model.joblib`. If absent, regenerate the original experiment with:

```powershell
.\.venv-check\Scripts\python.exe lesson_4.py
```

The replay pipeline runs ingestion, cleaning, feature calculation, feature checks, KBE scoring, and 2023 replay evaluation. It does not retrain the KBE model.

### Stop services

Stop Streamlit with `Ctrl+C`. Stop Kafka while preserving its volume:

```powershell
docker compose stop kafka
```

Avoid `docker compose down -v` if you want to preserve stored Kafka events. Preserve checkpoint directories to resume reading progress.

## Experimental KBE model

The pipeline combines `StandardScaler` and `LogisticRegression` with:

- Daily return: `Close / previous_close - 1`.
- Three-session momentum: `Close / Close.shift(3) - 1`.
- Rolling sample standard deviation of 20 daily returns.

The positive class means the adjusted closing price five trading sessions later is at least 3% below the current close (`future_return <= -0.03`). It does not mean any dip occurring during those five sessions.

Training uses observations before 2020 with label outcomes also ending before 2020. Validation covers 2020–2022, with label outcomes ending before 2023. The full test period covers 2023–2025 wherever five-session outcomes are available.

| Period | Model | Brier ↓ | ROC AUC ↑ | Average Precision ↑ |
|---|---|---:|---:|---:|
| Validation | Baseline | 0.186008 | 0.500000 | 0.226365 |
| Validation | Logistic | 0.180834 | 0.544467 | 0.275734 |
| Full test | Baseline | 0.129243 | 0.500000 | 0.151272 |
| Full test | Logistic | 0.132183 | 0.444223 | 0.133373 |

The constant baseline uses the training positive rate. Logistic improved the validation metrics but underperformed the baseline on all three full-test metrics. These results do not establish useful predictive performance.

### 2023 replay evaluation

Replay generated 250 KBE predictions. Evaluation retained 245 whose five-session outcomes ended within 2023; five observations were excluded.

| Model | Brier ↓ | ROC AUC ↑ | Average Precision ↑ |
|---|---:|---:|---:|
| Baseline | 0.153736 | 0.500000 | 0.183673 |
| Logistic | 0.155496 | 0.458222 | 0.168554 |

This uses a previously evaluated test period, not a new untouched holdout.

### Validation experiments

A volatility-only model achieved Brier 0.180236 versus 0.180834 for the three-feature model. Paired circular block bootstrap 95% intervals included zero with blocks of 5, 20, and 60 sessions. The comparison did not establish a clear advantage.

Annual expanding-window retraining achieved Brier 0.180953, ROC AUC 0.533613, and Average Precision 0.272486. It did not improve the aggregate validation metrics over the fixed model.

The original model remains an experimental reference. SPY is used for market-risk comparison, not scored with this KBE model.

## Fraud model comparison

The chronological split uses elapsed `Time`, keeping transactions with identical timestamps together:

| Split | Time boundary (seconds) | Transactions | Fraud cases |
|---|---|---:|---:|
| Train | `Time < 120396` | 170,882 | 360 |
| Validation | `120396 <= Time < 145248` | 56,963 | 57 |
| Test | `Time >= 145248` | 56,962 | 75 |

All candidates use `V1`–`V28` and `Amount`. `Time` defines the split; `Class` is never a model input. Scaling for Logistic and MLP is fitted on training data only.

| Validation model | Brier ↓ | ROC AUC ↑ | Average Precision ↑ |
|---|---:|---:|---:|
| Baseline | 0.001001 | 0.500000 | 0.001001 |
| Logistic | 0.000549 | 0.968405 | 0.650710 |
| XGBoost | 0.000325 | 0.981913 | 0.786476 |
| MLP | 0.000339 | 0.950579 | 0.758444 |

XGBoost was trained on `cuda:0` using 300 boosting rounds and depth 3. The MLP used architecture 29 → 64 → 32 → 1, dropout 0.1, and 20 fixed epochs on the RTX 3070. The MLP result is from one seeded run, not a multi-seed study.

At the same budget of 94 validation reviews:

| Model | Fraud detected | False alerts | Fraud missed |
|---|---:|---:|---:|
| Logistic | 44 | 50 | 13 |
| MLP | 43 | 51 | 14 |
| XGBoost | 46 | 48 | 11 |

XGBoost was selected using validation results. Its threshold was frozen at **0.080714114** before evaluating Test. The selection rule chose the largest alert count not exceeding 94 validation transactions, keeping equal-score groups together. The resulting policy uses `probability >= threshold` and requests manual review.

### Frozen-policy test results

| Model | Brier ↓ | ROC AUC ↑ | Average Precision ↑ |
|---|---:|---:|---:|
| Baseline | 0.001316 | 0.500000 | 0.001317 |
| XGBoost | 0.000391 | 0.982377 | 0.798469 |

| Decision outcome | Value |
|---|---:|
| Fraud detected (TP) | 59 |
| Fraud missed (FN) | 16 |
| Normal transactions flagged (FP) | 47 |
| Normal transactions not flagged (TN) | 56,840 |
| Total review alerts | 106 |
| Recall | 78.67% |
| Precision | 55.66% |
| Reviews per 1,000 transactions | 1.86 |

A frozen threshold does not guarantee a fixed number of alerts on future data. Average Precision is a ranking metric, not the fraction of fraud caught at the selected threshold.

## Verification

```powershell
.\.venv-check\Scripts\python.exe chek_risk.py
.\.venv-check\Scripts\python.exe check_replay_risk.py
.\.venv-check\Scripts\python.exe check_fraud_inference.py
```

- Risk checks cover recovery, unrecovered drawdown, and no drawdown.
- Replay checks compare dates, daily returns, momentum, and rolling volatility against independent pandas calculations from the original market CSVs, including initial missing values.
- Fraud inference checks compare scores and alerts for 1,000 held-out transactions with the saved test evaluation.
- Missing required columns, missing feature values, and negative amounts are rejected.

These checks require the original datasets and the relevant generated reports and models. Optional GPU checks are `check_xgboost_gpu.py` and a direct `torch.cuda.is_available()` check.

## Experiment scripts and artifact preservation

Fraud scripts are numbered by workflow stage:

| Script | Purpose |
|---|---|
| `fraud_01_inspect.py` | Inspect data and chronological boundaries |
| `fraud_02_train.py` | Train and evaluate Logistic on Validation |
| `fraud_03_thresholds.py` | Report Logistic validation thresholds |
| `fraud_04_xgboost.py` | Train GPU XGBoost and compare Validation metrics |
| `fraud_05_review_budget.py` | Compare candidates at 94 validation reviews |
| `fraud_06_mlp.py` | Train GPU MLP and compare Validation metrics |
| `fraud_07_freeze_policy.py` | Freeze the XGBoost validation decision policy |
| `fraud_08_test.py` | Evaluate the frozen model and policy on Test |

These training scripts write to their versioned artifact paths. Do not rerun them over the evaluated artifacts merely to start the dashboard. To reproduce experiments, use a separate working copy or change the output version paths consistently.

`fraud_07_freeze_policy.py` intentionally refuses to overwrite an existing policy. Preserve the model, metadata, policy, and evaluation reports together. A new model version requires validation and its own decision policy before final evaluation.

## Scope and limitations

- This is a local prototype with historical market analytics and a transaction fraud benchmark, not a deployed banking system.
- Kafka ingestion is streaming; downstream market cleaning, feature calculation, scoring, and report export currently run as batch jobs.
- The KBE model did not demonstrate useful full-test predictive performance. The project does not establish profitable trading performance.
- Fraud data covers only two days and uses precomputed anonymized PCA features. The original PCA fitting procedure cannot be reconstructed from the supplied data; the chronological split does not prove that upstream preprocessing was time-isolated.
- Fraud scores have not been separately calibrated as reliable individual probabilities.
- No long-term production monitoring, automatic transaction blocking, or live bank integration is implemented.
- Credit-risk/default modeling is not implemented.
