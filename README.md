# BankStreet — Risk Analytics

A Python dashboard comparing historical risk and performance
for KBE and SPY, with an archived machine-learning experiment.

## Features

- Select an analysis period.
- Compare adjusted-price performance from a common starting value.
- Measure historical drawdown and recovery duration.
- Compare daily volatility estimated over 20 trading sessions.
- Download the selected period's risk report as CSV.

## Data

Source: Yahoo Finance, downloaded using yfinance with
`auto_adjust=True`.

Local snapshots cover 2010-01-04 through 2025-12-31:

- `data/kbe_prices_v1.csv`
- `data/spy_prices_v1.csv`

The dashboard reads these snapshots without downloading fresh data.

## Run locally

From the project folder: Abderrahim amazal
Developed with Python 3.12.10.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard also requires the archived metrics file:
`results/experiment_v1/test_metrics.csv`.

## Check risk calculations

```powershell
python chek_risk.py
```

Checks cover recovery, an unrecovered drawdown, and no drawdown.

## Metric definitions

- Period return: change between the first and last selected
  trading-session adjusted closes.
- Historical drawdown: decline from the highest adjusted close
  observed since the beginning of the available history.
- Volatility: daily return standard deviation over 20 trading
  sessions, using `ddof=1`; it is not annualized.
- Recovery duration: calendar days from the deepest selected
  trough to the first subsequent close reaching its previous peak.

Drawdown and volatility are calculated before slicing the
selected period. Recovery searches can extend beyond the selected
end date, through the end of the available data.

## ML experiment

Target: a KBE adjusted-close return of -3% or worse after
5 trading sessions, rather than any dip during those sessions.

Training: before 2020.
Validation: 2020–2022.
Test: 2023–2025.

Rows whose labels cross split boundaries are excluded.

On the test set, logistic regression underperformed the constant
training-rate baseline:

| Model | Brier score | ROC-AUC | Average precision |
|---|---:|---:|---:|
| Constant baseline | 0.129243 | 0.500000 | 0.151272 |
| Logistic regression | 0.132183 | 0.444223 | 0.133373 |

These results do not establish useful predictive performance.

## Scope

This version provides historical market-risk analytics.
Credit-risk modeling and fraud detection are not implemented.

## Historical Replay and Risk Pipeline

BankStreet AI replays historical KBE and SPY prices through Kafka.
Spark Structured Streaming ingests the events into raw Parquet files.
Subsequent batch jobs remove duplicate events, calculate features,
and export reports for the Streamlit dashboard.

This is historical replay, not a live market feed. The dashboard
displays the results of the latest completed processing run.

### Processing stages

1. Python producer sends price events to Kafka.
2. Spark ingests new messages using checkpointed offsets.
3. The cleaning job removes identical business events and rejects
   conflicting values for the same event ID.
4. Spark calculates daily returns, three-session momentum, and
   rolling volatility over 20 daily returns.
5. A saved scikit-learn pipeline generates KBE probabilities.
6. Streamlit displays risk indicators, probabilities, and evaluation.

Checkpointing tracks Kafka reading progress. Business-level
deduplication handles events sent again by the producer.

### Verification

Spark feature values were compared with independently calculated
pandas values from the original CSV files.

Checks cover dates, daily returns, three-session momentum, and
20-session sample standard deviation, including missing values
during the initial history window.

## Experimental Machine Learning

The KBE model combines StandardScaler and LogisticRegression.

Features:
- Daily return.
- Three-session momentum.
- Rolling standard deviation of 20 daily returns.

The positive class means that the closing price five trading
sessions later is at least 3% below the current closing price.
It does not represent any price decline occurring within that window.

Training uses observations before 2020, with label outcomes also
ending before 2020. Validation covers 2020–2022. The original test
period starts in 2023.

### Original model results

| Period | Model | Brier ↓ | ROC AUC ↑ | Average Precision ↑ |
|---|---|---:|---:|---:|
| Validation | Baseline | 0.186008 | 0.500000 | 0.226365 |
| Validation | Logistic | 0.180834 | 0.544467 | 0.275734 |
| Full test | Baseline | 0.129243 | 0.500000 | 0.151272 |
| Full test | Logistic | 0.132183 | 0.444223 | 0.133373 |

The model improved on the validation baseline but underperformed
the baseline on all three full-test metrics.

### 2023 replay evaluation

The replay generated 250 KBE predictions. Evaluation retained 245
observations whose five-session outcomes ended within 2023.

| Model | Brier ↓ | ROC AUC ↑ | Average Precision ↑ |
|---|---:|---:|---:|
| Baseline | 0.153736 | 0.500000 | 0.183673 |
| Logistic | 0.155496 | 0.458222 | 0.168554 |

This replay uses a previously evaluated test period; it is not
a new untouched holdout.

### Validation experiments

A volatility-only model achieved Brier 0.180236, compared with
0.180834 for the three-feature model.

Paired circular block bootstrap intervals for this Brier difference
included zero at block lengths of 5, 20, and 60 sessions.
The comparison did not establish a clear advantage.

Annual expanding-window retraining achieved Brier 0.180953,
ROC AUC 0.533613, and Average Precision 0.272486.
It did not improve the aggregate validation metrics over the
fixed model.

The original model remains the experimental reference. These
experiments do not establish reliable predictive performance.

## Scope and Limitations

- Kafka ingestion uses streaming; downstream cleaning, feature
  calculation, scoring, and report generation currently use batch jobs.
- ML scoring is implemented for KBE.
- SPY is used for market risk comparison.
- Fraud detection is not implemented in this version.
- The project demonstrates data processing, verification, and honest
  model evaluation; it does not establish profitable trading performance.