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