import pandas as pd

from bankstreet.risk import summarize_drawdown

close = pd.Series(
    [100.0, 80.0, 90.0, 100.0],
    index=pd.date_range("2020-01-01", periods=4)
)

result = summarize_drawdown(close)

assert abs(result["drawdown_pct"] - (-20.0)) < 1e-10
assert result["peak_date"] == pd.Timestamp("2020-01-01")
assert result["trough_date"] == pd.Timestamp("2020-01-02")
assert result["recovery_date"] == pd.Timestamp("2020-01-04")
assert result["trough_to_recovery_days"] == 2

print("Drawdown and recovery check passed.")

unrecovered_close = close.iloc[:3]

result = summarize_drawdown(unrecovered_close)

assert abs(result["drawdown_pct"] - (-20.0)) < 1e-10
assert pd.isna(result["recovery_date"])
assert pd.isna(result["trough_to_recovery_days"])

print("Unrecovered drawdown check passed.")



rising_close = pd.Series(
    [100.0, 105.0, 110.0],
    index=pd.date_range("2020-01-01", periods=3)
)

result = summarize_drawdown(rising_close)

assert result["drawdown_pct"] == 0.0
assert pd.isna(result["trough_date"])
assert pd.isna(result["recovery_date"])
assert pd.isna(result["trough_to_recovery_days"])

print("No-drawdown check passed.")