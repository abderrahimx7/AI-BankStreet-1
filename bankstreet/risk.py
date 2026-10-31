import pandas as pd


def calculate_drawdown(close):
    running_peak = close.cummax()
    return close / running_peak - 1


def calculate_rolling_volatility(close, window=20):
    daily_returns = close.pct_change(fill_method=None)

    return daily_returns.rolling(
        window=window,
        min_periods=window
    ).std(ddof=1)


def summarize_drawdown(close, start=None, end=None):
    close = close.dropna().sort_index()

    if close.empty:
        raise ValueError("No prices available.")

    if not close.index.is_unique:
        raise ValueError("Price dates must be unique.")

    # Calculate using the full history first
    drawdown = calculate_drawdown(close)
    period_drawdown = drawdown.loc[start:end]

    if period_drawdown.empty:
        raise ValueError("No prices in the selected period.")

    if period_drawdown.min() == 0:
        return {
            "drawdown_pct": 0.0,
            "peak_date": pd.NaT,
            "trough_date": pd.NaT,
            "recovery_date": pd.NaT,
            "trough_to_recovery_days": pd.NA
        }
    trough_date = period_drawdown.idxmin()
    trough_value = close.loc[trough_date]

    history = close.loc[:trough_date]
    peak_value = history.max()

    # Last occurrence of that peak before the trough
    peak_date = history.loc[
        history == peak_value
    ].index[-1]

    # Search all available dates after the trough
    recovered = close.loc[trough_date:]
    recovered = recovered.loc[recovered >= peak_value]

    recovery_date = (
        recovered.index[0]
        if not recovered.empty
        else pd.NaT
    )

    recovery_days = (
        (recovery_date - trough_date).days
        if pd.notna(recovery_date)
        else pd.NA
    )

    return {
        "drawdown_pct": period_drawdown.loc[trough_date] * 100,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "trough_to_recovery_days": recovery_days
    }


