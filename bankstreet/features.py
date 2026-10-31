import pandas as pd


def build_features(prices):
    features = pd.DataFrame(index=prices.index)

    features["daily_return"] = (
        prices["Close"].pct_change(fill_method=None)
    )

    features["momentum"] = (
        prices["Close"] / prices["Close"].shift(3) - 1
    )

    features["volatility_20d"] = (
        features["daily_return"]
        .rolling(window=20, min_periods=20)
        .std(ddof=1)
    )

    return features