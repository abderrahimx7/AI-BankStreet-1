import pandas as pd


def build_labels(prices, horizon=5, threshold=-0.03):
    labels = pd.DataFrame(index=prices.index)

    labels["future_return"] = (
        prices["Close"].shift(-horizon)
        / prices["Close"]
        - 1
    )

    labels["label_end_date"] = pd.Series(
        prices.index,
        index=prices.index
    ).shift(-horizon)

    labels["target"] = pd.Series(
        pd.NA,
        index=prices.index,
        dtype="Int64"
    )

    known_future = labels["future_return"].notna()

    labels.loc[known_future, "target"] = (
        labels.loc[known_future, "future_return"] <= threshold
    ).astype(int)

    return labels