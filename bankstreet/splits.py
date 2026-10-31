import pandas as pd


def split_by_time(
    dataset,
    validation_start="2020-01-01",
    test_start="2023-01-01"
):
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)

    if validation_start >= test_start:
        raise ValueError("Validation must start before Test.")

    train = dataset.loc[
        (dataset.index < validation_start)
        & (dataset["label_end_date"] < validation_start)
    ].copy()

    validation = dataset.loc[
        (dataset.index >= validation_start)
        & (dataset.index < test_start)
        & (dataset["label_end_date"] < test_start)
    ].copy()

    test = dataset.loc[
        dataset.index >= test_start
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("One of the time splits is empty.")

    return train, validation, test