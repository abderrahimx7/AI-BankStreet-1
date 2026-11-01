from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


project_folder = Path(__file__).resolve().parent

spark = (
    SparkSession.builder
    .appName("BankStreetVerifyReplayRisk")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    actual = spark.read.parquet(
        (
            project_folder
            / "data"
            / "streaming"
            / "market_risk"
        ).as_posix()
    ).toPandas()

    for ticker in ["KBE", "SPY"]:
        observed = actual.loc[actual["ticker"] == ticker].copy()
        observed["date"] = pd.to_datetime(observed["date"])
        observed = (
            observed.sort_values("date")
            .reset_index(drop=True)
        )

        if observed.empty:
            raise ValueError(f"No results found for {ticker}.")

        sample_size = len(observed)

        expected = pd.read_csv(
            project_folder / "data" / f"{ticker.lower()}_prices_v1.csv",
            parse_dates=["Date"],
        )

        expected = (
            expected.sort_values("Date")
            .head(sample_size)
            .reset_index(drop=True)
        )

        expected["daily_return"] = expected["Close"].pct_change(
            fill_method=None
        )
        expected["momentum"] = (
            expected["Close"] / expected["Close"].shift(3) - 1
        )

        expected["volatility_20d"] = (
            expected["daily_return"]
            .rolling(window=20, min_periods=20)
            .std(ddof=1)
        )

        pd.testing.assert_series_equal(
            observed["date"].astype("datetime64[ns]"),
            expected["Date"].astype("datetime64[ns]"),
            check_names=False,
        )

        for column in ["daily_return", "volatility_20d"]:
            np.testing.assert_allclose(
                observed[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
                err_msg=f"{ticker}: mismatch in {column}",
            )

        print(
            f"{ticker}: {sample_size} rows checked. "
            "Dates, returns and volatility match pandas."
        )

    print("Replay risk verification passed.")

finally:
    spark.stop()