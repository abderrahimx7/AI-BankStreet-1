from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd 
import numpy as np

project_folder = Path(__file__).resolve().parent

spark = (
    SparkSession.builder
    .appName("BankStreetPrices")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    tables = []

    for ticker in ["KBE", "SPY"]:
        raw = (
            spark.read
            .option("header", True)
            .csv(
                str(
                    project_folder
                    / "data"
                    / f"{ticker.lower()}_prices_v1.csv"
                )
            )
        )

        typed = raw.select(
            F.to_date("Date", "yyyy-MM-dd").alias("date"),
            F.lit(ticker).alias("ticker"),
            F.col("Open").cast("double").alias("open"),
            F.col("High").cast("double").alias("high"),
            F.col("Low").cast("double").alias("low"),
            F.col("Close").cast("double").alias("close"),
            F.col("Volume").cast("long").alias("volume")
        )

        tables.append(typed)

    # Combine both tables after the loop
    prices = tables[0].unionByName(tables[1])

    prices.select(
        "date", "ticker", "close", "volume"
    ).orderBy("date", "ticker").show(10)

    prices.groupBy("ticker").count().show()
    print("Total rows:", prices.count())

    # Check missing values
    missing_counts = prices.select([
        F.sum(
            F.when(F.col(column).isNull(), 1).otherwise(0)
        ).alias(column)
        for column in prices.columns
    ])

    print("Missing values:")
    missing_counts.show()

    # Check duplicate ticker-date pairs
    duplicates = (
        prices.groupBy("ticker", "date")
        .count()
        .filter(F.col("count") > 1)
    )

    duplicate_count = duplicates.count()
    print("Duplicate ticker-date groups:", duplicate_count)

    assert all(
        value == 0
        for value in missing_counts.first()
    ), "Missing values detected."

    assert duplicate_count == 0, "Duplicate prices detected."

    print("Price quality checks passed.")
    output_path = (
        project_folder
        / "data"
        / "processed"
        / "prices_v2"
    )

    if not output_path.exists():
        prices.write.mode("errorifexists").parquet(
            output_path.as_posix()
        )

    saved_prices = spark.read.parquet(
        output_path.as_posix()
    )

    assert output_path.is_dir()
    assert saved_prices.count() == 8048

    print("Prices available at:", output_path)
    by_ticker_date = (
        Window
        .partitionBy("ticker")
        .orderBy("date")
    )

    returns = saved_prices.withColumn(
        "previous_close",
        F.lag("close", 1).over(by_ticker_date)
    )

    returns = returns.withColumn(
        "daily_return",
        F.col("close") / F.col("previous_close") - 1
    )

    returns.select(
        "date",
        "ticker",
        F.round("close", 4).alias("close"),
        F.round(
            F.col("daily_return") * 100, 4
        ).alias("daily_return_pct")
    ).orderBy("date", "ticker").show(10)
    rolling_20 = by_ticker_date.rowsBetween(-19, 0)

    risk_features = returns.withColumn(
        "return_count_20",
        F.count("daily_return").over(rolling_20)
    )

    risk_features = risk_features.withColumn(
        "volatility_20d",
        F.when(
            F.col("return_count_20") == 20,
            F.stddev_samp("daily_return").over(rolling_20)
        )
    )

    risk_features.filter(
        F.col("volatility_20d").isNotNull()
    ).select(
        "date",
        "ticker",
        F.round("volatility_20d", 6).alias("volatility_20d")
    ).orderBy("date", "ticker").show(10)
    period_risk = risk_features.filter(
        F.col("date").between(
            "2018-03-09",
            "2021-02-16"
        )
    )

    period_risk.groupBy("ticker").agg(
        F.round(
            F.max("volatility_20d") * 100,
            2
        ).alias("max_daily_volatility_pct")
    ).orderBy("ticker").show()
    for ticker in ["KBE", "SPY"]:
        source = pd.read_csv(
            project_folder
            / "data"
            / f"{ticker.lower()}_prices_v1.csv",
            index_col="Date",
            parse_dates=["Date"]
        ).sort_index()

        expected = (
            source["Close"]
            .pct_change(fill_method=None)
            .rolling(20, min_periods=20)
            .std(ddof=1)
        )

        spark_rows = (
            risk_features
            .filter(F.col("ticker") == ticker)
            .select("date", "volatility_20d")
            .orderBy("date")
            .collect()
        )

        actual = pd.Series(
            [
                row["volatility_20d"]
                for row in spark_rows
            ],
            index=pd.to_datetime([
                row["date"]
                for row in spark_rows
            ]),
            dtype="float64"
        )

        assert actual.index.equals(expected.index)

        np.testing.assert_allclose(
            actual.to_numpy(),
            expected.to_numpy(),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True
        )

        print(f"{ticker}: Spark/Pandas volatility check passed.")
    history_window = by_ticker_date.rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow
    )

    risk_features = risk_features.withColumn(
        "running_peak",
        F.max("close").over(history_window)
    )

    risk_features = risk_features.withColumn(
        "drawdown",
        F.col("close") / F.col("running_peak") - 1
    )
    features_path = (
        project_folder
        / "data"
        / "processed"
        / "risk_features_v1"
    )

    if not features_path.exists():
        risk_features.write.mode("errorifexists").parquet(
            features_path.as_posix()
        )
        print("Risk features saved.")
    else:
        print("Existing risk features retained.")

    saved_features = spark.read.parquet(
        features_path.as_posix()
    )

    assert saved_features.count() == 8048

    saved_features.select(
        "date", "ticker",
        "daily_return", "volatility_20d",
        "running_peak", "drawdown"
    ).orderBy("date", "ticker").show(6)

    print("Risk features read-back check passed.")

    risk_features.filter(
        F.col("date").between(
            "2018-03-09",
            "2021-02-16"
        )
    ).groupBy("ticker").agg(
        F.round(
            F.min("drawdown") * 100,
            2
        ).alias("deepest_drawdown_pct")
    ).orderBy("ticker").show()
finally:
    spark.stop()