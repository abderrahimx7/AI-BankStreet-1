from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


project_folder = Path(__file__).resolve().parent
streaming_folder = project_folder / "data" / "streaming"

spark = (
    SparkSession.builder
    .appName("BankStreetReplayReturns")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    prices = spark.read.parquet(
        (streaming_folder / "market_prices_clean").as_posix()
    )

    # A separate chronological sequence for each ticker.
    price_window = Window.partitionBy("ticker").orderBy("date")

    returns = (

        prices
        .withColumn(
            "previous_close",
            F.lag("close", 1).over(price_window)
        )
        .withColumn(
            "momentum",
            F.col("close") / F.lag("close", 3).over(price_window) - 1
        )
        .withColumn(
            "daily_return",
            F.col("close") / F.col("previous_close") - 1
        )
    )

    # Multiply by 100 only for display.
    returns.select(
        "ticker",
        "date",
        "close",
        "previous_close",
        F.round(
            F.col("daily_return") * 100, 4
        ).alias("daily_return_pct"),
    ).orderBy("ticker", "date").show(20, truncate=False)

    returns.groupBy("ticker").agg(
        F.count("*").alias("price_rows"),
        F.count("daily_return").alias("return_rows"),
    ).orderBy("ticker").show()

    returns.write.mode("overwrite").parquet(
        (streaming_folder / "market_returns").as_posix()
    )

    print("Daily returns saved.")

finally:
    spark.stop()