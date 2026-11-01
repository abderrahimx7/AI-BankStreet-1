from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


project_folder = Path(__file__).resolve().parent
streaming_folder = project_folder / "data" / "streaming"

spark = (
    SparkSession.builder
    .appName("BankStreetReplayVolatility")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    returns = spark.read.parquet(
        (streaming_folder / "market_returns").as_posix()
    )

    rolling_window = (
        Window.partitionBy("ticker")
        .orderBy("date")
        .rowsBetween(-19, 0)
    )

    risk = (
        returns
        .withColumn(
            "returns_in_window",
            F.count("daily_return").over(rolling_window)
        )
        .withColumn(
            "volatility_20d",
            F.when(
                F.col("returns_in_window") == 20,
                F.stddev_samp("daily_return").over(rolling_window)
            )
        )
    )

    risk.select(
        "ticker",
        "date",
        "returns_in_window",
        F.round(
            F.col("volatility_20d") * 100, 4
        ).alias("volatility_20d_pct"),
    ).orderBy("ticker", "date").show(60, truncate=False)

    risk.groupBy("ticker").agg(
        F.count("*").alias("price_rows"),
        F.count("volatility_20d").alias("volatility_rows"),
    ).orderBy("ticker").show()

    risk.write.mode("overwrite").parquet(
        (streaming_folder / "market_risk").as_posix()
    )
    report_folder = project_folder / "results" / "replay"
    report_folder.mkdir(parents=True, exist_ok=True)

    risk.orderBy("ticker", "date").toPandas().to_csv(
        report_folder / "market_risk.csv",
        index=False,
    )
    print("Rolling volatility saved.")

finally:
    spark.stop()