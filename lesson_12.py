from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

project_folder = Path(__file__).resolve().parent

spark = (
    SparkSession.builder
    .appName("BankStreetAnnualRisk")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    features_path = (
        project_folder
        / "data"
        / "processed"
        / "risk_features_v1"
    )

    features = spark.read.parquet(
        features_path.as_posix()
    )

    annual_risk = (
        features
        .withColumn("year", F.year("date"))
        .groupBy("ticker", "year")
        .agg(
            F.count("*").alias("sessions"),
            F.min("drawdown").alias("deepest_drawdown"),
            F.max("volatility_20d").alias("max_daily_volatility")
        )
    )

    annual_risk.select(
        "ticker",
        "year",
        "sessions",
        F.round(
            F.col("deepest_drawdown") * 100, 2
        ).alias("deepest_drawdown_pct"),
        F.round(
            F.col("max_daily_volatility") * 100, 2
        ).alias("max_daily_volatility_pct")
    ).orderBy("year", "ticker").show(40)
    report_path = (
        project_folder
        / "results"
        / "spark"
        / "annual_risk_v1"
    )

    if not report_path.exists():
        annual_risk.write.mode("errorifexists").parquet(
            report_path.as_posix()
        )

    print("Annual risk report available at:", report_path)
finally:
    spark.stop()