from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

project_folder = Path(__file__).resolve().parent

spark = (
    SparkSession.builder
    .appName("BankStreetKafkaCheck")
    .master("local[2]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.2"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    raw_events = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "127.0.0.1:9092")
        .option("subscribe", "market-prices")
        .option("startingOffsets", "earliest")
        .load()
    )

    schema = """
        event_id STRING,
        ticker STRING,
        date DATE,
        close DOUBLE,
        volume BIGINT,
        source STRING
    """

    events = raw_events.select(
        "partition",
        "offset",
        F.from_json(
            F.col("value").cast("string"),
            schema
        ).alias("event")
    ).select(
        "partition",
        "offset",
        "event.*"
    )

    query = (
        events.writeStream
        .format("parquet")
        .outputMode("append")
        .option(
            "path",
            (
                project_folder
                / "data"
                / "streaming"
                / "market_prices_raw"
            ).as_posix()
        )
        .option(
            "checkpointLocation",
            (
                project_folder
                / "checkpoints"
                / "market_prices_raw_v1"
            ).as_posix()
        )
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    print("Kafka events saved to Parquet.")

    query.awaitTermination()
    print("Kafka streaming read completed.")

finally:
    spark.stop()