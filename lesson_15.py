from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


project_folder = Path(__file__).resolve().parent
streaming_folder = project_folder / "data" / "streaming"

spark = (
    SparkSession.builder
    .appName("BankStreetCleanReplay")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    raw = spark.read.parquet(
        (streaming_folder / "market_prices_raw").as_posix()
    )

    # Keep business fields; Kafka offsets differ between re-sends.
    columns = [
        "event_id", "ticker", "date",
        "close", "volume", "source",
    ]
    prices = raw.select(*columns)

    missing = prices.filter(
        F.expr(" OR ".join(f"{name} IS NULL" for name in columns))
    ).count()

    if missing:
        raise ValueError(f"Found {missing} rows with missing fields.")

    # Remove identical copies across all business fields.
    variants = prices.dropDuplicates()

    # Different values sharing the same identity require investigation.
    conflicts = (
        variants.groupBy("event_id")
        .count()
        .filter(F.col("count") > 1)
    )

    if conflicts.limit(1).count():
        conflicts.show(truncate=False)
        raise ValueError("Conflicting values for the same event_id.")

    raw_count = prices.count()
    clean_count = variants.count()

    print("Raw events:", raw_count)
    print("Unique events:", clean_count)
    print("Duplicate copies removed:", raw_count - clean_count)

    variants.orderBy("date", "ticker").show(20, truncate=False)

    output_path = streaming_folder / "market_prices_clean"

    variants.write.mode("overwrite").parquet(output_path.as_posix())

    print("Clean market prices saved.")

finally:
    spark.stop()