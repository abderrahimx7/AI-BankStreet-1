import sys
import os

# Use the same Python for Spark workers
os.environ["PYSPARK_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("BankStreetCheck")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("Spark version:", spark.version)

    numbers = spark.range(5)
    numbers.show()

    assert numbers.count() == 5
    print("Spark execution check passed.")
finally:
    spark.stop()