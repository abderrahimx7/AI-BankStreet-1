import csv
import json
from pathlib import Path
from kafka import KafkaProducer

project_folder = Path(__file__).resolve().parent

with (
    project_folder / "data" / "kbe_prices_v1.csv"
).open(encoding="utf-8-sig", newline="") as file:
    row = next(csv.DictReader(file))

event = {
    "event_id": f"KBE:{row['Date']}",
    "ticker": "KBE",
    "date": row["Date"],
    "close": float(row["Close"]),
    "volume": int(row["Volume"]),
    "source": "historical_replay"
}

producer = KafkaProducer(
    bootstrap_servers="127.0.0.1:9092",
    key_serializer=lambda key: key.encode("utf-8"),
    value_serializer=lambda value: json.dumps(
        value, allow_nan=False
    ).encode("utf-8"),
    acks="all"
)

try:
    metadata = producer.send(
        "market-prices",
        key=event["ticker"],
        value=event
    ).get(timeout=30)

    print("Event:", event)
    print("Partition:", metadata.partition)
    print("Offset:", metadata.offset)
    print("Kafka acknowledged the event.")
finally:
    producer.close()