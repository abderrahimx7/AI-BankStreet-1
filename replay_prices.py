import csv
import json
import time
from datetime import date
from pathlib import Path

from kafka import KafkaProducer


project_folder = Path(__file__).resolve().parent

ROWS_PER_TICKER = 250
DELAY_SECONDS = 0.002

files = {
    "KBE": "kbe_prices_v1.csv",
    "SPY": "spy_prices_v1.csv",
}

events = []

for ticker, filename in files.items():
    path = project_folder / "data" / filename

    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    # Sort before selecting the first 10 trading sessions.
    rows.sort(key=lambda row: date.fromisoformat(row["Date"]))

    rows = [
        row for row in rows
        if date.fromisoformat(row["Date"]) < date(2024, 1, 1)
    ]

    for row in rows:
        events.append({
            "event_id": f"{ticker}:{row['Date']}",
            "ticker": ticker,
            "date": row["Date"],
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
            "source": "historical_replay",
        })

# Interleave the two tickers by historical date.
events.sort(key=lambda event: (event["date"], event["ticker"]))

producer = KafkaProducer(
    bootstrap_servers="127.0.0.1:9092",
    key_serializer=lambda key: key.encode("utf-8"),
    value_serializer=lambda value: json.dumps(
        value, allow_nan=False
    ).encode("utf-8"),
    acks="all",
)

try:
    for index, event in enumerate(events, start=1):
        metadata = producer.send(
            "market-prices",
            key=event["ticker"],
            value=event,
        ).get(timeout=30)

        print(
            f"{index}/{len(events)} | "
            f"{event['event_id']} | "
            f"partition={metadata.partition} | "
            f"offset={metadata.offset}"
        )

        time.sleep(DELAY_SECONDS)

    print(f"Historical replay completed: {len(events)} events.")

finally:
    producer.close()