from pathlib import Path
import pandas as pd
import yfinance as yf

prices_path = (
    Path(__file__).resolve().parent
    / "data"
    / "spy_prices_v1.csv"
)

if prices_path.exists():
    spy = pd.read_csv(
        prices_path,
        index_col="Date",
        parse_dates=["Date"]
    )
    print("Loaded existing snapshot.")
else:
    spy = yf.download(
        "SPY",
        start="2010-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=True,
        multi_level_index=False,
        threads=False,
        timeout=30,
        progress=False
    )

    if spy is None or spy.empty:
        raise RuntimeError("SPY download failed.")

    spy = spy.sort_index()

    prices_path.parent.mkdir(parents=True, exist_ok=True)

    spy.to_csv(
        prices_path,
        index=True,
        index_label="Date"
    )

    print("Snapshot saved:", prices_path)

spy = spy.sort_index()

print(spy.head())
print("\nSessions:", len(spy))
print("\nMissing values:")
print(spy.isna().sum())