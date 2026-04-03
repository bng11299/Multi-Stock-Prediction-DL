import yfinance as yf
import pandas as pd
from pathlib import Path
from time import sleep

DATA_DIR = Path("data/raw")

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA"
]

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"


def download_data():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_data = []

    for ticker in TICKERS:

        print(f"Downloading {ticker}")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False
        )

        df["Ticker"] = ticker
        all_data.append(df)

        sleep(1)  # prevents API/cache locking

    data = pd.concat(all_data)

    data.to_csv(DATA_DIR / "stocks.csv")

    print("Saved to data/raw/stocks.csv")


if __name__ == "__main__":
    download_data()