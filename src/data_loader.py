import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA"
]

START = "2015-01-01"
END = "2024-12-31"

SEQ_LEN = 60
PRED_HORIZON = 30


def download_data():

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_prices = []

    for ticker in TICKERS:

        print(f"Downloading {ticker}")

        df = yf.download(
            ticker,
            start=START,
            end=END,
            progress=False
        )

        prices = df["Close"].squeeze()
        prices.name = ticker

        all_prices.append(prices)

    data = pd.concat(all_prices, axis=1)

    data = data.dropna()

    data.to_csv(RAW_DIR / "close_prices.csv")

    return data


def normalize(data):

    return (data - data.mean()) / data.std()


def create_sequences(data):

    values = data.values

    X = []
    y = []

    for i in range(len(values) - SEQ_LEN - PRED_HORIZON):

        X.append(values[i:i+SEQ_LEN])
        y.append(values[i+SEQ_LEN+PRED_HORIZON])

    return np.array(X), np.array(y)


def build_dataset():

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    data = download_data()

    data = normalize(data)

    X, y = create_sequences(data)

    np.save(PROC_DIR / "X.csv", X)
    np.save(PROC_DIR / "y.csv", y)

    print("Dataset built successfully")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    build_dataset()