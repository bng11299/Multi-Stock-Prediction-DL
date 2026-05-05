import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

START = "2015-01-01"
END = "2024-12-31"

SEQ_LEN = 60
PRED_HORIZON = 30


# ---------------------------
# Download
# ---------------------------
def download_data():

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    prices = []

    for ticker in TICKERS:

        print(f"Downloading {ticker}")

        df = yf.download(ticker, start=START, end=END, progress=False)

        s = df["Close"].squeeze()
        s.name = ticker

        prices.append(s)

    data = pd.concat(prices, axis=1).dropna()

    data.to_csv(RAW_DIR / "close_prices.csv")

    return data


# ---------------------------
# Feature Engineering
# ---------------------------
def compute_features(prices):

    features = []

    for col in prices.columns:

        p = prices[col]

        df = pd.DataFrame(index=prices.index)

        # --- RETURNS ---
        df["return"] = np.log(p / p.shift(1))

        # --- MOVING AVERAGES ---
        df["ma_5"] = p.rolling(5).mean()
        df["ma_20"] = p.rolling(20).mean()

        # --- VOLATILITY ---
        df["volatility"] = df["return"].rolling(20).std()

        # --- RSI ---
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))

        # rename columns with ticker prefix
        df.columns = [f"{col}_{c}" for c in df.columns]

        features.append(df)

    features = pd.concat(features, axis=1)

    # Shift all features so prediction at time t only uses info available before t
    features = features.shift(1)
    
    return features.dropna()


# ---------------------------
# Targets (future returns)
# ---------------------------
def compute_targets(prices):

    future_returns = np.log(prices.shift(-PRED_HORIZON) / prices)

    return future_returns.dropna()


# ---------------------------
# Sequence builder
# ---------------------------
def create_sequences(features, target, seq_length=60):
    X, y = [], []

    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])

    return np.array(X), np.array(y)


# ---------------------------
# Main
# ---------------------------
def build_dataset():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    prices = download_data()

    features = compute_features(prices)
    features = features.shift(1)

    targets = compute_targets(prices)

    common_index = features.index.intersection(targets.index)
    features = features.loc[common_index]
    targets = targets.loc[common_index]

    features = features.replace([np.inf, -np.inf], np.nan)
    targets = targets.replace([np.inf, -np.inf], np.nan)

    mask = ~np.isnan(features).any(axis=1) & ~np.isnan(targets).any(axis=1)

    features = features[mask]
    targets = targets[mask]

    X, y = create_sequences(features.values, targets.values, SEQ_LEN)

    np.save(PROC_DIR / "X.npy", X)
    np.save(PROC_DIR / "y.npy", y)

    print("Dataset built successfully")
    print("Features shape:", X.shape)
    print("Targets shape:", y.shape)


if __name__ == "__main__":
    build_dataset()