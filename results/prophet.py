"""
Multi-Stock Prophet Forecasting Model
Features: Price, Returns, Moving Averages, Volatility, RSI
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

#CONFIG
STOCKS = ["AAPL", "MSFT", "GOOGL"]   # Add or remove tickers
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
FORECAST_HORIZON = 30                 # Days to forecast (configurable)
RESULTS_DIR = "results/prophet"
# ─────────────────────────────────────────


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download stock data from yfinance."""
    print(f"  Fetching {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns, moving averages, volatility, and RSI."""

    # Daily returns
    df["returns"] = df["close"].pct_change()

    # Moving averages
    df["ma_7"] = df["close"].rolling(window=7).mean()
    df["ma_21"] = df["close"].rolling(window=21).mean()
    df["ma_50"] = df["close"].rolling(window=50).mean()

    # Volatility (21-day rolling std of returns)
    df["volatility"] = df["returns"].rolling(window=21).std()

    # RSI (14-day)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df = df.dropna()
    return df


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe for Prophet (ds/y columns + regressors)."""
    prophet_df = df.reset_index().rename(columns={"Date": "ds", "close": "y"})
    # Ensure ds is tz-naive
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
    return prophet_df


def build_and_train(prophet_df: pd.DataFrame) -> Prophet:
    """Build Prophet model with extra regressors and fit."""
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )

    # Add feature regressors
    for col in ["returns", "ma_7", "ma_21", "ma_50", "volatility", "rsi"]:
        model.add_regressor(col)

    model.fit(prophet_df)
    return model


def make_forecast(model: Prophet, prophet_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create future dataframe and predict."""
    future = model.make_future_dataframe(periods=horizon)

    # Fill regressors for future dates using last known values
    regressor_cols = ["returns", "ma_7", "ma_21", "ma_50", "volatility", "rsi"]
    for col in regressor_cols:
        last_val = prophet_df[col].iloc[-1]
        future[col] = prophet_df.set_index("ds")[col].reindex(future["ds"]).fillna(last_val).values

    forecast = model.predict(future)
    return forecast


def evaluate(prophet_df: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Compute MAE and RMSE on the training period."""
    merged = prophet_df[["ds", "y"]].merge(forecast[["ds", "yhat"]], on="ds")
    mae = mean_absolute_error(merged["y"], merged["yhat"])
    rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}


def plot_results(model: Prophet, forecast: pd.DataFrame, ticker: str, out_dir: str):
    """Save forecast and components plots."""
    # Forecast plot
    fig1 = model.plot(forecast, figsize=(12, 5))
    plt.title(f"{ticker} — Prophet Forecast")
    fig1.savefig(os.path.join(out_dir, f"{ticker}_forecast.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Components plot
    fig2 = model.plot_components(forecast, figsize=(12, 8))
    fig2.savefig(os.path.join(out_dir, f"{ticker}_components.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"    Plots saved.")


def plot_features(df: pd.DataFrame, ticker: str, out_dir: str):
    """Save a feature overview chart (price, RSI, volatility, returns)."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"{ticker} — Feature Overview", fontsize=14)

    axes[0].plot(df.index, df["close"], label="Close Price")
    axes[0].plot(df.index, df["ma_7"], label="MA-7", linestyle="--", alpha=0.7)
    axes[0].plot(df.index, df["ma_21"], label="MA-21", linestyle="--", alpha=0.7)
    axes[0].plot(df.index, df["ma_50"], label="MA-50", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)

    axes[1].plot(df.index, df["returns"], color="steelblue")
    axes[1].set_ylabel("Daily Returns")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")

    axes[2].plot(df.index, df["volatility"], color="orange")
    axes[2].set_ylabel("Volatility (21d)")

    axes[3].plot(df.index, df["rsi"], color="purple")
    axes[3].axhline(70, color="red", linewidth=0.8, linestyle="--", label="Overbought (70)")
    axes[3].axhline(30, color="green", linewidth=0.8, linestyle="--", label="Oversold (30)")
    axes[3].set_ylabel("RSI (14)")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{ticker}_features.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Feature plot saved.")


def save_results(forecast: pd.DataFrame, metrics: dict, ticker: str, out_dir: str):
    """Save forecast CSV and print metrics."""
    # Save forecast to CSV
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    forecast[cols].to_csv(os.path.join(out_dir, f"{ticker}_forecast.csv"), index=False)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{"Ticker": ticker, **metrics}])
    metrics_path = os.path.join(out_dir, "metrics_summary.csv")
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"    Metrics — MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}")
    print(f"    Forecast CSV saved.")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_metrics = []

    for ticker in STOCKS:
        print(f"\n{'='*40}")
        print(f"Processing: {ticker}")
        print(f"{'='*40}")

        # Per-ticker output folder
        ticker_dir = os.path.join(RESULTS_DIR, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        # Pipeline
        df = fetch_data(ticker, START_DATE, END_DATE)
        df = add_features(df)
        prophet_df = prepare_prophet_df(df)
        model = build_and_train(prophet_df)
        forecast = make_forecast(model, prophet_df, FORECAST_HORIZON)
        metrics = evaluate(prophet_df, forecast)

        # Outputs
        plot_results(model, forecast, ticker, ticker_dir)
        plot_features(df, ticker, ticker_dir)
        save_results(forecast, metrics, ticker, ticker_dir)

        all_metrics.append({"Ticker": ticker, **metrics})

    # Print summary table
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    summary_df = pd.DataFrame(all_metrics)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run()
