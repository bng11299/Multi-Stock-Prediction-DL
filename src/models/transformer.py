"""
Multi-Stock Temporal Fusion Transformer (TFT) Forecasting Model
Features: Price, Returns, Moving Averages, Volatility, RSI
Mirrors the Prophet pipeline structure for easy comparison.

Dependencies:
    pip install pytorch-forecasting pytorch-lightning yfinance scikit-learn matplotlib pandas
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
STOCKS           = ["AAPL", "MSFT", "GOOGL"]   # Add or remove tickers
START_DATE       = "2020-01-01"
END_DATE         = "2024-12-31"
FORECAST_HORIZON = 30                           # Days to forecast
RESULTS_DIR      = "results/tft"

# TFT hyper-parameters
MAX_ENCODER_LENGTH = 90       # Look-back window (days)
BATCH_SIZE         = 64
MAX_EPOCHS         = 50
LEARNING_RATE      = 3e-3
HIDDEN_SIZE        = 64       # TFT hidden layer size
ATTENTION_HEADS    = 4
DROPOUT            = 0.1
# ─────────────────────────────────────────────────────────────────────────────


# ── DATA ─────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from yfinance."""
    print(f"  Fetching {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns, moving averages, volatility, and RSI — identical to Prophet pipeline."""

    df["returns"]    = df["close"].pct_change()
    df["ma_7"]       = df["close"].rolling(window=7).mean()
    df["ma_21"]      = df["close"].rolling(window=21).mean()
    df["ma_50"]      = df["close"].rolling(window=50).mean()
    df["volatility"] = df["returns"].rolling(window=21).std()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(window=14).mean()
    loss  = -delta.clip(upper=0).rolling(window=14).mean()
    rs    = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df = df.dropna()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features used as known future inputs by TFT."""
    df["day_of_week"]  = df.index.dayofweek.astype(float)
    df["day_of_month"] = df.index.day.astype(float)
    df["month"]        = df.index.month.astype(float)
    df["week_of_year"] = df.index.isocalendar().week.astype(float)
    return df


def build_combined_df(stocks: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch and combine all tickers into a single long-format DataFrame.
    TFT benefits from training on all stocks simultaneously (shared patterns).
    """
    frames = []
    for ticker in stocks:
        df = fetch_data(ticker, start, end)
        df = add_features(df)
        df = add_time_features(df)
        df["ticker"] = ticker
        df = df.reset_index().rename(columns={"Date": "date"})
        # Integer time index per group (required by pytorch-forecasting)
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined

# ─────────────────────────────────────────────────────────────────────────────


# ── DATASET ──────────────────────────────────────────────────────────────────

CONTINUOUS_REGRESSORS = ["returns", "ma_7", "ma_21", "ma_50", "volatility", "rsi", "volume"]
KNOWN_FUTURE_REALS    = ["day_of_week", "day_of_month", "month", "week_of_year"]


def make_datasets(combined: pd.DataFrame, forecast_horizon: int, max_encoder_length: int):
    """
    Split into train / validation TimeSeriesDataSet objects.
    Validation = last `forecast_horizon` steps per ticker.
    """
    max_time_idx = combined["time_idx"].max()
    training_cutoff = max_time_idx - forecast_horizon

    training = TimeSeriesDataSet(
        combined[combined["time_idx"] <= training_cutoff],
        time_idx                 = "time_idx",
        target                   = "close",
        group_ids                = ["ticker"],
        max_encoder_length       = max_encoder_length,
        max_prediction_length    = forecast_horizon,
        static_categoricals      = ["ticker"],
        time_varying_known_reals = KNOWN_FUTURE_REALS + ["time_idx"],
        time_varying_unknown_reals = CONTINUOUS_REGRESSORS + ["close"],
        target_normalizer        = GroupNormalizer(groups=["ticker"], transformation="softplus"),
        add_relative_time_idx    = True,
        add_target_scales        = True,
        add_encoder_length       = True,
        allow_missing_timesteps  = True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        combined,
        predict       = True,
        stop_randomization = True,
    )

    return training, validation


def make_dataloaders(training: TimeSeriesDataSet, validation: TimeSeriesDataSet):
    train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)
    return train_loader, val_loader

# ─────────────────────────────────────────────────────────────────────────────


# ── MODEL ────────────────────────────────────────────────────────────────────

def build_model(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Construct TFT from dataset metadata."""
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate          = LEARNING_RATE,
        hidden_size            = HIDDEN_SIZE,
        attention_head_size    = ATTENTION_HEADS,
        dropout                = DROPOUT,
        hidden_continuous_size = 32,
        loss                   = QuantileLoss(),        # Probabilistic output
        log_interval           = 10,
        reduce_on_plateau_patience = 4,
    )
    print(f"  Model parameters: {tft.hparams}")
    return tft


def train_model(
    tft: TemporalFusionTransformer,
    train_loader,
    val_loader,
    out_dir: str,
) -> pl.Trainer:
    """Train TFT with early stopping and checkpointing."""
    early_stop = EarlyStopping(
        monitor  = "val_loss",
        patience = 5,
        mode     = "min",
        verbose  = False,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath   = out_dir,
        filename  = "best_tft",
        monitor   = "val_loss",
        save_top_k = 1,
        mode      = "min",
    )

    trainer = pl.Trainer(
        max_epochs        = MAX_EPOCHS,
        accelerator       = "auto",
        gradient_clip_val = 0.1,
        callbacks         = [early_stop, checkpoint_cb],
        enable_progress_bar = True,
        logger            = False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer


def load_best_model(out_dir: str, training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Load the best checkpoint saved during training."""
    ckpt_path = os.path.join(out_dir, "best_tft.ckpt")
    best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    return best_model

# ─────────────────────────────────────────────────────────────────────────────


# ── FORECASTING ──────────────────────────────────────────────────────────────

def make_forecast(
    best_model: TemporalFusionTransformer,
    validation: TimeSeriesDataSet,
    val_loader,
    combined: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Generate predictions for the validation / forecast window.
    Returns a DataFrame with columns: date, yhat, yhat_lower, yhat_upper.
    """
    predictions = best_model.predict(val_loader, return_y=True, return_index=True)

    # predictions.output is shape [N, forecast_horizon, n_quantiles]
    # Default quantiles for QuantileLoss: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    pred_array = predictions.output.numpy()  # [N_samples, horizon, quantiles]
    index_df   = predictions.index           # DataFrame with ticker, time_idx

    ticker_mask = index_df["ticker"] == ticker
    if ticker_mask.sum() == 0:
        print(f"    Warning: no predictions found for {ticker}")
        return pd.DataFrame()

    pred_ticker = pred_array[ticker_mask.values]  # [n_windows, horizon, quantiles]

    # Take the last window (most recent forecast)
    last_pred = pred_ticker[-1]  # [horizon, quantiles]

    # Quantile indices: 0.1 → idx 1, 0.5 → idx 3, 0.9 → idx 5
    yhat       = last_pred[:, 3]  # median
    yhat_lower = last_pred[:, 1]  # 10th percentile
    yhat_upper = last_pred[:, 5]  # 90th percentile

    # Reconstruct dates for the forecast window
    ticker_dates = combined[combined["ticker"] == ticker].sort_values("time_idx")
    last_date    = ticker_dates["date"].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON)

    forecast_df = pd.DataFrame({
        "date":       future_dates[:len(yhat)],
        "yhat":       yhat,
        "yhat_lower": yhat_lower,
        "yhat_upper": yhat_upper,
    })
    return forecast_df

# ─────────────────────────────────────────────────────────────────────────────


# ── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(
    best_model: TemporalFusionTransformer,
    val_loader,
    combined: pd.DataFrame,
    ticker: str,
) -> dict:
    """Compute MAE and RMSE on the validation window."""
    predictions = best_model.predict(val_loader, return_y=True, return_index=True)

    pred_array = predictions.output.numpy()
    actuals    = predictions.y[0].numpy()      # [N, horizon]
    index_df   = predictions.index

    ticker_mask = index_df["ticker"] == ticker
    if ticker_mask.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan")}

    pred_median = pred_array[ticker_mask.values, :, 3].flatten()
    act_flat    = actuals[ticker_mask.values].flatten()

    # Denormalise if needed — pytorch-forecasting returns scaled values;
    # for a rough in-sample metric we compare at the scaled level.
    mae  = round(float(mean_absolute_error(act_flat, pred_median)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(act_flat, pred_median))), 4)
    return {"MAE": mae, "RMSE": rmse}

# ─────────────────────────────────────────────────────────────────────────────


# ── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_forecast(
    combined: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ticker: str,
    out_dir: str,
):
    """Plot historical close + forecast with confidence interval."""
    hist = combined[combined["ticker"] == ticker].sort_values("date")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hist["date"], hist["close"], label="Historical", color="steelblue", linewidth=1.2)
    ax.plot(forecast_df["date"], forecast_df["yhat"], label="Forecast (median)", color="tomato", linewidth=1.5)
    ax.fill_between(
        forecast_df["date"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        alpha=0.25,
        color="tomato",
        label="80% PI",
    )
    ax.set_title(f"{ticker} — TFT Forecast", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{ticker}_forecast.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Forecast plot saved.")


def plot_features(combined: pd.DataFrame, ticker: str, out_dir: str):
    """Feature overview chart — mirrors the Prophet pipeline plot."""
    df = combined[combined["ticker"] == ticker].sort_values("date").set_index("date")

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"{ticker} — Feature Overview", fontsize=14)

    axes[0].plot(df.index, df["close"],   label="Close Price")
    axes[0].plot(df.index, df["ma_7"],    label="MA-7",  linestyle="--", alpha=0.7)
    axes[0].plot(df.index, df["ma_21"],   label="MA-21", linestyle="--", alpha=0.7)
    axes[0].plot(df.index, df["ma_50"],   label="MA-50", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)

    axes[1].plot(df.index, df["returns"], color="steelblue")
    axes[1].set_ylabel("Daily Returns")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")

    axes[2].plot(df.index, df["volatility"], color="orange")
    axes[2].set_ylabel("Volatility (21d)")

    axes[3].plot(df.index, df["rsi"], color="purple")
    axes[3].axhline(70, color="red",   linewidth=0.8, linestyle="--", label="Overbought (70)")
    axes[3].axhline(30, color="green", linewidth=0.8, linestyle="--", label="Oversold (30)")
    axes[3].set_ylabel("RSI (14)")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{ticker}_features.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Feature plot saved.")


def plot_attention(
    best_model: TemporalFusionTransformer,
    val_loader,
    ticker: str,
    out_dir: str,
):
    """
    Plot TFT variable importance — a unique diagnostic not available in Prophet.
    Shows which features the model pays most attention to.
    """
    interpretation = best_model.interpret_output(
        best_model.predict(val_loader, return_attention=True),
        reduction="sum",
    )

    fig = best_model.plot_interpretation(interpretation)
    fig.suptitle(f"{ticker} — TFT Variable Importance", fontsize=12)
    fig.savefig(os.path.join(out_dir, f"{ticker}_attention.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Attention/importance plot saved.")

# ─────────────────────────────────────────────────────────────────────────────


# ── SAVING ───────────────────────────────────────────────────────────────────

def save_results(
    forecast_df: pd.DataFrame,
    metrics: dict,
    ticker: str,
    out_dir: str,
):
    """Save forecast CSV and append to metrics summary — mirrors Prophet pipeline."""
    if not forecast_df.empty:
        forecast_df.to_csv(os.path.join(out_dir, f"{ticker}_forecast.csv"), index=False)
        print(f"    Forecast CSV saved.")

    metrics_df   = pd.DataFrame([{"Ticker": ticker, **metrics}])
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    if os.path.exists(metrics_path):
        existing   = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"    Metrics — MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}")

# ─────────────────────────────────────────────────────────────────────────────


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Build combined dataset (TFT trains across all tickers jointly)
    print("\nBuilding combined dataset for all tickers...")
    combined = build_combined_df(STOCKS, START_DATE, END_DATE)

    # 2. Create TimeSeriesDataSet objects
    print("\nPreparing TimeSeriesDataSet...")
    training, validation = make_datasets(combined, FORECAST_HORIZON, MAX_ENCODER_LENGTH)
    train_loader, val_loader = make_dataloaders(training, validation)
    print(f"  Training samples:   {len(training)}")
    print(f"  Validation samples: {len(validation)}")

    # 3. Build and train model (shared across all tickers)
    print("\nBuilding TFT model...")
    tft     = build_model(training)
    trainer = train_model(tft, train_loader, val_loader, RESULTS_DIR)

    # 4. Load best checkpoint
    best_model = load_best_model(RESULTS_DIR, training)
    best_model.eval()

    # 5. Per-ticker outputs
    all_metrics = []
    for ticker in STOCKS:
        print(f"\n{'='*40}")
        print(f"Post-processing: {ticker}")
        print(f"{'='*40}")

        ticker_dir = os.path.join(RESULTS_DIR, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        forecast_df = make_forecast(best_model, validation, val_loader, combined, ticker)
        metrics     = evaluate(best_model, val_loader, combined, ticker)

        plot_forecast(combined, forecast_df, ticker, ticker_dir)
        plot_features(combined, ticker, ticker_dir)

        # Optional: attention plot (comment out if pytorch-forecasting version
        # doesn't support return_attention keyword)
        try:
            plot_attention(best_model, val_loader, ticker, ticker_dir)
        except Exception as e:
            print(f"    Attention plot skipped: {e}")

        save_results(forecast_df, metrics, ticker, ticker_dir)
        all_metrics.append({"Ticker": ticker, **metrics})

    # 6. Summary
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    print(pd.DataFrame(all_metrics).to_string(index=False))


if __name__ == "__main__":
    run()