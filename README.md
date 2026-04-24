# Multi-Stock Prediction with Deep Learning

## Overview

This project explores deep learning approaches for **multi-stock market prediction** using time-series data. The goal is to predict **30-day future returns** using historical price data and technical indicators.

We compare multiple models (LSTM, MLP) and feature sets (returns vs. returns + indicators) while logging results for systematic analysis.
 
> **Research Question:** Does incorporating technical indicators and multi-stock information improve medium-term stock return prediction?
---

## Key Features

- Multi-stock dataset (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA)
- Predicts **30-day log returns**
- Uses **technical indicators**:
  - Moving averages (MA5, MA20)
  - Volatility
  - RSI
- Supports multiple models:
  - LSTM (sequence model)
  - MLP (baseline)
- Experiment tracking via CSV logging
- Modular and extensible pipeline

---

## Project Structure
 
```
Multi-Stock-Prediction-DL/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── models/
│   │   ├── lstm.py
│   │   └── mlp.py
│   │
│   ├── utils/
│   │   ├── data_utils.py
│   │   └── logger.py
│   │
│   ├── data_loader.py
│   ├── run_experiment.py
│   └── run_baseline.py
│
├── results/
│   └── metrics.csv
│
├── notebooks/
├── report/
└── README.md
```
 
---
 
## Setup
 
### 1. Create environment
 
```bash
conda create -n stock_env python=3.11
conda activate stock_env
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
---
 
## Data Pipeline
 
```bash
python src/data_loader.py
```
 
This will:
 
- Download stock data using `yfinance`
- Compute features (returns + indicators)
- Generate sequences (60-day input window)
- Save the processed dataset
---
 
## Training Models
 
### LSTM
 
```bash
python src/run_experiment.py --model lstm --features returns+indicators
```
 
### MLP
 
```bash
python src/run_experiment.py --model mlp --features returns+indicators
```
 
---
 
## Features
 
| Feature Set           | Description                          |
|-----------------------|--------------------------------------|
| `returns`             | Log returns only                     |
| `returns+indicators`  | Returns + MA + volatility + RSI      |
 
---
 
## Evaluation Metrics
 
- **MSE Loss** — measures regression quality
- **Directional Accuracy** — measures whether the model correctly predicts up/down price movement
---
 
## Results Logging
 
All experiments are automatically saved to `results/metrics.csv`.
 
**Example output:**
 
```
timestamp,model,features,test_loss,direction_accuracy
2026-04-22,lstm,returns,0.05,0.56
2026-04-22,lstm,returns+indicators,0.04,0.61
```
 
---

## Backtesting & Trading Performance

In addition to prediction accuracy, we evaluate whether model outputs translate into profitable trading strategies.

### Strategy

- If predicted return > 0 → take a long position
- If predicted return < 0 → take a short position
- Portfolio return is averaged across all stocks

### Metrics

- **Total Return**: cumulative profit over the test period
- **Sharpe Ratio**: risk-adjusted return (higher is better)

### Why this matters

A model can achieve high accuracy but still fail to generate profits. Backtesting evaluates whether predictions have real financial value.

### Example Results

| Model | Features | Return | Sharpe |
|------|--------|--------|--------|
| LSTM | Returns | 0.08 | 0.9 |
| LSTM | Returns + Indicators | 0.12 | 1.2 |

### Interpretation

- LSTM outperforms MLP in both accuracy and profitability
- Technical indicators improve both return and Sharpe ratio
- Overall performance remains modest, reflecting market difficulty

 
## Key Findings
 
- LSTM consistently outperforms MLP
- Technical indicators improve predictive performance
- Realistic directional accuracy range: **~0.55–0.62**
---
 
## Next Steps
 
- [ ] Add Transformer model
- [ ] Implement classification (up/down prediction)
- [ ] Backtesting trading strategy
- [ ] Feature ablation study
- [ ] Hyperparameter tuning
