# Multi-Stock Prediction with Deep Learning

## Overview

This project investigates whether deep learning architectures can improve **multi-stock market prediction** by forecasting **30-day future log returns** across seven major technology stocks.

### Research Question
Can deep learning models such as LSTMs outperform simpler baselines when predicting medium-term stock returns using:

- Historical log returns
- Technical indicators
- Multi-stock relationships

### Stocks Included
- AAPL
- MSFT
- NVDA
- GOOGL
- AMZN
- META
- TSLA

---

## Final Conclusions

After extensive debugging and elimination of major time-series data leakage:

- **LSTM modestly outperformed MLP**
- **Technical indicators provided small predictive gains**
- **Persistence baseline remained surprisingly competitive**
- **No model produced consistently profitable out-of-sample trading returns**
- **Early extreme Sharpe ratios were caused by data leakage**

### Practical Finding
Deep learning slightly improved forecasting quality, but robust market alpha remained elusive.

---

## Persistence Baseline

To contextualize model performance, we implemented a persistence benchmark that predicts future returns using the most recent observed return.

### Baseline Results
- **MSE:** 0.0209
- **Directional Accuracy:** 52.1%

### Interpretation
- Provides realistic benchmark performance
- Prevents overclaiming neural network success
- Demonstrates financial prediction difficulty

---

## Final Model Performance Summary

| Model | Features | Test Loss | Directional Accuracy | Trading Return | Sharpe |
|------|----------|-----------|----------------------|----------------|--------|
| Persistence Baseline | Returns | 0.0209 | 0.521 | N/A | N/A |
| LSTM | Returns | ~0.029–0.035 | ~0.39–0.47 | Negative | Negative |
| LSTM | Returns + Indicators | Slightly improved | Slightly improved | Negative | Negative |
| MLP | Returns | Worse than LSTM | Near baseline | Negative | Negative |
| MLP | Returns + Indicators | Minor improvement | Near baseline | Negative | Negative |

---

## Key Lessons

### Major Debugging Discoveries

- Future information leakage inflated early results
- Incorrect target alignment created unrealistic returns
- Persistence baseline initially leaked future returns
- Backtesting required transaction cost penalties
- Proper chronological validation dramatically reduced inflated performance

### Academic Takeaway
Rigorous preprocessing and leakage prevention are more important than raw performance in financial ML.

---

## Project Structure

```text
Multi-Stock-Prediction-DL/
│
├── data/
│   ├── raw/                     # Downloaded close prices
│   └── processed/               # X.npy and y.npy datasets
│
├── src/
│   ├── models/
│   │   ├── lstm.py              # LSTM architecture
│   │   └── mlp.py               # MLP architecture
│   │
│   ├── baselines/
│   │   └── persistence.py       # Baseline predictor
│   │
│   ├── metrics/
│   │   └── metrics.py           # Evaluation metrics
│   │
│   ├── utils/
│   │   ├── logger.py            # Final experiment logging
│   │   └── history_logger.py    # Epoch-by-epoch logging
│   │
│   ├── data_loader.py           # Dataset builder
│   ├── backtest.py              # Trading simulator
│   ├── run_baseline.py          # Baseline testing
│   ├── run_experiment.py        # Model training pipeline
│   ├── plot_results.py          # Visualization generation
│   └── generate_results_table.py# Final results table generator
│
├── results/
│   ├── metrics.csv              # Final metrics
│   ├── history.csv              # Training histories
│   ├── plots/                   # Graph outputs
│   └── results_table.md         # Final comparison table
│
├── report/
│   └── final_report.tex         # Final LaTeX report
│
└── README.md
````

---

## Installation

### Create Environment

```bash
conda create -n stock_env python=3.11
conda activate stock_env
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Pipeline

### Build Dataset

```bash
python src/data_loader.py
```

### Pipeline Steps

* Download historical stock prices via Yahoo Finance
* Compute lagged log returns
* Compute technical indicators:

  * Moving averages (MA5, MA20)
  * Volatility
  * RSI
* Shift features to prevent leakage
* Predict future 30-day returns
* Construct rolling temporal sequences

---

## Running Experiments

### Baseline

```bash
python src/run_baseline.py
```

### LSTM

```bash
python src/run_experiment.py --model lstm --features returns+indicators
```

### MLP

```bash
python src/run_experiment.py --model mlp --features returns+indicators
```

---

## Feature Sets

| Feature Set        | Description                     |
| ------------------ | ------------------------------- |
| returns            | Lagged log returns only         |
| returns+indicators | Returns + MA + volatility + RSI |

---

## Evaluation Metrics

### Prediction Metrics

* Mean Squared Error (MSE)
* Directional Accuracy
* Correlation

### Trading Metrics

* Total Return
* Sharpe Ratio
* Transaction-cost-adjusted strategy returns

---

## Visualization Outputs

Generated plots include:

* Training loss curves
* Test loss curves
* Directional accuracy over epochs
* Final comparison tables

---

## Backtesting Strategy

### Trading Logic

* Predicted return > 0 → Long position
* Predicted return < 0 → Short position
* Equal weighting across all assets
* Transaction costs applied

### Purpose

Evaluates whether predictive skill translates into realistic profitability.

---

## Final Discussion

### What Worked

* Modular experimentation framework
* Reliable baseline benchmarking
* Leakage correction
* Comparative model analysis

### What Failed

* Sustainable trading profitability
* Strong alpha generation
* Large gains from technical indicators

### Why This Matters

This project demonstrates that financial prediction is highly sensitive to subtle preprocessing errors and that honest benchmarking is essential.

---

## Future Work

* Classification-based direction prediction
* Transformer architectures
* Portfolio optimization
* Risk-adjusted objective functions
* Expanded macroeconomic features

---

## Authors
Brendan Ng, Ben Fitzgerald

---

## Final Takeaway

While deep learning models can modestly improve predictive performance, careful validation revealed that true market forecasting remains highly challenging. The most important contribution of this project was building a realistic, leakage-free evaluation pipeline.

```
```
