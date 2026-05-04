# Multi-Stock Prediction with Deep Learning
## Overview
This project investigates whether deep learning models can predict **30-day
future stock returns** across multiple major technology stocks using historical
market data.
### Research Question
Can sequence models like LSTMs outperform simpler baselines when forecasting
medium-term stock returns using:
- Historical log returns
- Technical indicators
- Multi-stock temporal relationships
### Stocks Used
- AAPL
- MSFT
- NVDA
- GOOGL
- AMZN
- META
- TSLA
---
## Final Conclusions
After extensive debugging and elimination of severe time-series data leakage:
- **LSTM outperformed MLP modestly**
1.
2.
3.
1
- **Technical indicators provided small predictive improvements**
- **Persistence baseline remained highly competitive**
- **No model produced robust trading profitability after realistic corrections**
- **Early extreme Sharpe ratios were caused by dataset leakage**
### Practical Finding
Deep learning can slightly improve predictive accuracy, but generating
consistent market alpha remains extremely difficult.
---
## Final Verified Baseline
### Persistence Baseline
Predicts future return using the most recent observed return.
- **MSE:** 0.0209
- **Directional Accuracy:** 52.1%
This serves as the core benchmark for evaluating whether neural networks add
real value.
---
## Final Model Summary
| Model | Features | Test Loss | Directional Accuracy | Total Return | Sharpe |
|------|----------|-----------|----------------------|--------------|--------|
| Persistence | Returns | 0.0209 | 0.521 | N/A | N/A |
| LSTM | Returns | ~0.029–0.035 | ~0.39–0.47 | Negative | Negative |
| LSTM | Returns + Indicators | Slightly better loss | Slightly better |
Negative | Negative |
| MLP | Returns | Worse than LSTM | Near baseline | Negative | Negative |
| MLP | Returns + Indicators | Minor improvement | Near baseline | Negative |
Negative |
---
## Project Structure
```text
Multi-Stock-Prediction-DL/
│
├── data/
│ ├── raw/ # Downloaded close prices
│ └── processed/ # Final X.npy and y.npy datasets
2
│
├── src/
│ ├── models/
│ │ ├── lstm.py # LSTM architecture
│ │ └── mlp.py # MLP baseline architecture
│ │
│ ├── baselines/
│ │ └── persistence.py # Persistence benchmark
│ │
│ ├── metrics/
│ │ └── metrics.py # Directional accuracy
│ │
│ ├── utils/
│ │ ├── logger.py # Final experiment logger
│ │ └── history_logger.py # Epoch-by-epoch logger
│ │
│ ├── backtest.py # Trading strategy simulation
│ ├── data_loader.py # Dataset construction pipeline
│ ├── run_baseline.py # Baseline testing
│ ├── run_experiment.py # Model training + evaluation
│ ├── plot_results.py # Training curve generation
│ └── generate_results_table.py# Final results tables
│
├── results/
│ ├── metrics.csv # Final summary metrics
│ ├── history.csv # Epoch logs
│ ├── plots/ # Generated graphs
│ └── results_table.md # Final comparison table
│
├── report/
│ └── final_report.tex # LaTeX submission report
│
└── README.md
