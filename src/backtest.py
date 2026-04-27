import numpy as np


def backtest_strategy(y_true, y_pred):
    """
    y_true: actual returns (N, num_stocks)
    y_pred: predicted returns (N, num_stocks)
    """

    # Convert predictions into positions
    k = 2
    positions = np.zeros_like(y_pred)

    for t in range(len(y_pred)):
        top = np.argsort(y_pred[t])[-k:]
        bottom = np.argsort(y_pred[t])[:k]

        positions[t, top] = 1
        positions[t, bottom] = -1

    # Strategy returns
    strategy_returns = positions * y_true
    strategy_returns = np.clip(strategy_returns, -0.1, 0.1)
    cost = 0.001
    strategy_returns -= cost

    # Average across stocks
    strategy_returns = strategy_returns.mean(axis=1)

    # Cumulative return
    cumulative_return = np.cumprod(1 + strategy_returns)

    return {
        "strategy_returns": strategy_returns,
        "cumulative_return": cumulative_return,
        "total_return": cumulative_return[-1] - 1,
        "sharpe_ratio": compute_sharpe(strategy_returns)
    }


def compute_sharpe(returns):
    if returns.std() == 0:
        return 0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)