import numpy as np

def backtest_strategy(y_true, y_pred, transaction_cost=0.001):

    # convert to numpy if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. positions from predictions
    positions = np.sign(y_pred)

    # 2. returns from true market movement
    strategy_returns = positions * y_true

    # 3. transaction costs (based on position changes)
    turnover = np.abs(np.diff(positions, axis=0))

    # align lengths (diff removes first step)
    strategy_returns = strategy_returns[1:]

    strategy_returns -= turnover * transaction_cost

    # 4. metrics
    total_return = np.prod(1 + strategy_returns) - 1
    sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio
    }