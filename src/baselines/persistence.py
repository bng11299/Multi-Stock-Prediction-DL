import numpy as np
from metrics.metrics import directional_accuracy

def persistence_baseline(X, y):
    """
    Use each stock's most recent return as prediction.
    """
    
    return_indices = [i * 5 for i in range(7)]  # every ticker's return column
    pred = X[:, -1, return_indices]

    mse = np.mean((pred - y) ** 2)
    acc = directional_accuracy(pred, y)

    return mse, acc