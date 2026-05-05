import numpy as np
from metrics.metrics import directional_accuracy


def persistence_baseline(X, y):
    """
    Predict future return as the most recent observed return.
    X shape: [samples, seq_len, num_stocks]
    y shape: [samples, num_stocks]
    """

    # Last timestep returns for each stock
    #return_indices = [i * 5 for i in range(7)]  # every ticker's return feature
    return_indices = [i * 5 for i in range(7)]
    pred = X[:, -1, return_indices]

    mse = np.mean((pred - y) ** 2)
    acc = directional_accuracy(pred, y)
    print("Baseline pred range:", pred.min(), pred.max())
    print("True range:", y.min(), y.max())
    return mse, acc