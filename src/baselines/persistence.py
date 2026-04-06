import numpy as np
import torch
from metrics.metrics import directional_accuracy


def persistence_baseline(X, y):

    # last observed price in the sequence
    pred = X[:, -1, :]

    pred = torch.tensor(pred, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    mse = torch.mean((pred - y) ** 2).item()
    acc = directional_accuracy(pred, y, X[:, -1, :])

    return mse, acc