import pandas as pd
import numpy as np


def create_sequences(data, seq_length=60, pred_horizon=30):
    X = []
    y = []

    for i in range(len(data) - seq_length - pred_horizon):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length+pred_horizon])

    return np.array(X), np.array(y)