import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMModel
from utils import create_sequences

SEQ_LENGTH = 60
PRED_HORIZON = 30
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001


def load_data():

    df = pd.read_csv("data/raw/stocks.csv", index_col=0)

    prices = df.filter(like="Close").values

    X, y = create_sequences(prices, SEQ_LENGTH, PRED_HORIZON)

    return X, y


def train():

    X, y = load_data()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(input_size=X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):

        total_loss = 0

        for xb, yb in loader:

            pred = model(xb).squeeze()

            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    train()