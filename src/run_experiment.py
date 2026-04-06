import sys
sys.path.append("src")

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.lstm import LSTMModel
from models.mlp import MLPModel
from metrics.metrics import directional_accuracy


BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001


def load_data():

    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    return X, y


def get_model(name, seq_len, num_stocks):

    if name == "lstm":
        return LSTMModel(num_stocks)

    if name == "mlp":
        return MLPModel(seq_len, num_stocks)

    raise ValueError("Unknown model")


def train(model, train_loader, test_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for X_batch, y_batch in train_loader:

            pred = model(X_batch)

            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_loss, acc = evaluate(model, test_loader, loss_fn)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss {total_loss/len(train_loader):.4f} | "
            f"Test Loss {test_loss:.4f} | "
            f"Dir Acc {acc:.3f}"
        )


def evaluate(model, loader, loss_fn):

    model.eval()

    total_loss = 0
    preds = []
    trues = []
    lasts = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            pred = model(X_batch)

            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()

            last_price = X_batch[:, -1, :]

            preds.append(pred)
            trues.append(y_batch)
            lasts.append(last_price)

    pred = torch.cat(preds)
    true = torch.cat(trues)
    last = torch.cat(lasts)

    acc = directional_accuracy(pred, true, last)

    return total_loss / len(loader), acc


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="lstm")

    args = parser.parse_args()

    X, y = load_data()

    seq_len = X.shape[1]
    num_stocks = X.shape[2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE
    )

    model = get_model(args.model, seq_len, num_stocks)

    print("Running model:", args.model)

    train(model, train_loader, test_loader)


if __name__ == "__main__":
    main()