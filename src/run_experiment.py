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
    """Load the preprocessed sliding-window features and next-step targets."""

    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    return X, y


def get_model(name, seq_len, input_size, output_size):
    """Build the requested architecture with dimensions inferred from the data."""

    if name == "lstm":
        return LSTMModel(input_size, output_size)

    if name == "mlp":
        return MLPModel(seq_len, input_size, output_size)

    raise ValueError("Unknown model")


def train(model, train_loader, test_loader):
    """Train for a fixed number of epochs and report loss plus directional accuracy."""

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for X_batch, y_batch in train_loader:

            pred = model(X_batch)

            loss = loss_fn(pred, y_batch)

            # Standard optimize step for the current mini-batch.
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
    """Run the model in inference mode and aggregate metrics across the loader."""

    model.eval()

    total_loss = 0
    preds = []
    trues = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            pred = model(X_batch)

            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()

            preds.append(pred)
            trues.append(y_batch)

    pred = torch.cat(preds)
    true = torch.cat(trues)

    acc = directional_accuracy(pred, true)

    return total_loss / len(loader), acc


def main():
    """Parse CLI args, prepare datasets, train the model, and save weights."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="lstm")

    args = parser.parse_args()

    X, y = load_data()

    seq_len = X.shape[1]
    num_stocks = X.shape[2]

    input_size = X.shape[2]   # ~35 features
    output_size = y.shape[1]  # 7 stocks
    seq_len = X.shape[1]

    # Preserve chronological order so future information never leaks into training.
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

    model = get_model(args.model, seq_len, input_size, output_size)

    print("Running model:", args.model)

    train(model, train_loader, test_loader)
    # Save the learned parameters so the same architecture can be reloaded later.
    torch.save(model.state_dict(), f"results/{args.model}_model.pt")


if __name__ == "__main__":
    main()
