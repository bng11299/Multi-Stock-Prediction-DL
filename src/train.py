import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.lstm import LSTMModel

from model import LSTMModel

BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
TEST_SPLIT = 0.2


def load_dataset():

    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    return X, y


def prepare_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, shuffle=False
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, test_loader, X.shape[2]


from metrics import directional_accuracy


def evaluate(model, loader, loss_fn):

    model.eval()

    total_loss = 0
    all_pred = []
    all_true = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            pred = model(X_batch)

            loss = loss_fn(pred, y_batch)

            total_loss += loss.item()

            all_pred.append(pred)
            all_true.append(y_batch)

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)

    acc = directional_accuracy(pred, true)

    return total_loss / len(loader), acc


def train():

    X, y = load_dataset()

    train_loader, test_loader, input_size = prepare_data(X, y)

    model = LSTMModel(input_size=input_size)

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

        train_loss = total_loss / len(train_loader)
        test_loss, acc = evaluate(model, test_loader, loss_fn)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Direction Acc: {acc:.3f}"
        )


if __name__ == "__main__":
    train()