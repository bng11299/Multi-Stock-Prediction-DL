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
from utils.history_logger import log_epoch

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


def train(model, train_loader, test_loader, model_name, features):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    final_loss = None
    final_acc = None

    for epoch in range(EPOCHS):

        
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                continue

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_loss, acc, y_true, y_pred = evaluate(model, test_loader, loss_fn)


        final_loss = test_loss
        final_acc = acc

        log_epoch(
            model_name=model_name,
            features=features,
            epoch=epoch+1,
            train_loss=total_loss/len(train_loader),
            test_loss=test_loss,
            direction_acc=acc
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss {total_loss/len(train_loader):.4f} | "
            f"Test Loss {test_loss:.4f} | "
            f"Dir Acc {acc:.3f}"
        )

    from backtest import backtest_strategy

    results = backtest_strategy(y_true, y_pred)
    print("y_true range:", y_true.min(), y_true.max())
    print("y_pred range:", y_pred.min(), y_pred.max())
    print("\nFinal Backtest Results:")
    print(f"Total Return: {results['total_return']:.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")

    return final_loss, final_acc, results


def evaluate(model, test_loader, loss_fn):
    """Run the model in inference mode and aggregate metrics across the loader."""

    all_preds = []
    all_targets = []

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch)

            loss = loss_fn(pred, y_batch)
            test_loss += loss.item()

            # STORE predictions
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())


    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    acc = directional_accuracy(y_pred, y_true)
    print("Correlation(pred, true):", np.corrcoef(y_pred.flatten(), y_true.flatten())[0,1])

    return test_loss / len(test_loader), acc, y_true, y_pred


def main():
    """Parse CLI args, prepare datasets, train the model, and save weights."""
    from utils.logger import log_results

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--features", default="returns")

    args = parser.parse_args()

    X, y = load_data()

    print("NaNs in X:", np.isnan(X).sum())
    print("NaNs in y:", np.isnan(y).sum())

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
   
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_shape = X_train.shape
    X_test_shape = X_test.shape

    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train_shape)

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test_shape)

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

    test_loss, acc, results = train(
        model,
        train_loader,
        test_loader,
        model_name=args.model,
        features=args.features
    )

    torch.save(model.state_dict(), f"results/{args.model}_model.pt")

    log_results(
        model_name=args.model,
        features=args.features,
        test_loss=test_loss,
        direction_acc=acc,
        total_return=results["total_return"],
        sharpe=results["sharpe_ratio"]
    )


if __name__ == "__main__":
    main()
