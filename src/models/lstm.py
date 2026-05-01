"""
lstm.py — Multi-Stock LSTM: Regression & Binary Classification
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Sequence model that uses the final hidden state to forecast all stocks.

    Supports two modes:
      - regression:      output is a raw price value per stock (MSELoss)
      - classification:  output is a logit per stock (BCEWithLogitsLoss)
                         Apply torch.sigmoid() at inference to get probabilities.
    """

    def __init__(
        self,
        input_size:     int,
        output_size:    int,
        hidden_size:    int  = 64,
        num_layers:     int  = 2,
        dropout:        float = 0.2,
        classification: bool  = False,
    ):
        """
        Args:
            input_size:     Number of features per time step.
            output_size:    Number of stocks being predicted simultaneously.
            hidden_size:    LSTM hidden state dimension.
            num_layers:     Number of stacked LSTM layers.
            dropout:        Dropout between LSTM layers (ignored if num_layers == 1).
            classification: If True, the model is used for binary direction prediction.
                            The forward pass is identical; the flag is stored so the
                            training loop can select the correct loss and apply sigmoid
                            at inference time.
        """
        super().__init__()
        self.classification = classification

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input sequence and map the last time step to predictions.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits / predictions: (batch, output_size)
            - For classification, pass through torch.sigmoid() at inference.
            - For regression, use directly.
        """
        out, _ = self.lstm(x)

        # The last output summarises the full historical window for each sample.
        out = out[:, -1, :]
        out = self.dropout(out)

        return self.fc(out)


# ── Loss helper ───────────────────────────────────────────────────────────────

def get_loss(classification: bool):
    """Return the appropriate loss function for the task."""
    if classification:
        return nn.BCEWithLogitsLoss()   # numerically stable sigmoid + BCE
    return nn.MSELoss()


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_proba(model: LSTMModel, x: torch.Tensor) -> torch.Tensor:
    """
    Run inference and return probabilities (classification) or raw values (regression).

    Args:
        model: Trained LSTMModel.
        x:     Input tensor (batch, seq_len, input_size).

    Returns:
        (batch, output_size) — probabilities in [0, 1] if classification, else raw.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if model.classification:
            return torch.sigmoid(logits)
        return logits


# ── Example training loop skeleton ───────────────────────────────────────────

def train_one_epoch(model: LSTMModel, loader, optimizer, device: str = "cpu") -> float:
    """
    Single training epoch. Works for both regression and classification.

    Args:
        model:     LSTMModel instance.
        loader:    DataLoader yielding (x, y) batches.
                   y should be float targets (prices) or float binary labels (0.0 / 1.0).
        optimizer: e.g. torch.optim.Adam(model.parameters(), lr=1e-3)
        device:    "cpu" or "cuda"

    Returns:
        Mean loss over the epoch.
    """
    criterion = get_loss(model.classification)
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss  = criterion(preds, y.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # helps LSTM stability
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_classification(model: LSTMModel, loader, device: str = "cpu") -> dict:
    """
    Compute accuracy and log-loss on a validation set (classification only).

    Returns:
        dict with 'accuracy' and 'log_loss'.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss

    assert model.classification, "evaluate_classification requires classification=True"

    all_probs, all_labels = [], []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    probs_arr  = np.concatenate(all_probs,  axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    acc = accuracy_score(labels_arr.flatten().round(), (probs_arr.flatten() >= 0.5).astype(int))
    ll  = log_loss(labels_arr.flatten(), probs_arr.flatten())

    return {"accuracy": round(acc, 4), "log_loss": round(ll, 4)}