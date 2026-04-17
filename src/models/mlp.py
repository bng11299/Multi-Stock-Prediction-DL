import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """Baseline feed-forward model that treats the input window as one flat vector."""

    def __init__(self, seq_len, input_size, output_size):
        super().__init__()

        # Each sample contains a full sequence of prices for every tracked stock.
        input_dim = seq_len * input_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        """Flatten the time window, then predict one next value per stock."""

        # Collapse the sequence and stock dimensions into a single feature vector.
        x = x.reshape(x.shape[0], -1)

        return self.net(x)
