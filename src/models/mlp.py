import torch
import torch.nn as nn


class MLPModel(nn.Module):

    def __init__(self, seq_len, num_stocks):
        super().__init__()

        input_dim = seq_len * num_stocks

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_stocks)
        )

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)

        return self.net(x)