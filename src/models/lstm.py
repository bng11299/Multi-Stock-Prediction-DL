import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Sequence model that uses the final hidden state to forecast all stocks."""

    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Encode the input sequence and map the last time step to predictions."""

        out, _ = self.lstm(x)

        # The last output summarizes the full historical window for each sample.
        out = out[:, -1, :]

        return self.fc(out)
