# src/models/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Positional Encoding ───────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings — outperforms fixed sinusoidal on financial
    data where the model benefits from learning position-specific patterns.
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return self.dropout(x + self.pe(positions))


# ── Stochastic Depth ──────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """
    Randomly drops entire residual branches during training (DropPath).
    Each sample in the batch is dropped independently.
    Acts as strong regularization for deep transformers.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (batch, 1, 1) so it broadcasts over seq_len and d_model
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = torch.floor(noise + keep_prob)
        return x * noise / keep_prob


# ── Gated Residual Connection ─────────────────────────────────────────────────

class GatedResidual(nn.Module):
    """
    Learns a scalar gate per feature that controls how much of the sublayer
    output to mix into the residual stream. When the gate → 0, the sublayer
    is effectively skipped; when → 1, it behaves like a normal residual.
    This is the mechanism TFT uses internally.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.dropout   = nn.Dropout(dropout)
        # Initialise gate bias to ~0.5 so training starts near a normal residual
        nn.init.constant_(self.gate_proj.bias, 0.5)

    def forward(self, residual: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([residual, sublayer_out], dim=-1)))
        return residual + gate * self.dropout(sublayer_out)


# ── Pre-Norm Transformer Encoder Layer ───────────────────────────────────────

class PreNormEncoderLayer(nn.Module):
    """
    Transformer encoder layer with:
      - Pre-LayerNorm (more stable gradients than post-norm)
      - Separate attention dropout
      - GELU feedforward (smoother than ReLU for financial signals)
      - Gated residual connections
      - Stochastic depth on both sublayers
    """
    def __init__(
        self,
        d_model:         int,
        nhead:           int,
        dim_feedforward: int,
        attn_dropout:    float = 0.1,
        ffn_dropout:     float = 0.1,
        drop_path:       float = 0.1,
    ):
        super().__init__()

        # Self-attention sublayer
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim    = d_model,
            num_heads    = nhead,
            dropout      = attn_dropout,   # separate from ffn dropout
            batch_first  = True,
        )
        self.gate1       = GatedResidual(d_model, dropout=ffn_dropout)
        self.drop_path1  = StochasticDepth(drop_path)

        # Feedforward sublayer
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),                          # smoother than ReLU
            nn.Dropout(ffn_dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(ffn_dropout),
        )
        self.gate2      = GatedResidual(d_model, dropout=ffn_dropout)
        self.drop_path2 = StochasticDepth(drop_path)

    def forward(
        self,
        x:           torch.Tensor,
        attn_mask:   torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ── Attention sublayer (pre-norm) ──────────────────────────────────
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask        = attn_mask,
            key_padding_mask = key_padding_mask,
            need_weights     = False,
        )
        x = self.gate1(x, self.drop_path1(attn_out))

        # ── Feedforward sublayer (pre-norm) ────────────────────────────────
        normed   = self.norm2(x)
        ffn_out  = self.ffn(normed)
        x = self.gate2(x, self.drop_path2(ffn_out))

        return x


# ── Full Model ────────────────────────────────────────────────────────────────

class TransformerModel(nn.Module):
    """
    Regularized causal Transformer for multi-stock regression.

    Interface is identical to LSTMModel:
        forward(x) -> (batch, output_size)
        x shape:      (batch, seq_len, input_size)

    Regularization stack:
      - Learnable positional embeddings          (better than fixed sinusoidal)
      - Separate attention dropout               (targets attention weights only)
      - Pre-LayerNorm                            (more stable than post-norm)
      - GELU activations                         (smoother gradient flow)
      - Gated residual connections               (learned skip strength)
      - Stochastic depth / DropPath              (drops entire layers randomly)
      - Causal attention mask                    (no future leakage)
      - Final LayerNorm before head              (stabilises output scale)
    """

    def __init__(
        self,
        input_size:      int,
        output_size:     int,
        d_model:         int   = 64,
        nhead:           int   = 4,
        num_layers:      int   = 3,
        dim_feedforward: int   = 256,
        attn_dropout:    float = 0.1,
        ffn_dropout:     float = 0.1,
        # Stochastic depth rate — linearly increases layer by layer
        # so early layers are more stable and deeper ones get more regularization
        max_drop_path:   float = 0.2,
        classification:  bool  = False,
    ):
        super().__init__()
        self.classification = classification

        # Input projection: raw features → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),      # normalise before adding positional info
        )

        self.pos_enc = LearnablePositionalEncoding(d_model, dropout=ffn_dropout)

        # Build layers with linearly increasing stochastic depth rates
        # Layer 0 gets the least regularization, layer N-1 gets the most
        drop_path_rates = [
            max_drop_path * i / max(num_layers - 1, 1)
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList([
            PreNormEncoderLayer(
                d_model         = d_model,
                nhead           = nhead,
                dim_feedforward = dim_feedforward,
                attn_dropout    = attn_dropout,
                ffn_dropout     = ffn_dropout,
                drop_path       = drop_path_rates[i],
            )
            for i in range(num_layers)
        ])

        # Final norm before projection head (standard in pre-norm transformers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, zero-init for output head bias."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular mask so each position can only attend to itself
        and earlier positions. Prevents any future leakage during training.
        Shape: (seq_len, seq_len), True means 'ignore this position'.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, output_size)
        """
        seq_len = x.size(1)

        x = self.input_proj(x)                          # (batch, seq_len, d_model)
        x = self.pos_enc(x)                             # add learnable positions

        causal_mask = self._causal_mask(seq_len, x.device)

        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)         # (batch, seq_len, d_model)

        x = self.norm(x)                                # final pre-head norm
        x = x[:, -1, :]                                 # last token → prediction
        return self.head(x)                             # (batch, output_size)


# ── Loss & inference helpers (mirror lstm.py) ─────────────────────────────────

def get_loss(classification: bool):
    if classification:
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


def predict_proba(model: TransformerModel, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if model.classification:
            return torch.sigmoid(logits)
        return logits