"""``DownbeatCRNN`` -- the acoustic model behind the bar-phase decoder."""
from __future__ import annotations

import torch
from torch import nn

from .model import FREQ_POOL, freq_pool_blocks

PARAM_BUDGET = 300_000


class DownbeatCRNN(nn.Module):
    def __init__(self, n_mels: int = 40, conv_channels: tuple = (32, 64, 64),
                 conv1d_channels: int = 64, rnn_hidden: int = 96,
                 rnn_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if rnn_layers < 1:
            raise ValueError(f"rnn_layers must be >= 1, got {rnn_layers}")

        blocks, freq = freq_pool_blocks(n_mels, conv_channels)

        self.n_mels = int(n_mels)
        self.conv_channels = tuple(int(channels) for channels in conv_channels)
        self.conv1d_channels = int(conv1d_channels)
        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = int(rnn_layers)
        self.freq_out = freq
        self.feature_dim = int(conv_channels[-1]) * freq

        self.conv = nn.Sequential(*blocks)
        self.temporal = nn.Sequential(
            nn.Conv1d(self.feature_dim, conv1d_channels, kernel_size=5, padding=2,
                      bias=False),
            nn.BatchNorm1d(conv1d_channels),
            nn.GELU(),
        )
        self.rnn = nn.GRU(conv1d_channels, rnn_hidden, num_layers=self.rnn_layers,
                          batch_first=True, bidirectional=True,
                          dropout=float(dropout) if self.rnn_layers > 1 else 0.0)
        self.dropout = nn.Dropout(float(dropout))
        self.downbeat_head = nn.Linear(2 * rnn_hidden, 1)

    def arch(self) -> dict:
        return {
            "n_mels": self.n_mels,
            "conv_channels": list(self.conv_channels),
            "conv1d_channels": self.conv1d_channels,
            "rnn_hidden": self.rnn_hidden,
            "rnn_layers": self.rnn_layers,
        }

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() != 3:
            raise ValueError(
                f"mel must be [batch, time, n_mels], got {tuple(mel.shape)}"
            )

        batched = mel.unsqueeze(1)
        conv_features = self.conv(batched)
        # flatten, not reshape: reading .shape freezes the time axis under ONNX tracing.
        stacked = conv_features.permute(0, 1, 3, 2).flatten(1, 2)
        temporal = self.temporal(stacked)
        recurrent, _ = self.rnn(temporal.transpose(1, 2))
        recurrent = self.dropout(recurrent)
        return self.downbeat_head(recurrent).squeeze(-1)


__all__ = ["DownbeatCRNN", "PARAM_BUDGET", "FREQ_POOL"]
