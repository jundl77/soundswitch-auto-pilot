"""``SectionCRNN`` -- the acoustic model behind the section decoder.

The shape of this network is dictated by what the decoder downstream needs, not
by what is fashionable:

*Convolutions pool frequency, never time.*  The boundary head has to say
*which frame* a section change lands on, to the ~0.5 s the annotations are
accurate to; a time-strided front-end would throw that resolution away before
the recurrent layers ever saw it.  Frequency is the axis that can be summarised
-- 40 mel bands fold to 5 over three blocks -- because "which bands are loud"
survives pooling and "exactly when" does not.

*The recurrence is bidirectional.*  The whole premise of the design is that the
window carries look-ahead the audience has not heard yet, so a buildup can be
told from a fake-out by whether the drop actually lands to its right.  A causal
model cannot do that; a backward GRU can.

*Two heads, two rates.*  Labels are pooled to ~10 Hz (``label_pool`` frames per
decision) because section identity does not change at 21 Hz and the pooled head
both halves the label loss's variance and matches the target grid the dataset
emits.  Boundaries stay at the full frame rate for the reason above.  The
pooling is a plain average over the GRU output rather than over posteriors: it
smooths *evidence*, and leaves the softmax -- the thing the ECE metric and the
decoder both read -- untouched at the rate it is scored on.

Capacity is deliberately small (~0.46 M parameters against a 1 M budget).  The
corpus is a few hundred tracks; the reference systems this design borrows from
reach SOTA-class results at ~300 K.  Under-parameterising is recoverable, and
the fastest thing on the list to change if the model underfits is the GRU width.

Nothing here is allowed to bake in the time axis: the ONNX graph exported in
Task 3 declares time dynamic so the decoder can push a whole track through in
one pass, and `flatten` (rather than an explicit `reshape` on read-back shape
values) is what keeps the traced graph honest about that.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# The plan's ceiling.  Printed on every run and asserted in the tests: an
# architecture edit that quietly triples the parameter count is a different
# experiment, not a tweak.
PARAM_BUDGET = 1_000_000

# Frequency pooling per conv block.  Three blocks => bands / 8.
FREQ_POOL = 2


class SectionCRNN(nn.Module):
    """Mel window -> (label logits at ~10 Hz, boundary logits at frame rate).

    Input is ``[batch, time, n_mels]`` -- the layout ``WindowDataset`` yields,
    collated.  Output is ``([batch, time // label_pool, n_classes],
    [batch, time])``.
    """

    def __init__(self, n_mels: int = 40, n_classes: int = 5,
                 conv_channels: tuple = (32, 64, 64), conv1d_channels: int = 128,
                 rnn_hidden: int = 128, label_pool: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if label_pool < 1:
            raise ValueError(f"label_pool must be >= 1, got {label_pool}")

        blocks: list = []
        in_channels = 1
        freq = int(n_mels)
        for out_channels in conv_channels:
            if freq % FREQ_POOL:
                raise ValueError(
                    f"{n_mels} mel bands do not survive {len(conv_channels)} rounds "
                    f"of frequency pooling by {FREQ_POOL} (stuck at {freq})"
                )
            blocks += [
                # bias=False: the BatchNorm that follows has its own shift, so a
                # conv bias would be a redundant parameter the optimiser has to
                # fight the normalisation for.
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(1, FREQ_POOL)),   # (time, freq)
            ]
            in_channels = out_channels
            freq //= FREQ_POOL

        self.n_mels = int(n_mels)
        self.n_classes = int(n_classes)
        self.label_pool = int(label_pool)
        self.freq_out = freq
        self.feature_dim = int(conv_channels[-1]) * freq

        self.conv = nn.Sequential(*blocks)
        # Kernel 5 over ~46 ms frames = a ~230 ms receptive field handed to the
        # GRU: enough to have already resolved a kick from a snare, cheap enough
        # that the recurrence is not spending its capacity on it.
        self.temporal = nn.Sequential(
            nn.Conv1d(self.feature_dim, conv1d_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(conv1d_channels),
            nn.GELU(),
        )
        self.rnn = nn.GRU(conv1d_channels, rnn_hidden, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(float(dropout))
        self.label_head = nn.Linear(2 * rnn_hidden, self.n_classes)
        self.boundary_head = nn.Linear(2 * rnn_hidden, 1)

    def forward(self, mel: torch.Tensor) -> tuple:
        if mel.dim() != 3:
            raise ValueError(
                f"mel must be [batch, time, n_mels], got {tuple(mel.shape)}"
            )

        x = mel.unsqueeze(1)                      # [B, 1, T, M]
        x = self.conv(x)                          # [B, C, T, M/8]
        # [B, C, T, F] -> [B, C*F, T].  `flatten` over the channel/freq pair
        # keeps the time axis symbolic under ONNX tracing; reading `x.shape` and
        # reshaping would freeze it.
        x = x.permute(0, 1, 3, 2).flatten(1, 2)   # [B, C*F, T]
        x = self.temporal(x)                      # [B, conv1d, T]
        x, _ = self.rnn(x.transpose(1, 2))        # [B, T, 2*hidden]
        x = self.dropout(x)

        boundary = self.boundary_head(x).squeeze(-1)          # [B, T]
        pooled = F.avg_pool1d(x.transpose(1, 2), self.label_pool,
                              self.label_pool).transpose(1, 2)
        labels = self.label_head(pooled)                      # [B, T/pool, C]
        return labels, boundary


def count_parameters(module: nn.Module) -> int:
    """Trainable parameters -- the number checked against ``PARAM_BUDGET``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
