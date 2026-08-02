"""``DownbeatCRNN`` -- the acoustic model behind the bar-phase decoder.

One head, one job: given a mel window, say how strongly each frame looks like
the start of a bar.  Everything structural -- which beat is phase 1, how the
phase advances through a dropout -- is the decoder's, so this network is *not*
asked to be a bar tracker.  It is asked to produce an activation whose peaks
rank downbeats above off-beats; the HMM downstream turns a ranking into a grid.

The shape is deliberately the section model's, minus a head and a size:

*Convolutions pool frequency, never time.*  The whole product is a 70 ms
tolerance -- a time-strided front-end would throw the resolution away before the
recurrence saw it.  The block stack is literally ``model.freq_pool_blocks``, the
same definition ``SectionCRNN`` builds from, so the two heads cannot drift into
disagreeing about what a mel front-end is.

*The recurrence is bidirectional and it is the point.*  A downbeat is not a
locally distinctive sound -- in four-on-the-floor every beat carries a kick.
What separates beat 1 from beat 3 is the bar-length pattern around it, and a
bidirectional GRU over a 16 s window sees eight bars of it in both directions.
That is why the GRU shrinks less than the 1D conv does: capacity spent on
periodicity is capacity spent on the actual task.

*Half the budget goes nowhere near the recurrence.*  The 1D conv is the biggest
single block in ``SectionCRNN`` (205 K of its 460 K) purely because it projects
320 conv features into 128 channels.  Narrowing that projection to 64 buys back
100 K parameters for free at this input rate; the GRU is what needs the width.

Capacity: ~252 K parameters against the spec's 300 K ceiling, roughly half
``SectionCRNN``.  Under-parameterising is the recoverable direction, and
``rnn_hidden`` / ``rnn_layers`` are the two knobs to reach for first if the
activation underfits -- reference downbeat trackers are recurrent-heavy and
convolutionally thin.

The time axis is never baked in: Task 3 exports this graph with a dynamic time
dimension so a whole track can be pushed through in one pass, which is why
``flatten`` (not a shape-reading ``reshape``) joins the channel and frequency
axes.
"""
from __future__ import annotations

import torch
from torch import nn

from .model import FREQ_POOL, freq_pool_blocks

# The spec's ceiling for this head ("target <=300k params").  Printed on every
# run and asserted in the tests: a widening edit that doubles the count is a
# different experiment, not a tweak.
PARAM_BUDGET = 300_000


class DownbeatCRNN(nn.Module):
    """Mel window -> per-frame downbeat logit.

    Input is ``[batch, time, n_mels]`` -- the layout ``DownbeatWindowDataset``
    yields, collated.  Output is ``[batch, time]``, one logit per mel frame, at
    the frame rate the targets are defined on.
    """

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
        # Kernel 5 over ~46 ms frames = a ~230 ms receptive field: one beat's
        # worth of transient shape resolved before the recurrence has to spend
        # capacity on it.
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
        """The constructor arguments that decide the tensor shapes.

        Stored in every checkpoint so the file describes itself.  A
        ``state_dict`` alone does not, and the failure that costs a day is the
        silent one: a geometry change that still loads.  Same contract as
        ``SectionCRNN.arch``, and the resume path compares it *before*
        ``load_state_dict``.
        """
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

        x = mel.unsqueeze(1)                      # [B, 1, T, M]
        x = self.conv(x)                          # [B, C, T, M/8]
        # `flatten` over the channel/freq pair keeps the time axis symbolic
        # under ONNX tracing; reading `x.shape` and reshaping would freeze it.
        x = x.permute(0, 1, 3, 2).flatten(1, 2)   # [B, C*F, T]
        x = self.temporal(x)                      # [B, conv1d, T]
        x, _ = self.rnn(x.transpose(1, 2))        # [B, T, 2*hidden]
        x = self.dropout(x)
        return self.downbeat_head(x).squeeze(-1)  # [B, T]


__all__ = ["DownbeatCRNN", "PARAM_BUDGET", "FREQ_POOL"]
