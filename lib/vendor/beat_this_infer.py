"""Beat This! inference path, vendored (MIT, (c) 2024 CP.JKU — see BEAT_THIS_LICENSE).

The minimal path the bar tracker needs from https://github.com/CPJKU/beat_this:
the BeatThis model, its log-mel frontend, and the checkpoint load.  Rewritten
against the tree the fine-tune ran on so the vendored forward is numerically
verified against the original on the shipped checkpoint (the parity record
lives beside the checkpoint); einops, rotary-embedding-torch and torchaudio are
replaced with plain torch + librosa's mel filterbank so the show gains no new
dependency.  State-dict key layout is preserved exactly — the fine-tuned
Lightning checkpoint loads strict.  Upstream's Windows path bug (NOTES.md in
the c2a_finetune campaign dir) lives in their dataset code, which is not
vendored.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

SAMPLE_RATE = 22050
FPS = 50
HOP_LENGTH = 441
N_FFT = 1024
N_MELS = 128
CHUNK_FRAMES = 1500
BORDER_FRAMES = 6


# --------------------------------------------------------------------------- #
# frontend — torchaudio.transforms.MelSpectrogram(power=1,
# normalized="frame_length", mel_scale="slaney") + log1p(1000 x), from
# beat_this/preprocessing.py, via torch.stft + librosa's slaney filterbank
# --------------------------------------------------------------------------- #
class LogMelSpect(nn.Module):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        import librosa

        fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                                 fmin=30.0, fmax=11000.0, htk=False, norm=None)
        self.register_buffer("fb", torch.tensor(fb, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.to(device)

    def forward(self, x: torch.Tensor, *, center: bool = True) -> torch.Tensor:
        spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP_LENGTH,
                          window=self.window, center=center,
                          pad_mode="reflect", return_complex=True)
        mag = spec.abs() / (N_FFT ** 0.5)
        return torch.log1p(1000.0 * (self.fb @ mag)).T


# --------------------------------------------------------------------------- #
# rotary embedding — rotary_embedding_torch.RotaryEmbedding(dim=head_dim),
# default theta 10000, interleaved-pair convention; the freqs live in the
# checkpoint as a parameter, so this mirrors that registration
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def rotate_queries_or_keys(self, t: torch.Tensor) -> torch.Tensor:
        seq = torch.arange(t.shape[-2], device=t.device, dtype=t.dtype)
        freqs = torch.einsum("n,f->nf", seq.type(self.freqs.dtype), self.freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        x = t.reshape(*t.shape[:-1], -1, 2)
        rotated = torch.stack((-x[..., 1], x[..., 0]), dim=-1).reshape(t.shape)
        return (t * freqs.cos() + rotated * freqs.sin()).type(t.dtype)


# --------------------------------------------------------------------------- #
# roformer — beat_this/model/roformer.py (itself adapted from lucidrains'
# BS-RoFormer, MIT), einops rearranges written out
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1) -> None:
        super().__init__()
        self.scale = size ** 0.5
        if dim >= 0:
            raise ValueError(f"dim must be negative, got {dim}")
        self.gamma = nn.Parameter(torch.ones((size,) + (1,) * (abs(dim) - 1)))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=self.dim) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, inner), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float,
                 rotary_embed: RotaryEmbedding) -> None:
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False),
                                    nn.Dropout(dropout))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        b, n, _ = x.shape
        qkv = self.to_qkv(x).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        gates = self.to_gates(x).permute(0, 2, 1).unsqueeze(-1)
        out = out * gates.sigmoid()
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, *, dim: int, depth: int, heads: int, dim_head: int,
                 attn_dropout: float, ff_dropout: float, ff_mult: int,
                 rotary_embed: RotaryEmbedding) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head,
                          dropout=attn_dropout, rotary_embed=rotary_embed),
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout),
            ]) for _ in range(depth)])
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# --------------------------------------------------------------------------- #
# the model — beat_this/model/beat_tracker.py, einops layers written out;
# module names preserved so the checkpoint's state_dict loads strict
# --------------------------------------------------------------------------- #
class _TransposeTF(nn.Module):
    def forward(self, x):          # b t f -> b f t
        return x.transpose(1, 2)


class _AddChannel(nn.Module):
    def forward(self, x):          # b f t -> b 1 f t
        return x.unsqueeze(1)


class _ConcatChannelFreq(nn.Module):
    def forward(self, x):          # b c f t -> b t (c f)
        b, c, f, t = x.shape
        return x.permute(0, 3, 1, 2).reshape(b, t, c * f)


class PartialFTTransformer(nn.Module):
    def __init__(self, dim: int, dim_head: int, n_head: int,
                 rotary_embed: RotaryEmbedding, dropout: float) -> None:
        super().__init__()
        self.attnF = Attention(dim, heads=n_head, dim_head=dim_head,
                               dropout=dropout, rotary_embed=rotary_embed)
        self.ffF = FeedForward(dim, dropout=dropout)
        self.attnT = Attention(dim, heads=n_head, dim_head=dim_head,
                               dropout=dropout, rotary_embed=rotary_embed)
        self.ffT = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * t, f, c)      # b c f t -> (b t) f c
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        x = x.reshape(b, t, f, c).permute(0, 2, 1, 3).reshape(b * f, t, c)
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        return x.reshape(b, f, t, c).permute(0, 3, 1, 2)     # (b f) t c -> b c f t


class SumHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> dict:
        beat, downbeat = self.beat_downbeat_lin(x).permute(2, 0, 1)
        beat = beat.float() + downbeat.float()
        return {"beat": beat, "downbeat": downbeat}


class BeatThis(nn.Module):
    def __init__(self, spect_dim: int = 128, transformer_dim: int = 512,
                 ff_mult: int = 4, n_layers: int = 6, head_dim: int = 32,
                 stem_dim: int = 32, dropout: dict | None = None,
                 sum_head: bool = True, partial_transformers: bool = True) -> None:
        super().__init__()
        if not (sum_head and partial_transformers):
            raise ValueError("only the shipped configuration is vendored "
                             "(sum_head and partial_transformers)")
        dropout = dropout or {"frontend": 0.1, "transformer": 0.2}
        rotary_embed = RotaryEmbedding(head_dim)

        stem = nn.Sequential(OrderedDict(
            rearrange_tf=_TransposeTF(),
            bn1d=nn.BatchNorm1d(spect_dim),
            add_channel=_AddChannel(),
            conv2d=nn.Conv2d(1, stem_dim, kernel_size=(4, 3), stride=(4, 1),
                             padding=(0, 1), bias=False),
            bn2d=nn.BatchNorm2d(stem_dim),
            activation=nn.GELU()))
        spect_dim //= 4

        blocks = []
        dim = stem_dim
        for _ in range(3):
            blocks.append(nn.Sequential(OrderedDict(
                partial=PartialFTTransformer(
                    dim=dim, dim_head=head_dim, n_head=dim // head_dim,
                    rotary_embed=rotary_embed, dropout=dropout["frontend"]),
                conv2d=nn.Conv2d(dim, dim * 2, kernel_size=(2, 3),
                                 stride=(2, 1), padding=(0, 1), bias=False),
                norm=nn.BatchNorm2d(dim * 2),
                activation=nn.GELU())))
            dim *= 2
            spect_dim //= 2
        self.frontend = nn.Sequential(OrderedDict(
            stem=stem, blocks=nn.Sequential(*blocks),
            concat=_ConcatChannelFreq(),
            linear=nn.Linear(dim * spect_dim, transformer_dim)))

        self.transformer_blocks = Transformer(
            dim=transformer_dim, depth=n_layers,
            heads=transformer_dim // head_dim, dim_head=head_dim,
            attn_dropout=dropout["transformer"],
            ff_dropout=dropout["transformer"], ff_mult=ff_mult,
            rotary_embed=rotary_embed)

        self.task_heads = SumHead(transformer_dim)

    def forward(self, x: torch.Tensor) -> dict:
        return self.task_heads(self.transformer_blocks(self.frontend(x)))


MODEL_HPARAMS = ("spect_dim", "transformer_dim", "ff_mult", "n_layers",
                 "head_dim", "stem_dim", "dropout", "sum_head",
                 "partial_transformers")


def load_checkpoint(path, device="cpu") -> dict:
    """weights_only load of the fine-tuned Lightning checkpoint.

    Its hparams carry a Path and numpy scalars (the c2a_finetune campaign's own
    artifact), so exactly those types are allow-listed.
    """
    import pathlib

    import numpy as np

    import importlib

    # The checkpoint was written under numpy 2, so its pickle names the
    # _core paths; the (object, name) form covers a numpy 1.x venv where the
    # object's own module is numpy.core.
    multiarray = importlib.import_module("numpy._core.multiarray")
    torch.serialization.add_safe_globals(
        [pathlib.WindowsPath, pathlib.PosixPath,
         (multiarray.scalar, "numpy._core.multiarray.scalar"),
         np.dtype, np.dtypes.Float64DType])
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path, device="cpu") -> tuple:
    """(model, hparams) — rebuilt from the checkpoint's own hyper_parameters."""
    checkpoint = load_checkpoint(path, device)
    hparams = {key: value for key, value in
               checkpoint["hyper_parameters"].items() if key in MODEL_HPARAMS}
    model = BeatThis(**hparams)
    state = {key[len("model."):]: value
             for key, value in checkpoint["state_dict"].items()
             if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), hparams
