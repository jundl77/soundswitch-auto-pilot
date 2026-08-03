"""Frozen copy of training/nn/ceiling's extractor: never refactored to share code with the module it checks, re-copied only when that extractor changes."""
from __future__ import annotations

import math

import numpy as np

ENCODER_SAMPLE_RATE = 24000
ENCODER_SAMPLES_PER_FRAME = 320
ENCODER_RECEPTIVE_FIELD = 400
ENCODER_FRAME_RATE_HZ = ENCODER_SAMPLE_RATE / ENCODER_SAMPLES_PER_FRAME

SAMPLE_RATE = 44100
BUFFER_SIZE = 512
POOL_BUFFERS = 4
FRAME_SEC = POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE
LABEL_POOL = 2
LABEL_FRAME_SEC = FRAME_SEC * LABEL_POOL


def chunk_samples(seconds: float, sample_rate: int = ENCODER_SAMPLE_RATE) -> int:
    frames = int(math.floor(seconds * sample_rate / ENCODER_SAMPLES_PER_FRAME))
    return frames * ENCODER_SAMPLES_PER_FRAME


def encoder_frame_times(n_frames: int, *,
                        frame_rate_hz: float = ENCODER_FRAME_RATE_HZ,
                        samples_per_frame: int = ENCODER_SAMPLES_PER_FRAME,
                        receptive_field: int = ENCODER_RECEPTIVE_FIELD,
                        sample_rate: int = ENCODER_SAMPLE_RATE,
                        offset_samples: int = 0) -> np.ndarray:
    del frame_rate_hz
    starts = np.arange(int(n_frames), dtype=np.float64) * samples_per_frame
    return (offset_samples + starts + receptive_field / 2.0) / float(sample_rate)


def encoder_frames(n_samples: int) -> int:
    if int(n_samples) < ENCODER_RECEPTIVE_FIELD:
        return 0
    return 1 + (int(n_samples) - ENCODER_RECEPTIVE_FIELD) // ENCODER_SAMPLES_PER_FRAME


def pass_schedule(n_samples: int, *, length: int, hop: int, margin_samples: dict):
    margins = sorted(margin_samples)
    lo = {margin: 0 for margin in margins}
    step = 1
    while any(lo[margin] < n_samples for margin in margins):
        end = min(step * hop, n_samples)
        step += 1
        final = end >= n_samples
        spans = {}
        for margin in margins:
            hi = n_samples if final else end - margin_samples[margin]
            if hi > lo[margin]:
                spans[margin] = (lo[margin], hi)
        if not spans:
            if final:
                break
            continue
        yield max(0, end - length), end, spans
        for margin, (_, hi) in spans.items():
            lo[margin] = hi


class Accumulator:
    def __init__(self, n_pooled: int, n_layers: int, dim: int) -> None:
        self.sums = np.zeros((n_pooled, n_layers, dim), dtype=np.float64)
        self.counts = np.zeros(n_pooled, dtype=np.int64)

    def add(self, stacked: np.ndarray, times: np.ndarray, lo_sec: float,
            hi_sec: float) -> None:
        keep = (times >= lo_sec) & (times < hi_sec)
        if not keep.any():
            return
        cell = np.floor(times[keep] / LABEL_FRAME_SEC).astype(np.int64)
        inside = (cell >= 0) & (cell < len(self.counts))
        np.add.at(self.sums, cell[inside], stacked[keep][inside].astype(np.float64))
        np.add.at(self.counts, cell[inside], 1)

    def finish(self) -> tuple:
        reached = self.counts > 0
        pooled = np.zeros_like(self.sums)
        pooled[reached] = self.sums[reached] / self.counts[reached][:, None, None]
        filled = int((~reached).sum())
        if filled:
            if not reached.any():
                raise RuntimeError("no encoder frame reached any label cell")
            source = np.maximum.accumulate(
                np.where(reached, np.arange(len(reached)), -1))
            source[source < 0] = int(np.flatnonzero(reached)[0])
            pooled = pooled[source]
        return pooled.astype(np.float32), filled


def extract(encoder, audio, *, margin_sec: float, hop_sec: float,
            buffer_sec: float, n_pooled: int) -> np.ndarray:
    length = chunk_samples(buffer_sec)
    hop = chunk_samples(hop_sec)
    margins = {float(margin_sec): chunk_samples(margin_sec)}
    state = Accumulator(n_pooled, encoder.n_layers, encoder.dim)
    for start, end, spans in pass_schedule(len(audio), length=length, hop=hop,
                                           margin_samples=margins):
        segment = np.ascontiguousarray(audio[start:end], dtype=np.float32)
        times = encoder_frame_times(encoder_frames(len(segment)),
                                    offset_samples=start,
                                    sample_rate=ENCODER_SAMPLE_RATE)
        wanted = np.zeros(len(times), dtype=bool)
        for lo, hi in spans.values():
            wanted |= ((times >= lo / ENCODER_SAMPLE_RATE)
                       & (times < hi / ENCODER_SAMPLE_RATE))
        keep = np.flatnonzero(wanted)
        stacked = encoder.frames(segment, start, keep)
        kept_times = times[keep]
        for lo, hi in spans.values():
            state.add(stacked, kept_times, lo / ENCODER_SAMPLE_RATE,
                      hi / ENCODER_SAMPLE_RATE)
    return state.finish()[0]
