"""Construct a replayable mertcells sidecar from a phase-B F3 feature sidecar.

The F3 sidecars carry the pooled embeddings and the stream geometry but not the
cell cache's replay bookkeeping. The bookkeeping is a pure function of the
framing, the source sample count and the instants the chain was reset at; the
features are not reconstructible (F3 decoded via ffmpeg at 24 kHz, the live
path decodes at 44.1 kHz and resamples), so a constructed sidecar is a
transplant and its decision-level fidelity is measured, not assumed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.analyser.mert_stream import (_CONTEXT_BLOCKS, _WORK_BLOCKS,
                                      encoder_samples)
from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
from simulate.cell_cache import _write_archive

ENCODER_RATE = 24000
_DIVISOR = math.gcd(SAMPLE_RATE, ENCODER_RATE)
_UP = ENCODER_RATE // _DIVISOR
_DOWN = SAMPLE_RATE // _DIVISOR
_CONTEXT = _DOWN * _CONTEXT_BLOCKS
_WORK = _DOWN * _WORK_BLOCKS
_EMIT = _WORK * _UP // _DOWN


def read_f3(path):
    with np.load(path) as archive:
        return {
            "emb": np.asarray(archive["emb"]),
            "label_frame_sec": float(archive["label_frame_sec"]),
            "margin_sec": float(archive["stream_margin_sec"]),
            "hop_sec": float(archive["stream_hop_sec"]),
            "buffer_sec": float(archive["stream_buffer_sec"]),
        }


def source_samples_for(written_target: int, reset_at: int) -> int:
    blocks = math.ceil(written_target / _EMIT)
    return reset_at + _CONTEXT + blocks * _WORK


def bookkeeping(total_pushed: int, reset_at: int, *, margin_sec: float,
                hop_sec: float, label_frame_sec: float):
    hop = encoder_samples(hop_sec)
    margin = encoder_samples(margin_sec)
    triggers = []
    while True:
        k = len(triggers)
        due_at = source_samples_for((k + 1) * hop, reset_at)
        trigger = BUFFER_SIZE * math.ceil(due_at / BUFFER_SIZE)
        if trigger > total_pushed:
            break
        triggers.append(trigger)
    offsets = [0]
    indices: list = []
    seen: list = []
    emitted = 0
    for k in range(len(triggers)):
        hi = (k + 1) * hop - margin
        limit = max(0, math.floor(hi / ENCODER_RATE / label_frame_sec))
        for index in range(emitted, limit):
            indices.append(index)
            seen.append((k + 1) * hop / ENCODER_RATE)
        emitted = max(emitted, limit)
        offsets.append(len(indices))
    return (np.asarray(triggers, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(seen, dtype=np.float64))


def construct(f3_path, out_path, *, key: dict, total_pushed: int,
              reset_at: int, cell_shift: int = 0) -> dict:
    record = read_f3(f3_path)
    triggers, offsets, indices, seen = bookkeeping(
        int(total_pushed), int(reset_at), margin_sec=record["margin_sec"],
        hop_sec=record["hop_sec"], label_frame_sec=record["label_frame_sec"])
    emb = record["emb"]
    available = len(emb) - cell_shift
    if len(indices) > available:
        keep = int(np.searchsorted(indices, available))
        indices, seen = indices[:keep], seen[:keep]
        last_pass = int(np.searchsorted(offsets, keep, side="left"))
        offsets = np.minimum(offsets, keep)[:last_pass + 1]
        triggers = triggers[:last_pass]
    features = emb[cell_shift:cell_shift + len(indices)]
    features = features.reshape(len(indices), -1).astype(np.float32)
    _write_archive(Path(out_path), {
        "key": np.str_(json.dumps(key, sort_keys=True)),
        "total_pushed": np.int64(total_pushed),
        "pass_trigger": triggers,
        "pass_offset": offsets,
        "cell_index": indices,
        "cell_seen_sec": seen,
        "cell_features": features,
    })
    return {"passes": len(triggers), "cells": len(indices),
            "cell_shift": cell_shift}


def reset_from_first_trigger(first_trigger: int, *, hop_sec: float,
                             margin_sec: float) -> int:
    hop = encoder_samples(hop_sec)
    latest = first_trigger - (source_samples_for(hop, 0))
    return BUFFER_SIZE * (latest // BUFFER_SIZE)
