"""The live bar tracker: trailing-window Beat This inference, downbeat logits out.

The #330 causal fit's cc_infer geometry, run incrementally: every 2.5 s of new
audio one forward over the trailing 30 s of log-mel frames, emitting the
newest margin-safe frames exactly once.  Frames are on beat_this's own 50 fps
grid anchored at song start; each is computed once, from a soxr-resampled
slice with enough context that the values match a whole-stream resample (the
measured context requirement is in the artifact record's parity block).  The
net never emits the grid — the chunks become per-beat evidence in
``lib/engine/bar_phase.py``.

A tracker failure is not a section-stage failure: construction problems with
artifacts present are fatal (a wrong model must not light a room), but a
forward that starts failing mid-show marks the tracker dead and the decoder
runs on the counting grid — never through the watchdog's shed door, which
would hold the whole section stage hostage to the optional stage (#332).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from lib.analyser.mert_stream import RingOverrun, SampleRing
from lib.audio_config import SAMPLE_RATE as SOURCE_SAMPLE_RATE

TRACKER_DIR = "bar_tracker"
RECORD_NAME = "bar_tracker.json"

FPS = 50
CHUNK_FRAMES = 1500
MARGIN_FRAMES = 6
STRIDE_SEC = 2.5
STRIDE_FRAMES = int(round(STRIDE_SEC * FPS))

_N_FFT = 1024
_HOP_22K = 441
_SRC_PER_FRAME = 882          # 44.1 kHz source samples per 50 fps frame
_RESAMPLE_CTX_SRC = 1024      # measured: >=256 matches a whole-stream soxr resample
_STRIDE_SRC = STRIDE_FRAMES * _SRC_PER_FRAME
# The last frame of pass p needs source samples through end_f*882 - 882 + 1024
# (its STFT support) plus the resampler context.
_PASS_PAD_SRC = _N_FFT - _SRC_PER_FRAME + _RESAMPLE_CTX_SRC
_RING_STRIDES = 8

GEOMETRY = {"sample_rate": 22050, "fps": FPS, "chunk_frames": CHUNK_FRAMES,
            "margin_frames": MARGIN_FRAMES, "stride_sec": STRIDE_SEC}


class TrackerChunk(NamedTuple):
    frame_lo: int
    frame_hi: int
    db_logits: np.ndarray
    avail_sec: float


class TrackerArtifacts(NamedTuple):
    checkpoint: Path
    record: Path

    def missing(self) -> list:
        return [str(path) for path in self if not path.exists()]


def artifacts(generation_dir) -> TrackerArtifacts:
    root = Path(generation_dir) / TRACKER_DIR
    record_path = root / RECORD_NAME
    checkpoint = None
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            checkpoint = root / record["checkpoint"]
        except (OSError, ValueError, KeyError):
            checkpoint = None
    return TrackerArtifacts(checkpoint=checkpoint or root / "checkpoint.ckpt",
                            record=record_path)


def tracker_present(generation_dir) -> bool:
    return not artifacts(generation_dir).missing()


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_record(generation_dir) -> dict:
    found = artifacts(generation_dir)
    absent = found.missing()
    if absent:
        raise FileNotFoundError(f"bar tracker artifacts missing: "
                                f"{', '.join(absent)}")
    record = json.loads(found.record.read_text(encoding="utf-8"))
    actual = _sha256(found.checkpoint)
    if actual != record["sha256"]:
        raise RuntimeError(
            f"bar tracker checkpoint sha {actual} is not the {record['sha256']} "
            f"the record beside it names — refusing to track bars with an "
            f"unverified model")
    if record["geometry"] != GEOMETRY:
        raise RuntimeError(
            f"bar tracker record geometry {record['geometry']} disagrees with "
            f"the runtime's {GEOMETRY} — the checkpoint was recorded for a "
            f"different framing")
    return record


def load_bar_tracker(generation_dir, *, device: str | None = None):
    from lib.analyser.mert_stream import best_device
    from lib.vendor import beat_this_infer

    record = load_record(generation_dir)
    found = artifacts(generation_dir)
    where = device or best_device()
    model, hparams = beat_this_infer.load_model(found.checkpoint, where)
    recorded = record.get("arch")
    if recorded is not None and recorded != hparams:
        raise RuntimeError(f"bar tracker arch {hparams} is not the recorded "
                           f"{recorded}")
    logging.info(f'[tracker] {found.checkpoint.name} on {where} '
                 f'(sha {record["sha256"][:12]})')
    return BarTrackerStream(model, device=where)


class BarTrackerStream:
    def __init__(self, model, *, device: str = "cpu",
                 source_rate: int = SOURCE_SAMPLE_RATE) -> None:
        if int(source_rate) != SOURCE_SAMPLE_RATE:
            raise ValueError(f"the tracker's framing constants assume "
                             f"{SOURCE_SAMPLE_RATE} Hz source audio, "
                             f"not {source_rate}")
        from lib.vendor.beat_this_infer import LogMelSpect

        self._model = model
        self._device = device
        self._mel = LogMelSpect(device)
        from lib.vendor.beat_this_infer import N_MELS
        self._n_mels = N_MELS
        self._ring = SampleRing(_RING_STRIDES * _STRIDE_SRC)
        from collections import deque

        self.alive = True
        self.passes = 0
        self.pass_sec: deque = deque(maxlen=256)
        self.reset()

    def reset(self) -> None:
        self._ring.reset()
        self.passes = 0
        self._n_frames = 0
        self._valid_from = 0
        self._prev_safe = 0
        self._window = np.zeros((CHUNK_FRAMES, self._n_mels), dtype=np.float32)

    @property
    def samples_seen(self) -> int:
        return self._ring.written

    def feed(self, samples) -> None:
        if not self.alive:
            return
        self._ring.write(np.asarray(samples, dtype=np.float32).reshape(-1))

    def due(self) -> bool:
        return (self.alive
                and self._ring.written >= self._next_needed())

    def _next_needed(self) -> int:
        return (self.passes + 1) * _STRIDE_SRC + _PASS_PAD_SRC

    def run_pass(self) -> list:
        if not self.due():
            return []
        import time

        started = time.perf_counter()
        try:
            chunk = self._one_pass()
        except RingOverrun:
            self.resync()
            return []
        except Exception as error:
            self.alive = False
            logging.warning(f'[tracker] forward failed ({error!r}) — the bar '
                            f'tracker is dead for this run; the decoder is on '
                            f'the counting grid')
            return []
        self.pass_sec.append(time.perf_counter() - started)
        self.passes += 1
        return [] if chunk is None else [chunk]

    def resync(self) -> None:
        """Skip to the live edge after a stall: the hole is never back-filled."""
        passes = max(self.passes,
                     (self._ring.written - _PASS_PAD_SRC) // _STRIDE_SRC)
        skipped_to = passes * STRIDE_FRAMES
        if skipped_to > self._n_frames:
            logging.warning(f'[tracker] resync: skipping frames '
                            f'[{self._n_frames}, {skipped_to}) — evidence for '
                            f'that span is lost')
            self._n_frames = skipped_to
            self._valid_from = skipped_to
            self._prev_safe = skipped_to
        self.passes = passes

    def _one_pass(self):
        import torch

        end_f = (self.passes + 1) * STRIDE_FRAMES
        self._extend_frames(end_f)
        window = self._assemble_window(end_f)
        with torch.inference_mode():
            x = torch.from_numpy(window).to(self._device).unsqueeze(0)
            logits = self._model(x)["downbeat"][0].float().cpu().numpy()
        safe_end = end_f - MARGIN_FRAMES
        lo = max(self._prev_safe, self._valid_from, end_f - CHUNK_FRAMES)
        if safe_end <= lo:
            return None
        self._prev_safe = safe_end
        offset = end_f - CHUNK_FRAMES
        return TrackerChunk(lo, safe_end,
                            np.ascontiguousarray(
                                logits[lo - offset:safe_end - offset]),
                            self._avail_sec(end_f))

    @staticmethod
    def _avail_sec(end_f: int) -> float:
        return (end_f * _SRC_PER_FRAME + _PASS_PAD_SRC) / float(SOURCE_SAMPLE_RATE)

    def _extend_frames(self, end_f: int) -> None:
        import soxr
        import torch

        nf = self._n_frames
        if end_f <= nf:
            return
        lo22 = nf * _HOP_22K - (_N_FFT // 2)
        hi22 = (end_f - 1) * _HOP_22K + (_N_FFT // 2)
        src_lo = max(0, 2 * max(0, lo22) - _RESAMPLE_CTX_SRC)
        src_hi = 2 * hi22 + _RESAMPLE_CTX_SRC
        segment = self._ring.snapshot(src_lo, src_hi)
        resampled = soxr.resample(segment, SOURCE_SAMPLE_RATE, 22050)
        start22 = src_lo // 2
        span = resampled[max(0, lo22) - start22:hi22 - start22]
        if lo22 < 0:
            # torch's own reflect padding at the head of the whole stream.
            span = np.concatenate([span[1:1 - lo22][::-1], span])
        with torch.inference_mode():
            frames = self._mel(torch.tensor(span, dtype=torch.float32,
                                            device=self._device),
                               center=False).cpu().numpy()
        if frames.shape[0] != end_f - nf:
            raise RuntimeError(f"framing arithmetic drifted: {frames.shape[0]} "
                               f"frames for [{nf}, {end_f})")
        self._window[np.arange(nf, end_f) % CHUNK_FRAMES] = frames
        self._n_frames = end_f

    def _assemble_window(self, end_f: int) -> np.ndarray:
        out = np.zeros((CHUNK_FRAMES, self._window.shape[1]), dtype=np.float32)
        indices = np.arange(end_f - CHUNK_FRAMES, end_f)
        held = indices >= self._valid_from
        out[held] = self._window[indices[held] % CHUNK_FRAMES]
        return out
