"""The bar tracker's chunks, recorded beside the audio and replayed without a GPU.

The mertcells pattern one stage over (see simulate/cell_cache.py): the trigger
is a position in the call sequence, the key carries the checkpoint sha, the
framing, the decode path and the audio's identity, and the archive bytes are a
pure function of the contents.
"""
from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

from lib.analyser import bar_tracker
from simulate.cell_cache import _write_archive

SCHEMA = "bar-tracker/1"
SUFFIX = "bartracker.npz"

_TRACKER_SOURCES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[1] / "lib" / "analyser" / "bar_tracker.py",
    Path(__file__).resolve().parents[1] / "lib" / "vendor" / "beat_this_infer.py",
)

log = logging.getLogger(__name__)


def tracker_sha() -> str:
    digest = hashlib.sha256()
    for path in _TRACKER_SOURCES:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def sidecar_path(audio_path, decode_path: str) -> Path:
    audio_path = Path(audio_path)
    return audio_path.with_name(f"{audio_path.name}.{decode_path}.{SUFFIX}")


def cache_key(record: dict, *, source_rate: int, audio_path,
              decode_path: str, backend: dict | None = None) -> dict:
    from lib import section_chain

    stat = Path(audio_path).stat()
    return {
        "schema": SCHEMA,
        "extractor": tracker_sha(),
        "decode": str(decode_path),
        "checkpoint_sha256": record["sha256"],
        "geometry": dict(record["geometry"]),
        # The tracker runs fp32, but which device computed a chunk is part of
        # what it is -- cell_cache's own rule.
        "backend": dict(backend if backend is not None
                        else section_chain.resolve_backend(fp16=False)),
        "source_rate": int(source_rate),
        "audio_size": stat.st_size,
        "audio_mtime": stat.st_mtime,
    }


def miss_reason(stored: dict, wanted: dict) -> str | None:
    for field, reason in (("schema", "miss_schema"),
                          ("extractor", "miss_extractor"),
                          ("decode", "miss_decode_path"),
                          ("checkpoint_sha256", "miss_checkpoint"),
                          ("geometry", "miss_geometry"),
                          ("backend", "miss_backend"),
                          ("source_rate", "miss_source_rate"),
                          ("audio_size", "miss_audio_changed"),
                          ("audio_mtime", "miss_audio_changed")):
        if stored.get(field) != wanted.get(field):
            return reason
    return None


def open_replay(path, key: dict, expected_samples: int | None = None):
    path = Path(path)
    if not path.exists():
        return None, "miss_new"
    try:
        with np.load(path) as archive:
            stored = json.loads(str(archive["key"]))
            reason = miss_reason(stored, key)
            if reason is not None:
                return None, reason
            total = int(archive["total_pushed"])
            if expected_samples is not None and total < int(expected_samples):
                return None, "miss_truncated"
            return TrackerReplay(archive["pass_trigger"],
                                 archive["chunk_offset"],
                                 archive["frame_lo"], archive["frame_hi"],
                                 archive["avail_sec"], archive["logits"],
                                 archive["logit_offset"], total), "hit"
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None, "miss_unreadable"


class TrackerReplay:
    alive = True

    def __init__(self, triggers, chunk_offsets, frame_lo, frame_hi, avail_sec,
                 logits, logit_offsets, total_pushed: int) -> None:
        self._triggers = np.asarray(triggers, dtype=np.int64)
        self._chunk_offsets = np.asarray(chunk_offsets, dtype=np.int64)
        self._frame_lo = np.asarray(frame_lo, dtype=np.int64)
        self._frame_hi = np.asarray(frame_hi, dtype=np.int64)
        self._avail_sec = np.asarray(avail_sec, dtype=np.float64)
        self._logits = np.asarray(logits, dtype=np.float32)
        self._logit_offsets = np.asarray(logit_offsets, dtype=np.int64)
        self._total_pushed = int(total_pushed)
        self._pushed = 0
        self._cursor = 0
        self.pass_sec: deque = deque(maxlen=1)

    def feed(self, samples) -> None:
        self._pushed += len(samples)
        if self._pushed > self._total_pushed:
            from simulate.cell_cache import TruncatedRecording

            raise TruncatedRecording(
                f"this tracker recording covers {self._total_pushed} source "
                f"samples and the run has pushed {self._pushed} — delete the "
                f"sidecar or re-record it over the whole file")

    def due(self) -> bool:
        return (self._cursor < len(self._triggers)
                and self._pushed >= int(self._triggers[self._cursor]))

    def run_pass(self) -> list:
        if not self.due():
            return []
        index = self._cursor
        self._cursor += 1
        lo, hi = (int(self._chunk_offsets[index]),
                  int(self._chunk_offsets[index + 1]))
        return [self._chunk(row) for row in range(lo, hi)]

    def _chunk(self, row: int) -> bar_tracker.TrackerChunk:
        a, b = int(self._logit_offsets[row]), int(self._logit_offsets[row + 1])
        return bar_tracker.TrackerChunk(int(self._frame_lo[row]),
                                        int(self._frame_hi[row]),
                                        self._logits[a:b],
                                        float(self._avail_sec[row]))

    def reset(self) -> None:
        pass

    def resync(self) -> None:
        pass


class TrackerRecorder:
    def __init__(self, tracker, path, key: dict) -> None:
        self.tracker = tracker
        self.path = Path(path)
        self._key = key
        self._pushed = 0
        self._triggers: list = []
        self._chunk_offsets: list = [0]
        self._frame_lo: list = []
        self._frame_hi: list = []
        self._avail_sec: list = []
        self._logits: list = []
        self._logit_offsets: list = [0]

    @property
    def alive(self) -> bool:
        return self.tracker.alive

    @property
    def pass_sec(self):
        return self.tracker.pass_sec

    def feed(self, samples) -> None:
        self._pushed += len(samples)
        self.tracker.feed(samples)

    def due(self) -> bool:
        return self.tracker.due()

    def run_pass(self) -> list:
        chunks = self.tracker.run_pass()
        self._triggers.append(self._pushed)
        for chunk in chunks:
            self._frame_lo.append(int(chunk.frame_lo))
            self._frame_hi.append(int(chunk.frame_hi))
            self._avail_sec.append(float(chunk.avail_sec))
            self._logits.append(np.asarray(chunk.db_logits, dtype=np.float32))
            self._logit_offsets.append(self._logit_offsets[-1]
                                       + len(chunk.db_logits))
        self._chunk_offsets.append(len(self._frame_lo))
        return chunks

    def reset(self) -> None:
        self.tracker.reset()

    def resync(self) -> None:
        self.tracker.resync()

    def stop(self) -> None:
        self.save()

    def save(self) -> None:
        logits = (np.concatenate(self._logits) if self._logits
                  else np.zeros(0, dtype=np.float32))
        written = _write_archive(self.path, {
            "key": np.str_(json.dumps(self._key, sort_keys=True)),
            "total_pushed": np.int64(self._pushed),
            "pass_trigger": np.asarray(self._triggers, dtype=np.int64),
            "chunk_offset": np.asarray(self._chunk_offsets, dtype=np.int64),
            "frame_lo": np.asarray(self._frame_lo, dtype=np.int64),
            "frame_hi": np.asarray(self._frame_hi, dtype=np.int64),
            "avail_sec": np.asarray(self._avail_sec, dtype=np.float64),
            "logits": logits,
            "logit_offset": np.asarray(self._logit_offsets, dtype=np.int64),
        })
        if written:
            log.info(f'[tracker-cache] wrote {len(self._frame_lo)} chunks over '
                     f'{len(self._triggers)} passes → {self.path}')
