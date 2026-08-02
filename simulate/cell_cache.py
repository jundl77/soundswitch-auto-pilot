"""The extractor's cells, recorded beside the audio and replayed without a GPU."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import secrets
import zipfile
from pathlib import Path

import numpy as np

from lib.analyser.mert_stream import Cell
from lib.analyser.section_model import PosteriorStream

SCHEMA = "mert-cells/1"
SUFFIX = "mertcells.npz"

# 1980-01-01 is the zip format's own epoch: the fixed timestamp keeps clocks out of the bytes.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

_EXTRACTOR_SOURCES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[1] / "lib" / "analyser" / "mert_stream.py",
)


def extractor_sha() -> str:
    """Source bytes rather than a git sha: an uncommitted edit invalidates cells too."""
    digest = hashlib.sha256()
    for path in _EXTRACTOR_SOURCES:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]

_TOP_LEVEL_MISSES = (("schema", "miss_schema"),
                     ("extractor", "miss_extractor"),
                     ("decode", "miss_decode_path"),
                     ("source_rate", "miss_source_rate"),
                     ("audio_size", "miss_audio_changed"),
                     ("audio_mtime", "miss_audio_changed"))
_GROUP_MISSES = (("encoder", "miss_encoder"), ("framing", "miss_framing"),
                 ("backend", "miss_backend"))

log = logging.getLogger(__name__)


class TruncatedRecording(RuntimeError):
    ...


def sidecar_path(audio_path, decode_path: str) -> Path:
    audio_path = Path(audio_path)
    return audio_path.with_name(f"{audio_path.name}.{decode_path}.{SUFFIX}")


def cache_key(geometry, *, source_rate: int, audio_path, decode_path: str,
              backend: dict | None = None) -> dict:
    from lib import section_chain

    stat = Path(audio_path).stat()
    return {
        "schema": SCHEMA,
        "extractor": extractor_sha(),
        "decode": str(decode_path),
        "encoder": {
            "model_id": geometry.model_id,
            "revision": geometry.revision,
            "layers": [int(layer) for layer in geometry.layers],
            "encoder_sha": geometry.encoder_sha,
        },
        "framing": {
            "margin_sec": float(geometry.margin_sec),
            "hop_sec": float(geometry.hop_sec),
            "buffer_sec": float(geometry.buffer_sec),
            "label_frame_sec": float(geometry.label_frame_sec),
        },
        "backend": dict(backend if backend is not None
                        else section_chain.resolve_backend()),
        "source_rate": int(source_rate),
        "audio_size": stat.st_size,
        "audio_mtime": stat.st_mtime,
    }


def miss_reason(stored: dict, wanted: dict) -> str | None:
    for field, reason in _TOP_LEVEL_MISSES:
        if stored.get(field) != wanted.get(field):
            return reason
    for group, reason in _GROUP_MISSES:
        if stored.get(group) != wanted.get(group):
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
            if not _internally_consistent(archive):
                return None, "miss_schema"
            return Replay(archive["pass_trigger"], archive["pass_offset"],
                          archive["cell_index"], archive["cell_seen_sec"],
                          archive["cell_features"],
                          float(key["framing"]["label_frame_sec"]),
                          total), "hit"
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None, "miss_unreadable"


def _internally_consistent(archive) -> bool:
    offsets = archive["pass_offset"]
    cells = len(archive["cell_index"])
    return (len(offsets) == len(archive["pass_trigger"]) + 1
            and (len(offsets) == 0 or (int(offsets[0]) == 0
                                       and int(offsets[-1]) == cells))
            and len(archive["cell_seen_sec"]) == cells
            and len(archive["cell_features"]) == cells)


class Replay:
    def __init__(self, triggers, offsets, indices, seen_sec, features,
                 label_frame_sec: float, total_pushed: int = 0) -> None:
        self._triggers = np.asarray(triggers, dtype=np.int64)
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._indices = np.asarray(indices, dtype=np.int64)
        self._seen_sec = np.asarray(seen_sec, dtype=np.float64)
        self._features = np.asarray(features, dtype=np.float32)
        self._label_frame_sec = float(label_frame_sec)
        self._total_pushed = int(total_pushed)
        self._pushed = 0
        self._cursor = 0

    def push_audio(self, samples) -> None:
        self._pushed += len(samples)
        if self._pushed > self._total_pushed:
            raise TruncatedRecording(
                f"this recording covers {self._total_pushed} source samples "
                f"and the run has pushed {self._pushed} -- it was cut short "
                f"(a duration_sec bound saves a sidecar exactly like a "
                f"complete run does).  Delete the sidecar, or re-record it "
                f"over the whole file.")

    def due(self) -> bool:
        return (self._cursor < len(self._triggers)
                and self._pushed >= int(self._triggers[self._cursor]))

    def run_pass(self) -> list:
        if not self.due():
            return []
        index = self._cursor
        self._cursor += 1
        lo, hi = int(self._offsets[index]), int(self._offsets[index + 1])
        return [self._cell(row) for row in range(lo, hi)]

    def reset(self) -> None:
        pass

    def _cell(self, row: int) -> Cell:
        index = int(self._indices[row])
        return Cell(index, (index + 1) * self._label_frame_sec,
                    self._features[row], float(self._seen_sec[row]))


class Recorder:
    def __init__(self, stream, path, key: dict) -> None:
        self.stream = stream
        self.path = Path(path)
        self._key = key
        self._pushed = 0
        self._triggers: list = []
        self._offsets: list = [0]
        self._indices: list = []
        self._seen_sec: list = []
        self._features: list = []

    @property
    def geometry(self):
        return self.stream.geometry

    def push_audio(self, samples) -> None:
        self._pushed += len(samples)
        self.stream.push_audio(samples)

    def due(self) -> bool:
        return self.stream.due()

    def run_pass(self) -> list:
        cells = self.stream.run_pass()
        self._triggers.append(self._pushed)
        for cell in cells:
            self._indices.append(int(cell.index))
            self._seen_sec.append(float(cell.audio_seen_sec))
            self._features.append(np.asarray(cell.features, dtype=np.float32))
        self._offsets.append(len(self._indices))
        return cells

    def reset(self) -> None:
        self.stream.reset()

    def save(self) -> None:
        dim = len(self._features[0]) if self._features else 0
        features = (np.stack(self._features) if self._features
                    else np.zeros((0, dim), dtype=np.float32))
        written = _write_archive(self.path, {
            "key": np.str_(json.dumps(self._key, sort_keys=True)),
            "total_pushed": np.int64(self._pushed),
            "pass_trigger": np.asarray(self._triggers, dtype=np.int64),
            "pass_offset": np.asarray(self._offsets, dtype=np.int64),
            "cell_index": np.asarray(self._indices, dtype=np.int64),
            "cell_seen_sec": np.asarray(self._seen_sec, dtype=np.float64),
            "cell_features": features,
        })
        if written:
            log.info(f'[cells] wrote {len(self._indices)} cells over '
                     f'{len(self._triggers)} passes → {self.path}')


class RecordingStream(PosteriorStream):
    def stop(self) -> None:
        self.stream.save()


def recording_chain(chain, path, key: dict):
    return chain._replace(
        stream=RecordingStream(Recorder(chain.stream.stream, path, key),
                               chain.stream.model))


def _write_archive(path: Path, arrays: dict) -> bool:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, value in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(value),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue())
    tmp = path.with_name(f"{path.name}.{os.getpid()}."
                         f"{secrets.token_hex(4)}.part")
    try:
        tmp.write_bytes(buffer.getvalue())
        tmp.replace(path)
        return True
    except OSError as error:
        tmp.unlink(missing_ok=True)
        log.warning(f'[cells] could not publish {path.name} ({error!r}) — the '
                    f'run is unaffected, the next one pays for the GPU again')
        return False
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
