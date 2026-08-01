"""D12: the extractor's cells, cached beside the audio -- the decode cache one
layer up.

`simulate file` runs the identical NN path to production, and the only part of
it that needs a GPU is the MERT encoder.  Recording the cells it emits and
replaying them makes every later run pure CPU, and makes the honest determinism
claim the `.npy` decode cache already set the precedent for: **byte-identical
reports given cached sidecars**, with everything downstream of the extractor
bit-exact and the extractor itself replayed.

**The trigger is a position in the call sequence, not song time.**  A pass is
recorded against the number of source samples pushed when it ran, so a replay
needs neither the resampler, the ring, nor the schedule -- and a `reset()`
mid-run (the engine does one at each song boundary) needs no special handling,
because the recorded triggers already span it.  A `Replay` is therefore
single-use and `open_replay` hands back a fresh one per simulation.

**The decode path is part of the identity, twice** (#161).  The simulation
decodes mp3s with librosa and the corpus pipeline with ffmpeg, and the two move
13.2% of near-boundary decisions -- so a corpus sidecar replayed into a sim (or
the reverse) would be a silently different measurement.  It is in the filename,
which makes the collision structurally impossible, and in the key, which makes
it a named miss rather than a wrong answer.

The threaded stage is deliberately not cached: `--ui` and `realtime` exist to
run the real GPU thread, and a replay there would prove nothing about it.
"""
from __future__ import annotations

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

# 1980-01-01, the zip epoch: see training/nn/infer.py's `save_posteriors`, whose
# reasoning this writer follows -- np.savez inherits its timestamp from a CPython
# default and its compressed form folds the zlib build into the bytes, so a
# determinism claim resting on either is a claim about one machine.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

_TOP_LEVEL_MISSES = (("schema", "miss_schema"),
                     ("decode", "miss_decode_path"),
                     ("source_rate", "miss_source_rate"),
                     ("audio_size", "miss_audio_changed"),
                     ("audio_mtime", "miss_audio_changed"))
_GROUP_MISSES = (("encoder", "miss_encoder"), ("framing", "miss_framing"),
                 ("backend", "miss_backend"))

log = logging.getLogger(__name__)


class TruncatedRecording(RuntimeError):
    """A replay was pushed past the audio its recording covers.

    The recorder saves on every normal stop, and `run_simulation` stops on a
    `duration_sec` bound as readily as on the end of the file -- so a run cut
    short leaves a sidecar that is indistinguishable, by key, from a complete
    one.  Replayed into a full run it serves cells until the recording ends and
    then serves nothing at all, for the rest of the track, in silence: the
    decoder stops receiving posteriors and the show holds one intent with no
    fault anywhere.  Loud and stopped beats quiet and wrong.
    """


def sidecar_path(audio_path, decode_path: str) -> Path:
    audio_path = Path(audio_path)
    return audio_path.with_name(f"{audio_path.name}.{decode_path}.{SUFFIX}")


def cache_key(geometry, *, source_rate: int, audio_path, decode_path: str,
              backend: dict | None = None) -> dict:
    """Everything a cell depends on, in the three groups that fail differently.

    `encoder` is which weights produced the features, `framing` is the geometry
    they were pooled under, and `backend` is the arithmetic that ran them.
    Split so a miss says which one moved rather than "something".

    **The backend belongs here for the same reason the decode path does.**  It
    is chosen at build time, not requested: `build_section_chain` resolves the
    device off `best_device()` and the precision off its `fp16` default, so the
    same call produces cuda-fp16 cells on this box and cpu-fp32 cells on a box
    with no GPU.  Those are different numbers, and without this a sidecar
    recorded on one replays as the other's answer -- a wrong measurement with
    nothing to say so.  It is resolved through the chain's own helper rather
    than recomputed, so the key and the builder cannot drift apart.
    """
    from lib import section_chain

    stat = Path(audio_path).stat()
    return {
        "schema": SCHEMA,
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
    """``None`` when the sidecar may be replayed, else the name of the miss."""
    for field, reason in _TOP_LEVEL_MISSES:
        if stored.get(field) != wanted.get(field):
            return reason
    for group, reason in _GROUP_MISSES:
        if stored.get(group) != wanted.get(group):
            return reason
    return None


def open_replay(path, key: dict, expected_samples: int | None = None):
    """``(Replay, "hit")``, or ``(None, reason)`` -- never an exception.

    ``expected_samples`` is how much audio this run will push, when the caller
    can say.  A recording that does not cover it is ``miss_truncated`` and is
    re-recorded rather than served short.  A caller that cannot say still gets
    the guarantee, one layer later: `Replay` refuses at the sample the
    recording stops covering instead of quietly emitting nothing.
    """
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
    """Do the five arrays describe one recording?

    Checked at load, where it is a named miss and costs a re-record, rather
    than at the first `run_pass` that indexes past the end -- which happens
    minutes into a batch, out of the middle of a simulation, as an IndexError
    with nothing in it about a cache.  A sidecar can be inconsistent without
    being unreadable: a writer killed between members, or a schema that grew a
    column while keeping its name.
    """
    offsets = archive["pass_offset"]
    cells = len(archive["cell_index"])
    return (len(offsets) == len(archive["pass_trigger"]) + 1
            and (len(offsets) == 0 or (int(offsets[0]) == 0
                                       and int(offsets[-1]) == cells))
            and len(archive["cell_seen_sec"]) == cells
            and len(archive["cell_features"]) == cells)


class Replay:
    """The recorded cells, handed back on the schedule they were emitted on.

    Single-use: `reset` is deliberately not a rewind, because the recording
    already contains whatever the live stream did after its own resets.
    """

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
    """A pass-through over the live extractor that writes what it emitted.

    Only the four methods the synchronous consumer uses are forwarded.  The
    shed path's `resync` is not one of them: the fast simulation has no
    watchdog and cannot shed, so a resync reaching a recording would be a new
    fact about the pipeline and should stop rather than be silently cached.
    """

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
    """The chain's stream when this run is the one paying for the GPU.

    `stop` is where the sidecar is written, because `SectionChain.stop` is
    already the one place that knows whether a chain owns something that has to
    be wound down, and nothing above it should have to remember.
    """

    def stop(self) -> None:
        self.stream.save()


def recording_chain(chain, path, key: dict):
    """The shipped chain, re-headed onto a recorder over its own extractor."""
    return chain._replace(
        stream=RecordingStream(Recorder(chain.stream.stream, path, key),
                               chain.stream.model))


def _write_archive(path: Path, arrays: dict) -> bool:
    """Publish the sidecar atomically, or log why it could not be.

    **The temp name carries the writer's identity.**  It used to be one fixed
    `.part` beside the audio, and the corpus batch and the benchmark both run a
    pool over the same files: two writers then interleaved on one temp file and
    published each other's bytes.  On Windows the second `replace` onto a file
    the first still has open raises `PermissionError` instead -- out of
    `run_simulation`, after every expensive thing in the run had already
    succeeded.

    **A cache is an optimisation, so failing to publish one is not a failure.**
    The simulation that just finished is complete and correct whether or not
    its cells reach the disk; the next run simply pays for the GPU again.
    """
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
