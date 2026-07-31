"""The online student, one cell at a time -- the deployed decision.

Ported from phase-b's `OnlineCRNN.step` and the graph
`training/nn/ceiling/online_export.py` exports from it. The model is
bidirectional over a bounded window: cell ``t`` is decided from a ring holding
``t-reach .. t+future+reach``, plus the forward GRU's carried state, which is
the whole past of the song at no extra cost. So the live path holds two things
and nothing else -- a ring of feature cells and one state tensor.

**The ring is primed from the corpus mean, never zeros** (D10). Zero raw
features are a confident out-of-distribution input after the model's own input
affine; the corpus mean is the one row it reads as no information, and it is
what the whole-track pass pads its edges with.

**The session is pinned single-threaded.** That is a determinism contract, not
a performance choice: a threaded reduction sums in whatever order the pool
finishes in, and float addition is not associative. Throughput, when it is
wanted, comes from running tracks in parallel over separate sessions.

**The graph is verified against its recorded sha at construction**, not at the
first beat -- a show that discovers its model is the wrong one halfway through
a set has already played the wrong lights.

The session is built through an injectable factory. `session` is the only
definition of the pinned options and is the default; the seam exists because
building a synthetic ONNX graph for a unit test needs the `onnx` package, which
is not a dependency of the show, and because Task 11 has to inject a session
that faults.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnxruntime as ort

FRAMES_INPUT = "frames"
STATE_INPUT = "state"
LABEL_OUTPUT = "label_logits"
BOUNDARY_OUTPUT = "boundary_logit"
STATE_OUTPUT = "next_state"

_GEOMETRY_FIELDS = ("window_cells", "input_dim", "rnn_hidden", "future_cells",
                    "future_sec", "label_frame_sec", "sha256")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HeadGeometry:
    window_cells: int
    input_dim: int
    rnn_hidden: int
    future_cells: int
    future_sec: float
    label_frame_sec: float
    sha256: str

    @property
    def conv_reach_cells(self) -> int:
        return self.window_cells - self.future_cells - 1


def load_head_geometry(onnx_path) -> HeadGeometry:
    """The graph's own record of its shape -- a retyped constant drifts."""
    meta_path = Path(str(onnx_path) + ".json")
    record = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [field for field in _GEOMETRY_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{meta_path} records no {', '.join(missing)}")
    return HeadGeometry(window_cells=int(record["window_cells"]),
                        input_dim=int(record["input_dim"]),
                        rnn_hidden=int(record["rnn_hidden"]),
                        future_cells=int(record["future_cells"]),
                        future_sec=float(record["future_sec"]),
                        label_frame_sec=float(record["label_frame_sec"]),
                        sha256=str(record["sha256"]))


def session_options() -> ort.SessionOptions:
    """Split out from `session` because a constructed InferenceSession does not
    report the options it was built with, so this is the only object a test can
    assert on."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def session(path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), session_options(),
                                providers=["CPUExecutionProvider"])


class Posterior(NamedTuple):
    index: int
    time_sec: float
    posterior: np.ndarray
    boundary: float


class SectionModel:
    """One posterior per label cell, from the ring and the carried state."""

    def __init__(self, onnx_path, *, mean, geometry: HeadGeometry | None = None,
                 expected_sha: str | None = None, session_factory=None) -> None:
        self.geometry = geometry or load_head_geometry(onnx_path)
        wanted = expected_sha or self.geometry.sha256
        found = sha256_file(onnx_path)
        if found != wanted:
            raise RuntimeError(f"the graph at {onnx_path} hashes to {found}, "
                               f"not the {wanted} it is recorded as")
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        if len(mean) != self.geometry.input_dim:
            raise ValueError(f"the affine is {len(mean)}-dim, the graph's "
                             f"input_dim is {self.geometry.input_dim}")
        self._mean = mean
        self._session = (session_factory or session)(onnx_path)
        self.reset()

    def reset(self) -> None:
        self._ring = np.repeat(self._mean[None], self.geometry.window_cells,
                               axis=0)
        self._state = np.zeros((1, 1, self.geometry.rnn_hidden),
                               dtype=np.float32)
        self._pushed = 0
        self._flushed = False

    def push(self, features) -> Posterior | None:
        row = np.asarray(features, dtype=np.float32).reshape(-1)
        if len(row) != self.geometry.input_dim:
            raise ValueError(f"a cell is {len(row)}-dim, the graph's input_dim "
                             f"is {self.geometry.input_dim}")
        self._ring[:-1] = self._ring[1:]
        self._ring[-1] = row
        self._pushed += 1
        index = self._pushed - 1 - self.geometry.future_cells
        return None if index < 0 else self._step(index)

    def flush(self) -> list:
        """Drain the cells still inside the future window at a song boundary."""
        if self._flushed:
            return []
        self._flushed = True
        out = [self.push(self._mean) for _ in range(self.geometry.future_cells)]
        return [item for item in out if item is not None]

    def _step(self, index: int) -> Posterior:
        frames = np.ascontiguousarray(self._ring[None], dtype=np.float32)
        label, boundary, self._state = self._session.run(
            [LABEL_OUTPUT, BOUNDARY_OUTPUT, STATE_OUTPUT],
            {FRAMES_INPUT: frames, STATE_INPUT: self._state})
        return Posterior(index, index * self.geometry.label_frame_sec,
                         _softmax(label[0]),
                         _sigmoid(boundary.reshape(-1)[0]))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = np.exp(logits - logits.max())
    return (shifted / shifted.sum()).astype(np.float32)


def _sigmoid(logit) -> float:
    return float(1.0 / (1.0 + np.exp(-float(logit))))
