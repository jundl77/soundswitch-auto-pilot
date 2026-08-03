"""The online student, one cell at a time -- the deployed decision."""
from __future__ import annotations

import hashlib
import json
from collections import deque
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
    backward_cells: int | None = None

    @property
    def conv_reach_cells(self) -> int:
        return self.window_cells - self.future_cells - 1


def load_head_geometry(onnx_path) -> HeadGeometry:
    meta_path = Path(str(onnx_path) + ".json")
    record = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [field for field in _GEOMETRY_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{meta_path} records no {', '.join(missing)}")
    backward = record.get("arch", {}).get("backward_cells")
    return HeadGeometry(window_cells=int(record["window_cells"]),
                        input_dim=int(record["input_dim"]),
                        rnn_hidden=int(record["rnn_hidden"]),
                        future_cells=int(record["future_cells"]),
                        future_sec=float(record["future_sec"]),
                        label_frame_sec=float(record["label_frame_sec"]),
                        sha256=str(record["sha256"]),
                        backward_cells=None if backward is None
                        else int(backward))


def check_head_geometry(geometry: HeadGeometry) -> None:
    if geometry.conv_reach_cells < 0:
        raise ValueError(f"a {geometry.window_cells}-cell window cannot hold "
                         f"{geometry.future_cells} future cells and the present "
                         f"one: conv reach {geometry.conv_reach_cells}")
    spanned = geometry.future_cells * geometry.label_frame_sec
    if abs(geometry.future_sec - spanned) > geometry.label_frame_sec / 2.0:
        raise ValueError(f"future_sec {geometry.future_sec} is not the "
                         f"{spanned} that {geometry.future_cells} cells of "
                         f"{geometry.label_frame_sec} span")
    if geometry.backward_cells is not None:
        implied = 2 * geometry.future_cells - geometry.backward_cells + 1
        if implied != geometry.window_cells:
            raise ValueError(f"window_cells {geometry.window_cells} is not the "
                             f"{implied} its future and backward reach imply")


def check_graph_geometry(session, geometry: HeadGeometry) -> None:
    declared = getattr(session, "get_inputs", None)
    produced = getattr(session, "get_outputs", None)
    if not callable(declared) or not callable(produced):
        raise ValueError("the session cannot declare its own shapes, so the "
                         "graph and the sidecar cannot be cross-checked")
    inputs = {port.name: list(port.shape) for port in declared()}
    outputs = {port.name: list(port.shape) for port in produced()}
    missing = [name for name in (FRAMES_INPUT, STATE_INPUT) if name not in inputs]
    missing += [name for name in (LABEL_OUTPUT, BOUNDARY_OUTPUT, STATE_OUTPUT)
                if name not in outputs]
    if missing:
        raise ValueError(f"the graph declares no {', '.join(missing)}")
    _axis(inputs[FRAMES_INPUT], -2, geometry.window_cells, "window_cells")
    _axis(inputs[FRAMES_INPUT], -1, geometry.input_dim, "input_dim")
    _axis(inputs[STATE_INPUT], -1, geometry.rnn_hidden, "rnn_hidden")
    _axis(outputs[STATE_OUTPUT], -1, geometry.rnn_hidden, "rnn_hidden")


def _axis(shape: list, axis: int, wanted: int, field: str) -> None:
    found = shape[axis] if len(shape) >= abs(axis) else None
    if isinstance(found, int) and found != wanted:
        raise ValueError(f"the graph's {field} is {found}, not the {wanted} "
                         f"its sidecar records")


def session_options() -> ort.SessionOptions:
    # Single-threaded is a determinism contract: float addition is not associative.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def session(path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), session_options(),
                                providers=["CPUExecutionProvider"])


class Flushed(RuntimeError):
    ...


class Posterior(NamedTuple):
    index: int
    time_sec: float
    posterior: np.ndarray
    boundary: float


class SectionModel:
    def __init__(self, onnx_path, *, mean, geometry: HeadGeometry | None = None,
                 expected_sha: str | None = None, session_factory=None) -> None:
        self.geometry = geometry or load_head_geometry(onnx_path)
        check_head_geometry(self.geometry)
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
        check_graph_geometry(self._session, self.geometry)
        self.reset()

    def reset(self) -> None:
        self._ring = np.repeat(self._mean[None], self.geometry.window_cells,
                               axis=0)
        self._state = np.zeros((1, 1, self.geometry.rnn_hidden),
                               dtype=np.float32)
        self._window: deque = deque()
        self._last_index: int | None = None
        self._flushed = False

    def push(self, features, index: int | None = None) -> Posterior | None:
        if self._flushed:
            raise Flushed("the model was flushed at a song boundary; reset it "
                          "before pushing the next song's cells")
        return self._push(features, index)

    def _push(self, features, index: int | None = None) -> Posterior | None:
        row = np.asarray(features, dtype=np.float32).reshape(-1)
        if len(row) != self.geometry.input_dim:
            raise ValueError(f"a cell is {len(row)}-dim, the graph's input_dim "
                             f"is {self.geometry.input_dim}")
        self._ring[:-1] = self._ring[1:]
        self._ring[-1] = row
        self._last_index = (0 if self._last_index is None
                            else self._last_index + 1) if index is None \
            else int(index)
        self._window.append(self._last_index)
        if len(self._window) <= self.geometry.future_cells:
            return None
        return self._step(self._window.popleft())

    def flush(self) -> list:
        if self._flushed:
            raise Flushed("the model has already been flushed; reset it first")
        self._flushed = True
        out = [self._push(self._mean) for _ in range(self.geometry.future_cells)]
        return [item for item in out if item is not None]

    def _step(self, index: int) -> Posterior:
        frames = np.ascontiguousarray(self._ring[None], dtype=np.float32)
        label, boundary, self._state = self._session.run(
            [LABEL_OUTPUT, BOUNDARY_OUTPUT, STATE_OUTPUT],
            {FRAMES_INPUT: frames, STATE_INPUT: self._state})
        return Posterior(index, (index + 1) * self.geometry.label_frame_sec,
                         _softmax(label[0]),
                         _sigmoid(boundary.reshape(-1)[0]))


class Drained(NamedTuple):
    gap: bool
    posteriors: list


class PosteriorStream:
    def __init__(self, stream, model: SectionModel) -> None:
        self.stream = stream
        self.model = model

    def push_audio(self, samples) -> Drained:
        self.feed(samples)
        posteriors = []
        while self.due():
            posteriors.extend(self.run_pass())
        return Drained(False, posteriors)

    def feed(self, samples) -> None:
        self.stream.push_audio(samples)

    def due(self) -> bool:
        return self.stream.due()

    def run_pass(self) -> list:
        out = [self.model.push(cell.features, cell.index)
               for cell in self.stream.run_pass()]
        return [item for item in out if item is not None]

    def resync(self):
        record = self.stream.resync()
        self.model.reset()
        return record

    def set_encoder(self, encoder) -> None:
        self.stream.set_encoder(encoder)

    def reset(self) -> None:
        self.stream.reset()
        self.model.reset()


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = np.exp(logits - logits.max())
    return (shifted / shifted.sum()).astype(np.float32)


def _sigmoid(logit) -> float:
    return float(1.0 / (1.0 + np.exp(-float(logit))))
