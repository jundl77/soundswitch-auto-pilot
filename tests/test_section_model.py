"""The online student's deployed step: a ring of cells in, one posterior out.

The load-bearing test is phase-b's streaming-equivalence test, ported: the live
path is not a re-implementation of the offline one, it is the same graph fed the
same tensors, and this is where that claim is checked. A streaming path whose
numbers are merely close to the offline model's is the defect the whole
train==deploy discipline exists to prevent.

The unit cases run a fake session with the deployed signature -- same input and
output names, same carried state -- so the ring-buffer and state bookkeeping is
exercised without the shipped 1.4 M-parameter student and without loading a
graph at all. The real graph is checked under `integration`, and that is where
the equivalence claim is actually settled.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from lib.analyser import section_model as S

WINDOW = 12
FUTURE = 9
DIM = 6
HIDDEN = 4
CLASSES = 5


def _softmax(x):
    shifted = np.exp(np.asarray(x, dtype=np.float32)
                     - np.asarray(x, dtype=np.float32).max())
    return (shifted / shifted.sum()).astype(np.float32)


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-float(x))))


class _Port:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    """The deployed signature, as a pure function of (window, state).

    Standing in for a synthetic ONNX graph, which cannot be built without the
    `onnx` package -- not a dependency of the show, and not worth becoming one
    to give a bookkeeping test a different arithmetic backend. What these tests
    are about is the ring and the carried state; the real graph is checked
    under `integration`, which is where a graph question belongs.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.into = rng.normal(size=(DIM, HIDDEN)).astype(np.float32) * 0.4
        self.carry = rng.normal(size=(HIDDEN, HIDDEN)).astype(np.float32) * 0.6
        self.label = rng.normal(size=(HIDDEN, CLASSES)).astype(np.float32)
        self.edge = rng.normal(size=(HIDDEN, 1)).astype(np.float32)
        self.weights = np.linspace(0.1, 1.0, WINDOW,
                                   dtype=np.float32).reshape(1, WINDOW, 1)
        self.calls = 0

    def get_inputs(self):
        return [_Port(S.FRAMES_INPUT, ["batch", WINDOW, DIM]),
                _Port(S.STATE_INPUT, [1, "batch", HIDDEN])]

    def get_outputs(self):
        return [_Port(S.LABEL_OUTPUT, ["batch", CLASSES]),
                _Port(S.BOUNDARY_OUTPUT, ["batch", 1]),
                _Port(S.STATE_OUTPUT, [1, "batch", HIDDEN])]

    def run(self, names, feeds):
        assert names == [S.LABEL_OUTPUT, S.BOUNDARY_OUTPUT, S.STATE_OUTPUT]
        frames = feeds[S.FRAMES_INPUT]
        state = feeds[S.STATE_INPUT]
        assert frames.shape == (1, WINDOW, DIM), frames.shape
        assert state.shape == (1, 1, HIDDEN), state.shape
        self.calls += 1
        pooled = (frames * self.weights).sum(axis=1)
        hidden = np.tanh(pooled @ self.into + state[0] @ self.carry)
        return [hidden @ self.label, hidden @ self.edge,
                hidden[None].astype(np.float32)]


@pytest.fixture
def tiny(tmp_path):
    """A file whose bytes hash to what its sidecar records."""
    path = tmp_path / "tiny_step.onnx"
    path.write_bytes(b"not a real graph, but a real sha")
    meta = {"sha256": S.sha256_file(path), "window_cells": WINDOW,
            "input_dim": DIM, "rnn_hidden": HIDDEN, "future_cells": FUTURE,
            "future_sec": FUTURE * 0.25, "label_frame_sec": 0.25}
    Path(str(path) + ".json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")
    return path


@pytest.fixture
def geometry(tiny):
    return S.load_head_geometry(tiny)


@pytest.fixture
def mean():
    return (np.arange(DIM, dtype=np.float32) * 0.3 - 0.5)


def _model(tiny, mean, **kwargs):
    kwargs.setdefault("session_factory", lambda _path: FakeSession())
    return S.SectionModel(tiny, mean=mean, **kwargs)


def _cells(count, seed=1):
    return np.random.default_rng(seed).normal(
        size=(count, DIM)).astype(np.float32)


def _reference(path, geometry, cells, mean, session=None):
    """The offline windowed pass: the same graph, windows built explicitly."""
    session = session or S.session(path)
    reach = geometry.conv_reach_cells
    padded = np.concatenate([np.repeat(np.asarray(mean)[None], reach, axis=0),
                             cells,
                             np.repeat(np.asarray(mean)[None],
                                       geometry.future_cells, axis=0)])
    state = np.zeros((1, 1, geometry.rnn_hidden), dtype=np.float32)
    out = []
    for index in range(len(cells)):
        frames = np.ascontiguousarray(
            padded[index:index + geometry.window_cells][None],
            dtype=np.float32)
        label, boundary, state = session.run(
            [S.LABEL_OUTPUT, S.BOUNDARY_OUTPUT, S.STATE_OUTPUT],
            {S.FRAMES_INPUT: frames, S.STATE_INPUT: state})
        out.append((_softmax(label[0]), _sigmoid(boundary.reshape(-1)[0])))
    return out


def _stream(model, cells):
    out = []
    for row in cells:
        posterior = model.push(row)
        if posterior is not None:
            out.append(posterior)
    out.extend(model.flush())
    return out


# --------------------------------------------------------------------------- #
# TRAIN == DEPLOY
# --------------------------------------------------------------------------- #


def test_the_streaming_posteriors_reproduce_the_offline_windowed_pass(
        tiny, geometry, mean):
    cells = _cells(60)
    live = _stream(_model(tiny, mean), cells)
    offline = _reference(tiny, geometry, cells, mean, FakeSession())

    assert [item.index for item in live] == list(range(len(cells)))
    for item, (posterior, boundary) in zip(live, offline):
        assert np.abs(item.posterior - posterior).max() < 1e-6, item.index
        assert abs(item.boundary - boundary) < 1e-6, item.index


def test_the_forward_state_is_carried_across_steps(tiny, geometry, mean):
    """Reset only at a song boundary -- a per-step reset is a different model."""
    cells = _cells(40)
    carried = _stream(_model(tiny, mean), cells)

    cold = []
    for row in cells:
        model = _model(tiny, mean)
        cold.extend(_stream(model, np.repeat(row[None], 1, axis=0)))
    assert np.abs(carried[-1].posterior - cold[-1].posterior).max() > 1e-4


def test_the_feature_ring_is_primed_from_the_corpus_mean_not_zeros(
        tiny, geometry, mean):
    """Zeros are not silence after the input affine -- they are a confident,
    out-of-distribution input (D10)."""
    cells = _cells(30)
    primed = _stream(_model(tiny, mean), cells)
    zeroed = _reference(tiny, geometry, cells,
                        np.zeros(DIM, dtype=np.float32), FakeSession())
    assert np.abs(primed[0].posterior - zeroed[0][0]).max() > 1e-5


def test_no_posterior_is_emitted_until_the_future_window_has_arrived(
        tiny, mean):
    model = _model(tiny, mean)
    cells = _cells(FUTURE + 3)
    emitted = [model.push(row) is not None for row in cells]
    assert emitted == [False] * FUTURE + [True] * 3


def test_flush_drains_the_tail_cells(tiny, mean):
    model = _model(tiny, mean)
    live = [model.push(row) for row in _cells(20)]
    seen = len([item for item in live if item is not None])
    tail = model.flush()
    assert [item.index for item in tail] == list(range(seen, 20))


def test_a_flush_is_terminal_until_the_model_is_reset(tiny, mean):
    """Pushes after a flush kept emitting -- from a ring holding the padding
    rows the flush put there, at indices that carried straight on -- and the
    next song's flush returned nothing, dropping its last four seconds. Two
    silent failures where MertStream, the other half of the same pipeline,
    raises.
    """
    model = _model(tiny, mean)
    _stream(model, _cells(20))
    with pytest.raises(S.Flushed):
        model.push(_cells(1)[0])
    with pytest.raises(S.Flushed):
        model.flush()
    model.reset()
    assert model.push(_cells(1)[0]) is None


def test_reset_returns_the_model_to_its_cold_state(tiny, mean):
    cells = _cells(35)
    model = _model(tiny, mean)
    first = _stream(model, cells)
    model.reset()
    second = _stream(model, cells)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.array_equal(a.posterior, b.posterior)
        assert a.boundary == b.boundary


def test_posteriors_are_stamped_on_the_label_grid(tiny, geometry, mean):
    live = _stream(_model(tiny, mean), _cells(25))
    for item in live:
        assert item.time_sec == pytest.approx(item.index
                                              * geometry.label_frame_sec)


def test_a_posterior_is_a_distribution_and_the_boundary_a_probability(
        tiny, mean):
    for item in _stream(_model(tiny, mean), _cells(25)):
        assert item.posterior.shape == (CLASSES,)
        assert item.posterior.dtype == np.float32
        assert float(item.posterior.sum()) == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= item.boundary <= 1.0


# --------------------------------------------------------------------------- #
# The graph is verified at startup, not at the first beat
# --------------------------------------------------------------------------- #


def test_the_head_geometry_is_read_from_the_shipped_json(geometry):
    assert geometry.window_cells == WINDOW
    assert geometry.future_cells == FUTURE
    assert geometry.input_dim == DIM
    assert geometry.rnn_hidden == HIDDEN
    assert geometry.label_frame_sec == 0.25


def test_the_conv_reach_is_derived_from_the_window_not_retyped(geometry):
    assert geometry.conv_reach_cells == WINDOW - FUTURE - 1
    assert geometry.conv_reach_cells == 2


def test_a_graph_whose_bytes_do_not_match_the_recorded_sha_is_refused(
        tiny, mean):
    with pytest.raises(RuntimeError, match="graph"):
        _model(tiny, mean, expected_sha="0" * 64)


def test_the_recorded_sha_is_checked_against_the_file_at_startup(
        tmp_path, tiny, mean):
    copy = tmp_path / "tampered.onnx"
    copy.write_bytes(Path(tiny).read_bytes() + b"\x00")
    meta = json.loads(Path(str(tiny) + ".json").read_text(encoding="utf-8"))
    Path(str(copy) + ".json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="graph"):
        _model(copy, mean)


def test_a_json_missing_a_geometry_field_is_refused(tmp_path, tiny):
    copy = tmp_path / "partial.onnx"
    copy.write_bytes(Path(tiny).read_bytes())
    meta = json.loads(Path(str(tiny) + ".json").read_text(encoding="utf-8"))
    del meta["future_cells"]
    Path(str(copy) + ".json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="future_cells"):
        S.load_head_geometry(copy)


def _retyped(tmp_path, tiny, **changes):
    """The same graph bytes with an edited sidecar -- the sha covers only the
    .onnx, so this is what a wrong-geometry model looks like from outside."""
    copy = tmp_path / "regeometried.onnx"
    copy.write_bytes(Path(tiny).read_bytes())
    meta = json.loads(Path(str(tiny) + ".json").read_text(encoding="utf-8"))
    meta.update(changes)
    meta["sha256"] = S.sha256_file(copy)
    Path(str(copy) + ".json").write_text(json.dumps(meta), encoding="utf-8")
    return copy


def test_a_window_the_graph_does_not_declare_is_refused_at_startup(
        tmp_path, tiny, mean):
    """Reproduced against the shipped graph: window_cells 46 -> 45 constructed
    cleanly and raised inside onnxruntime at push #44 -- mid-show, which is
    exactly what verifying at construction is supposed to prevent."""
    with pytest.raises(ValueError, match="window_cells"):
        _model(_retyped(tmp_path, tiny, window_cells=WINDOW - 1), mean)


def test_a_state_width_the_graph_does_not_declare_is_refused_at_startup(
        tmp_path, tiny, mean):
    with pytest.raises(ValueError, match="rnn_hidden"):
        _model(_retyped(tmp_path, tiny, rnn_hidden=HIDDEN + 1), mean)


def test_a_future_window_that_contradicts_its_own_seconds_is_refused(
        tmp_path, tiny, mean):
    """The graph shape cannot see this one: future_cells 43 -> 41 on the shipped
    model constructed AND ran, stamping every posterior two cells early and
    moving probabilities by up to 0.155, with nothing raising anywhere."""
    with pytest.raises(ValueError, match="future_sec"):
        _model(_retyped(tmp_path, tiny, future_cells=FUTURE - 2), mean)


def test_a_window_that_cannot_hold_its_own_future_is_refused(
        tmp_path, tiny, mean):
    with pytest.raises(ValueError, match="conv reach"):
        _model(_retyped(tmp_path, tiny, window_cells=FUTURE,
                        future_cells=FUTURE), mean)


def test_a_session_that_cannot_describe_itself_is_a_failure_not_a_skip(
        tiny, mean):
    """The seam the reviewer used to demonstrate the hole is the one place the
    check must not quietly turn itself off."""
    class _Mute(FakeSession):
        get_inputs = None
        get_outputs = None

    with pytest.raises(ValueError, match="declare"):
        S.SectionModel(tiny, mean=mean, session_factory=lambda _p: _Mute())


def test_an_affine_of_the_wrong_width_is_refused(tiny):
    with pytest.raises(ValueError, match="input_dim"):
        S.SectionModel(tiny, mean=np.zeros(DIM + 1, dtype=np.float32))


def test_a_feature_row_of_the_wrong_width_is_refused(tiny, mean):
    model = _model(tiny, mean)
    with pytest.raises(ValueError, match="input_dim"):
        model.push(np.zeros(DIM + 2, dtype=np.float32))


# --------------------------------------------------------------------------- #
# The determinism contract
# --------------------------------------------------------------------------- #


def test_the_session_is_pinned_to_one_thread_and_sequential_execution():
    """A threaded reduction sums in whatever order the pool finishes in, and
    float addition is not associative."""
    options = S.session_options()
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL


# --------------------------------------------------------------------------- #
# The shipped student (integration)
# --------------------------------------------------------------------------- #


SHIPPED_ONNX_SHA = ("f1fe6ef7c3cc0dede24a7d572841b3eb2c381f12"
                    "3868f67dcf0e1d0298aa33b4")


def _shipped():
    import run_eval_set

    directory = (Path(run_eval_set.corpus_dir()) / "models" / "phase_b")
    onnx = directory / "student_kd_t2_w05_s1234" / "online_step.onnx"
    affine = directory / "input_affine_F3.npz"
    if not onnx.exists() or not affine.exists():
        pytest.skip(f"shipping artifacts absent under {directory} -- "
                    f"they live in the gitignored corpus data directory")
    return onnx, affine


@pytest.mark.integration
def test_the_shipped_graph_is_the_one_the_frontier_was_measured_on():
    onnx, _affine = _shipped()
    head = S.load_head_geometry(onnx)
    assert head.sha256 == SHIPPED_ONNX_SHA
    assert S.session(onnx).get_providers() == ["CPUExecutionProvider"]
    assert S.sha256_file(onnx) == SHIPPED_ONNX_SHA
    assert (head.window_cells, head.future_cells, head.input_dim) == (46, 43, 2048)


@pytest.mark.integration
def test_the_shipped_student_streams_exactly_the_offline_windowed_pass():
    from lib.analyser import mert_stream as M

    onnx, affine = _shipped()
    head = S.load_head_geometry(onnx)
    mean, _std = M.load_input_affine(affine)
    cells = np.random.default_rng(11).normal(
        size=(120, head.input_dim)).astype(np.float32)

    live = _stream(S.SectionModel(onnx, mean=mean, expected_sha=SHIPPED_ONNX_SHA),
                   cells)
    offline = _reference(onnx, head, cells, mean)
    assert len(live) == len(cells)
    for item, (posterior, boundary) in zip(live, offline):
        assert np.abs(item.posterior - posterior).max() < 1e-6, item.index
        assert abs(item.boundary - boundary) < 1e-6, item.index


@pytest.mark.integration
def test_the_shipped_student_is_deterministic_across_two_runs_in_one_process():
    onnx, affine = _shipped()
    from lib.analyser import mert_stream as M

    head = S.load_head_geometry(onnx)
    mean, _std = M.load_input_affine(affine)
    cells = np.random.default_rng(3).normal(
        size=(90, head.input_dim)).astype(np.float32)
    runs = [_stream(S.SectionModel(onnx, mean=mean), cells) for _ in range(2)]
    for a, b in zip(*runs):
        assert np.array_equal(a.posterior, b.posterior)
        assert a.boundary == b.boundary


def test_sha256_file_reads_the_bytes_on_disk(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"abc")
    assert S.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()
