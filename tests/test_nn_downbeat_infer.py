"""Tests for the downbeat ONNX export and the activation sidecars
(``training/nn/downbeat_infer.py``).

Everything Task 4 scores is read out of these ``.npz`` files, and a defect here
is invisible downstream: the decoder will happily phase-label an activation that
is shifted by a frame, averaged over the wrong windows, or produced by last
week's checkpoint, and every number in the verdict will be wrong in a way no
later assertion can catch.  So the checks are aimed at the parts that fail
*silently*.

**The graph.**  ``torch.onnx.export`` reports success even when the dynamo path
has baked the traced time length into a GRU model, and this architecture has the
same GRU the section head's pre-flight caught it on.  The declared axes are read
back out of the file and the graph is run at a length it never saw.

**The numbers.**  A golden inference against a committed reference pins torch,
the exporter and onnxruntime together, on a model built from a numpy seed so it
runs on a machine with no corpus.

**The geometry.**  Stub sessions whose output encodes either the global frame
index (so a one-frame shift is visible) or the position within the window (so a
leaked window edge is visible).  Real inference cannot test this: every plausible
bug still produces a plausible-looking activation.

**The aggregation.**  Beat instants are jittered on purpose.  A tracker's beats
do not land on the mel grid and do not land where the annotator put them, so a
window that only works on exact instants is a window that only works offline.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training extra not synced")
pytest.importorskip("onnx", reason="training extra not synced")
pytest.importorskip("onnxruntime", reason="training extra not synced")

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.dataset import FRAME_SEC, WINDOW_FRAMES, load_sidecar  # noqa: E402
from nn.downbeat_decoder import aggregate_at_beats  # noqa: E402
from nn.downbeat_infer import (  # noqa: E402
    EDGE_FRAMES,
    HOP_FRAMES,
    INPUT_NAME,
    MODEL_FILE,
    OUTPUT_NAME,
    TIME_AXIS,
    live_beat_times,
    build_from_checkpoint,
    declared_axes,
    default_checkpoint,
    export_model,
    infer_track,
    lag_profile,
    load_downbeat_checkpoint,
    run_window,
    save_sidecar,
    session,
    sidecar_arrays,
    sidecar_is_current,
)
from nn.downbeat_model import DownbeatCRNN  # noqa: E402

MEL_BANDS = 40
GOLDEN_FILE = Path(__file__).resolve().parent / "data" / "nn_downbeat_onnx_golden.npz"
GOLDEN_TOLERANCE = 1e-5
PARITY_TOLERANCE = 1e-4
GAIN = 2.0

DATA_DIR = Path(__file__).resolve().parents[1] / "training" / "data" / "raveform"
if not DATA_DIR.exists():   # the worktree has no data dir; the main checkout does
    DATA_DIR = (Path(__file__).resolve().parents[2] / "soundswitch-auto-pilot"
                / "training" / "data" / "raveform")

needs_corpus = pytest.mark.skipif(
    not (DATA_DIR / "features").is_dir(),
    reason=f"no corpus at {DATA_DIR} (gitignored)",
)


# --------------------------------------------------------------------------- #
# A reproducible stand-in for the trained model
# --------------------------------------------------------------------------- #


def seeded_model(seed: int = 20260727) -> DownbeatCRNN:
    """The real architecture with weights drawn from a numpy seed.

    Same recipe as the section head's golden model, and for the same reason:
    numpy's ``default_rng`` stream is a documented, versioned guarantee, so the
    reference survives exactly the torch upgrade it exists to notice.  The gain
    is what keeps a random GRU from contracting to a constant, which would make
    the reference pin nothing.
    """
    model = DownbeatCRNN()
    state = model.state_dict()
    normalised = {name.rsplit(".", 1)[0] for name in state if name.endswith("running_mean")}
    moments = {"weight": (1.0, 0.25), "bias": (0.0, 0.25),
               "running_mean": (0.0, 0.25), "running_var": (1.0, 0.5)}

    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for name, tensor in sorted(state.items()):
            if not tensor.dtype.is_floating_point:
                continue
            prefix, leaf = name.rsplit(".", 1)
            draw = rng.random(tensor.numel()) * 2.0 - 1.0
            if prefix in normalised:
                centre, spread = moments[leaf]
                draw = centre + spread * draw
            elif tensor.dim() > 1:
                draw = GAIN * draw / math.sqrt(tensor[0].numel())     # fan-in
            else:
                draw = 0.05 * draw
            tensor.copy_(torch.from_numpy(draw.reshape(tensor.shape)).float())
    return model.eval()


def seeded_mel(frames: int = WINDOW_FRAMES, seed: int = 4242) -> np.ndarray:
    """A synthetic log-mel window on roughly the scale the sidecars carry."""
    rng = np.random.default_rng(seed)
    time = np.arange(frames, dtype=np.float64)[:, None]
    band = np.arange(MEL_BANDS, dtype=np.float64)[None, :]
    centre = 8.0 + 20.0 * (0.5 + 0.5 * np.sin(time * 0.01))
    sweep = np.exp(-0.5 * ((band - centre) / 4.0) ** 2)
    pulse = 1.0 + 0.8 * np.sin(time * 0.6)
    mel = 2.5 * sweep * pulse + 0.3 * rng.random((frames, MEL_BANDS))
    return np.maximum(mel, 0.0).astype(np.float32)[None]


def exported(tmp_path, model=None, **kwargs):
    path = tmp_path / MODEL_FILE
    export_model(model or seeded_model(), path, **kwargs)
    return path, session(path)


@pytest.fixture(scope="session")
def graph(tmp_path_factory):
    """One export shared by the tests that are not testing the exporter."""
    return exported(tmp_path_factory.mktemp("downbeat_graph"))


# --------------------------------------------------------------------------- #
# Stub sessions -- the only way to see the window geometry
# --------------------------------------------------------------------------- #


def _logit(value):
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(value / (1.0 - value))


class IndexSession:
    """Echoes the mel back as an activation, so a frame that moved is visible.

    The caller writes the *global* frame index into the mel; the returned
    activation is that value, so the stitched track must come back as the
    identity.  A one-frame slice error anywhere in the window loop shows up as a
    step in a straight line.
    """

    def run(self, _outputs, feed):
        return [_logit(feed[INPUT_NAME][:, :, 0])]


class PositionSession:
    """Answers by *position inside the window*, so a leaked edge is visible."""

    def __init__(self, window_frames=WINDOW_FRAMES, edge_frames=EDGE_FRAMES):
        self.window_frames = window_frames
        self.edge_frames = edge_frames

    def run(self, _outputs, feed):
        position = np.arange(feed[INPUT_NAME].shape[1])
        edge = (position < self.edge_frames) | (position >= self.window_frames - self.edge_frames)
        return [_logit(np.where(edge, 0.99, 0.01))[None].repeat(feed[INPUT_NAME].shape[0], 0)]


def index_mel(n_frames: int) -> np.ndarray:
    """Mel whose every band carries the frame's own position in ``[0, 1)``."""
    return np.repeat((np.arange(n_frames) / n_frames)[:, None],
                     MEL_BANDS, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Export: the graph
# --------------------------------------------------------------------------- #


def test_export_declares_a_dynamic_time_axis(tmp_path):
    """The failure this guards is silent: the dynamo exporter bakes the traced
    length into a GRU graph and still reports success."""
    path, _sess = exported(tmp_path)

    axes = declared_axes(path)

    assert axes[INPUT_NAME] == ["batch", TIME_AXIS, MEL_BANDS]
    assert axes[OUTPUT_NAME] == ["batch", TIME_AXIS]


@pytest.mark.parametrize("frames", [WINDOW_FRAMES, 512, 64])
def test_exported_graph_runs_at_a_length_it_was_not_traced_at(graph, frames):
    _path, sess = graph

    logits = run_window(sess, seeded_mel(frames))

    assert logits.shape == (1, frames)


def test_golden_onnx_inference_matches_the_saved_reference(tmp_path):
    """One uncached export + inference against a committed reference, so an
    exporter or onnxruntime upgrade fails here rather than as an unexplained
    metric change in Task 4."""
    _path, sess = exported(tmp_path)

    logits = run_window(sess, seeded_mel())

    with np.load(GOLDEN_FILE) as reference:
        assert logits.shape == tuple(reference["downbeat_logits"].shape)
        assert np.abs(logits - reference["downbeat_logits"]).max() < GOLDEN_TOLERANCE

    # The reference is only worth its tolerance if the output moves when the
    # input does; a contracted random net would pass the comparison above while
    # pinning nothing at all.
    other = run_window(sess, seeded_mel(seed=99))
    assert np.abs(logits - other).max() > 1000 * GOLDEN_TOLERANCE


@pytest.mark.parametrize("frames", [WINDOW_FRAMES, 512, 64])
def test_torch_and_onnx_agree_on_synthetic_windows(graph, frames):
    model = seeded_model()
    _path, sess = graph
    mel = seeded_mel(frames, seed=frames)

    logits = run_window(sess, mel)
    with torch.no_grad():
        want = model(torch.from_numpy(mel))

    assert np.abs(logits - want.numpy()).max() < PARITY_TOLERANCE


def test_the_session_is_the_pinned_single_threaded_one(graph):
    """The determinism contract is one definition, in ``export_onnx.session``."""
    from nn.export_onnx import session as section_session

    from nn.downbeat_infer import session as downbeat_session

    assert downbeat_session is section_session


# --------------------------------------------------------------------------- #
# The checkpoint payload -- pinned, because it differs from the section head's
# --------------------------------------------------------------------------- #


def make_checkpoint(tmp_path, **overrides):
    model = DownbeatCRNN()
    payload = {"model": model.state_dict(), "arch": model.arch(), "config": {},
               "epoch": 20, "f1": 0.5512, "metrics": {"f1": 0.5512},
               "pos_weight": 9.355}
    payload.update(overrides)
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path


def test_the_downbeat_checkpoint_payload_shape_is_pinned(tmp_path):
    """``best.pt`` here is *flat* -- ``["metrics"]["f1"]`` and ``["f1"]`` are both
    floats -- while the section head's nests one level deeper.  Nothing else in
    the tree pins that, so a loader written against the wrong shape would read a
    dict where it expects a number and only notice in a report."""
    state = load_downbeat_checkpoint(make_checkpoint(tmp_path))

    assert isinstance(state["f1"], float)
    assert isinstance(state["metrics"]["f1"], float)
    assert isinstance(state["pos_weight"], float)
    assert set(state["arch"]) == {"n_mels", "conv_channels", "conv1d_channels",
                                  "rnn_hidden", "rnn_layers"}


@pytest.mark.parametrize("missing", ["arch", "model", "pos_weight"])
def test_a_checkpoint_missing_a_required_field_is_named_not_guessed(tmp_path, missing):
    payload = torch.load(make_checkpoint(tmp_path), map_location="cpu",
                         weights_only=False)
    del payload[missing]
    path = tmp_path / "broken.pt"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match=missing):
        load_downbeat_checkpoint(path)


def test_build_from_checkpoint_accepts_the_json_round_tripped_arch(tmp_path):
    """``arch`` survives a JSON round trip in ``config.json``, so it arrives with
    lists where the constructor took tuples.  A constructor that reinterpreted
    one of its fields would build a differently shaped net that still loads."""
    path = make_checkpoint(tmp_path)
    state = load_downbeat_checkpoint(path)
    state["arch"] = json.loads(json.dumps(state["arch"]))

    model = build_from_checkpoint(state)

    assert model.arch() == state["arch"]
    assert not model.training


def test_an_arch_the_weights_do_not_fit_is_refused(tmp_path):
    state = load_downbeat_checkpoint(make_checkpoint(tmp_path))
    state["arch"] = dict(state["arch"], rnn_hidden=32)

    with pytest.raises(RuntimeError):
        build_from_checkpoint(state)


# --------------------------------------------------------------------------- #
# Window geometry
# --------------------------------------------------------------------------- #


def test_every_frame_of_a_track_is_voted_on(graph):
    _path, sess = graph

    track = infer_track(sess, seeded_mel(1200)[0])

    assert track.n_frames == 1200
    assert np.all(track.coverage >= 1)
    assert track.activation.shape == (1200,)


def test_the_stitched_activation_is_the_identity_on_an_index_mel():
    """A one-frame slice error anywhere in the window loop puts a step in what
    must be a straight line."""
    n_frames = 1000
    track = infer_track(IndexSession(), index_mel(n_frames))

    assert np.abs(track.activation - np.arange(n_frames) / n_frames).max() < 1e-6


def test_no_window_edge_reaches_a_frame_another_window_could_have_voted_on():
    """The spec forbids reading a window's outermost frames -- cold GRU state,
    no context on one side.  The only frames allowed to see one are the track's
    own first and last, which no window's interior can reach."""
    n_frames = 1200
    track = infer_track(PositionSession(), np.zeros((n_frames, MEL_BANDS), np.float32))

    interior = track.coverage >= 2
    assert np.abs(track.activation[interior] - 0.01).max() < 1e-6
    lonely = np.flatnonzero(~interior)
    assert lonely.min() < EDGE_FRAMES
    assert lonely.max() >= n_frames - EDGE_FRAMES


def test_a_track_shorter_than_one_window_is_padded_not_refused(graph):
    _path, sess = graph

    track = infer_track(sess, seeded_mel(100)[0])

    assert track.n_frames == 100
    assert track.windows == 1


def test_the_geometry_travels_with_the_result(graph):
    """A sidecar that recorded the module defaults while holding non-default
    numbers is worse than an unlabelled one -- the cache key would accept it."""
    _path, sess = graph

    track = infer_track(sess, seeded_mel(800)[0], hop_frames=8, edge_frames=10)

    assert (track.hop_frames, track.edge_frames) == (8, 10)


def test_a_hop_wider_than_the_window_interior_is_refused(graph):
    _path, sess = graph

    with pytest.raises(ValueError, match="uncovered"):
        infer_track(sess, seeded_mel(800)[0], hop_frames=WINDOW_FRAMES)


# --------------------------------------------------------------------------- #
# Where the activation sits against the grid
# --------------------------------------------------------------------------- #


def test_lag_profile_finds_an_injected_shift():
    """The instrument the aggregation window is chosen with: it has to be able to
    see an offset before it is trusted to report that there is none."""
    activation = np.zeros(500, dtype=np.float64)
    marks = np.arange(20, 480, 16)
    activation[marks + 2] = 1.0

    profile = lag_profile(activation, marks, radius=4)

    assert int(np.argmax(profile)) - 4 == 2


# --------------------------------------------------------------------------- #
# Sidecars
# --------------------------------------------------------------------------- #


def make_track(n_frames=300):
    rng = np.random.default_rng(3)
    from nn.downbeat_infer import TrackActivation
    return TrackActivation(
        rng.random(n_frames).astype(np.float32),
        np.full(n_frames, 7, dtype=np.uint16),
        n_frames, 12, WINDOW_FRAMES, HOP_FRAMES, EDGE_FRAMES)


def make_arrays(model_sha="a" * 64):
    track = make_track()
    beats = {"live": (np.array([1.0, 2.0]), np.array([0.3, 0.7]),
                       np.array([3, 3], dtype=np.int32)),
             "expert": (np.array([1.1]), np.array([0.5]),
                        np.array([3], dtype=np.int32))}
    return sidecar_arrays(track, beats, model_sha, pos_weight=9.355)


def test_two_writes_of_one_sidecar_are_byte_identical(tmp_path):
    """np.savez's reproducibility is a CPython default, not an API promise, and
    a determinism claim resting on it is a claim about this machine."""
    arrays = make_arrays()
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"

    save_sidecar(first, arrays)
    save_sidecar(second, arrays)

    assert first.read_bytes() == second.read_bytes()


def test_a_sidecar_round_trips_both_conditions(tmp_path):
    path = tmp_path / "t.npz"
    save_sidecar(path, make_arrays())

    with np.load(path) as archive:
        assert archive["live_beat_time"].tolist() == [1.0, 2.0]
        assert archive["expert_beat_score"].tolist() == [0.5]
        assert float(archive["pos_weight"]) == pytest.approx(9.355)
        assert float(archive["frame_sec"]) == pytest.approx(FRAME_SEC)


def test_the_stored_beat_scores_are_reproducible_from_the_stored_curve(tmp_path):
    """The half-beat decode aggregates its own candidates off ``activation``, so
    the cached per-beat scores and a fresh aggregation have to be the same thing.
    If they ever diverge, a subdivision-1 and a subdivision-2 decode of one track
    would be reading two different models."""
    path = tmp_path / "t.npz"
    track = make_track(600)
    beats = np.arange(2.0, 20.0, 0.47)
    scores, counts = aggregate_at_beats(track.activation, beats, FRAME_SEC, FRAME_SEC)
    save_sidecar(path, sidecar_arrays(
        track, {"live": (beats, scores, counts), "expert": (beats, scores, counts)},
        "a" * 64, 9.355))

    with np.load(path) as archive:
        again, _counts = aggregate_at_beats(archive["activation"],
                                            archive["live_beat_time"],
                                            float(archive["frame_sec"]),
                                            float(archive["t0"]))
        assert np.array_equal(again, archive["live_beat_score"])


def test_the_sidecar_carries_no_bar_phase():
    """The expert *phase* is truth, and truth must not ride into a test-split
    artifact.  Beat instants are the diagnostic condition's input; the phase is
    what Task 4 scores against, read from the annotations."""
    assert not any("phase" in key for key in make_arrays())


def test_sidecar_is_current_keys_on_the_model_and_the_geometry(tmp_path):
    path = tmp_path / "t.npz"
    save_sidecar(path, make_arrays("a" * 64))

    assert sidecar_is_current(path, "a" * 64)
    assert not sidecar_is_current(path, "b" * 64)
    assert not sidecar_is_current(tmp_path / "absent.npz", "a" * 64)


def test_a_geometry_change_invalidates_the_cache(tmp_path):
    path = tmp_path / "t.npz"
    arrays = make_arrays()
    arrays["hop_frames"] = np.int32(HOP_FRAMES + 2)
    save_sidecar(path, arrays)

    assert not sidecar_is_current(path, "a" * 64)


# --------------------------------------------------------------------------- #
# The live beat stream
# --------------------------------------------------------------------------- #


def test_live_beats_are_read_from_the_cached_sim_report(tmp_path):
    """The live condition's input is the production pipeline's own beat stream,
    bit for bit -- not a re-derivation."""
    import gzip

    reports = tmp_path / "reports"
    reports.mkdir()
    payload = {"report": {"beats": [{"t": 1.5, "bpm": 128.0}, {"t": 2.0}]}}
    with gzip.open(reports / "abc.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    assert live_beat_times(tmp_path, "abc").tolist() == [1.5, 2.0]


# --------------------------------------------------------------------------- #
# Split hygiene on the CLI
# --------------------------------------------------------------------------- #


def _val_only_ids(monkeypatch):
    """Stand in for the corpus's splits so the guard is testable without one."""
    import nn.downbeat_infer as module

    monkeypatch.setattr(module, "split_ids",
                        lambda _dir, splits=("val", "test"): sorted(
                            {"v1", "v2"} if list(splits) == ["val"]
                            else {"v1", "v2", "t1"}))


@pytest.mark.parametrize("argv", [
    ["--lag-profile", "--splits", "val", "test"],
    ["--lag-profile", "--ids", "t1"],
])
def test_the_lag_profile_refuses_the_test_split(monkeypatch, argv):
    """It reads annotated bar phase to *choose a decoder parameter*, which makes
    it a tuning measurement -- and a bare run used to default to val+test.  The
    guard checks split membership rather than the flag, because an explicit
    ``--ids`` walks straight past a flag-level check."""
    from nn.downbeat_infer import main

    _val_only_ids(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code != 0


def test_the_lag_profile_defaults_to_val_alone(monkeypatch, capsys):
    """The default is the thing that was wrong, so the default is what is pinned."""
    import nn.downbeat_infer as module

    _val_only_ids(monkeypatch)
    seen = {}

    def record(_data_dir, ids, **_kwargs):
        seen["ids"] = list(ids)
        return {"tracks": 0, "offsets": [0], "downbeat_mean": [0.0],
                "offbeat_mean": [0.0], "peak_histogram": [0],
                "argmax_offset": 0, "modal_peak_offset": 0}

    monkeypatch.setattr(module, "measure_lag", record)

    assert module.main(["--lag-profile"]) == 0
    assert seen["ids"] == ["v1", "v2"]


def test_generation_still_covers_val_and_test(monkeypatch):
    """The quarantine is on the *tuning* read, not on producing inputs -- a
    guard that also blocked sidecar generation would be the wrong fix."""
    import nn.downbeat_infer as module

    _val_only_ids(monkeypatch)
    seen = {}

    def record(_data_dir, **kwargs):
        seen["ids"] = list(kwargs["ids"])
        return {"tracks": 0, "computed": 0, "cached": 0, "failed": 0, "frames": 0,
                "windows": 0, "bytes": 0, "wall_seconds": 0.0, "model_sha": "x" * 64}

    monkeypatch.setattr(module, "generate", record)

    assert module.main([]) == 0
    assert seen["ids"] == ["t1", "v1", "v2"]


def test_a_missing_report_names_the_track(tmp_path):
    (tmp_path / "reports").mkdir()

    with pytest.raises(RuntimeError, match="abc"):
        live_beat_times(tmp_path, "abc")


# --------------------------------------------------------------------------- #
# The real artifacts (skipped without the gitignored corpus)
# --------------------------------------------------------------------------- #


@needs_corpus
def test_torch_and_onnx_agree_on_real_windows_from_the_trained_checkpoint(tmp_path):
    checkpoint = default_checkpoint(DATA_DIR)
    if not checkpoint.exists():
        pytest.skip(f"no checkpoint at {checkpoint}")
    model = build_from_checkpoint(load_downbeat_checkpoint(checkpoint))
    _path, sess = exported(tmp_path, model)

    sidecars = sorted((DATA_DIR / "features").glob("*.npz"))[:2]
    windows = [load_sidecar(sidecars[0])[:WINDOW_FRAMES][None],
               load_sidecar(sidecars[1])[5000:5000 + WINDOW_FRAMES][None],
               load_sidecar(sidecars[0])[1000:1000 + 512][None]]

    for mel in windows:
        logits = run_window(sess, np.ascontiguousarray(mel, dtype=np.float32))
        with torch.no_grad():
            want = model(torch.from_numpy(np.ascontiguousarray(mel, np.float32)))
        assert np.abs(logits - want.numpy()).max() < PARITY_TOLERANCE
