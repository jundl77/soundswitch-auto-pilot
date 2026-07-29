"""Tests for the ONNX export and the posterior sidecars
(``training/nn/export_onnx.py``, ``training/nn/infer.py``).

Everything downstream of Task 3 -- priors, decoder, sweeps, the final
evaluation -- reads cached ``.npz`` files and never runs a network again.  That
buys a sweep that costs seconds, and it means a defect *in these two modules* is
invisible: the decoder will happily decode posteriors that are shifted by a
frame, averaged over the wrong windows, or produced by last week's checkpoint,
and every number in the final report will be wrong in a way no assertion
downstream can catch.  So the checks here are about the three things that fail
silently.

**The graph.**  ``torch.onnx.export`` reports success even when it has baked the
time axis to the length it traced -- the pre-flight proved it for this
architecture.  The declared axes are read back out of the file, and the graph is
run at a length it never saw.

**The numbers.**  A golden inference against a saved reference pins torch, the
exporter and onnxruntime together: an upgrade that moves any output past 1e-5
fails here rather than showing up as a slightly different macro-F1 in Task 6.
The model it exercises is built from a numpy seed rather than from a checkpoint,
so it runs anywhere -- CI has no data directory.

**The aggregation.**  Window geometry is asserted against stub sessions whose
outputs encode either the global frame index (so a one-frame shift is visible)
or the position within the window (so a leaked edge frame is visible).  Real
inference cannot test this: every plausible bug still produces plausible
posteriors.

The handful of tests that need the trained checkpoint and the real corpus are
skipped when the (gitignored) data directory is absent, which is every machine
but the one that trained it.
"""
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training extra not synced")
pytest.importorskip("onnx", reason="training extra not synced")
onnxruntime = pytest.importorskip("onnxruntime", reason="training extra not synced")

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.dataset import (  # noqa: E402
    FRAME_SEC,
    LABEL_POOL,
    NUM_CLASSES,
    WINDOW_FRAMES,
    load_sidecar,
)
from nn.export_onnx import (  # noqa: E402
    BOUNDARY_OUTPUT,
    INPUT_NAME,
    LABEL_OUTPUT,
    MODEL_FILE,
    TIME_AXIS,
    build_from_checkpoint,
    declared_axes,
    default_checkpoint,
    export_model,
    load_checkpoint,
    model_dir,
    run_window,
    session,
    session_options,
    sha256_file,
)
from nn.infer import (  # noqa: E402
    EDGE_FRAMES,
    EDGE_SEC,
    HOP_FRAMES,
    HOP_SEC,
    POSTERIORS_DIR,
    TrackPosteriors,
    contribution_span,
    generate,
    infer_track,
    posterior_arrays,
    save_posteriors,
    sidecar_is_current,
    usable_frames,
    window_offsets,
)
from nn.model import SectionCRNN  # noqa: E402

MEL_BANDS = 40
GOLDEN_FILE = Path(__file__).resolve().parent / "data" / "nn_onnx_golden.npz"
GOLDEN_TOLERANCE = 1e-5
PARITY_TOLERANCE = 1e-4
# Weight gain for the seeded stand-in model -- see `seeded_model`.
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


def seeded_model(seed: int = 20260726) -> SectionCRNN:
    """The real architecture with weights drawn from a numpy seed.

    Not ``torch.manual_seed``: numpy's ``default_rng`` stream is a documented,
    versioned guarantee, so the golden reference stays meaningful across torch
    upgrades -- which is precisely the class of change the golden test exists to
    notice.

    The draws are fan-in scaled with a gain, and the normalisation layers get
    their statistics around 0 and 1, because a badly conditioned random net is
    not a weaker test -- it is a *different* one.  At unit gain a random GRU
    contracts: measured output range 0.12 logits and two completely different
    inputs land 0.001 apart, so a 1e-5 reference would pin almost nothing about
    the input path.  ``GAIN`` puts the logits on the +-3 scale a trained model
    produces, and the two-input sensitivity at 0.08 -- four orders of magnitude
    above the tolerance.  A normalisation layer is recognised by the
    ``running_mean`` next to it, so the rule survives an architecture edit that
    renumbers the ``Sequential``.
    """
    model = SectionCRNN()
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
    """A synthetic log-mel window on roughly the scale the sidecars carry.

    Structured, not white: a slow spectral sweep under a beat-rate amplitude.
    Noise alone makes the conv front-end's output near-constant over time and
    the golden reference degenerates into "did it return the same flat number",
    which would not notice a temporal bug at all.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(frames, dtype=np.float64)[:, None]
    band = np.arange(MEL_BANDS, dtype=np.float64)[None, :]
    centre = 8.0 + 20.0 * (0.5 + 0.5 * np.sin(time * 0.01))
    sweep = np.exp(-0.5 * ((band - centre) / 4.0) ** 2)
    pulse = 1.0 + 0.8 * np.sin(time * 0.6)
    mel = 2.5 * sweep * pulse + 0.3 * rng.random((frames, MEL_BANDS))
    return np.maximum(mel, 0.0).astype(np.float32)[None]


def exported(tmp_path, model=None, **kwargs):
    """A freshly exported graph and its session -- never a cached artifact."""
    path = tmp_path / MODEL_FILE
    export_model(model or seeded_model(), path, **kwargs)
    return path, session(path)


@pytest.fixture(scope="session")
def graph(tmp_path_factory):
    """One export shared by every test that is not testing the exporter.

    ``torch.onnx.export`` costs about a second, and this file would otherwise
    pay it seventeen times.  The tests that must see a cold export -- the golden
    inference above all, which the plan requires to be uncached -- call
    ``exported`` themselves and do not take this fixture.
    """
    return exported(tmp_path_factory.mktemp("graph"))


# --------------------------------------------------------------------------- #
# Export: the graph
# --------------------------------------------------------------------------- #


def test_export_declares_a_dynamic_time_axis(tmp_path):
    """The failure this guards is silent: the dynamo exporter bakes time=348
    into this architecture and still reports success."""
    path, _sess = exported(tmp_path)

    axes = declared_axes(path)

    assert axes[INPUT_NAME] == ["batch", TIME_AXIS, MEL_BANDS]
    assert axes[LABEL_OUTPUT] == ["batch", "time_pooled", NUM_CLASSES]
    assert axes[BOUNDARY_OUTPUT] == ["batch", TIME_AXIS]


@pytest.mark.parametrize("frames", [WINDOW_FRAMES, 512, 64])
def test_exported_graph_runs_at_a_length_it_was_not_traced_at(graph, frames):
    _path, sess = graph

    label, boundary = run_window(sess, seeded_mel(frames))

    assert label.shape == (1, frames // LABEL_POOL, NUM_CLASSES)
    assert boundary.shape == (1, frames)


def test_export_writes_nothing_when_the_graph_is_rejected(tmp_path, monkeypatch):
    """A half-written model.onnx would be picked up by the next run as valid."""
    import nn.export_onnx as export_onnx

    monkeypatch.setattr(export_onnx, "declared_axes", lambda _p: {"mel": [1, 348, 40]})
    with pytest.raises(RuntimeError, match="specialized"):
        export_onnx.export_model(seeded_model(), tmp_path / MODEL_FILE)

    assert list(tmp_path.iterdir()) == []


def test_session_is_pinned_to_one_thread_on_the_cpu(graph):
    """`intra_op_num_threads = 1` is the determinism contract: a threaded
    reduction sums in completion order, and float addition is not associative."""
    _path, sess = graph
    options = session_options()

    assert sess.get_providers() == ["CPUExecutionProvider"]
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == onnxruntime.ExecutionMode.ORT_SEQUENTIAL


def test_two_onnx_runs_are_bit_identical(graph):
    _path, sess = graph
    mel = seeded_mel()

    first = run_window(sess, mel)
    second = run_window(sess, mel)

    assert first[0].tobytes() == second[0].tobytes()
    assert first[1].tobytes() == second[1].tobytes()


# --------------------------------------------------------------------------- #
# Export: verify-then-build
# --------------------------------------------------------------------------- #


def test_build_from_checkpoint_uses_the_checkpoint_geometry(tmp_path):
    """The dangerous mismatch is `label_pool`: it changes no tensor shape, so a
    wrong value loads cleanly and decodes at half the frame rate."""
    donor = SectionCRNN(label_pool=4)
    checkpoint = {"model": donor.state_dict(), "arch": donor.arch()}

    model = build_from_checkpoint(checkpoint)

    assert model.arch() == donor.arch()
    assert model.label_pool == 4
    assert not model.training


def test_build_from_checkpoint_refuses_a_checkpoint_without_an_arch_block():
    with pytest.raises(RuntimeError, match="arch"):
        build_from_checkpoint({"model": SectionCRNN().state_dict()})


def test_build_from_checkpoint_refuses_weights_that_do_not_fit():
    checkpoint = {"model": SectionCRNN(rnn_hidden=64).state_dict(),
                  "arch": SectionCRNN().arch()}

    with pytest.raises(RuntimeError):
        build_from_checkpoint(checkpoint)


def test_a_published_checkpoint_loads_without_executing_it(tmp_path):
    """`load_checkpoint` runs the restricted unpickler (`weights_only=True`).

    A checkpoint is a pickle: read the unrestricted way it is arbitrary code,
    and these are read from a shared data directory rather than produced in the
    same breath.  Everything the export path touches -- the state dict, the arch
    block, config primitives -- survives the restriction, so nothing is traded
    for it.  The published `best.pt` files load under it unchanged.
    """
    donor = SectionCRNN(label_pool=4)
    path = tmp_path / "best.pt"
    torch.save({"model": donor.state_dict(), "arch": donor.arch(),
                "config": {"lr": 0.001, "run_name": "v1b", "epochs": 40},
                "epoch": 7, "metrics": {"macro_f1": 0.5}}, path)

    checkpoint = load_checkpoint(path)

    assert build_from_checkpoint(checkpoint).arch() == donor.arch()
    assert checkpoint["config"]["run_name"] == "v1b"


@needs_corpus
def test_the_shipped_checkpoint_really_loads_under_the_restriction():
    """The test above proves the SHAPE is loadable; this proves the file on disk
    is -- the one the exported graph was actually built from."""
    path = default_checkpoint(DATA_DIR)
    if not path.exists():
        pytest.skip(f"no checkpoint at {path}")

    checkpoint = load_checkpoint(path)

    assert "arch" in checkpoint
    assert build_from_checkpoint(checkpoint).arch() == checkpoint["arch"]


# --------------------------------------------------------------------------- #
# Golden inference + torch parity
# --------------------------------------------------------------------------- #


def test_golden_onnx_inference_matches_the_saved_reference(tmp_path):
    """One uncached export + inference against a committed reference.

    Nothing is memoised: the graph is exported into ``tmp_path`` on every run,
    so an exporter or onnxruntime upgrade that moves any output past 1e-5 fails
    here -- where it is one number to look at -- instead of surfacing as an
    unexplained metric change three tasks later.
    """
    _path, sess = exported(tmp_path)

    label, boundary = run_window(sess, seeded_mel())

    with np.load(GOLDEN_FILE) as reference:
        assert label.shape == tuple(reference["label_logits"].shape)
        assert np.abs(label - reference["label_logits"]).max() < GOLDEN_TOLERANCE
        assert np.abs(boundary - reference["boundary_logits"]).max() < GOLDEN_TOLERANCE

    # The reference is only worth its tolerance if the outputs move when the
    # input does; a contracted random net would pass the comparison above while
    # pinning nothing.  A different window must land orders of magnitude away.
    other_label, other_boundary = run_window(sess, seeded_mel(seed=99))
    assert np.abs(label - other_label).max() > 1000 * GOLDEN_TOLERANCE
    assert np.abs(boundary - other_boundary).max() > 1000 * GOLDEN_TOLERANCE


@pytest.mark.parametrize("frames", [WINDOW_FRAMES, 512, 64])
def test_torch_and_onnx_agree_on_synthetic_windows(graph, frames):
    model = seeded_model()
    _path, sess = graph
    mel = seeded_mel(frames, seed=frames)

    label, boundary = run_window(sess, mel)
    with torch.no_grad():
        want_label, want_boundary = model(torch.from_numpy(mel))

    assert np.abs(label - want_label.numpy()).max() < PARITY_TOLERANCE
    assert np.abs(boundary - want_boundary.numpy()).max() < PARITY_TOLERANCE


@needs_corpus
def test_torch_and_onnx_agree_on_three_real_windows(tmp_path):
    """The parity the plan actually asks for: the trained checkpoint, real mel,
    and one window at a length the graph was never traced at."""
    checkpoint = default_checkpoint(DATA_DIR)
    if not checkpoint.exists():
        pytest.skip(f"no checkpoint at {checkpoint}")
    model = build_from_checkpoint(load_checkpoint(checkpoint))
    _path, sess = exported(tmp_path, model)

    sidecars = sorted((DATA_DIR / "features").glob("*.npz"))[:2]
    mel_a = load_sidecar(sidecars[0])
    mel_b = load_sidecar(sidecars[1])
    windows = [
        mel_a[:WINDOW_FRAMES][None],
        mel_b[5000:5000 + WINDOW_FRAMES][None],
        mel_a[1000:1000 + 512][None],          # a length never exported
    ]

    for mel in windows:
        label, boundary = run_window(sess, mel)
        with torch.no_grad():
            want_label, want_boundary = model(torch.from_numpy(np.ascontiguousarray(mel)))
        assert np.abs(label - want_label.numpy()).max() < PARITY_TOLERANCE
        assert np.abs(boundary - want_boundary.numpy()).max() < PARITY_TOLERANCE


# --------------------------------------------------------------------------- #
# Window geometry
# --------------------------------------------------------------------------- #


def test_hop_is_the_runtime_cadence_snapped_to_the_pooled_grid():
    """An odd hop would put the label head's output between two cells of the
    track-wide pooled grid, and the shift would be silent."""
    assert HOP_FRAMES % LABEL_POOL == 0
    assert abs(HOP_FRAMES * FRAME_SEC - HOP_SEC) <= FRAME_SEC


def test_edge_margin_is_at_least_the_full_second_the_spec_asks_for():
    assert EDGE_FRAMES % LABEL_POOL == 0
    assert EDGE_FRAMES * FRAME_SEC >= EDGE_SEC
    assert 2 * EDGE_FRAMES + HOP_FRAMES <= WINDOW_FRAMES


def test_window_offsets_end_exactly_at_the_end_of_the_track():
    """The tail is covered by a window clamped back onto its predecessor, which
    is why aggregation has to average per frame rather than concatenate."""
    frames = 1000

    offsets = window_offsets(frames, window_frames=100, hop_frames=40)

    assert offsets[0] == 0
    assert offsets[-1] == frames - 100          # 900, not the hop's 880
    assert offsets[-1] - offsets[-2] < 40       # the deliberate re-overlap


def test_window_offsets_does_not_duplicate_an_exact_landing():
    offsets = window_offsets(400, window_frames=100, hop_frames=30)

    assert offsets[-1] == 300
    assert len(offsets) == len(set(offsets))


def test_window_offsets_of_a_track_shorter_than_one_window_is_a_single_window():
    assert window_offsets(50, window_frames=100, hop_frames=30) == [0]


def test_contribution_spans_cover_every_frame_exactly_once_or_more():
    frames, window, hop, edge = 1000, 100, 30, 10
    offsets = window_offsets(frames, window_frames=window, hop_frames=hop)

    covered = np.zeros(frames, dtype=int)
    for index, offset in enumerate(offsets):
        lo, hi = contribution_span(offset, n_frames=frames, first=index == 0,
                                   last=index == len(offsets) - 1,
                                   window_frames=window, edge_frames=edge)
        covered[lo:hi] += 1

    assert covered.min() >= 1, "the hop/edge geometry leaves frames unvoted on"


def test_an_interior_window_never_votes_on_its_own_edges():
    lo, hi = contribution_span(300, n_frames=1000, first=False, last=False,
                               window_frames=100, edge_frames=10)

    assert (lo, hi) == (310, 390)


def test_the_first_and_last_window_donate_the_margin_nothing_else_can_reach():
    first = contribution_span(0, n_frames=1000, first=True, last=False,
                              window_frames=100, edge_frames=10)
    last = contribution_span(900, n_frames=1000, first=False, last=True,
                             window_frames=100, edge_frames=10)

    assert first == (0, 90)
    assert last == (910, 1000)


def test_usable_frames_truncates_to_whole_pooled_groups():
    assert usable_frames(LABEL_POOL * 50 + 1) == LABEL_POOL * 50
    assert usable_frames(LABEL_POOL * 50) == LABEL_POOL * 50


# --------------------------------------------------------------------------- #
# Aggregation, against stub sessions
# --------------------------------------------------------------------------- #


class _StubSession:
    """A session whose outputs are a function the aggregation must reproduce.

    ``run`` receives the mel window the caller sliced, so encoding the global
    frame index into mel band 0 lets the stub answer in *track* coordinates --
    which is the only way to see a one-frame aggregation shift.
    """

    def __init__(self, label_fn, boundary_fn):
        self.label_fn = label_fn
        self.boundary_fn = boundary_fn
        self.calls = 0

    def run(self, _names, feeds):
        mel = feeds["mel"]
        index = mel[0, :, 0].astype(np.float64)     # global frame index, or -1 when padded
        position = np.arange(mel.shape[1], dtype=np.float64)
        self.calls += 1
        boundary = self.boundary_fn(index, position)[None]
        label = self.label_fn(index, position)[None]
        return [label.astype(np.float32), boundary.astype(np.float32)]


def _indexed_mel(frames: int) -> np.ndarray:
    mel = np.zeros((frames, MEL_BANDS), dtype=np.float32)
    mel[:, 0] = np.arange(frames, dtype=np.float32)
    return mel


def _pooled(values: np.ndarray) -> np.ndarray:
    """Frame-rate values -> one column per class at the pooled rate."""
    grouped = values.reshape(-1, LABEL_POOL).mean(axis=1)
    logits = np.zeros((len(grouped), NUM_CLASSES), dtype=np.float64)
    logits[:, 0] = grouped
    return logits


def test_aggregation_places_every_frame_at_its_own_track_position():
    """The strongest available statement about alignment: when the model's
    answer depends only on *which* frame it is, averaging over 100 overlapping
    windows must return that same answer, frame for frame."""
    frames = 3 * WINDOW_FRAMES
    truth = np.sin(np.arange(frames) * 0.01)
    stub = _StubSession(lambda index, _p: _pooled(np.sin(index * 0.01)),
                        lambda index, _p: np.sin(index * 0.01))

    result = infer_track(stub, _indexed_mel(frames))

    want = 1.0 / (1.0 + np.exp(-truth))
    assert np.abs(result.boundary - want).max() < 1e-6
    assert result.windows == len(window_offsets(frames))


def test_aggregation_never_reads_a_window_edge():
    """A stub that screams only at its own outer margin: if any interior frame
    of the track picks that up, the never-read-the-edge rule is not enforced."""
    frames = 3 * WINDOW_FRAMES
    def edge_only(_index, position):
        loud = ((position < EDGE_FRAMES)
                | (position >= WINDOW_FRAMES - EDGE_FRAMES)).astype(np.float64)
        return loud * 40.0 - 20.0
    stub = _StubSession(lambda i, p: _pooled(edge_only(i, p)), edge_only)

    result = infer_track(stub, _indexed_mel(frames))

    interior = result.boundary[EDGE_FRAMES:frames - EDGE_FRAMES]
    assert interior.max() < 1e-6, "a window edge leaked into the aggregate"
    # The two ends are the documented exception: no window's interior reaches
    # them, so the first and last window donate their margin rather than leave
    # the decoder an undefined posterior.
    assert result.boundary[:EDGE_FRAMES].min() > 0.99
    assert result.boundary[frames - EDGE_FRAMES:].min() > 0.99


def test_aggregation_averages_the_reoverlapped_tail_instead_of_concatenating():
    """A track length is not a whole number of hops, so the final window is
    clamped back and re-covers frames its predecessor already voted on."""
    frames = WINDOW_FRAMES + 3 * HOP_FRAMES + LABEL_POOL     # not hop-aligned
    stub = _StubSession(lambda index, _p: _pooled(index * 0.0),
                        lambda index, _p: np.full_like(index, 2.0))

    result = infer_track(stub, _indexed_mel(frames))

    assert len(result.boundary) == frames
    assert result.coverage.sum() > frames          # overlap really happened
    # Every frame is a mean of identical values, so the mean is that value --
    # a concatenation would have produced a longer array, a sum a larger one.
    assert np.abs(result.boundary - 1.0 / (1.0 + math.exp(-2.0))).max() < 1e-6


def test_coverage_counts_the_windows_that_voted_on_each_frame():
    frames = 3 * WINDOW_FRAMES
    stub = _StubSession(lambda index, _p: _pooled(index * 0.0),
                        lambda index, _p: np.zeros_like(index))

    result = infer_track(stub, _indexed_mel(frames))

    interior = (WINDOW_FRAMES - 2 * EDGE_FRAMES) // HOP_FRAMES
    assert result.coverage.max() == interior
    assert result.coverage[0] == 1                 # only the first window reaches it
    assert result.coverage.dtype == np.uint16


def test_label_posteriors_are_probabilities_on_the_pooled_grid():
    frames = 2 * WINDOW_FRAMES
    rng = np.random.default_rng(3)
    stub = _StubSession(
        lambda index, _p: rng.normal(size=(len(index) // LABEL_POOL, NUM_CLASSES)),
        lambda index, _p: np.zeros_like(index))

    result = infer_track(stub, _indexed_mel(frames))

    assert result.label_post.shape == (frames // LABEL_POOL, NUM_CLASSES)
    assert result.label_post.dtype == np.float32
    assert np.abs(result.label_post.sum(axis=1) - 1.0).max() < 1e-6
    assert (result.label_post >= 0.0).all()


def test_a_sidecar_records_the_geometry_it_was_actually_run_with():
    """`infer_track` takes the geometry as arguments, so writing the module
    constants into the file would make a non-default run describe itself as a
    default one -- and `sidecar_is_current` would then accept it, handing the
    decoder a hop it is not expecting with nothing to notice."""
    frames = 3 * WINDOW_FRAMES
    hop, edge = 4 * HOP_FRAMES, 2 * EDGE_FRAMES
    stub = _StubSession(lambda index, _p: _pooled(index * 0.0),
                        lambda index, _p: np.zeros_like(index))

    result = infer_track(stub, _indexed_mel(frames), hop_frames=hop, edge_frames=edge)
    arrays = posterior_arrays(result, "0" * 64)

    assert (result.hop_frames, result.edge_frames) == (hop, edge)
    assert int(arrays["hop_frames"]) == hop
    assert int(arrays["edge_frames"]) == edge
    assert int(arrays["window_frames"]) == WINDOW_FRAMES
    assert result.windows == len(window_offsets(frames, hop_frames=hop))


def test_a_sidecar_from_a_non_default_geometry_is_not_treated_as_current(tmp_path):
    path = tmp_path / "a.npz"
    frames = 3 * WINDOW_FRAMES
    stub = _StubSession(lambda index, _p: _pooled(index * 0.0),
                        lambda index, _p: np.zeros_like(index))
    result = infer_track(stub, _indexed_mel(frames), hop_frames=4 * HOP_FRAMES)

    save_posteriors(path, posterior_arrays(result, "0" * 64))

    assert not sidecar_is_current(path, "0" * 64)


@pytest.mark.parametrize("kwargs", [
    {"hop_frames": HOP_FRAMES + 1},
    {"edge_frames": EDGE_FRAMES + 1},
    {"window_frames": WINDOW_FRAMES + 1},
])
def test_infer_track_refuses_a_geometry_that_breaks_pool_alignment(kwargs):
    """An unaligned window or hop shifts every label posterior by half a pooled
    frame, and the slice arithmetic would keep working."""
    stub = _StubSession(lambda index, _p: _pooled(index * 0.0),
                        lambda index, _p: np.zeros_like(index))

    with pytest.raises(ValueError, match="multiple of the label pooling factor"):
        infer_track(stub, _indexed_mel(2 * WINDOW_FRAMES), **kwargs)


def test_infer_track_refuses_a_hop_wider_than_the_window_interior():
    """Frames no window votes on would divide by a zero coverage count."""
    stub = _StubSession(lambda index, _p: _pooled(index * 0.0),
                        lambda index, _p: np.zeros_like(index))

    with pytest.raises(ValueError, match="usable window interior"):
        infer_track(stub, _indexed_mel(2 * WINDOW_FRAMES),
                    hop_frames=WINDOW_FRAMES - 2 * EDGE_FRAMES + LABEL_POOL)


def test_a_track_shorter_than_one_window_is_padded_not_dropped():
    frames = WINDOW_FRAMES // 2
    stub = _StubSession(lambda index, _p: _pooled(index * 0.0),
                        lambda index, _p: np.full_like(index, 1.0))

    result = infer_track(stub, _indexed_mel(frames))

    assert result.windows == 1
    assert len(result.boundary) == frames
    assert result.coverage.min() == 1


def test_an_odd_frame_count_is_truncated_to_the_pooled_grid():
    frames = 2 * WINDOW_FRAMES + 1
    stub = _StubSession(lambda index, _p: _pooled(index * 0.0),
                        lambda index, _p: np.zeros_like(index))

    result = infer_track(stub, _indexed_mel(frames))

    assert result.n_frames == frames - 1
    assert len(result.boundary) * 1.0 / LABEL_POOL == len(result.label_post)


# --------------------------------------------------------------------------- #
# Sidecar bytes
# --------------------------------------------------------------------------- #


def _arrays(seed=0, **geometry):
    rng = np.random.default_rng(seed)
    label = rng.random((10, NUM_CLASSES)).astype(np.float32)
    shape = {"window_frames": WINDOW_FRAMES, "hop_frames": HOP_FRAMES,
             "edge_frames": EDGE_FRAMES, **geometry}
    track = TrackPosteriors(label / label.sum(1, keepdims=True),
                            rng.random(20).astype(np.float32),
                            np.full(20, 3, dtype=np.uint16), 20, 7, **shape)
    return posterior_arrays(track, "0" * 64)


def test_sidecar_bytes_are_a_pure_function_of_their_contents(tmp_path):
    """Two identical runs must produce the same file, and half of that is the
    zip container rather than the numbers.

    ``np.savez`` happens to satisfy this on CPython 3.11 -- ``ZipFile.open``
    builds a ``ZipInfo`` whose ``date_time`` defaults to the 1980 epoch -- but
    that is an implementation default rather than an API promise, and
    ``savez_compressed`` additionally folds the zlib build into the bytes.  The
    writer under test owns the order, the epoch and the compression method
    itself so the guarantee is about the pipeline and not about this machine.
    """
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"

    save_posteriors(first, _arrays())
    save_posteriors(second, _arrays())

    assert first.read_bytes() == second.read_bytes()


def test_sidecar_members_carry_a_fixed_timestamp(tmp_path):
    path = tmp_path / "a.npz"
    save_posteriors(path, _arrays())

    with zipfile.ZipFile(path) as archive:
        stamps = {info.date_time for info in archive.infolist()}

    assert stamps == {(1980, 1, 1, 0, 0, 0)}


def test_sidecar_members_are_stored_uncompressed(tmp_path):
    """Deflate would put the zlib build inside a guarantee that is supposed to
    depend only on the numbers -- and ``np.savez_compressed``, the obvious
    alternative, does exactly that."""
    path = tmp_path / "a.npz"
    save_posteriors(path, _arrays())

    with zipfile.ZipFile(path) as archive:
        methods = {info.compress_type for info in archive.infolist()}

    assert methods == {zipfile.ZIP_STORED}


def test_sidecar_members_are_written_in_a_fixed_order(tmp_path):
    """Member order is part of the bytes; leaving it to dict iteration makes
    the file a function of how the caller happened to build its mapping."""
    path = tmp_path / "a.npz"
    arrays = _arrays()
    save_posteriors(path, arrays)

    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist()]

    assert names == [f"{name}.npy" for name in arrays]


def test_sidecar_round_trips_through_np_load(tmp_path):
    path = tmp_path / "a.npz"
    arrays = _arrays()
    save_posteriors(path, arrays)

    with np.load(path) as loaded:
        assert set(loaded.files) == set(arrays)
        assert np.array_equal(loaded["label_post"], arrays["label_post"])
        assert str(loaded["model_sha"]) == "0" * 64
        assert float(loaded["frame_sec"]) == pytest.approx(FRAME_SEC)
        assert float(loaded["t0"]) == pytest.approx(FRAME_SEC)
        assert int(loaded["hop_frames"]) == HOP_FRAMES


def test_the_two_time_bases_are_both_recorded():
    """`boundary` runs at frame rate from `t0`; `label_post` runs at the pooled
    rate and is stamped at the END of each group, as the targets were pooled.
    A decoder that assumed one origin would read every label a frame early."""
    arrays = _arrays()

    assert float(arrays["label_frame_sec"]) == pytest.approx(FRAME_SEC * LABEL_POOL)
    assert float(arrays["label_t0"]) == pytest.approx(
        float(arrays["t0"]) + (LABEL_POOL - 1) * FRAME_SEC)


def test_sidecar_is_current_only_for_this_model_and_this_geometry(tmp_path):
    path = tmp_path / "a.npz"
    save_posteriors(path, _arrays())

    assert sidecar_is_current(path, "0" * 64)
    assert not sidecar_is_current(path, "1" * 64)
    assert not sidecar_is_current(tmp_path / "missing.npz", "0" * 64)


def test_sidecar_is_current_rejects_a_changed_hop(tmp_path):
    path = tmp_path / "a.npz"
    save_posteriors(path, _arrays(hop_frames=HOP_FRAMES + LABEL_POOL))

    assert not sidecar_is_current(path, "0" * 64)


def test_sidecar_is_current_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "a.npz"
    save_posteriors(path, _arrays())
    path.write_bytes(path.read_bytes()[:64])

    assert not sidecar_is_current(path, "0" * 64)


# --------------------------------------------------------------------------- #
# The corpus run
# --------------------------------------------------------------------------- #


def _mini_corpus(tmp_path, graph, ids=("aaa", "bbb"),
                 frames=WINDOW_FRAMES + 10 * HOP_FRAMES):
    """A features/ directory and a graph -- the two inputs of a corpus run.

    Just long enough to need eleven windows: these tests are about the run's
    bookkeeping (cache key, failure handling, worker independence), and the
    aggregation itself is pinned exactly by the stub-session tests above.
    """
    from build_training_table import write_feature_sidecar

    data_dir = tmp_path / "corpus"
    rng = np.random.default_rng(7)
    for youtube_id in ids:
        mel = (rng.random((frames, MEL_BANDS)) * 3.0).astype(np.float32)
        write_feature_sidecar(data_dir / "features" / f"{youtube_id}.npz",
                              mel, FRAME_SEC, FRAME_SEC)
    model_path, _sess = graph
    return data_dir, model_path, list(ids)


def test_generate_writes_one_sidecar_per_track_and_a_manifest(tmp_path, graph):
    data_dir, model_path, ids = _mini_corpus(tmp_path, graph)

    manifest = generate(data_dir, model_path=model_path, ids=ids,
                        workers=1, progress_every=0)

    out = data_dir / POSTERIORS_DIR
    assert sorted(p.name for p in out.glob("*.npz")) == [f"{i}.npz" for i in ids]
    assert manifest["computed"] == len(ids) and manifest["failed"] == 0
    assert manifest["model_sha"] == sha256_file(model_path)
    assert manifest["hop_frames"] == HOP_FRAMES
    assert (out / "manifest.json").exists()


def test_generated_bytes_do_not_depend_on_how_many_workers_ran(tmp_path, graph):
    """The determinism claim in one assertion: parallelism is per track, so a
    12-thread run and a 1-thread run must produce identical files."""
    data_dir, model_path, ids = _mini_corpus(tmp_path, graph)

    generate(data_dir, model_path=model_path, ids=ids, out_dir=tmp_path / "serial",
             workers=1, progress_every=0)
    generate(data_dir, model_path=model_path, ids=ids, out_dir=tmp_path / "parallel",
             workers=4, progress_every=0)

    for youtube_id in ids:
        assert ((tmp_path / "serial" / f"{youtube_id}.npz").read_bytes()
                == (tmp_path / "parallel" / f"{youtube_id}.npz").read_bytes())


def test_generate_reuses_a_sidecar_this_model_already_produced(tmp_path, graph):
    data_dir, model_path, ids = _mini_corpus(tmp_path, graph)
    generate(data_dir, model_path=model_path, ids=ids, workers=1, progress_every=0)

    again = generate(data_dir, model_path=model_path, ids=ids,
                     workers=1, progress_every=0)

    assert again["cached"] == len(ids) and again["computed"] == 0


def test_generate_recomputes_when_the_model_changes(tmp_path, graph):
    data_dir, model_path, ids = _mini_corpus(tmp_path, graph)
    generate(data_dir, model_path=model_path, ids=ids, workers=1, progress_every=0)
    other = tmp_path / "other" / MODEL_FILE
    export_model(seeded_model(seed=11), other)

    manifest = generate(data_dir, model_path=other, ids=ids,
                        workers=1, progress_every=0)

    assert manifest["computed"] == len(ids) and manifest["cached"] == 0


def test_generate_records_a_broken_track_and_finishes_the_rest(tmp_path, graph):
    """A corpus run is hours long; one unreadable sidecar must not discard
    every other track's work -- but it must not vanish either."""
    data_dir, model_path, ids = _mini_corpus(tmp_path, graph, ids=("aaa", "bbb", "ccc"))
    (data_dir / "features" / "bbb.npz").write_bytes(b"not an npz")

    manifest = generate(data_dir, model_path=model_path, ids=ids,
                        workers=2, progress_every=0)

    assert manifest["failed"] == 1
    assert manifest["computed"] == 2
    broken = [r for r in manifest["records"] if r["youtube_id"] == "bbb"]
    assert broken and "error" in broken[0]
    assert not (data_dir / POSTERIORS_DIR / "bbb.npz").exists()


def test_a_failed_track_does_not_leave_an_older_models_sidecar_readable(tmp_path, graph):
    """The manifest says "failed" but `np.load` still succeeds -- so a reader
    would consume last week's posteriors believing they are this model's."""
    data_dir, model_path, ids = _mini_corpus(tmp_path, graph)
    stale = data_dir / POSTERIORS_DIR / "aaa.npz"
    save_posteriors(stale, _arrays())                  # a different model_sha
    (data_dir / "features" / "aaa.npz").write_bytes(b"not an npz")

    manifest = generate(data_dir, model_path=model_path, ids=ids,
                        workers=1, progress_every=0)

    assert manifest["failed"] == 1
    assert not stale.exists()
    assert [r for r in manifest["records"] if r["youtube_id"] == "aaa"][0]["removed_stale"]


def test_a_forced_recompute_that_fails_keeps_the_current_sidecar(tmp_path, graph):
    """The mirror image: under --force the file on disk is already this model's
    answer, so a transient read error must not destroy a good artifact."""
    data_dir, model_path, ids = _mini_corpus(tmp_path, graph)
    generate(data_dir, model_path=model_path, ids=ids, workers=1, progress_every=0)
    good = data_dir / POSTERIORS_DIR / "aaa.npz"
    before = good.read_bytes()
    (data_dir / "features" / "aaa.npz").write_bytes(b"not an npz")

    manifest = generate(data_dir, model_path=model_path, ids=ids, workers=1,
                        force=True, progress_every=0)

    assert manifest["failed"] == 1
    assert good.read_bytes() == before


# --------------------------------------------------------------------------- #
# The real artifacts, when this machine has them
# --------------------------------------------------------------------------- #


@needs_corpus
def test_the_exported_graph_matches_its_recorded_metadata():
    path = model_dir(DATA_DIR) / MODEL_FILE
    meta_path = path.with_name(path.name + ".json")
    if not meta_path.exists():
        pytest.skip("model.onnx has not been exported on this machine")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    assert meta["model_sha"] == sha256_file(path)
    assert meta["dynamo"] is False
    assert TIME_AXIS in meta["declared_axes"][INPUT_NAME]
