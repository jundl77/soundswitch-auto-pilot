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


def seeded_model(seed: int = 20260727) -> DownbeatCRNN:
    model = DownbeatCRNN()
    state = model.state_dict()
    normalised = {name.rsplit(".", 1)[0] for name in state if name.endswith("running_mean")}
    moments = {"weight": (1.0, 0.25), "bias": (0.0, 0.25),
               "running_mean": (0.0, 0.25), "running_var": (1.0, 0.5)}

    # numpy's default_rng stream is a versioned guarantee, so the golden
    # reference survives exactly the torch upgrade it exists to notice.
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
                fan_in = tensor[0].numel()
                draw = GAIN * draw / math.sqrt(fan_in)
            else:
                draw = 0.05 * draw
            tensor.copy_(torch.from_numpy(draw.reshape(tensor.shape)).float())
    return model.eval()


def seeded_mel(frames: int = WINDOW_FRAMES, seed: int = 4242) -> np.ndarray:
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
    return exported(tmp_path_factory.mktemp("downbeat_graph"))


def _logit(value):
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(value / (1.0 - value))


class IndexSession:
    def run(self, _outputs, feed):
        return [_logit(feed[INPUT_NAME][:, :, 0])]


class PositionSession:
    def __init__(self, window_frames=WINDOW_FRAMES, edge_frames=EDGE_FRAMES):
        self.window_frames = window_frames
        self.edge_frames = edge_frames

    def run(self, _outputs, feed):
        position = np.arange(feed[INPUT_NAME].shape[1])
        edge = (position < self.edge_frames) | (position >= self.window_frames - self.edge_frames)
        return [_logit(np.where(edge, 0.99, 0.01))[None].repeat(feed[INPUT_NAME].shape[0], 0)]


def index_mel(n_frames: int) -> np.ndarray:
    return np.repeat((np.arange(n_frames) / n_frames)[:, None],
                     MEL_BANDS, axis=1).astype(np.float32)


def test_export_declares_a_dynamic_time_axis(tmp_path):
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
    _path, sess = exported(tmp_path)

    logits = run_window(sess, seeded_mel())

    with np.load(GOLDEN_FILE) as reference:
        assert logits.shape == tuple(reference["downbeat_logits"].shape)
        assert np.abs(logits - reference["downbeat_logits"]).max() < GOLDEN_TOLERANCE

    logits_for_a_different_input = run_window(sess, seeded_mel(seed=99))
    assert np.abs(logits - logits_for_a_different_input).max() > 1000 * GOLDEN_TOLERANCE


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
    from nn.export_onnx import session as section_session

    from nn.downbeat_infer import session as downbeat_session

    assert downbeat_session is section_session


def make_checkpoint(tmp_path, **overrides):
    model = DownbeatCRNN()
    payload = {"model": model.state_dict(), "arch": model.arch(), "config": {},
               "epoch": 20, "f1": 0.5512, "metrics": {"f1": 0.5512},
               "pos_weight": 9.355}
    payload.update(overrides)
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path


def test_the_downbeat_checkpoint_payload_is_flat_where_the_section_head_nests(tmp_path):
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


def test_build_from_checkpoint_accepts_an_arch_whose_tuples_became_json_lists(tmp_path):
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


def test_every_frame_of_a_track_is_voted_on(graph):
    _path, sess = graph

    track = infer_track(sess, seeded_mel(1200)[0])

    assert track.n_frames == 1200
    assert np.all(track.coverage >= 1)
    assert track.activation.shape == (1200,)


def test_the_stitched_activation_is_the_identity_on_an_index_mel():
    n_frames = 1000
    track = infer_track(IndexSession(), index_mel(n_frames))

    assert np.abs(track.activation - np.arange(n_frames) / n_frames).max() < 1e-6


def test_no_window_edge_reaches_a_frame_another_window_could_have_voted_on():
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


def test_non_default_geometry_travels_with_the_result_rather_than_the_module_defaults(graph):
    _path, sess = graph

    track = infer_track(sess, seeded_mel(800)[0], hop_frames=8, edge_frames=10)

    assert (track.hop_frames, track.edge_frames) == (8, 10)


def test_a_hop_wider_than_the_window_interior_is_refused(graph):
    _path, sess = graph

    with pytest.raises(ValueError, match="uncovered"):
        infer_track(sess, seeded_mel(800)[0], hop_frames=WINDOW_FRAMES)


def test_lag_profile_finds_an_injected_shift():
    activation = np.zeros(500, dtype=np.float64)
    marks = np.arange(20, 480, 16)
    activation[marks + 2] = 1.0

    profile = lag_profile(activation, marks, radius=4)

    assert int(np.argmax(profile)) - 4 == 2


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


def test_the_sidecar_carries_no_bar_phase_so_no_truth_rides_into_a_test_split_artifact():
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


def test_live_beats_are_read_from_the_cached_sim_report_not_re_derived(tmp_path):
    import gzip

    reports = tmp_path / "reports"
    reports.mkdir()
    payload = {"report": {"beats": [{"t": 1.5, "bpm": 128.0}, {"t": 2.0}]}}
    with gzip.open(reports / "abc.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    assert live_beat_times(tmp_path, "abc").tolist() == [1.5, 2.0]


def _val_only_ids(monkeypatch):
    import nn.downbeat_infer as module

    monkeypatch.setattr(module, "split_ids",
                        lambda _dir, splits=("val", "test"): sorted(
                            {"v1", "v2"} if list(splits) == ["val"]
                            else {"v1", "v2", "t1"}))


@pytest.mark.parametrize("argv", [
    ["--lag-profile", "--splits", "val", "test"],
    ["--lag-profile", "--ids", "t1"],
])
def test_the_lag_profile_refuses_the_test_split_by_membership_not_by_flag(monkeypatch, argv):
    from nn.downbeat_infer import main

    _val_only_ids(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code != 0


def test_the_lag_profile_defaults_to_val_alone(monkeypatch, capsys):
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


def test_only_the_tuning_read_is_quarantined_so_generation_covers_val_and_test(monkeypatch):
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
