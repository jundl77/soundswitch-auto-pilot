"""Tests for the downbeat head and its training objective
(``training/nn/downbeat_model.py``, ``training/nn/downbeat_train.py``).
CPU only -- nothing here needs the 3070.

Three families, all chosen because they fail *silently* on a GPU run:

**Shapes and capacity.**  The head speaks at the full frame rate, the targets
arrive at the full frame rate, and the spec caps the model at 300 K parameters.
A head that emitted the pooled rate would still train -- broadcasting is happy
to oblige -- on targets shifted against the audio.

**The metric.**  Peak F1@+-70 ms is the number this task is judged on and the
number early stopping reads, so every part of it is checked against a
hand-computed answer: the picker's suppression, the matcher's greedy pass, the
threshold sweep's monotonic bookkeeping, and the stitching that turns overlapped
windows back into one curve.  A metric that is quietly optimistic is worse than
no metric.

**Calibration.**  Two ECE numbers are reported and they mean different things;
if the de-weighting transform is wrong, the run looks better calibrated than it
is and nobody downstream can tell.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training extra not synced")

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.dataset import FRAME_SEC, WINDOW_FRAMES  # noqa: E402
from nn.downbeat_dataset import DownbeatTargets, DownbeatWindowDataset  # noqa: E402
from nn.downbeat_model import PARAM_BUDGET, DownbeatCRNN  # noqa: E402
from nn.downbeat_train import (  # noqa: E402
    MIN_PEAK_DISTANCE_SEC,
    POSITIVE_THRESHOLD,
    TOLERANCE_SEC,
    accumulate_downbeat_stats,
    binary_ece,
    deweighted,
    evaluate,
    frame_times,
    match_events,
    peak_candidates,
    prf,
    stitch,
    sweep_peak_f1,
)
from nn.model import SectionCRNN, count_parameters, freq_pool_blocks  # noqa: E402
from nn.train import boundary_pos_weight, build_loader  # noqa: E402

from tests.test_nn_downbeat_dataset import with_grids  # noqa: E402

MEL_BANDS = 40


def _model(**kwargs) -> DownbeatCRNN:
    torch.manual_seed(0)
    return DownbeatCRNN(**kwargs).eval()


# --------------------------------------------------------------------------- #
# Model shape / capacity
# --------------------------------------------------------------------------- #


def test_forward_returns_one_logit_per_mel_frame():
    """The targets are at frame rate; a pooled head would train on shifted truth."""
    logits = _model()(torch.zeros(3, WINDOW_FRAMES, MEL_BANDS))

    assert logits.shape == (3, WINDOW_FRAMES)


@pytest.mark.parametrize("frames", [WINDOW_FRAMES, 512, 64])
def test_time_axis_is_not_baked_in(frames):
    """Task 3 pushes whole tracks through one graph, so any length must work."""
    assert _model()(torch.zeros(1, frames, MEL_BANDS)).shape == (1, frames)


def test_parameter_count_is_within_the_specs_budget():
    count = count_parameters(_model())

    assert count <= PARAM_BUDGET, f"{count} params exceeds the {PARAM_BUDGET} budget"
    # A floor too: an edit that dropped the GRU or the 1D conv would still pass
    # every shape test above.
    assert count > 150_000


def test_the_downbeat_head_is_smaller_than_the_section_head():
    """The spec's whole argument for a second model is that it is cheap."""
    assert count_parameters(_model()) < count_parameters(SectionCRNN()) / 1.5


def test_frequency_is_pooled_away_and_time_is_preserved():
    model = _model()

    assert model.freq_out == MEL_BANDS // 8
    assert model.feature_dim == 64 * (MEL_BANDS // 8)


def test_the_conv_front_end_is_the_section_models_own():
    """Shared, not copied: 'pool frequency, never time' has one definition."""
    blocks, freq = freq_pool_blocks(MEL_BANDS, (32, 64, 64))

    assert freq == MEL_BANDS // 8
    assert [type(layer).__name__ for layer in blocks[:4]] == [
        "Conv2d", "BatchNorm2d", "GELU", "MaxPool2d"]
    # Pooling is (time, freq) = (1, 2): the time axis must survive untouched.
    assert blocks[3].kernel_size == (1, 2)


# Frozen on 2026-07-27, when `freq_pool_blocks` was extracted from `SectionCRNN`
# for this head to share.  Every trained v1/v2 checkpoint on disk was written
# against exactly this list.
SECTION_STATE_DICT_KEYS = [
    "conv.0.weight",
    "conv.1.weight", "conv.1.bias", "conv.1.running_mean", "conv.1.running_var",
    "conv.4.weight",
    "conv.5.weight", "conv.5.bias", "conv.5.running_mean", "conv.5.running_var",
    "conv.8.weight",
    "conv.9.weight", "conv.9.bias", "conv.9.running_mean", "conv.9.running_var",
    "temporal.0.weight",
    "temporal.1.weight", "temporal.1.bias",
    "temporal.1.running_mean", "temporal.1.running_var",
    "rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0",
    "rnn.weight_ih_l0_reverse", "rnn.weight_hh_l0_reverse",
    "rnn.bias_ih_l0_reverse", "rnn.bias_hh_l0_reverse",
    "label_head.weight", "label_head.bias",
    "boundary_head.weight", "boundary_head.bias",
]


def test_section_state_dict_keys_are_frozen():
    """`freq_pool_blocks` was lifted out of `SectionCRNN` for this head to share.
    That refactor must not rename or reorder a single parameter, because every
    trained checkpoint on disk is a dict keyed by these strings and a rename
    surfaces as a `load_state_dict` failure in whichever task next tries to
    export or infer -- not here, where the cause is visible.

    This pin exists because the obvious candidate does not cover it: the ONNX
    golden test builds a *fresh* seeded `SectionCRNN` and never calls
    `load_state_dict`, and the one test that does load a real checkpoint
    (`test_torch_and_onnx_agree_on_three_real_windows`) is `@needs_corpus` and so
    skips on any machine without the gitignored corpus. This one runs everywhere,
    in milliseconds, and catches a rename anywhere in the module.
    """
    keys = [key for key in SectionCRNN().state_dict()
            if not key.endswith("num_batches_tracked")]

    assert keys == SECTION_STATE_DICT_KEYS


def test_the_two_heads_share_every_key_but_their_own_head():
    """The concrete statement of 'shared front end': the encoders are key-identical
    and only the heads differ. A drift here means the two models stopped sharing."""
    section = [k for k in SectionCRNN().state_dict() if "_head." not in k]
    downbeat = [k for k in DownbeatCRNN().state_dict() if "_head." not in k]

    assert section == downbeat


def test_mel_band_count_must_survive_the_frequency_pooling():
    with pytest.raises(ValueError, match="frequency"):
        DownbeatCRNN(n_mels=12)


def test_forward_rejects_a_mis_shaped_batch():
    with pytest.raises(ValueError, match="mel"):
        _model()(torch.zeros(2, 1, WINDOW_FRAMES, MEL_BANDS))


def test_arch_names_every_shape_deciding_argument():
    arch = _model().arch()

    assert arch == {"n_mels": 40, "conv_channels": [32, 64, 64],
                    "conv1d_channels": 64, "rnn_hidden": 96, "rnn_layers": 1}


@pytest.mark.parametrize("changed", [
    {"rnn_hidden": 64},
    {"rnn_layers": 2},
    {"conv1d_channels": 96},
    {"conv_channels": (32, 64, 32)},
])
def test_arch_distinguishes_a_checkpoint_that_must_not_be_loaded(changed):
    assert DownbeatCRNN(**changed).arch() != DownbeatCRNN().arch()


def test_eval_forward_is_deterministic():
    model = _model()
    mel = torch.randn(2, WINDOW_FRAMES, MEL_BANDS,
                      generator=torch.Generator().manual_seed(7))

    assert torch.equal(model(mel), model(mel))


def test_the_head_reaches_every_parameter():
    """A head wired to the wrong tensor still trains -- just not end to end."""
    model = DownbeatCRNN().train()
    model(torch.randn(2, WINDOW_FRAMES, MEL_BANDS)).square().mean().backward()

    missing = [name for name, param in model.named_parameters()
               if param.grad is None or not torch.isfinite(param.grad).all()]
    assert missing == []


def test_bidirectional_context_actually_reaches_backwards():
    """The premise of the design: a frame's activation depends on its future.

    A downbeat is not locally distinctive in four-on-the-floor; only the
    bar-length pattern around it is.  A causal model cannot use that, so if
    changing the tail of a window left the head untouched the recurrence would be
    silently unidirectional.

    The perturbation is deliberately close to the frame under test.  An
    *untrained* GRU's update gate sits at ~0.5, so its state halves per step and
    a 24-frame reach attenuates to ~1e-8 -- indistinguishable from float noise,
    and the first version of this test failed for exactly that reason rather than
    because the recurrence was one-directional.
    """
    model = _model()
    assert model.rnn.bidirectional is True

    mel = torch.zeros(1, 8, MEL_BANDS)
    before = model(mel)[0, 0].item()
    mel[0, 6:] = 5.0

    assert model(mel)[0, 0].item() != pytest.approx(before, abs=1e-6)


# --------------------------------------------------------------------------- #
# Target sparsity -> pos_weight
# --------------------------------------------------------------------------- #


def _targets(downbeat, mask=None) -> DownbeatTargets:
    downbeat = np.asarray(downbeat, dtype=np.float32)
    mask = np.ones(len(downbeat), dtype=bool) if mask is None else np.asarray(mask)
    return DownbeatTargets(downbeat, mask, np.zeros(0), np.zeros(0, np.int8),
                           np.zeros(0, np.int64))


def test_stats_count_only_supervised_mass():
    targets = _targets([1.0, 0.5, 9.0, 0.0], mask=[True, True, False, True])

    stats = accumulate_downbeat_stats([targets, targets])

    assert stats.positive == pytest.approx(3.0)
    assert stats.valid == 6
    assert stats.frames == 8
    assert stats.track_mass.tolist() == pytest.approx([0.5, 0.5])


def test_pos_weight_matches_the_carried_corpus_measurement():
    """Task 1 measured 0.0992 target mass per frame over 120 tracks.

    The section head's ratio form is reused deliberately, so the number is
    negatives-over-positives (~9.1) rather than 1/mass (~10.1); both are the
    same claim about sparsity and only one of them can be the code's.
    """
    stats = accumulate_downbeat_stats([_targets(np.full(10_000, 0.0992))])

    assert boundary_pos_weight(stats.positive, stats.valid) == pytest.approx(9.08, abs=0.01)


def test_a_track_with_no_supervision_does_not_divide_by_zero():
    stats = accumulate_downbeat_stats([_targets([1.0, 1.0], mask=[False, False])])

    assert stats.track_mass.tolist() == [0.0]
    assert boundary_pos_weight(stats.positive, stats.valid) == 1.0


# --------------------------------------------------------------------------- #
# Peak picking
# --------------------------------------------------------------------------- #


def test_peaks_are_local_maxima_in_time_order():
    scores = np.array([0.0, 0.9, 0.1, 0.0, 0.7, 0.2])

    assert peak_candidates(scores, min_distance=1).tolist() == [1, 4]


def test_suppression_keeps_the_higher_of_two_close_peaks():
    scores = np.array([0.0, 0.6, 0.0, 0.9, 0.0])

    assert peak_candidates(scores, min_distance=2).tolist() == [3]
    assert peak_candidates(scores, min_distance=1).tolist() == [1, 3]


def test_suppression_is_threshold_free_so_the_sweep_compares_operating_points():
    """The property the threshold sweep rests on: the peak set at a threshold is
    a *subset* of the peak set, never a differently-picked one."""
    rng = np.random.default_rng(3)
    scores = rng.random(500)
    picked = peak_candidates(scores, min_distance=6)

    for threshold in (0.2, 0.5, 0.8):
        subset = picked[scores[picked] >= threshold]
        assert set(subset.tolist()) <= set(picked.tolist())
        assert np.all(np.diff(subset) >= 6) or subset.size < 2


def test_a_plateau_collapses_to_its_centre_not_an_edge():
    """A saturating head emits runs of identical values; picking an endpoint
    would bias the reported instant by half the plateau -- up to 46 ms of a
    70 ms budget."""
    scores = np.array([0.0, 0.5, 0.5, 0.5, 0.0])

    assert peak_candidates(scores, min_distance=1).tolist() == [2]


def test_a_flat_activation_is_one_peak_not_a_comb():
    """A model that has said nothing must not be handed a peak every
    ``min_distance`` frames -- that alone would score well above chance."""
    assert peak_candidates(np.full(400, 0.5), min_distance=15).tolist() == [199]


def test_an_empty_activation_has_no_peaks():
    assert peak_candidates(np.zeros(0), min_distance=4).tolist() == []


def test_min_distance_covers_the_fastest_bar_in_the_corpus():
    """0.70 s cannot merge two real downbeats: the corpus tops out at 252 BPM,
    i.e. a 0.95 s bar."""
    assert MIN_PEAK_DISTANCE_SEC < 4 * 60.0 / 252.0


# --------------------------------------------------------------------------- #
# Event matching
# --------------------------------------------------------------------------- #


def test_matching_pairs_events_inside_the_tolerance():
    predicted = np.array([1.00, 2.00, 3.00])
    reference = np.array([1.05, 2.00, 3.20])

    assert match_events(predicted, reference, tolerance=0.070) == (2, 1, 1)


def test_matching_is_one_to_one():
    """Two predictions on one reference must not both score -- a jittery picker
    would otherwise buy recall for free."""
    predicted = np.array([1.00, 1.02])
    reference = np.array([1.01])

    assert match_events(predicted, reference, tolerance=0.070) == (1, 1, 0)


def test_matching_of_empty_sides():
    assert match_events(np.zeros(0), np.array([1.0]), 0.07) == (0, 0, 1)
    assert match_events(np.array([1.0]), np.zeros(0), 0.07) == (0, 1, 0)


def test_tolerance_is_the_verdicts_own():
    assert TOLERANCE_SEC == pytest.approx(0.070)


def test_prf_is_the_textbook_formula():
    scores = prf(tp=6, fp=2, fn=4)

    assert scores["precision"] == pytest.approx(0.75)
    assert scores["recall"] == pytest.approx(0.6)
    assert scores["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_prf_of_nothing_is_zero_not_a_crash():
    assert prf(0, 0, 0)["f1"] == 0.0


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #


def test_the_sweep_finds_the_threshold_that_deletes_the_false_positives():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    scores = np.array([0.9, 0.9, 0.1, 0.1])          # last two are noise
    reference = np.array([1.0, 2.0])

    result = sweep_peak_f1([(times, scores, reference)], thresholds=(0.05, 0.5))

    assert result["f1"] == pytest.approx(1.0)
    assert result["best_threshold"] == pytest.approx(0.5)
    assert result["curve"][0.05]["f1"] == pytest.approx(2 * 2 / (4 + 2))


def test_the_sweep_micro_averages_across_tracks():
    """Macro would let a 90-second track outvote a nine-minute one."""
    long_track = (np.arange(100) * 2.0, np.ones(100), np.arange(100) * 2.0)
    short_track = (np.array([1.0]), np.ones(1), np.array([50.0]))

    result = sweep_peak_f1([long_track, short_track], thresholds=(0.5,))

    # 100 hits, 1 false positive, 1 miss -- not the mean of 1.0 and 0.0.
    assert result["f1"] == pytest.approx(2 * 100 / (101 + 101))


def test_the_sweep_breaks_ties_toward_the_lower_threshold():
    """A tie means the extra confidence bought nothing; prefer the recall side."""
    track = (np.array([1.0]), np.array([0.9]), np.array([1.0]))

    assert sweep_peak_f1([track], thresholds=(0.1, 0.5))["best_threshold"] == 0.1


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_ece_is_zero_when_confidence_matches_frequency():
    probs = np.full(100, 0.3)
    labels = (np.arange(100) % 10) < 3

    assert binary_ece(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_one_for_a_confidently_wrong_head():
    assert binary_ece(np.ones(50), np.zeros(50, dtype=bool)) == pytest.approx(1.0)


def test_ece_of_nothing_is_zero_not_a_crash():
    assert binary_ece(np.zeros(0), np.zeros(0, dtype=bool)) == 0.0


def test_deweighting_undoes_exactly_a_log_pos_weight_logit_shift():
    """The transform has to be the analytic inverse of the reweighting, or the
    second calibration number is decoration."""
    logits = np.array([-3.0, -1.0, 0.0, 2.0])
    pos_weight = 9.0
    inflated = 1.0 / (1.0 + np.exp(-(logits + math.log(pos_weight))))

    recovered = deweighted(inflated, pos_weight)

    assert recovered == pytest.approx(1.0 / (1.0 + np.exp(-logits)), abs=1e-9)


def test_deweighting_at_unit_weight_changes_nothing():
    probs = np.array([0.1, 0.5, 0.9])

    assert deweighted(probs, 1.0) == pytest.approx(probs, abs=1e-9)


def test_deweighting_lowers_an_inflated_head():
    assert deweighted(np.array([0.5]), 9.0)[0] < 0.5


# --------------------------------------------------------------------------- #
# Frame times
# --------------------------------------------------------------------------- #


def test_frame_time_is_the_datasets_own_stamp():
    """Frame k carries t0 + k*dt with t0 == dt; an off-by-one here is 46 ms of
    systematic bias, two thirds of the tolerance budget."""
    assert frame_times([0, 1, 2]) == pytest.approx(
        [FRAME_SEC, 2 * FRAME_SEC, 3 * FRAME_SEC])


# --------------------------------------------------------------------------- #
# Stitching windows back into whole-track curves
# --------------------------------------------------------------------------- #


class _Slots:
    """Minimal stand-in for the dataset's window geometry."""

    def __init__(self, offsets, window_frames, track="t"):
        self._offsets = offsets
        self.window_frames = window_frames
        self._track = track

    def track_id_of(self, index):
        return self._track

    def window_offset(self, index):
        return self._offsets[index]


def test_stitching_places_each_window_at_its_own_offset():
    dataset = _Slots([0, 4], window_frames=4)
    sums = {"t": np.zeros(8)}
    counts = {"t": np.zeros(8, dtype=np.int64)}

    end = stitch(dataset, np.array([[1.0] * 4, [2.0] * 4]), 0, sums, counts)

    assert end == 2
    assert sums["t"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert counts["t"].tolist() == [1] * 8


def test_stitching_averages_the_re_overlapped_tail_window():
    """Eval-mode tiling clamps the final window back so it re-overlaps -- taking
    the later window would silently prefer one of two equal estimates on every
    track's last ~16 s, which is exactly where peak picking is fragile."""
    dataset = _Slots([0, 2], window_frames=4)
    sums = {"t": np.zeros(6)}
    counts = {"t": np.zeros(6, dtype=np.int64)}

    stitch(dataset, np.array([[1.0] * 4, [3.0] * 4]), 0, sums, counts)
    averaged = sums["t"] / np.maximum(counts["t"], 1)

    assert counts["t"].tolist() == [1, 1, 2, 2, 1, 1]
    assert averaged.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_stitching_clips_a_window_that_runs_past_the_end():
    dataset = _Slots([2], window_frames=4)
    sums = {"t": np.zeros(4)}
    counts = {"t": np.zeros(4, dtype=np.int64)}

    stitch(dataset, np.array([[5.0] * 4]), 0, sums, counts)

    assert sums["t"].tolist() == [0, 0, 5, 5]


# --------------------------------------------------------------------------- #
# The val loop, end to end on a synthetic corpus
# --------------------------------------------------------------------------- #


class _Oracle(torch.nn.Module):
    """A head handed the answer: its logit *is* the target, read off band 0."""

    def forward(self, mel):
        return torch.logit(mel[:, :, 0].clamp(1e-6, 1.0 - 1e-6))


def _target_as_mel(loader):
    """Substitute the target for the mel so ``_Oracle`` can be a real module.

    The alternative -- a module holding the batch in an attribute -- would let
    the test pass while ``evaluate`` fed the model something else entirely.
    """
    for _mel, target, mask in loader:
        yield target.unsqueeze(-1).expand(-1, -1, MEL_BANDS), target, mask


def test_evaluate_scores_a_perfect_head_at_f1_one(tmp_path):
    """The whole metric chain -- forward, stitch, pick, match, sweep -- against a
    model that has been handed the answer.  If any link is off by a frame this
    cannot reach 1.0."""
    data_dir, ids = with_grids(tmp_path, count=2, frames=1200, bars=60)
    dataset = DownbeatWindowDataset(data_dir, ids, augment=False)
    loader = build_loader(dataset, batch_size=8, shuffle=False, num_workers=0,
                          pin_memory=False, generator=None)

    metrics = evaluate(_Oracle(), _target_as_mel(loader), dataset,
                       torch.device("cpu"), pos_weight=9.0,
                       min_distance=max(1, int(round(MIN_PEAK_DISTANCE_SEC / FRAME_SEC))))

    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["downbeats"] > 20
    assert metrics["peaks"] == metrics["downbeats"]
    assert metrics["pr_auc"] == pytest.approx(1.0)
    # The valleys between the Gaussian bumps are exactly flat, so each one is a
    # plateau and contributes a near-zero candidate: the raw candidate count is
    # ~2x the downbeat count and only the thresholded `peaks` means anything.
    assert metrics["candidates"] > metrics["peaks"]


def test_evaluate_scores_a_dead_head_at_f1_zero(tmp_path):
    """A model that outputs a constant has no peaks to pick and must score 0 --
    not silently benefit from the picker inventing evenly spaced maxima."""
    data_dir, ids = with_grids(tmp_path, count=2, frames=1200, bars=60)
    dataset = DownbeatWindowDataset(data_dir, ids, augment=False)

    class Flat(torch.nn.Module):
        def forward(self, mel):
            return torch.zeros(mel.shape[0], mel.shape[1])

    loader = build_loader(dataset, batch_size=8, shuffle=False, num_workers=0,
                          pin_memory=False, generator=None)
    metrics = evaluate(Flat(), loader, dataset, torch.device("cpu"), pos_weight=9.0,
                       min_distance=max(1, int(round(MIN_PEAK_DISTANCE_SEC / FRAME_SEC))))

    # One flat plateau over the whole track collapses to a single peak per track.
    assert metrics["f1"] < 0.05
    assert metrics["positive_rate"] == pytest.approx(
        float((dataset.targets_for(ids[0]).downbeat >= POSITIVE_THRESHOLD).mean()),
        abs=0.02)
