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


def test_forward_returns_one_logit_per_mel_frame():
    logits = _model()(torch.zeros(3, WINDOW_FRAMES, MEL_BANDS))

    assert logits.shape == (3, WINDOW_FRAMES)


@pytest.mark.parametrize("frames", [WINDOW_FRAMES, 512, 64])
def test_time_axis_is_not_baked_in(frames):
    assert _model()(torch.zeros(1, frames, MEL_BANDS)).shape == (1, frames)


def test_parameter_count_sits_inside_the_specs_budget_and_above_a_floor():
    count = count_parameters(_model())

    assert count <= PARAM_BUDGET, f"{count} params exceeds the {PARAM_BUDGET} budget"
    assert count > 150_000


def test_the_downbeat_head_is_smaller_than_the_section_head():
    assert count_parameters(_model()) < count_parameters(SectionCRNN()) / 1.5


def test_frequency_is_pooled_away_and_time_is_preserved():
    model = _model()

    assert model.freq_out == MEL_BANDS // 8
    assert model.feature_dim == 64 * (MEL_BANDS // 8)


def test_the_conv_front_end_is_the_section_models_own():
    blocks, freq = freq_pool_blocks(MEL_BANDS, (32, 64, 64))

    assert freq == MEL_BANDS // 8
    assert [type(layer).__name__ for layer in blocks[:4]] == [
        "Conv2d", "BatchNorm2d", "GELU", "MaxPool2d"]
    keep_time_halve_freq = (1, 2)
    assert blocks[3].kernel_size == keep_time_halve_freq


# Frozen 2026-07-27: every trained v1/v2 checkpoint on disk is keyed by exactly this list.
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
    keys = [key for key in SectionCRNN().state_dict()
            if not key.endswith("num_batches_tracked")]

    assert keys == SECTION_STATE_DICT_KEYS


def test_the_two_heads_share_every_key_but_their_own_head():
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


def test_the_backward_pass_reaches_every_parameter_with_a_finite_gradient():
    model = DownbeatCRNN().train()
    model(torch.randn(2, WINDOW_FRAMES, MEL_BANDS)).square().mean().backward()

    missing = [name for name, param in model.named_parameters()
               if param.grad is None or not torch.isfinite(param.grad).all()]
    assert missing == []


def test_bidirectional_context_actually_reaches_backwards():
    model = _model()
    assert model.rnn.bidirectional is True

    # An untrained GRU's state halves per step: a 24-frame reach falls below float noise.
    mel = torch.zeros(1, 8, MEL_BANDS)
    before = model(mel)[0, 0].item()
    mel[0, 6:] = 5.0

    assert model(mel)[0, 0].item() != pytest.approx(before, abs=1e-6)


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
    # Task 1 measured 0.0992 target mass per frame over 120 tracks.
    stats = accumulate_downbeat_stats([_targets(np.full(10_000, 0.0992))])

    assert boundary_pos_weight(stats.positive, stats.valid) == pytest.approx(9.08, abs=0.01)


def test_a_track_with_no_supervision_does_not_divide_by_zero():
    stats = accumulate_downbeat_stats([_targets([1.0, 1.0], mask=[False, False])])

    assert stats.track_mass.tolist() == [0.0]
    assert boundary_pos_weight(stats.positive, stats.valid) == 1.0


def test_peaks_are_local_maxima_in_time_order():
    scores = np.array([0.0, 0.9, 0.1, 0.0, 0.7, 0.2])

    assert peak_candidates(scores, min_distance=1).tolist() == [1, 4]


def test_suppression_keeps_the_higher_of_two_close_peaks():
    scores = np.array([0.0, 0.6, 0.0, 0.9, 0.0])

    assert peak_candidates(scores, min_distance=2).tolist() == [3]
    assert peak_candidates(scores, min_distance=1).tolist() == [1, 3]


def test_the_peak_set_at_a_threshold_is_a_subset_of_the_unthresholded_peak_set():
    rng = np.random.default_rng(3)
    scores = rng.random(500)
    picked = peak_candidates(scores, min_distance=6)

    for threshold in (0.2, 0.5, 0.8):
        subset = picked[scores[picked] >= threshold]
        assert set(subset.tolist()) <= set(picked.tolist())
        assert np.all(np.diff(subset) >= 6) or subset.size < 2


def test_a_plateau_collapses_to_its_centre_not_an_edge():
    scores = np.array([0.0, 0.5, 0.5, 0.5, 0.0])

    assert peak_candidates(scores, min_distance=1).tolist() == [2]


def test_a_flat_activation_is_one_peak_not_a_comb():
    assert peak_candidates(np.full(400, 0.5), min_distance=15).tolist() == [199]


def test_an_empty_activation_has_no_peaks():
    assert peak_candidates(np.zeros(0), min_distance=4).tolist() == []


def test_min_distance_cannot_merge_two_downbeats_of_the_fastest_corpus_bar():
    assert MIN_PEAK_DISTANCE_SEC < 4 * 60.0 / 252.0


def test_matching_pairs_events_inside_the_tolerance():
    predicted = np.array([1.00, 2.00, 3.00])
    reference = np.array([1.05, 2.00, 3.20])

    assert match_events(predicted, reference, tolerance=0.070) == (2, 1, 1)


def test_matching_is_one_to_one():
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


def test_the_sweep_finds_the_threshold_that_deletes_the_false_positives():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    reference = np.array([1.0, 2.0])

    result = sweep_peak_f1([(times, scores, reference)], thresholds=(0.05, 0.5))

    assert result["f1"] == pytest.approx(1.0)
    assert result["best_threshold"] == pytest.approx(0.5)
    assert result["curve"][0.05]["f1"] == pytest.approx(2 * 2 / (4 + 2))


def test_the_sweep_micro_averages_across_tracks_instead_of_taking_a_macro_mean():
    long_track = (np.arange(100) * 2.0, np.ones(100), np.arange(100) * 2.0)
    short_track = (np.array([1.0]), np.ones(1), np.array([50.0]))

    result = sweep_peak_f1([long_track, short_track], thresholds=(0.5,))

    assert result["f1"] == pytest.approx(2 * 100 / (101 + 101))


def test_the_sweep_breaks_ties_toward_the_lower_threshold():
    track = (np.array([1.0]), np.array([0.9]), np.array([1.0]))

    assert sweep_peak_f1([track], thresholds=(0.1, 0.5))["best_threshold"] == 0.1


def test_ece_is_zero_when_confidence_matches_frequency():
    probs = np.full(100, 0.3)
    labels = (np.arange(100) % 10) < 3

    assert binary_ece(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_one_for_a_confidently_wrong_head():
    assert binary_ece(np.ones(50), np.zeros(50, dtype=bool)) == pytest.approx(1.0)


def test_ece_of_nothing_is_zero_not_a_crash():
    assert binary_ece(np.zeros(0), np.zeros(0, dtype=bool)) == 0.0


def test_deweighting_undoes_exactly_a_log_pos_weight_logit_shift():
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


def test_frame_zero_is_stamped_one_frame_in_not_at_zero():
    assert frame_times([0, 1, 2]) == pytest.approx(
        [FRAME_SEC, 2 * FRAME_SEC, 3 * FRAME_SEC])


class _Slots:
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


class _Oracle(torch.nn.Module):
    def forward(self, mel):
        return torch.logit(mel[:, :, 0].clamp(1e-6, 1.0 - 1e-6))


def _target_as_mel(loader):
    for _mel, target, mask in loader:
        yield target.unsqueeze(-1).expand(-1, -1, MEL_BANDS), target, mask


def test_evaluate_scores_a_perfect_head_at_f1_one(tmp_path):
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
    assert metrics["candidates"] > metrics["peaks"]


def test_evaluate_scores_a_dead_head_at_f1_zero(tmp_path):
    data_dir, ids = with_grids(tmp_path, count=2, frames=1200, bars=60)
    dataset = DownbeatWindowDataset(data_dir, ids, augment=False)

    class Flat(torch.nn.Module):
        def forward(self, mel):
            return torch.zeros(mel.shape[0], mel.shape[1])

    loader = build_loader(dataset, batch_size=8, shuffle=False, num_workers=0,
                          pin_memory=False, generator=None)
    metrics = evaluate(Flat(), loader, dataset, torch.device("cpu"), pos_weight=9.0,
                       min_distance=max(1, int(round(MIN_PEAK_DISTANCE_SEC / FRAME_SEC))))

    assert metrics["f1"] < 0.05
    assert metrics["positive_rate"] == pytest.approx(
        float((dataset.targets_for(ids[0]).downbeat >= POSITIVE_THRESHOLD).mean()),
        abs=0.02)
