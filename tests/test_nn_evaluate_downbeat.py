import json
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.downbeat_decoder import (  # noqa: E402
    BEATS_PER_BAR,
    BarPhaseHMM,
    PhaseDecision,
    PhaseParams,
    candidate_grid,
    downbeat_times,
    phase_flips,
)
from nn.evaluate_downbeat import (  # noqa: E402
    INTERVAL_DEVIATION,
    REACH_LABELS,
    TOLERANCE_SEC,
    alignment_row,
    beat_anchored_flips,
    config_fingerprint,
    edges_from_downbeats,
    fold_to_beats,
    interval_deviation,
    lag_for,
    nearest_offset,
    phase_scores,
    reach_labels,
    rolling_median,
    score_downbeats,
    split_guard,
)

PERIOD = 0.46875                      # 128 BPM, the corpus's own tempo
BAR = BEATS_PER_BAR * PERIOD


def beats(n: int, *, period: float = PERIOD, start: float = 1.0,
          offset: float = 0.0) -> np.ndarray:
    return start + offset + period * np.arange(n, dtype=np.float64)


def grid_phases(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64) % BEATS_PER_BAR + 1


def clean_decisions(n: int, *, subdivision: int = 1, period: float = PERIOD,
                    start: float = 1.0, first_phase: int = 1) -> list:
    cycle = BEATS_PER_BAR * subdivision
    step = period / subdivision
    return [PhaseDecision(index, start + index * step,
                          (first_phase - 1 + index) % cycle + 1, 1.0, False)
            for index in range(n)]


def test_the_lag_holds_wall_clock_look_ahead_constant_across_subdivisions():
    assert lag_for(4, 1) == 4
    assert lag_for(4, 2) == 8
    assert lag_for(2, 2) == 4


def test_nearest_offset_is_signed_and_points_from_query_to_reference():
    reference = beats(8)
    assert np.allclose(nearest_offset(reference + 0.02, reference), 0.02)
    assert np.allclose(nearest_offset(reference - 0.02, reference), -0.02)


def test_nearest_offset_of_an_empty_reference_is_nan_not_zero():
    offsets = nearest_offset(beats(4), np.zeros(0))
    assert offsets.shape == (4,) and np.isnan(offsets).all()


def test_folding_an_offset_into_beats_wraps_to_the_half_open_half_beat():
    period = np.full(4, PERIOD)
    assert np.allclose(fold_to_beats(np.full(4, 0.5 * PERIOD), period), 0.5)
    assert np.allclose(fold_to_beats(np.full(4, 1.5 * PERIOD), period), 0.5)
    assert np.allclose(fold_to_beats(np.full(4, 0.25 * PERIOD), period), 0.25)
    assert np.allclose(fold_to_beats(np.full(4, -0.25 * PERIOD), period), -0.25)


def test_the_rolling_median_follows_a_tempo_ride_instead_of_averaging_it():
    values = np.concatenate([np.full(20, 0.5), np.full(20, 0.4)])
    rolled = rolling_median(values, 5)
    assert rolled[2] == pytest.approx(0.5)
    assert rolled[-3] == pytest.approx(0.4)
    assert len(rolled) == len(values)


def test_a_perfectly_aligned_beat_stream_reports_no_offset_and_full_coverage():
    expert = beats(64)
    row = alignment_row(expert, expert, expert[::BEATS_PER_BAR])
    assert row["median_offset_sec"] == pytest.approx(0.0)
    assert row["live_on_grid"] == pytest.approx(1.0)
    assert row["expert_covered"] == pytest.approx(1.0)
    assert row["ibi_ratio"] == pytest.approx(1.0)


def test_a_half_beat_locked_stream_is_reported_as_a_lock_not_as_jitter():
    expert = beats(64)
    row = alignment_row(expert + 0.5 * PERIOD, expert, expert[::BEATS_PER_BAR])
    assert abs(row["median_abs_phase"]) == pytest.approx(0.5, abs=0.01)
    assert row["phase_iqr"] == pytest.approx(0.0, abs=0.01)
    assert row["live_on_grid"] == pytest.approx(0.0, abs=0.02)


def test_a_jittered_stream_is_reported_as_spread_around_zero():
    rng = np.random.default_rng(7)
    expert = beats(200)
    row = alignment_row(expert + rng.uniform(-0.03, 0.03, expert.size), expert,
                        expert[::BEATS_PER_BAR])
    assert row["median_offset_sec"] == pytest.approx(0.0, abs=0.01)
    assert row["phase_iqr"] > 0.02
    assert row["live_on_grid"] > 0.9


def test_a_half_tempo_stream_is_visible_in_the_ibi_ratio():
    expert = beats(64)
    row = alignment_row(expert[::2], expert, expert[::BEATS_PER_BAR])
    assert row["ibi_ratio"] == pytest.approx(2.0, abs=0.01)
    assert row["live_on_grid"] == pytest.approx(1.0)
    assert row["expert_covered"] == pytest.approx(0.5, abs=0.02)


def test_an_aligned_stream_reaches_every_downbeat_on_the_beat_grid():
    expert = beats(64)
    labels = reach_labels(expert, expert[::BEATS_PER_BAR], expert)
    assert set(labels) == {"beat"}


def test_a_half_beat_lock_is_unreachable_on_beats_and_reached_by_midpoints():
    expert = beats(64)
    live = expert + 0.5 * PERIOD
    labels = reach_labels(live, expert[::BEATS_PER_BAR], expert)
    assert labels.count("midpoint") >= 14
    assert "beat" not in labels


def test_a_quarter_beat_lock_is_reachable_by_neither_and_named_as_a_fraction():
    expert = beats(64)
    live = expert + 0.25 * PERIOD
    labels = reach_labels(live, expert[::BEATS_PER_BAR], expert)
    assert set(labels) <= {"fraction_lock"}
    assert labels.count("fraction_lock") == len(labels)


def test_a_dropout_the_decoder_cannot_coast_is_named_a_dropout():
    expert = beats(128)
    live = np.concatenate([expert[:20], expert[60:]])
    labels = reach_labels(live, expert[::BEATS_PER_BAR], expert)
    assert "dropout" in labels
    assert "beat" in labels


def test_a_dropout_short_enough_to_coast_is_not_charged_as_a_miss():
    expert = beats(64)
    live = np.concatenate([expert[:20], expert[26:]])
    labels = reach_labels(live, expert[::BEATS_PER_BAR], expert)
    assert "dropout" not in labels
    assert "coast" in labels


def test_a_downbeat_outside_the_beat_stream_is_named_no_coverage():
    expert = beats(64)
    labels = reach_labels(expert[24:], expert[::BEATS_PER_BAR], expert)
    assert labels[0] == "no_coverage"
    assert labels[-1] == "beat"


def test_a_drifting_tempo_is_named_a_tempo_mismatch():
    expert = beats(96)
    live = beats(96, period=PERIOD * 1.03)
    labels = reach_labels(live, expert[::BEATS_PER_BAR], expert)
    assert labels.count("tempo_mismatch") > labels.count("fraction_lock")


def test_a_stream_at_exactly_double_tempo_is_not_a_tempo_mismatch():
    expert = beats(64)
    labels = reach_labels(beats(128, period=PERIOD / 2), expert[::BEATS_PER_BAR], expert)
    assert set(labels) == {"beat"}


def test_an_unsteady_residual_is_jitter_and_a_steady_one_is_a_lock():
    rng = np.random.default_rng(11)
    expert = beats(96)
    steady = reach_labels(expert + 0.25 * PERIOD, expert[::BEATS_PER_BAR], expert)
    noisy = reach_labels(expert + rng.uniform(-0.22, 0.22, expert.size) * PERIOD,
                         expert[::BEATS_PER_BAR], expert)
    assert set(steady) == {"fraction_lock"}
    assert noisy.count("jitter") > noisy.count("fraction_lock")


def test_a_correctly_paced_beat_stream_has_a_bar_rate_ratio_of_one():
    from nn.evaluate_downbeat import bar_rate_ratio

    expert = beats(64)
    downbeats = expert[::BEATS_PER_BAR]
    for subdivision in (1, 2):
        params = PhaseParams(subdivision=subdivision,
                             lag_beats=lag_for(1, subdivision))
        assert bar_rate_ratio(expert, downbeats, params) == pytest.approx(1.0, abs=0.03)


def test_inserted_beats_raise_the_bar_rate_and_that_is_a_precision_ceiling():
    from nn.evaluate_downbeat import bar_rate_ratio

    expert = beats(64)
    downbeats = expert[::BEATS_PER_BAR]
    padded = np.sort(np.concatenate([expert, expert[:-1] + 0.5 * PERIOD]))
    params = PhaseParams(subdivision=1, lag_beats=4)
    assert bar_rate_ratio(padded, downbeats, params) == pytest.approx(2.0, abs=0.05)


def test_a_coasted_gap_counts_toward_the_bar_rate_because_the_decoder_emits_it():
    from nn.evaluate_downbeat import bar_rate_ratio, decoder_instants

    expert = beats(64)
    holed = np.concatenate([expert[:20], expert[26:]])
    params = PhaseParams(subdivision=1, lag_beats=4)
    assert decoder_instants(holed, params).size > holed.size
    assert bar_rate_ratio(holed, expert[::BEATS_PER_BAR], params) == pytest.approx(
        1.0, abs=0.05)


def test_the_ceiling_grid_is_the_shipped_candidate_grid_at_subdivision_two():
    from nn.evaluate_downbeat import subdivided_grid

    times = beats(32)
    assert np.allclose(subdivided_grid(times, 2), candidate_grid(times, 2))
    assert np.allclose(subdivided_grid(times, 1), times)


def test_a_quarter_beat_lock_is_unreachable_at_two_and_reachable_at_four():
    from nn.evaluate_downbeat import grid_ceiling

    expert = beats(64)
    live = expert + 0.25 * PERIOD
    interior_downbeats = expert[::BEATS_PER_BAR][1:]
    assert grid_ceiling(live, interior_downbeats, 2) == pytest.approx(0.0)
    assert grid_ceiling(live, interior_downbeats, 4) == pytest.approx(1.0)


def test_the_residual_says_where_in_the_beat_the_downbeat_fell():
    from nn.evaluate_downbeat import downbeat_residuals

    expert = beats(64)
    downbeats = expert[::BEATS_PER_BAR]
    assert np.allclose(downbeat_residuals(expert, downbeats, expert), 0.0)
    assert np.allclose(downbeat_residuals(expert + 0.25 * PERIOD, downbeats, expert),
                       0.25, atol=0.01)
    assert np.allclose(downbeat_residuals(expert + 0.5 * PERIOD, downbeats, expert),
                       0.5, atol=0.01)


def test_every_reach_label_is_declared():
    expert = beats(64)
    labels = reach_labels(expert + 0.25 * PERIOD, expert[::BEATS_PER_BAR], expert)
    assert set(labels) <= set(REACH_LABELS)


def test_coasting_is_credited_only_when_the_decoder_would_actually_coast():
    expert = beats(96)
    short = np.concatenate([expert[:20], expert[23:]])
    long = np.concatenate([expert[:20], expert[60:]])
    assert "coast" in reach_labels(short, expert[::BEATS_PER_BAR], expert)
    assert "coast" not in reach_labels(long, expert[::BEATS_PER_BAR], expert)


def test_downbeat_scoring_matches_at_the_tolerance_and_not_a_millisecond_past():
    truth = beats(16, period=BAR)
    assert score_downbeats(truth + 0.069, truth)["f1"] == pytest.approx(1.0)
    assert score_downbeats(truth + 0.071, truth)["f1"] == pytest.approx(0.0)


def test_downbeat_scoring_counts_extra_predictions_against_precision():
    truth = beats(16, period=BAR)
    doubled = np.sort(np.concatenate([truth, truth + 0.5 * BAR]))
    score = score_downbeats(doubled, truth)
    assert score["recall"] == pytest.approx(1.0)
    assert score["precision"] == pytest.approx(0.5)


def test_phase_accuracy_is_zero_when_the_grid_is_shifted_by_one_beat():
    expert = beats(64)
    phases = grid_phases(64)
    aligned = phase_scores(clean_decisions(64), 1, expert, phases)
    shifted = phase_scores(clean_decisions(64, first_phase=2), 1, expert, phases)
    assert aligned["accuracy"] == pytest.approx(1.0)
    assert shifted["accuracy"] == pytest.approx(0.0)


def test_phase_accuracy_reports_its_own_coverage_rather_than_hiding_a_gap():
    expert = beats(64)
    phases = grid_phases(64)
    scores = phase_scores(clean_decisions(32), 1, expert, phases)
    assert scores["covered"] == 32
    assert scores["total"] == 64
    assert scores["coverage"] == pytest.approx(0.5)


def test_an_interstitial_position_is_scored_wrong_not_skipped():
    expert = beats(32)
    phases = grid_phases(32)
    decisions = [PhaseDecision(i, t, 2, 1.0, False) for i, t in enumerate(expert)]
    scores = phase_scores(decisions, 2, expert, phases)
    assert scores["covered"] == 32
    assert scores["accuracy"] == pytest.approx(0.0)
    assert scores["interstitial"] == 32


def test_confidence_gating_keeps_fewer_downbeats_and_scores_only_those():
    from nn.evaluate_downbeat import confidence_sweep

    truth = beats(8, period=BAR)
    decisions = [PhaseDecision(index, time, 1, 0.9 if index % 2 else 0.2, False)
                 for index, time in enumerate(truth)]
    rows = confidence_sweep(decisions, truth, thresholds=(0.0, 0.5))
    assert rows[0.0]["tp"] == 8 and rows[0.0]["fp"] == 0
    assert rows[0.5]["tp"] == 4 and rows[0.5]["fn"] == 4
    assert rows[0.5]["kept"] == 4 and rows[0.5]["total"] == 8


def test_confidence_gating_can_raise_precision_when_confidence_is_informative():
    from nn.evaluate_downbeat import confidence_sweep

    truth = beats(8, period=BAR)
    good = [PhaseDecision(i, t, 1, 0.9, False) for i, t in enumerate(truth)]
    junk = [PhaseDecision(100 + i, t + 0.5 * BAR, 1, 0.2, False)
            for i, t in enumerate(truth[:-1])]
    decisions = sorted(good + junk, key=lambda d: d.time)
    rows = confidence_sweep(decisions, truth, thresholds=(0.0, 0.5))
    open_gate = rows[0.0]
    shut_gate = rows[0.5]
    assert open_gate["fp"] == 7 and shut_gate["fp"] == 0


def test_a_perfectly_regular_grid_has_exactly_zero_interval_deviations():
    result = interval_deviation(beats(64, period=BAR))
    assert result["events"] == 0
    assert result["per_minute"] == pytest.approx(0.0)


def test_one_short_bar_is_one_interval_deviation_event():
    downbeats = beats(64, period=BAR)
    downbeats[32:] -= 0.5 * BAR
    result = interval_deviation(downbeats)
    assert result["events"] >= 1
    assert result["per_minute"] > 0.0


def test_the_deviation_threshold_is_relative_to_the_track_not_absolute():
    slow = interval_deviation(beats(64, period=2 * BAR))
    fast = interval_deviation(beats(64, period=BAR))
    assert slow["events"] == fast["events"] == 0
    ride = beats(64, period=BAR) * (1.0 + 0.5 * INTERVAL_DEVIATION)
    assert interval_deviation(ride)["events"] == 0


def test_interval_deviation_is_reported_per_minute_of_the_track_it_measured():
    downbeats = beats(64, period=BAR)
    result = interval_deviation(downbeats)
    span = downbeats[-1] - downbeats[0]
    assert result["minutes"] == pytest.approx(span / 60.0)


def test_beat_anchored_flips_agree_with_phase_flips_when_nothing_was_coasted():
    decisions = clean_decisions(64)
    assert beat_anchored_flips(decisions, 1)["flips"] == phase_flips(decisions, 1)
    broken = list(decisions)
    broken[30] = broken[30]._replace(phase=broken[30].phase % 4 + 1)
    assert beat_anchored_flips(broken, 1)["flips"] == phase_flips(broken, 1)


def test_a_stable_off_beat_lock_reads_as_stable_at_beat_rate():
    decisions = clean_decisions(128, subdivision=2, first_phase=2)
    assert beat_anchored_flips(decisions, 2)["flips"] == 0
    assert beat_anchored_flips(decisions, 2)["pairs"] == 63


def test_a_coasted_candidate_breaks_the_pair_instead_of_faking_a_flip():
    decisions = clean_decisions(16, subdivision=1)
    decisions[8] = decisions[8]._replace(virtual=True)
    result = beat_anchored_flips(decisions, 1)
    assert result["breaks"] == 1
    assert result["flips"] == 0
    assert result["pairs"] == 13


def test_predicted_edges_close_the_last_bar_at_the_median_interval():
    downbeats = beats(16, period=BAR)
    edges = edges_from_downbeats(downbeats)
    assert len(edges) == len(downbeats) + 1
    assert edges[-1] - edges[-2] == pytest.approx(BAR)


def test_predicted_edges_refuse_a_grid_too_short_to_decode_on():
    with pytest.raises(RuntimeError, match="bar grid"):
        edges_from_downbeats(np.array([1.0]))


def test_predicted_edges_are_sorted_even_if_the_decoder_emitted_out_of_order():
    edges = edges_from_downbeats(np.array([3.0, 1.0, 2.0]))
    assert np.all(np.diff(edges) > 0)


def test_the_config_fingerprint_changes_with_every_knob_that_changes_a_number():
    base = config_fingerprint(PhaseParams(), "live", refine=False)
    assert base != config_fingerprint(PhaseParams(flip_penalty=3.0), "live", refine=False)
    assert base != config_fingerprint(PhaseParams(), "expert", refine=False)
    assert base != config_fingerprint(PhaseParams(), "live", refine=True)
    assert base == config_fingerprint(PhaseParams(), "live", refine=False)


def test_the_fingerprint_is_a_readable_record_not_only_a_hash():
    record = config_fingerprint(PhaseParams(flip_penalty=3.0, subdivision=2,
                                            lag_beats=8), "live", refine=True)
    payload = json.loads(record["config"])
    assert payload["flip_penalty"] == 3.0
    assert payload["subdivision"] == 2
    assert payload["condition"] == "live"
    assert payload["refine"] is True
    assert len(record["sha256"]) == 64


def test_a_tuning_mode_refuses_test_ids_by_membership_not_by_flag(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps(
        {"train": ["t1"], "val": ["v1", "v2"], "test": ["x1"]}), encoding="utf-8")
    assert split_guard(tmp_path, ["v1", "v2"], "val") == ["v1", "v2"]
    with pytest.raises(RuntimeError, match="tuning read"):
        split_guard(tmp_path, ["v1", "x1"], "val")
    assert split_guard(tmp_path, ["x1"], "test", reason="labelled test") == ["x1"]
    with pytest.raises(RuntimeError, match="labelled test"):
        split_guard(tmp_path, ["x1", "v1"], "test", reason="labelled test")


def synthetic_stream(n_bars: int, *, period: float, start: float,
                     phase_offset: int = 0) -> tuple:
    times = start + period * np.arange(4 * n_bars, dtype=np.float64)
    index = (np.arange(4 * n_bars) + phase_offset) % BEATS_PER_BAR
    scores = np.where(index == 0, 0.95, 0.05)
    return times, scores, times[index == 0]


def relock_bars(params, *, gap_beats: float, new_period: float,
                phase_offset: int) -> int | None:
    first, first_scores, _ = synthetic_stream(16, period=PERIOD, start=1.0)
    start = first[-1] + PERIOD * (1.0 + gap_beats)
    second, second_scores, truth = synthetic_stream(
        16, period=new_period, start=start, phase_offset=phase_offset)
    times = np.concatenate([first, second])
    scores = np.concatenate([first_scores, second_scores])
    if params.subdivision == 2:
        from nn.downbeat_decoder import candidate_grid as grid_of
        dense = grid_of(times, 2)
        dense_scores = np.full(dense.size, 0.05)
        dense_scores[0::2] = scores
        times, scores = dense, dense_scores
    predicted = downbeat_times(BarPhaseHMM(params).decode(times, scores))
    hit = [index for index, moment in enumerate(truth)
           if np.any(np.abs(predicted - moment) <= TOLERANCE_SEC)]
    locked = [index for index in range(len(truth))
              if all(later in hit for later in range(index, len(truth)))]
    return min(locked) if locked else None


def test_re_lock_after_a_deck_transition_is_bought_with_the_flip_penalty():
    cuts = (dict(gap_beats=1.9, new_period=0.5, phase_offset=2),
            dict(gap_beats=1.9, new_period=PERIOD, phase_offset=1),
            dict(gap_beats=0.0, new_period=0.5, phase_offset=3))

    responsive = PhaseParams(lag_beats=4, subdivision=2, flip_penalty=2.0)
    assert [relock_bars(responsive, **cut) for cut in cuts] == [0, 0, 0]

    # A measured limitation, not a preference: a phase-only cut never re-locks here.
    committed = PhaseParams(lag_beats=2, subdivision=1, flip_penalty=9.0)
    tempo_change, same_tempo, hard_cut = (relock_bars(committed, **cut) for cut in cuts)
    assert tempo_change == 0
    assert same_tempo is None and hard_cut is None

    shipped = PhaseParams()
    assert [relock_bars(shipped, **cut) for cut in cuts] == [0, None, None]


def test_the_phase_re_locks_after_a_deck_transition_to_a_new_tempo_and_offset():
    first, first_scores, _ = synthetic_stream(16, period=PERIOD, start=1.0)
    gap = first[-1] + 0.9
    second, second_scores, second_downbeats = synthetic_stream(
        16, period=0.5, start=gap, phase_offset=2)
    times = np.concatenate([first, second])
    scores = np.concatenate([first_scores, second_scores])

    params = PhaseParams(lag_beats=4, subdivision=1, flip_penalty=3.0)
    decisions = BarPhaseHMM(params).decode(times, scores)
    predicted = downbeat_times(decisions)

    late = predicted[predicted >= second_downbeats[0] - 0.05]
    matched = [t for t in second_downbeats
               if np.any(np.abs(late - t) <= TOLERANCE_SEC)]
    assert len(matched) >= len(second_downbeats) - 2
    deck_a_bar = PERIOD * BEATS_PER_BAR
    assert not np.allclose(np.diff(late), deck_a_bar)


def test_a_stopped_deck_does_not_manufacture_downbeats_across_the_silence():
    first, first_scores, _ = synthetic_stream(8, period=PERIOD, start=1.0)
    second, second_scores, _ = synthetic_stream(8, period=PERIOD, start=first[-1] + 30.0)
    times = np.concatenate([first, second])
    scores = np.concatenate([first_scores, second_scores])
    decisions = BarPhaseHMM(PhaseParams(lag_beats=4, flip_penalty=3.0)).decode(times, scores)
    predicted = downbeat_times(decisions)
    inside = predicted[(predicted > first[-1] + 1.0) & (predicted < second[0] - 1.0)]
    assert inside.size == 0


def test_decoding_from_memory_is_the_shipped_decode_track_exactly(tmp_path):
    from nn.downbeat_decoder import decode_track
    from nn.evaluate_downbeat import decode_evidence, read_sidecar

    frame_sec = 0.0464399
    times = beats(64)
    activation = np.zeros(1600, dtype=np.float32)
    frames = np.rint((times[::BEATS_PER_BAR] - frame_sec) / frame_sec).astype(int)
    activation[frames] = 0.93
    path = tmp_path / "track.npz"
    np.savez(path, activation=activation, frame_sec=np.float64(frame_sec),
             t0=np.float64(frame_sec), model_sha=np.str_("test"),
             live_beat_time=times, live_beat_score=np.full(times.size, 0.3),
             expert_beat_time=times, expert_beat_score=np.full(times.size, 0.3))

    sidecar = read_sidecar(path)
    for subdivision in (1, 2):
        for refine in (False, True):
            params = PhaseParams(lag_beats=lag_for(4, subdivision),
                                 subdivision=subdivision, flip_penalty=3.0)
            assert (decode_evidence(sidecar, "live", params, refine=refine)
                    == decode_track(path, "live", params, refine=refine))


def test_the_candidate_grid_the_ceiling_is_measured_on_is_the_one_that_decodes():
    times = beats(32)
    dense = candidate_grid(times, 2)
    params = PhaseParams(lag_beats=lag_for(4, 2), subdivision=2)
    decisions = BarPhaseHMM(params).decode(dense, np.full(dense.size, np.nan))
    assert np.allclose([d.time for d in decisions], dense)
