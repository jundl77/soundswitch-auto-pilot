"""Tests for the downbeat verdict evaluator (``training/nn/evaluate_downbeat.py``).

This module produces the numbers a gate is read against, so the failure it must
not have is a *plausible* number.  Three shapes of that, and one test group each:

**A ceiling attributed to the wrong cause.**  The live condition is bounded by
where aubio puts beats, and the whole owner decision rests on splitting that
bound into dropped beats, other-fraction locks, tempo mismatch and jitter.  Each
category is therefore tested against a beat stream *built* to exhibit exactly
that failure and nothing else -- a classifier that answered "fraction lock" to
everything would pass any single-case test.

**A stability metric that measures the input instead of the grid.**
``phase_flips`` at subdivision 2 counts aubio's own insertions and deletions, so
this module reports the two metrics review recommended instead.  Both are tested
for the property that matters: a *stable* grid on a *noisy* beat stream must read
as stable.

**A metric that agrees with the truth for the wrong reason.**  Phase accuracy on
a decode shifted by one beat must be 0, not 0.75; an interval-deviation rate on a
perfectly regular grid must be exactly 0 rather than merely small.

Everything here is synthetic and pure numpy: the corpus measurements live behind
the CLI, and a test that needs 215 sidecars is not a test.
"""
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
    """A decode that never flips, at the given subdivision."""
    cycle = BEATS_PER_BAR * subdivision
    step = period / subdivision
    return [PhaseDecision(index, start + index * step,
                          (first_phase - 1 + index) % cycle + 1, 1.0, False)
            for index in range(n)]


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #


def test_the_lag_holds_wall_clock_look_ahead_constant_across_subdivisions():
    # The trap Task 3 fell into: lag counts CANDIDATES, so a fixed lag halves the
    # look-ahead when the grid doubles.  This is the one function allowed to know.
    assert lag_for(4, 1) == 4
    assert lag_for(4, 2) == 8
    assert lag_for(2, 2) == 4


def test_nearest_offset_is_signed_and_points_from_query_to_reference():
    reference = beats(8)
    assert np.allclose(nearest_offset(reference + 0.02, reference), 0.02)
    assert np.allclose(nearest_offset(reference - 0.02, reference), -0.02)


def test_nearest_offset_of_an_empty_reference_is_nan_not_zero():
    # Zero would read as "perfectly aligned" for a track with no beats at all.
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


# --------------------------------------------------------------------------- #
# Alignment: what aubio does to the grid
# --------------------------------------------------------------------------- #


def test_a_perfectly_aligned_beat_stream_reports_no_offset_and_full_coverage():
    expert = beats(64)
    row = alignment_row(expert, expert, expert[::BEATS_PER_BAR])
    assert row["median_offset_sec"] == pytest.approx(0.0)
    assert row["aubio_on_grid"] == pytest.approx(1.0)
    assert row["expert_covered"] == pytest.approx(1.0)
    assert row["ibi_ratio"] == pytest.approx(1.0)


def test_a_half_beat_locked_stream_is_reported_as_a_lock_not_as_jitter():
    # The measured failure mode: a steady offset of half a beat with no spread.
    expert = beats(64)
    row = alignment_row(expert + 0.5 * PERIOD, expert, expert[::BEATS_PER_BAR])
    assert abs(row["median_abs_phase"]) == pytest.approx(0.5, abs=0.01)
    assert row["phase_iqr"] == pytest.approx(0.0, abs=0.01)
    assert row["aubio_on_grid"] == pytest.approx(0.0, abs=0.02)


def test_a_jittered_stream_is_reported_as_spread_around_zero():
    rng = np.random.default_rng(7)
    expert = beats(200)
    row = alignment_row(expert + rng.uniform(-0.03, 0.03, expert.size), expert,
                        expert[::BEATS_PER_BAR])
    assert row["median_offset_sec"] == pytest.approx(0.0, abs=0.01)
    assert row["phase_iqr"] > 0.02
    assert row["aubio_on_grid"] > 0.9


def test_a_half_tempo_stream_is_visible_in_the_ibi_ratio():
    expert = beats(64)
    row = alignment_row(expert[::2], expert, expert[::BEATS_PER_BAR])
    assert row["ibi_ratio"] == pytest.approx(2.0, abs=0.01)
    # Every aubio beat is on the grid; half the expert beats are uncovered.
    assert row["aubio_on_grid"] == pytest.approx(1.0)
    assert row["expert_covered"] == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------- #
# The residual decomposition -- the heart of the owner package
# --------------------------------------------------------------------------- #


def test_an_aligned_stream_reaches_every_downbeat_on_the_beat_grid():
    expert = beats(64)
    labels = reach_labels(expert, expert[::BEATS_PER_BAR], expert)
    assert set(labels) == {"beat"}


def test_a_half_beat_lock_is_unreachable_on_beats_and_reached_by_midpoints():
    expert = beats(64)
    aubio = expert + 0.5 * PERIOD
    labels = reach_labels(aubio, expert[::BEATS_PER_BAR], expert)
    # Every interior downbeat sits exactly on a midpoint of the aubio stream.
    assert labels.count("midpoint") >= 14
    assert "beat" not in labels


def test_a_quarter_beat_lock_is_reachable_by_neither_and_named_as_a_fraction():
    expert = beats(64)
    aubio = expert + 0.25 * PERIOD          # 117 ms -- outside the 70 ms tolerance
    labels = reach_labels(aubio, expert[::BEATS_PER_BAR], expert)
    assert set(labels) <= {"fraction_lock"}
    assert labels.count("fraction_lock") == len(labels)


def test_a_dropout_the_decoder_cannot_coast_is_named_a_dropout():
    expert = beats(128)
    aubio = np.concatenate([expert[:20], expert[60:]])     # 40 beats missing
    labels = reach_labels(aubio, expert[::BEATS_PER_BAR], expert)
    assert "dropout" in labels
    # and the reached ones are still reached: a hole is local, not global.
    assert "beat" in labels


def test_a_dropout_short_enough_to_coast_is_not_charged_as_a_miss():
    """The decoder's own coasting rebuilds a whole-multiple gap exactly.

    Worth pinning as a *ceiling* property and not only as decoder behaviour: a
    residual analysis that ignored coasting would attribute recall the live
    condition actually has to a cause it does not have.
    """
    expert = beats(64)
    aubio = np.concatenate([expert[:20], expert[26:]])     # 6 beats missing
    labels = reach_labels(aubio, expert[::BEATS_PER_BAR], expert)
    assert "dropout" not in labels
    assert "coast" in labels


def test_a_downbeat_outside_the_beat_stream_is_named_no_coverage():
    expert = beats(64)
    labels = reach_labels(expert[24:], expert[::BEATS_PER_BAR], expert)
    assert labels[0] == "no_coverage"
    assert labels[-1] == "beat"


def test_a_drifting_tempo_is_named_a_tempo_mismatch():
    # 3 % fast: the offset slides through every fraction of the beat, so no fixed
    # candidate grid can hold it.  This is the category the plan calls tempo drift.
    expert = beats(96)
    aubio = beats(96, period=PERIOD * 1.03)
    labels = reach_labels(aubio, expert[::BEATS_PER_BAR], expert)
    assert labels.count("tempo_mismatch") > labels.count("fraction_lock")


def test_a_stream_at_exactly_double_tempo_is_not_a_tempo_mismatch():
    # It puts beats ON the annotated instants, so every downbeat is reachable --
    # calling that a mismatch would attribute a cost that does not exist.
    expert = beats(64)
    labels = reach_labels(beats(128, period=PERIOD / 2), expert[::BEATS_PER_BAR], expert)
    assert set(labels) == {"beat"}


def test_an_unsteady_residual_is_jitter_and_a_steady_one_is_a_lock():
    # The actionable split: a steady off-grid offset is a lock a different
    # candidate grid could capture; an unsteady one is only fixed upstream.
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
    """The second ceiling: a flip-free decode emits one downbeat per cycle, so a
    beat stream with surplus beats produces surplus downbeats no phase model can
    retract.  Half again as many beats is half again as many bars."""
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
    # The coasted instants are back in the stream, so the rate is ~1 again --
    # measuring the raw beat count instead would report a grid running slow.
    assert decoder_instants(holed, params).size > holed.size
    assert bar_rate_ratio(holed, expert[::BEATS_PER_BAR], params) == pytest.approx(
        1.0, abs=0.05)


def test_the_ceiling_grid_is_the_shipped_candidate_grid_at_subdivision_two():
    # The bound analysis and the decoder must agree about what "the grid" is at
    # the subdivision that actually decodes, or the bound is about a different
    # system than the result.
    from nn.evaluate_downbeat import subdivided_grid

    times = beats(32)
    assert np.allclose(subdivided_grid(times, 2), candidate_grid(times, 2))
    assert np.allclose(subdivided_grid(times, 1), times)


def test_a_quarter_beat_lock_is_unreachable_at_two_and_reachable_at_four():
    from nn.evaluate_downbeat import grid_ceiling

    expert = beats(64)
    aubio = expert + 0.25 * PERIOD
    # Interior downbeats only: the first one lies before the shifted stream
    # starts, which is a coverage question rather than a grid-density one.
    downbeats = expert[::BEATS_PER_BAR][1:]
    assert grid_ceiling(aubio, downbeats, 2) == pytest.approx(0.0)
    assert grid_ceiling(aubio, downbeats, 4) == pytest.approx(1.0)


def test_the_residual_says_where_in_the_beat_the_downbeat_fell():
    # The number that separates a quarter-beat lock from a triplet feel; a bare
    # "unreachable" count cannot tell the two apart and they have different fixes.
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
    short = np.concatenate([expert[:20], expert[23:]])      # 3 beats -- coastable
    long = np.concatenate([expert[:20], expert[60:]])       # 40 beats -- a discontinuity
    assert "coast" in reach_labels(short, expert[::BEATS_PER_BAR], expert)
    assert "coast" not in reach_labels(long, expert[::BEATS_PER_BAR], expert)


# --------------------------------------------------------------------------- #
# Scoring a decode
# --------------------------------------------------------------------------- #


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
    # At subdivision 2 a beat committed to a half-beat position has NO bar phase.
    # Dropping those would flatter a decoder that locked to aubio's off-beats.
    expert = beats(32)
    phases = grid_phases(32)
    decisions = [PhaseDecision(i, t, 2, 1.0, False) for i, t in enumerate(expert)]
    scores = phase_scores(decisions, 2, expert, phases)
    assert scores["covered"] == 32
    assert scores["accuracy"] == pytest.approx(0.0)
    assert scores["interstitial"] == 32


# --------------------------------------------------------------------------- #
# Stability, in the two units review asked for
# --------------------------------------------------------------------------- #


def test_a_perfectly_regular_grid_has_exactly_zero_interval_deviations():
    result = interval_deviation(beats(64, period=BAR))
    assert result["events"] == 0
    assert result["per_minute"] == pytest.approx(0.0)


def test_one_short_bar_is_one_interval_deviation_event():
    downbeats = beats(64, period=BAR)
    downbeats[32:] -= 0.5 * BAR              # one bar swallowed: two bad intervals meet
    result = interval_deviation(downbeats)
    assert result["events"] >= 1
    assert result["per_minute"] > 0.0


def test_the_deviation_threshold_is_relative_to_the_track_not_absolute():
    slow = interval_deviation(beats(64, period=2 * BAR))
    fast = interval_deviation(beats(64, period=BAR))
    assert slow["events"] == fast["events"] == 0
    # A ride within the threshold is not an event at either tempo.
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
    # The whole point of the metric: aubio locked half a beat off produces a
    # perfectly steady bar grid, and `phase_flips` at subdivision 2 must not be
    # the number the gate reads.
    decisions = clean_decisions(128, subdivision=2, first_phase=2)
    assert beat_anchored_flips(decisions, 2)["flips"] == 0
    assert beat_anchored_flips(decisions, 2)["pairs"] == 63


def test_a_coasted_candidate_breaks_the_pair_instead_of_faking_a_flip():
    decisions = clean_decisions(16, subdivision=1)
    decisions[8] = decisions[8]._replace(virtual=True)
    result = beat_anchored_flips(decisions, 1)
    assert result["breaks"] == 1                # the one pair the coast sits inside
    assert result["flips"] == 0
    assert result["pairs"] == 13                # 15 beats -> 14 pairs, minus the break


# --------------------------------------------------------------------------- #
# The show ablation's grid adapter
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Provenance and split hygiene
# --------------------------------------------------------------------------- #


def test_the_config_fingerprint_changes_with_every_knob_that_changes_a_number():
    base = config_fingerprint(PhaseParams(), "aubio", refine=False)
    assert base != config_fingerprint(PhaseParams(flip_penalty=3.0), "aubio", refine=False)
    assert base != config_fingerprint(PhaseParams(), "expert", refine=False)
    assert base != config_fingerprint(PhaseParams(), "aubio", refine=True)
    assert base == config_fingerprint(PhaseParams(), "aubio", refine=False)


def test_the_fingerprint_is_a_readable_record_not_only_a_hash():
    record = config_fingerprint(PhaseParams(flip_penalty=3.0, subdivision=2,
                                            lag_beats=8), "aubio", refine=True)
    payload = json.loads(record["config"])
    assert payload["flip_penalty"] == 3.0
    assert payload["subdivision"] == 2
    assert payload["condition"] == "aubio"
    assert payload["refine"] is True
    assert len(record["sha256"]) == 64


def test_a_tuning_mode_refuses_test_ids_by_membership_not_by_flag(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps(
        {"train": ["t1"], "val": ["v1", "v2"], "test": ["x1"]}), encoding="utf-8")
    assert split_guard(tmp_path, ["v1", "v2"], "val") == ["v1", "v2"]
    with pytest.raises(RuntimeError, match="not in val"):
        split_guard(tmp_path, ["v1", "x1"], "val")


# --------------------------------------------------------------------------- #
# The deck-transition proxy
# --------------------------------------------------------------------------- #


def synthetic_stream(n_bars: int, *, period: float, start: float,
                     phase_offset: int = 0) -> tuple:
    """Beat times, a confident-on-the-downbeat score, and where those downbeats are.

    The truth comes back from the same expression that built the scores on
    purpose: a re-lock test whose expectation is derived separately can assert
    the wrong instants and look like a decoder failure, which is exactly what the
    first draft of this test did.
    """
    times = start + period * np.arange(4 * n_bars, dtype=np.float64)
    index = (np.arange(4 * n_bars) + phase_offset) % BEATS_PER_BAR
    scores = np.where(index == 0, 0.95, 0.05)
    return times, scores, times[index == 0]


def test_the_phase_re_locks_after_a_deck_transition_to_a_new_tempo_and_offset():
    """A cut from one deck to another: new tempo, new bar phase, small gap.

    The corpus is single tracks, so this is the only evidence the plan can offer
    that the decoder survives a real set.  The claim is bounded on purpose: it
    must re-lock *within a few bars*, not instantly, and it must then STAY locked.
    """
    first, first_scores, _ = synthetic_stream(16, period=PERIOD, start=1.0)
    gap = first[-1] + 0.9                                  # a beat and a half of silence
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
    # Allow the first two bars of deck B to be wrong; everything after must be right.
    assert len(matched) >= len(second_downbeats) - 2
    # And the re-lock must be a re-lock, not a coincidence: deck A's phase, run
    # forward through the cut, would have put the downbeats somewhere else.
    assert not np.allclose(np.diff(late), PERIOD * BEATS_PER_BAR)


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
    """The sweep decodes off arrays it already holds; that must not be a fork.

    Every level in this report comes out of ``decode_evidence``, so if it and
    ``decode_track`` ever disagree the report is measuring a decoder that does
    not exist.  Checked at both subdivisions and with refinement on, because the
    two branches read different fields of the sidecar.
    """
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
             aubio_beat_time=times, aubio_beat_score=np.full(times.size, 0.3),
             expert_beat_time=times, expert_beat_score=np.full(times.size, 0.3))

    sidecar = read_sidecar(path)
    for subdivision in (1, 2):
        for refine in (False, True):
            params = PhaseParams(lag_beats=lag_for(4, subdivision),
                                 subdivision=subdivision, flip_penalty=3.0)
            assert (decode_evidence(sidecar, "aubio", params, refine=refine)
                    == decode_track(path, "aubio", params, refine=refine))


def test_the_candidate_grid_the_ceiling_is_measured_on_is_the_one_that_decodes():
    # A ceiling measured on a different grid than the decoder uses is not a ceiling.
    times = beats(32)
    dense = candidate_grid(times, 2)
    params = PhaseParams(lag_beats=lag_for(4, 2), subdivision=2)
    decisions = BarPhaseHMM(params).decode(dense, np.full(dense.size, np.nan))
    assert np.allclose([d.time for d in decisions], dense)
