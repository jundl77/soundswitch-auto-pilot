import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.downbeat_decoder import (  # noqa: E402
    AGG_HI_FRAMES,
    AGG_LO_FRAMES,
    BEATS_PER_BAR,
    DEFAULT_FLIP_PENALTY,
    DEFAULT_LAG_BEATS,
    MAX_COAST_BEATS,
    BarPhaseHMM,
    PhaseParams,
    aggregate_at_beats,
    bar_phase,
    candidate_grid,
    decode_track,
    downbeat_times,
    nearest_frames,
    phase_flips,
    refine_instants,
)

TEMPO_SEC = 0.46875           # 128 BPM, the corpus's modal tempo
FRAME_SEC = 0.05


def beat_grid(n_beats: int, tempo: float = TEMPO_SEC, start: float = 1.0) -> np.ndarray:
    return start + tempo * np.arange(n_beats, dtype=np.float64)


def clean_scores(n_beats: int, offset: int = 0, high: float = 0.90,
                 low: float = 0.05) -> np.ndarray:
    index = np.arange(n_beats)
    return np.where((index - offset) % BEATS_PER_BAR == 0, high, low)


def ambiguous_scores(n_beats: int, offset: int = 0, high: float = 0.90,
                     low: float = 0.05) -> np.ndarray:
    index = np.arange(n_beats)
    return np.where((index - offset) % (BEATS_PER_BAR // 2) == 0, high, low)


def phases_of(decisions) -> list:
    return [d.phase for d in decisions]


def match_rate(predicted, reference, tolerance: float = 0.070) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if not reference.size:
        return 0.0
    if not predicted.size:
        return 0.0
    distance = np.abs(reference[:, None] - predicted[None, :]).min(axis=1)
    return float(np.mean(distance <= tolerance))


def test_the_candidate_grid_is_the_beat_stream_at_subdivision_one():
    beats = beat_grid(10)

    assert np.array_equal(candidate_grid(beats, 1), beats)


def test_subdivision_two_interleaves_the_midpoints():
    beats = np.array([1.0, 2.0, 4.0])

    dense = candidate_grid(beats, 2)

    assert dense.tolist() == [1.0, 1.5, 2.0, 3.0, 4.0]


def test_an_unmeasured_subdivision_is_refused_not_approximated():
    with pytest.raises(ValueError, match="subdivision"):
        candidate_grid(beat_grid(8), 3)


def test_a_candidate_maps_to_the_frame_whose_stamp_is_nearest():
    times = np.array([FRAME_SEC, 2 * FRAME_SEC, 2 * FRAME_SEC + 0.4 * FRAME_SEC])

    assert nearest_frames(times, 100, FRAME_SEC, FRAME_SEC).tolist() == [0, 1, 1]


def test_a_beat_before_the_first_frame_stamp_is_frame_zero_not_a_sentinel():
    assert nearest_frames(np.array([0.0, 1e-9]), 100, FRAME_SEC,
                          FRAME_SEC).tolist() == [0, 0]


def test_a_beat_past_the_end_of_the_activation_gets_the_sentinel():
    assert nearest_frames(np.array([1000.0]), 100, FRAME_SEC, FRAME_SEC).tolist() == [-1]


def test_aggregation_takes_the_peak_of_the_window_not_its_mean():
    activation = np.zeros(50)
    activation[10] = 0.8

    scores, counts = aggregate_at_beats(activation, np.array([FRAME_SEC * 11]),
                                        FRAME_SEC, FRAME_SEC)

    assert scores[0] == pytest.approx(0.8)
    assert counts[0] == AGG_HI_FRAMES - AGG_LO_FRAMES + 1


@pytest.mark.parametrize("jitter_ms", [-50, -25, 0, 25, 50])
def test_the_aggregation_window_absorbs_beat_timing_jitter(jitter_ms):
    frame_sec = 0.04644                      # the real mel grid, where it matters
    activation = np.zeros(400)
    downbeats = np.arange(20, 380, 32)
    activation[downbeats] = 0.9
    jittered = frame_sec * (downbeats + 1) + jitter_ms / 1000.0

    scores, _counts = aggregate_at_beats(activation, jittered, frame_sec, frame_sec)

    assert np.all(scores == pytest.approx(0.9))


def test_a_candidate_with_no_frames_scores_nan_rather_than_zero():
    scores, counts = aggregate_at_beats(np.zeros(50), np.array([1000.0]),
                                        FRAME_SEC, FRAME_SEC)

    assert np.isnan(scores[0])
    assert counts[0] == 0


@pytest.mark.parametrize("offset", range(BEATS_PER_BAR))
def test_a_clean_activation_recovers_the_true_phase_from_any_start(offset):
    times = beat_grid(64)
    scores = clean_scores(64, offset)

    decisions = BarPhaseHMM().decode(times, scores)

    assert len(decisions) == 64
    expected = [1 + (index - offset) % BEATS_PER_BAR for index in range(64)]
    assert phases_of(decisions) == expected


def test_phase_advances_cyclically_when_nothing_forces_a_flip():
    decisions = BarPhaseHMM().decode(beat_grid(40), clean_scores(40))

    assert phase_flips(decisions) == 0
    for previous, current in zip(decisions, decisions[1:]):
        assert current.phase == previous.phase % BEATS_PER_BAR + 1


def test_downbeat_times_are_the_beat_instants_not_frame_centres():
    times = beat_grid(32, start=1.234567)
    decisions = BarPhaseHMM().decode(times, clean_scores(32))

    predicted = downbeat_times(decisions)

    assert np.allclose(predicted, times[::BEATS_PER_BAR])
    assert predicted.dtype == np.float64


def test_a_half_bar_ambiguous_track_does_not_alternate():
    times = beat_grid(80)
    scores = ambiguous_scores(80)

    decisions = BarPhaseHMM().decode(times, scores)

    assert phase_flips(decisions) == 0
    assert len(downbeat_times(decisions)) == pytest.approx(80 / BEATS_PER_BAR, abs=1)


def test_zero_flip_penalty_lets_adversarial_evidence_alternate():
    times = beat_grid(80)
    scores = ambiguous_scores(80)

    free = BarPhaseHMM(PhaseParams(flip_penalty=0.0)).decode(times, scores)
    held = BarPhaseHMM(PhaseParams(flip_penalty=DEFAULT_FLIP_PENALTY)).decode(times, scores)

    assert phase_flips(free) > 0
    assert phase_flips(held) == 0


def test_a_single_anomalous_beat_cannot_move_the_grid():
    times = beat_grid(64)
    scores = clean_scores(64)
    scores[17] = 0.99

    decisions = BarPhaseHMM().decode(times, scores)

    assert phases_of(decisions) == [1 + index % BEATS_PER_BAR for index in range(64)]


def test_a_sustained_reharmonisation_is_followed_after_paying_the_penalty():
    times = beat_grid(96)
    scores = np.concatenate([clean_scores(48, offset=0), clean_scores(48, offset=2)])

    decisions = BarPhaseHMM(PhaseParams(flip_penalty=2.0)).decode(times, scores)

    assert phase_flips(decisions) == 1
    assert phases_of(decisions)[-8:] == [1 + (index - 2) % BEATS_PER_BAR
                                         for index in range(88, 96)]


def test_a_flip_must_pay_for_itself_inside_the_look_ahead():
    times = beat_grid(96)
    scores = np.concatenate([clean_scores(48, offset=0), clean_scores(48, offset=2)])
    # 5.1 nats per bar against a 14-nat penalty: one bar cannot fund the flip, three can.
    short = BarPhaseHMM(PhaseParams(lag_beats=4)).decode(times, scores)
    long = BarPhaseHMM(PhaseParams(lag_beats=12)).decode(times, scores)

    assert phase_flips(short) == 0
    assert phase_flips(long) == 1


def test_nothing_commits_until_the_lag_is_full():
    decoder = BarPhaseHMM(PhaseParams(lag_beats=5))
    times = beat_grid(12)
    scores = clean_scores(12)

    emitted = [len(decoder.push(times[i], scores[i])) for i in range(12)]

    assert emitted[:5] == [0, 0, 0, 0, 0]
    assert sum(emitted) == 12 - 5
    assert len(decoder.flush()) == 5


def test_flush_is_idempotent():
    decoder = BarPhaseHMM()
    for time, score in zip(beat_grid(20), clean_scores(20)):
        decoder.push(time, score)

    first = decoder.flush()

    assert first
    assert decoder.flush() == []


def test_a_committed_decision_survives_an_adversarial_future():
    times = beat_grid(120)
    honest = clean_scores(120)
    params = PhaseParams(lag_beats=DEFAULT_LAG_BEATS)

    decoder = BarPhaseHMM(params)
    committed = []
    for index in range(60):
        committed.extend(decoder.push(times[index], honest[index]))
    hostile = clean_scores(120, offset=2, high=0.999, low=0.001)
    for index in range(60, 120):
        decoder.push(times[index], hostile[index])
    decoder.flush()

    reference = BarPhaseHMM(params).decode(times[:60], honest[:60])
    assert committed == reference[:len(committed)]


@pytest.mark.parametrize("prefix", [20, 40, 60, 90])
def test_a_longer_track_never_changes_what_a_shorter_one_already_committed(prefix):
    rng = np.random.default_rng(5)
    times = beat_grid(120)
    scores = np.clip(clean_scores(120) + rng.normal(0, 0.25, 120), 0.01, 0.99)
    settled = len(times) - DEFAULT_LAG_BEATS

    short = BarPhaseHMM().decode(times[:prefix], scores[:prefix])
    long = BarPhaseHMM().decode(times, scores)

    overlap = min(prefix - DEFAULT_LAG_BEATS, settled)
    assert overlap > 0
    assert short[:overlap] == long[:overlap]


def test_reset_makes_a_used_decoder_indistinguishable_from_a_fresh_one():
    times = beat_grid(40)
    scores = clean_scores(40, offset=1)

    decoder = BarPhaseHMM()
    decoder.decode(beat_grid(25, start=90.0), clean_scores(25, offset=3))
    decoder.reset()
    reused = [d.phase for d in decoder.decode(times, scores)]

    assert reused == phases_of(BarPhaseHMM().decode(times, scores))


@pytest.mark.parametrize("dropped", [1, 2, 3, 4, 7])
def test_phase_survives_a_dropout_and_the_gap_is_coasted(dropped):
    n_beats = 64
    times = beat_grid(n_beats)
    scores = clean_scores(n_beats)
    keep = np.ones(n_beats, dtype=bool)
    keep[20:20 + dropped] = False

    decisions = BarPhaseHMM().decode(times[keep], scores[keep])

    real = [d for d in decisions if not d.virtual]
    assert [d.phase for d in real] == [1 + index % BEATS_PER_BAR
                                       for index in np.flatnonzero(keep)]
    assert sum(1 for d in decisions if d.virtual) == dropped


def test_a_coasted_beat_lands_on_the_tempo_grid():
    times = beat_grid(48)
    keep = np.ones(48, dtype=bool)
    keep[16:19] = False

    decisions = BarPhaseHMM().decode(times[keep], clean_scores(48)[keep])

    coasted = np.array([d.time for d in decisions if d.virtual])
    assert np.allclose(coasted, times[16:19], atol=1e-9)


def test_a_gap_too_long_to_coast_is_a_discontinuity_not_a_phantom_bar_grid():
    times = np.concatenate([beat_grid(16), beat_grid(16, start=600.0)])

    decisions = BarPhaseHMM().decode(times, np.tile(clean_scores(16), 2))

    assert not any(d.virtual for d in decisions)
    assert len(decisions) == 32


def test_the_longest_coastable_gap_is_still_coasted():
    tempo = TEMPO_SEC
    gap = tempo * MAX_COAST_BEATS
    times = np.concatenate([beat_grid(8), beat_grid(8, start=1.0 + 7 * tempo + gap)])

    decisions = BarPhaseHMM().decode(times, np.tile(clean_scores(8), 2))

    assert sum(1 for d in decisions if d.virtual) == MAX_COAST_BEATS - 1


def test_the_phase_re_locks_after_a_discontinuity():
    first = beat_grid(48)
    second = beat_grid(48, tempo=0.5, start=900.0)
    confident = dict(high=0.9999, low=0.0001)
    scores = np.concatenate([clean_scores(48, **confident),
                             clean_scores(48, offset=1, **confident)])

    decisions = BarPhaseHMM().decode(np.concatenate([first, second]), scores)

    tail = [d.phase for d in decisions[-8:]]
    assert tail == [1 + (index - 1) % BEATS_PER_BAR for index in range(88, 96)]


def test_a_double_tempo_warmup_does_not_capture_the_tempo_estimate():
    warmup = 1.0 + 0.5 * TEMPO_SEC * np.arange(12)
    real = warmup[-1] + TEMPO_SEC * np.arange(1, 81)
    times = np.concatenate([warmup, real])

    decisions = BarPhaseHMM().decode(times, clean_scores(times.size))

    virtual = sum(1 for d in decisions if d.virtual)
    assert virtual <= 2 * MAX_COAST_BEATS
    bars = np.diff(downbeat_times(decisions))
    assert np.median(bars) == pytest.approx(BEATS_PER_BAR * TEMPO_SEC, rel=0.05)


def test_an_ordinary_beat_stream_coasts_nothing():
    decisions = BarPhaseHMM().decode(beat_grid(60), clean_scores(60))

    assert not any(d.virtual for d in decisions)


def test_jittered_beats_do_not_trigger_coasting():
    rng = np.random.default_rng(7)
    times = beat_grid(80) + rng.uniform(-0.05, 0.05, 80)
    times = np.sort(times)

    decisions = BarPhaseHMM().decode(times, clean_scores(80))

    assert not any(d.virtual for d in decisions)


def test_a_nan_scored_beat_keeps_the_phase_advancing():
    times = beat_grid(40)
    scores = clean_scores(40)
    scores[12:20] = np.nan

    decisions = BarPhaseHMM().decode(times, scores)

    assert phases_of(decisions) == [1 + index % BEATS_PER_BAR for index in range(40)]


def test_an_entirely_evidence_free_track_still_produces_a_legal_grid():
    decisions = BarPhaseHMM().decode(beat_grid(32), np.full(32, np.nan))

    assert len(decisions) == 32
    assert phase_flips(decisions) == 0


def test_confidence_is_a_bounded_distribution_over_the_four_phases():
    decisions = BarPhaseHMM().decode(beat_grid(40), clean_scores(40))

    values = np.array([d.confidence for d in decisions])
    assert np.all(values > 1.0 / BEATS_PER_BAR - 1e-9)
    assert np.all(values <= 1.0 + 1e-9)


def test_confidence_collapses_to_chance_when_there_is_no_evidence():
    clean = BarPhaseHMM().decode(beat_grid(40), clean_scores(40))
    blind = BarPhaseHMM().decode(beat_grid(40), np.full(40, np.nan))

    assert np.mean([d.confidence for d in blind]) == pytest.approx(1.0 / BEATS_PER_BAR)
    assert np.mean([d.confidence for d in clean]) > 0.9


def test_confidence_reads_a_half_bar_ambiguity_as_a_two_way_tie():
    ambiguous = BarPhaseHMM().decode(beat_grid(60), ambiguous_scores(60))

    middle = [d.confidence for d in ambiguous[8:-8]]
    assert np.mean(middle) == pytest.approx(0.5, abs=0.05)


def test_subdivision_two_gives_the_bar_eight_positions():
    dense = candidate_grid(beat_grid(64), 2)
    scores = np.full(len(dense), 0.05)
    scores[::8] = 0.95

    decisions = BarPhaseHMM(PhaseParams(subdivision=2)).decode(dense, scores)

    assert phases_of(decisions) == [1 + index % 8 for index in range(len(dense))]
    assert phase_flips(decisions, subdivision=2) == 0


def test_a_half_beat_locked_beat_stream_is_recovered_at_subdivision_two():
    beats = beat_grid(64)
    truth = beats + TEMPO_SEC / 2.0
    dense = candidate_grid(beats, 2)
    scores = np.where(np.isin(np.round(dense, 9), np.round(truth[::4], 9)), 0.95, 0.05)

    on_beat = BarPhaseHMM(PhaseParams(subdivision=1)).decode(beats, np.full(64, 0.05))
    half = BarPhaseHMM(PhaseParams(subdivision=2)).decode(dense, scores)

    assert match_rate(downbeat_times(on_beat), truth[::4]) == 0.0
    assert match_rate(downbeat_times(half), truth[::4]) > 0.9


def test_bar_phase_names_the_beats_and_refuses_to_name_the_half_beats():
    assert [bar_phase(p, 1) for p in range(1, 5)] == [1, 2, 3, 4]
    assert [bar_phase(p, 2) for p in range(1, 9)] == [1, 0, 2, 0, 3, 0, 4, 0]


def test_phase_flips_needs_the_subdivision_it_was_decoded_at():
    dense = candidate_grid(beat_grid(32), 2)
    scores = np.full(len(dense), 0.05)
    scores[::8] = 0.95
    decisions = BarPhaseHMM(PhaseParams(subdivision=2)).decode(dense, scores)

    assert phase_flips(decisions, subdivision=2) == 0
    assert phase_flips(decisions, subdivision=1) > 0


def test_an_unmeasured_subdivision_is_refused_by_the_decoder():
    with pytest.raises(ValueError, match="subdivision"):
        BarPhaseHMM(PhaseParams(subdivision=4))


def test_refinement_moves_an_instant_onto_the_activation_peak():
    activation = np.zeros(200)
    activation[40] = 0.6
    activation[41] = 1.0
    activation[42] = 0.8
    decisions = [d for d in BarPhaseHMM().decode(
        np.array([FRAME_SEC * 42.0]), np.array([1.0]))]

    refined = refine_instants(decisions, activation, FRAME_SEC, FRAME_SEC)

    assert refined[0].time > FRAME_SEC * 42.0
    assert refined[0].time == pytest.approx(FRAME_SEC * 42.0, abs=0.5 * FRAME_SEC)


def test_refinement_changes_nothing_but_the_time():
    times = beat_grid(40)
    scores = clean_scores(40)
    activation = np.zeros(500)
    activation[nearest_frames(times[::4], 500, FRAME_SEC, FRAME_SEC)] = 0.9
    decisions = BarPhaseHMM().decode(times, scores)

    refined = refine_instants(decisions, activation, FRAME_SEC, FRAME_SEC)

    assert [d._replace(time=0.0) for d in refined] == \
           [d._replace(time=0.0) for d in decisions]


def test_refinement_cannot_move_an_instant_outside_its_own_evidence_window():
    monotone_rising_activation = np.linspace(0.0, 1.0, 500)
    times = beat_grid(20)
    decisions = BarPhaseHMM().decode(times, np.full(20, 0.5))

    refined = refine_instants(decisions, monotone_rising_activation, FRAME_SEC, FRAME_SEC)

    moved = np.abs(np.array([d.time for d in refined]) - times)
    assert np.all(moved <= AGG_HI_FRAMES * FRAME_SEC + 1e-9)


def test_beat_times_must_increase():
    with pytest.raises(ValueError, match="increasing"):
        BarPhaseHMM().decode(np.array([1.0, 2.0, 1.5]), np.full(3, 0.5))


def test_scores_must_match_the_beats():
    with pytest.raises(ValueError, match="score"):
        BarPhaseHMM().decode(beat_grid(10), np.full(9, 0.5))


def test_an_empty_beat_stream_decodes_to_nothing():
    assert BarPhaseHMM().decode(np.zeros(0), np.zeros(0)) == []


def test_a_negative_flip_penalty_is_refused():
    with pytest.raises(ValueError, match="flip_penalty"):
        BarPhaseHMM(PhaseParams(flip_penalty=-1.0))


def test_two_decodes_of_one_input_agree_exactly():
    times = beat_grid(200)
    rng = np.random.default_rng(11)
    scores = np.clip(clean_scores(200) + rng.normal(0, 0.2, 200), 0.0, 1.0)

    first = BarPhaseHMM().decode(times, scores)
    second = BarPhaseHMM().decode(times, scores)

    assert first == second


def test_the_decode_path_imports_without_torch():
    probe = (
        "import sys, importlib;"
        "sys.modules['torch'] = None;"
        "importlib.import_module('nn.downbeat_decoder');"
        "assert 'nn.dataset' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(TRAINING_DIR),
                            capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_decode_track_at_subdivision_two_builds_its_own_candidates(tmp_path):
    beats = np.arange(1.0, 20.0, TEMPO_SEC)
    activation = np.zeros(500, dtype=np.float32)
    midpoint_downbeats = 0.5 * (beats[:-1] + beats[1:])[::BEATS_PER_BAR]
    activation[nearest_frames(midpoint_downbeats, 500, FRAME_SEC, FRAME_SEC)] = 0.95
    path = tmp_path / "track.npz"
    np.savez(path, activation=activation, frame_sec=np.float64(FRAME_SEC),
             t0=np.float64(FRAME_SEC), live_beat_time=beats,
             live_beat_score=np.full(len(beats), 0.02))

    on_beat = decode_track(path, "live", PhaseParams(subdivision=1))
    half = decode_track(path, "live", PhaseParams(subdivision=2))

    assert match_rate(downbeat_times(on_beat), midpoint_downbeats) == 0.0
    assert match_rate(downbeat_times(half), midpoint_downbeats) > 0.9


def write_sidecar(path, **arrays):
    base = {"activation": np.zeros(600, dtype=np.float32),
            "frame_sec": np.float64(FRAME_SEC), "t0": np.float64(FRAME_SEC)}
    np.savez(path, **{**base, **arrays})
    return path


def test_decode_track_reads_a_sidecar_and_decodes_the_named_condition(tmp_path):
    times = beat_grid(48)
    path = write_sidecar(tmp_path / "track.npz", live_beat_time=times,
                         live_beat_score=clean_scores(48),
                         expert_beat_time=times[::2],
                         expert_beat_score=np.full(24, np.nan))

    decisions = decode_track(path, "live")

    assert phases_of(decisions) == [1 + index % BEATS_PER_BAR for index in range(48)]
    assert len(decode_track(path, "expert")) == 24


def test_decode_track_names_an_unknown_condition(tmp_path):
    path = write_sidecar(tmp_path / "track.npz", live_beat_time=np.zeros(1),
                         live_beat_score=np.zeros(1))

    with pytest.raises(KeyError, match="madmom"):
        decode_track(path, "madmom")
