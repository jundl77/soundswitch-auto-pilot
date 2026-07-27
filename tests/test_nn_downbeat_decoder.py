"""Tests for the bar-phase decoder (``training/nn/downbeat_decoder.py``).

The decoder is the component that turns a *ranking* of beats into a bar grid, so
almost every defect it can have is a defect that still produces plausible output:
a phase that alternates 1-3-1-3 forever, a commit that quietly changes after it
was emitted, a dropout that slides the whole grid by a beat.  None of those raise
and none of them look wrong in isolation, so the tests here are properties rather
than examples:

**Phase is cyclic and the cycle is the answer.**  Fed a clean activation the
decoder must recover the true offset from all four possible starting phases -- a
decoder that always answers "phase 1 first" would pass a single-example test.

**The flip penalty is the lever, and it is tested as one.**  A half-bar-ambiguous
activation (equal mass on beats 1 and 3, which is Task 2's measured error mode)
must decode to a *consistent* phase at the shipped penalty and must be *allowed*
to alternate at zero penalty.  Only the pair proves the knob is connected.

**Immutability is a structural claim, so it is tested structurally.**  Committed
decisions are re-checked against a decode whose future is replaced by adversarial
data: if any of them moves, the fixed-lag pruning is not doing what the name says.

**Coasting is tested by deletion.**  Beats are removed from a known-good stream
and the phase after the gap must still be right, which is the only way to see
that the tempo estimate carried the phase rather than luck.

**The half-beat grid is tested against the failure it exists for.**  A beat
stream locked half a beat off the music is unrecoverable on its own grid and
ordinary on the eight-position one, so the pair of decodes is the test -- a
single "it produces plausible output" assertion would pass either way.

Pure numpy, no corpus, no torch -- deliberately, because the decode path must
stay importable by a process that has neither.
"""
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
FRAME_SEC = 0.05              # a round stand-in; the arithmetic is what is tested


# --------------------------------------------------------------------------- #
# Synthetic beat streams
# --------------------------------------------------------------------------- #


def beat_grid(n_beats: int, tempo: float = TEMPO_SEC, start: float = 1.0) -> np.ndarray:
    return start + tempo * np.arange(n_beats, dtype=np.float64)


def clean_scores(n_beats: int, offset: int = 0, high: float = 0.90,
                 low: float = 0.05) -> np.ndarray:
    """High activation on every downbeat, low everywhere else.

    ``offset`` is the index of the first downbeat, so the four values exercise
    every starting phase a track can open on (254 corpus grids start mid-bar).
    """
    index = np.arange(n_beats)
    return np.where((index - offset) % BEATS_PER_BAR == 0, high, low)


def ambiguous_scores(n_beats: int, offset: int = 0, high: float = 0.90,
                     low: float = 0.05) -> np.ndarray:
    """Task 2's measured error mode: equal mass on beats 1 and 3."""
    index = np.arange(n_beats)
    return np.where((index - offset) % (BEATS_PER_BAR // 2) == 0, high, low)


def phases_of(decisions) -> list:
    return [d.phase for d in decisions]


def match_rate(predicted, reference, tolerance: float = 0.070) -> float:
    """Share of ``reference`` instants with a prediction inside the tolerance."""
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if not reference.size:
        return 0.0
    if not predicted.size:
        return 0.0
    distance = np.abs(reference[:, None] - predicted[None, :]).min(axis=1)
    return float(np.mean(distance <= tolerance))


# --------------------------------------------------------------------------- #
# Candidate instants and their evidence
# --------------------------------------------------------------------------- #


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
    """Task 1's contract: frame 0 pools the audio in ``(t0 - frame_sec, t0]``,
    which is exactly where such a beat lives.  ``-1`` means past the end and
    nothing else."""
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
    """Aubio's instants wobble against the annotator's by tens of milliseconds.
    A window that only finds the peak on an exact instant is a window that only
    works on the expert grid."""
    frame_sec = 0.04644                      # the real mel grid, where it matters
    activation = np.zeros(400)
    downbeats = np.arange(20, 380, 32)
    activation[downbeats] = 0.9
    jittered = frame_sec * (downbeats + 1) + jitter_ms / 1000.0

    scores, _counts = aggregate_at_beats(activation, jittered, frame_sec, frame_sec)

    assert np.all(scores == pytest.approx(0.9))


def test_a_candidate_with_no_frames_scores_nan_rather_than_zero():
    """No evidence is not evidence of no downbeat -- the decoder tells them
    apart, and NaN is how."""
    scores, counts = aggregate_at_beats(np.zeros(50), np.array([1000.0]),
                                        FRAME_SEC, FRAME_SEC)

    assert np.isnan(scores[0])
    assert counts[0] == 0


# --------------------------------------------------------------------------- #
# The cycle
# --------------------------------------------------------------------------- #


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
    """The decoder's whole point over naive peak picking: it emits the beat's own
    continuous instant, so nothing is quantised to the 46 ms mel grid."""
    times = beat_grid(32, start=1.234567)
    decisions = BarPhaseHMM().decode(times, clean_scores(32))

    predicted = downbeat_times(decisions)

    assert np.allclose(predicted, times[::BEATS_PER_BAR])
    assert predicted.dtype == np.float64


# --------------------------------------------------------------------------- #
# The flip penalty -- the lever Task 2's error analysis named
# --------------------------------------------------------------------------- #


def test_a_half_bar_ambiguous_track_does_not_alternate():
    """52 % of the head's false positives land on phase 3 (task-2 §7b).  The
    cyclic transition structure is what stops 1-3-1-3 from being a legal read."""
    times = beat_grid(80)
    scores = ambiguous_scores(80)

    decisions = BarPhaseHMM().decode(times, scores)

    assert phase_flips(decisions) == 0
    # It must pick one of the two half-bar offsets and hold it, not both.
    assert len(downbeat_times(decisions)) == pytest.approx(80 / BEATS_PER_BAR, abs=1)


def test_zero_flip_penalty_lets_adversarial_evidence_alternate():
    """The companion to the test above: with the penalty removed the decoder
    *is* free to chase the local evidence, so the consistency above is the knob
    doing work rather than a structural impossibility."""
    times = beat_grid(80)
    scores = ambiguous_scores(80)

    free = BarPhaseHMM(PhaseParams(flip_penalty=0.0)).decode(times, scores)
    held = BarPhaseHMM(PhaseParams(flip_penalty=DEFAULT_FLIP_PENALTY)).decode(times, scores)

    assert phase_flips(free) > 0
    assert phase_flips(held) == 0


def test_a_single_anomalous_beat_cannot_move_the_grid():
    times = beat_grid(64)
    scores = clean_scores(64)
    scores[17] = 0.99            # one off-beat shouting louder than the downbeats

    decisions = BarPhaseHMM().decode(times, scores)

    assert phases_of(decisions) == [1 + index % BEATS_PER_BAR for index in range(64)]


def test_a_sustained_reharmonisation_is_followed_after_paying_the_penalty():
    """The penalty is hysteresis, not a lock: evidence that persists must win, or
    a deck transition would strand the grid for the rest of the set."""
    times = beat_grid(96)
    scores = np.concatenate([clean_scores(48, offset=0), clean_scores(48, offset=2)])

    decisions = BarPhaseHMM(PhaseParams(flip_penalty=2.0)).decode(times, scores)

    assert phase_flips(decisions) == 1
    assert phases_of(decisions)[-8:] == [1 + (index - 2) % BEATS_PER_BAR
                                         for index in range(88, 96)]


def test_a_flip_must_pay_for_itself_inside_the_look_ahead():
    """``flip_penalty`` and ``lag_beats`` are coupled, and it is not obvious.

    A commit is final, so evidence *after* the look-ahead window can never fund a
    flip however long it persists: the decoder can only act on what it has seen.
    A penalty above roughly one look-ahead window's worth of contrast therefore
    reads as "never flip", and the same penalty at a longer lag reads as "follow
    the music".  Task 4 has to sweep the pair, not either alone.
    """
    times = beat_grid(96)
    scores = np.concatenate([clean_scores(48, offset=0), clean_scores(48, offset=2)])
    # 5.1 nats of contrast per bar against a 14-nat penalty: one bar of
    # look-ahead cannot pay for it, three bars can.
    short = BarPhaseHMM(PhaseParams(lag_beats=4)).decode(times, scores)
    long = BarPhaseHMM(PhaseParams(lag_beats=12)).decode(times, scores)

    assert phase_flips(short) == 0
    assert phase_flips(long) == 1


# --------------------------------------------------------------------------- #
# Fixed lag and immutability
# --------------------------------------------------------------------------- #


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
    """The structural claim: once a decision is emitted, no later audio can
    revise it.  Replacing the future with the *opposite* phase evidence is the
    strongest form of the test -- a decoder that merely read the backtrace
    without pruning would happily change its mind here."""
    times = beat_grid(120)
    honest = clean_scores(120)
    params = PhaseParams(lag_beats=DEFAULT_LAG_BEATS)

    decoder = BarPhaseHMM(params)
    committed = []
    for index in range(60):
        committed.extend(decoder.push(times[index], honest[index]))
    # Everything from here on argues for a half-bar shift, loudly and forever.
    hostile = clean_scores(120, offset=2, high=0.999, low=0.001)
    for index in range(60, 120):
        decoder.push(times[index], hostile[index])
    decoder.flush()

    reference = BarPhaseHMM(params).decode(times[:60], honest[:60])
    # Whole decisions, not just phases: the confidence must not move either, or
    # a consumer reading it would see a committed beat change its mind.
    assert committed == reference[:len(committed)]


@pytest.mark.parametrize("prefix", [20, 40, 60, 90])
def test_a_longer_track_never_changes_what_a_shorter_one_already_committed(prefix):
    """Prefix stability as a property, over an activation with real ambiguity in
    it -- the case where two decodes could plausibly diverge."""
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


# --------------------------------------------------------------------------- #
# Coasting through aubio's dropouts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dropped", [1, 2, 3, 4, 7])
def test_phase_survives_a_dropout_and_the_gap_is_coasted(dropped):
    """Aubio loses beats under heavy sidechain compression.  The phase after the
    gap must still be the true one, and the missing instants must be filled by
    tempo so the bar grid does not simply stop."""
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
    """A gap of minutes is a stopped deck, not a dropout.  Coasting it would
    manufacture downbeats out of silence -- every one of them a false positive --
    so past the cap the decoder emits nothing and re-locks on the far side."""
    times = np.concatenate([beat_grid(16), beat_grid(16, start=600.0)])

    decisions = BarPhaseHMM().decode(times, np.tile(clean_scores(16), 2))

    assert not any(d.virtual for d in decisions)
    assert len(decisions) == 32


def test_the_longest_coastable_gap_is_still_coasted():
    """The cap is a threshold, so its inside edge is worth pinning: one beat
    fewer than the cap must still produce a filled grid."""
    tempo = TEMPO_SEC
    gap = tempo * MAX_COAST_BEATS       # MAX_COAST_BEATS - 1 missing beats
    times = np.concatenate([beat_grid(8), beat_grid(8, start=1.0 + 7 * tempo + gap)])

    decisions = BarPhaseHMM().decode(times, np.tile(clean_scores(8), 2))

    assert sum(1 for d in decisions if d.virtual) == MAX_COAST_BEATS - 1


def test_the_phase_re_locks_after_a_discontinuity():
    """The deck-transition proxy, at the shipped defaults: a new grid at a new
    tempo and a new phase is picked up when the evidence for it is strong enough
    to pay the penalty inside the look-ahead."""
    first = beat_grid(48)
    second = beat_grid(48, tempo=0.5, start=900.0)
    confident = dict(high=0.9999, low=0.0001)
    scores = np.concatenate([clean_scores(48, **confident),
                             clean_scores(48, offset=1, **confident)])

    decisions = BarPhaseHMM().decode(np.concatenate([first, second]), scores)

    tail = [d.phase for d in decisions[-8:]]
    assert tail == [1 + (index - 1) % BEATS_PER_BAR for index in range(88, 96)]


def test_an_ordinary_beat_stream_coasts_nothing():
    decisions = BarPhaseHMM().decode(beat_grid(60), clean_scores(60))

    assert not any(d.virtual for d in decisions)


def test_jittered_beats_do_not_trigger_coasting():
    """Aubio's instants wobble; a 10 % tempo wobble is not a dropout."""
    rng = np.random.default_rng(7)
    times = beat_grid(80) + rng.uniform(-0.05, 0.05, 80)
    times = np.sort(times)

    decisions = BarPhaseHMM().decode(times, clean_scores(80))

    assert not any(d.virtual for d in decisions)


# --------------------------------------------------------------------------- #
# Evidence that is not there
# --------------------------------------------------------------------------- #


def test_a_beat_with_no_evidence_keeps_the_phase_advancing():
    """``NaN`` is what the aggregation emits for a beat past the end of the mel
    or with no covered frame: no evidence, not evidence of no downbeat."""
    times = beat_grid(40)
    scores = clean_scores(40)
    scores[12:20] = np.nan

    decisions = BarPhaseHMM().decode(times, scores)

    assert phases_of(decisions) == [1 + index % BEATS_PER_BAR for index in range(40)]


def test_an_entirely_evidence_free_track_still_produces_a_legal_grid():
    decisions = BarPhaseHMM().decode(beat_grid(32), np.full(32, np.nan))

    assert len(decisions) == 32
    assert phase_flips(decisions) == 0


# --------------------------------------------------------------------------- #
# Phase confidence
# --------------------------------------------------------------------------- #


def test_confidence_is_a_bounded_distribution_over_the_four_phases():
    decisions = BarPhaseHMM().decode(beat_grid(40), clean_scores(40))

    values = np.array([d.confidence for d in decisions])
    assert np.all(values > 1.0 / BEATS_PER_BAR - 1e-9)
    assert np.all(values <= 1.0 + 1e-9)


def test_confidence_collapses_to_chance_when_there_is_no_evidence():
    """The engine's documented fallback -- beat-snapping when the grid is unsure
    -- needs this number to actually drop, not just to exist."""
    clean = BarPhaseHMM().decode(beat_grid(40), clean_scores(40))
    blind = BarPhaseHMM().decode(beat_grid(40), np.full(40, np.nan))

    assert np.mean([d.confidence for d in blind]) == pytest.approx(1.0 / BEATS_PER_BAR)
    assert np.mean([d.confidence for d in clean]) > 0.9


def test_confidence_reads_a_half_bar_ambiguity_as_a_two_way_tie():
    """The decoder still has to *pick* on a 1-3-1-3 track, and the number that
    says the pick was a coin flip is the one the engine needs -- a decode that
    was right for the wrong reason must not look identical to a confident one."""
    ambiguous = BarPhaseHMM().decode(beat_grid(60), ambiguous_scores(60))

    middle = [d.confidence for d in ambiguous[8:-8]]
    assert np.mean(middle) == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------- #
# The half-beat state space
# --------------------------------------------------------------------------- #


def test_subdivision_two_gives_the_bar_eight_positions():
    dense = candidate_grid(beat_grid(64), 2)
    scores = np.full(len(dense), 0.05)
    scores[::8] = 0.95                      # a downbeat every eight candidates

    decisions = BarPhaseHMM(PhaseParams(subdivision=2)).decode(dense, scores)

    assert phases_of(decisions) == [1 + index % 8 for index in range(len(dense))]
    assert phase_flips(decisions, subdivision=2) == 0


def test_a_half_beat_locked_beat_stream_is_recovered_at_subdivision_two():
    """Aubio's measured failure: every emitted beat sits half a beat off the
    music.  On its own grid the downbeat is unreachable; on the half-beat grid it
    is simply an odd-numbered state the decoder can sit in."""
    beats = beat_grid(64)
    truth = beats + TEMPO_SEC / 2.0                     # where the bars really are
    dense = candidate_grid(beats, 2)
    # Evidence lands on the true downbeats, i.e. on midpoints of the beat stream.
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
    assert phase_flips(decisions, subdivision=1) > 0     # loud, not silent


def test_an_unmeasured_subdivision_is_refused_by_the_decoder():
    with pytest.raises(ValueError, match="subdivision"):
        BarPhaseHMM(PhaseParams(subdivision=4))


# --------------------------------------------------------------------------- #
# Continuous instant refinement (a separate, togglable effect)
# --------------------------------------------------------------------------- #


def test_refinement_moves_an_instant_onto_the_activation_peak():
    """The head's peaks are quantised to the mel grid; a third of the +-70 ms
    budget is spent there before the decoder acts."""
    activation = np.zeros(200)
    activation[40] = 0.6
    activation[41] = 1.0
    activation[42] = 0.8                      # true peak just past frame 41
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
    activation = np.linspace(0.0, 1.0, 500)   # monotone: the peak is always right
    times = beat_grid(20)
    decisions = BarPhaseHMM().decode(times, np.full(20, 0.5))

    refined = refine_instants(decisions, activation, FRAME_SEC, FRAME_SEC)

    moved = np.abs(np.array([d.time for d in refined]) - times)
    assert np.all(moved <= AGG_HI_FRAMES * FRAME_SEC + 1e-9)


# --------------------------------------------------------------------------- #
# Input contract
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_two_decodes_of_one_input_agree_exactly():
    times = beat_grid(200)
    rng = np.random.default_rng(11)
    scores = np.clip(clean_scores(200) + rng.normal(0, 0.2, 200), 0.0, 1.0)

    first = BarPhaseHMM().decode(times, scores)
    second = BarPhaseHMM().decode(times, scores)

    assert first == second


def test_the_decode_path_imports_without_torch():
    """`lib/` will import this module when the runtime plan wires it in, and torch
    is an optional extra.  A transitive import through ``downbeat_train`` or
    ``downbeat_dataset`` would not fail any other test in this file."""
    probe = (
        "import sys, importlib;"
        "sys.modules['torch'] = None;"
        "importlib.import_module('nn.downbeat_decoder');"
        "assert 'nn.dataset' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(TRAINING_DIR),
                            capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# The sidecar adapter
# --------------------------------------------------------------------------- #


def test_decode_track_at_subdivision_two_builds_its_own_candidates(tmp_path):
    """The half-beat grid is aggregated off the stored curve at decode time, so a
    subdivision-2 decode needs no second inference pass over the corpus."""
    beats = np.arange(1.0, 20.0, TEMPO_SEC)
    activation = np.zeros(500, dtype=np.float32)
    # Put the real downbeats on the MIDPOINTS -- aubio's measured failure mode.
    downbeats = 0.5 * (beats[:-1] + beats[1:])[::BEATS_PER_BAR]
    activation[nearest_frames(downbeats, 500, FRAME_SEC, FRAME_SEC)] = 0.95
    path = tmp_path / "track.npz"
    np.savez(path, activation=activation, frame_sec=np.float64(FRAME_SEC),
             t0=np.float64(FRAME_SEC), aubio_beat_time=beats,
             aubio_beat_score=np.full(len(beats), 0.02))

    on_beat = decode_track(path, "aubio", PhaseParams(subdivision=1))
    half = decode_track(path, "aubio", PhaseParams(subdivision=2))

    # The on-beat grid cannot reach the downbeats at all; the half-beat one can.
    assert match_rate(downbeat_times(on_beat), downbeats) == 0.0
    assert match_rate(downbeat_times(half), downbeats) > 0.9


def write_sidecar(path, **arrays):
    """A stand-in with the geometry every real sidecar carries."""
    base = {"activation": np.zeros(600, dtype=np.float32),
            "frame_sec": np.float64(FRAME_SEC), "t0": np.float64(FRAME_SEC)}
    np.savez(path, **{**base, **arrays})
    return path


def test_decode_track_reads_a_sidecar_and_decodes_the_named_condition(tmp_path):
    times = beat_grid(48)
    path = write_sidecar(tmp_path / "track.npz", aubio_beat_time=times,
                         aubio_beat_score=clean_scores(48),
                         expert_beat_time=times[::2],
                         expert_beat_score=np.full(24, np.nan))

    decisions = decode_track(path, "aubio")

    assert phases_of(decisions) == [1 + index % BEATS_PER_BAR for index in range(48)]
    assert len(decode_track(path, "expert")) == 24


def test_decode_track_names_an_unknown_condition(tmp_path):
    path = write_sidecar(tmp_path / "track.npz", aubio_beat_time=np.zeros(1),
                         aubio_beat_score=np.zeros(1))

    with pytest.raises(KeyError, match="madmom"):
        decode_track(path, "madmom")
