"""Bar-phase decoder: beat instants + downbeat activation in, a bar grid out.

The downbeat head answers "does this instant sound like the start of a bar" and
Task 2 measured exactly how far that gets you: **94.6 % of its peaks are
beat-locked and 52 % of its false positives land on beat 3**.  The head has
solved *is this a beat* and not *which beat*, and no better activation fixes
that, because in four-on-the-floor beat 3 is acoustically the same event as beat
1.  What separates them is the *sequence*: a track cannot alternate 1-3-1-3
without paying for a phase flip over and over.  So this module is a cyclic HMM
over the bar, at the rate of whatever candidate instants it is given, and the
flip penalty is the knob that turns a ranking into a grid.

**The cycle is four positions or eight, and the difference is the whole live
condition.**  Fed aubio's beats it is four (one per beat).  But aubio's dominant
failure on this corpus is not jitter, it is a steady **half-beat lock**: an aubio
beat sits within +-70 ms of an expert downbeat only 51.8 % of the time on val,
and de-shifting each track by its own median offset recovers nothing, because the
offset is half a beat rather than a latency.  A four-state decoder can only round
that away.  Admitting the midpoint of every consecutive pair as a candidate makes
the bar an eight-position cycle, turns "aubio is half a beat off" into a state the
decoder can occupy and *hold*, and lifts the reachable ceiling from 51.8 % to
85.2 %.  Midpoints stay causal -- the midpoint of beats n and n+1 exists as soon
as beat n+1 arrives -- so the live discipline is unchanged.  ``subdivision`` is
the switch; ``candidate_grid`` builds the instants.

**One real lever, and the reason is structural.**  ``flip_penalty`` is the cost
of not advancing the phase; ``downbeat_ref`` is the activation level at which a
beat stops arguing against being a downbeat.  Only the first of the two does much
work, and the argument is worth writing down: every legal cyclic path assigns
phase 1 to the same number of beats (+-1 at the ends), so shifting the emission by
a constant -- which is all ``downbeat_ref`` does -- shifts every path's score by
the same amount and cannot change the argmax.  It survives as a knob because a
path *with* flips can carry one more or one fewer downbeat, so the effect is real
but second order.  Anything Task 4 wants to tune, it should tune on the penalty.

**The activation is a ranking score, and the emission only uses its ordering.**
``pos_weight`` 9.355 inflated the head's sigmoid during training, so it is not
P(downbeat).  The emission is the log-odds of the score relative to
``downbeat_ref`` -- and that is not a workaround, it is the same model: adding
``[log p, log(1-p), log(1-p), log(1-p)]`` to the trellis differs from adding
``[logit(p), 0, 0, 0]`` by a per-beat constant, which is shared across states and
therefore invisible to Viterbi.  Undoing the training-time prior shift
analytically (``deweighted``) is exactly a constant subtraction in log-odds
space, i.e. one particular choice of ``downbeat_ref``
(``pos_weight / (1 + pos_weight)``) -- so the transform is not skipped here, it
is *exposed* as the parameter it collapses to.

**Coasting, because aubio drops beats.**  Under heavy sidechain compression the
beat stream gaps, and a decoder that only advances on arrivals loses the bar.
When a gap is a whole multiple of the running tempo the missing instants are
interpolated and pushed as evidence-free beats, so the phase walks through the
dropout and the grid stays dense.  Past ``MAX_COAST_BEATS`` the gap is not a
dropout -- it is a stopped deck or a track boundary -- and filling it would
manufacture downbeats out of silence, so nothing is emitted and the phase
re-locks from the far side's evidence.

**Fixed lag, then frozen -- and the frontier is carried, not backtraced.**  A
light show cannot un-fire a strobe, so a decision ``lag_beats`` candidates behind
the arrival head is final.  The section decoder reaches that by reading a
backtrace and then setting every disagreeing trellis state to -inf.  This one
carries state instead: ``delta`` scores the oldest *uncommitted* candidate,
conditioned on every commit before it, and a commit rolls a ``(frontier position,
head position)`` table forward over the buffered look-ahead.

Three reasons, none of them "the other construction is broken" -- it was measured
over this chain on real val tracks and it decodes within ~0.005 F1 of this one:

* **It is exact by construction.**  ``table.max(axis=1)`` is the best score
  reachable with each position at the frontier *given everything already
  emitted*, so the decision maximises over exactly the paths that are still
  legal.  The -inf rule is exact within the surviving set, which is the same
  thing whenever the backtrace has converged and not quite the same thing when it
  has not.
* **It hands back the phase posterior for free**, which is what the confidence
  output is read off (see below).  Recovering that from a pruned backtrace means
  a second trellis.
* **It is cheap here**: ``lag`` operations on a 4x4 or 8x8 array per candidate,
  with no backtrace to walk and no per-commit pruning pass.

A warning against generalising *either* rule: how many states survive a -inf
prune is a property of the chain, not of the pattern, and it is not intuitable.
Measured, the section decoder's own 48 ``(class, dwell)`` states collapse to a
single live state in 24 % of commits at its frozen config -- far more often than
this four-state cyclic chain does.  Measure it on the chain in front of you.

**``flip_penalty`` and ``lag_beats`` are coupled, and the coupling is not
obvious.**  A commit is final, so evidence arriving *after* the look-ahead window
can never fund a flip, however long it persists -- the decoder can only act on
what it has seen.  A penalty worth more than a look-ahead window's contrast is
therefore "never flip" no matter how the music argues, and the same penalty at a
longer lag is "follow the music".  Sweep the pair, never either alone.

Pure numpy and stdlib, deliberately: the runtime-integration plan imports this
into ``lib/``, torch is an optional extra, and a decode path that pulls a
training package is a decode path that cannot ship.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

# The corpus is 4/4 throughout, so the phase alphabet is a constant.  Stated here
# rather than imported from ``downbeat_dataset``: that module pulls the training
# dataset, and with it torch, which this one must never pay for.
BEATS_PER_BAR = 4

# Nats of evidence a phase flip has to buy: ~2.7 bars of confident activation (a
# 0.90 downbeat against 0.05 off-beats is 5.1 nats of log-odds contrast per bar).
# **Provisional, and it suits the EXPERT condition.**  A 20-track val sweep put
# the plateau at 12-16 with a sharp fall either side; the full 215-track sweep
# confirms the ordering (expert-driven F1 0.7063 at 14 against a 0.5592
# naive-picking floor on the same activations) but was too coarse to locate the
# plateau, and the 20-track *levels* were optimistic by ~0.17.
#
# **What it does NOT depend on is the subdivision, and what it DOES depend on is
# the condition.**  Measured at *matched wall-clock look-ahead* on all 215 val
# tracks (``lag_beats = 4 * subdivision``; see DEFAULT_LAG_BEATS for why that
# qualifier decides the answer): the expert condition prefers 14 at both
# subdivisions (0.7063 / 0.6864 against 0.6143 / 0.5998 at flip 3), while the
# aubio condition prefers ~3 at both (0.3502 / 0.4275 against 0.2712 / 0.3007 at
# flip 14).  A clean beat stream wants roughly 4-5x the penalty a noisy one does.
# Task 4 owns the tuning -- on all 215 val tracks, on the aubio-driven numbers the
# gates bind to, and jointly with ``lag_beats`` and ``subdivision``.
DEFAULT_FLIP_PENALTY = 14.0

# The sigmoid's own midpoint, the same neutral reference the section decoder uses
# for its equally uncalibrated boundary score.  The analytic de-weighting point is
# ``pos_weight / (1 + pos_weight)`` (0.903 for this checkpoint); see the module
# docstring for why the choice is second order.
DEFAULT_DOWNBEAT_REF = 0.5

# **This counts CANDIDATES, not beats, and the two differ once ``subdivision``
# does.**  At subdivision 1 a lag of 4 is one bar, ~1.9 s at 128 BPM, inside the
# runtime's 2.5 s look-ahead and committed early enough for the section decoder to
# quantise to it.  At subdivision 2 the same 4 is *two beats*, ~0.94 s -- half the
# look-ahead, which measurably costs F1 and confounds any sweep that varies the
# subdivision without compensating.  **Hold wall-clock look-ahead constant with
# ``lag_beats = 4 * subdivision``**; the name is kept because the streaming API
# counts what it is handed, but nothing else about it is a beat.
DEFAULT_LAG_BEATS = 4

# A gap this many times the running beat period is a dropout rather than tempo
# drift.  Below 1.5 an ordinary aubio wobble would insert phantom beats.
DEFAULT_COAST_RATIO = 1.5

# Four bars.  Aubio's documented failure is a few beats under sidechain
# compression; anything longer is a discontinuity and coasting it would invent
# downbeats that are all false positives.
MAX_COAST_BEATS = 16

# Beat periods behind the tempo estimate.  Two bars: long enough that one
# mistracked interval cannot move the median, short enough to follow a tempo ride.
DEFAULT_TEMPO_WINDOW = 8

# Candidate instants per beat.  1 decodes on the beat stream as given; 2 adds the
# midpoint of every consecutive pair, doubling the cycle to eight half-beat states.
# The amendment exists because aubio's dominant failure on this corpus is a steady
# HALF-BEAT lock, not jitter: an aubio beat sits within +-70 ms of an expert
# downbeat only 51.8 % of the time on val, and admitting the midpoints lifts that
# ceiling to 85.2 %.  A steady half-beat offset is then a state the decoder can
# occupy and hold, rather than an error it has to round away.  Midpoints stay
# causal -- the midpoint of beats n and n+1 exists as soon as beat n+1 arrives, on
# the same lag discipline as everything else.
DEFAULT_SUBDIVISION = 1

# Frames either side of a candidate instant that count as evidence for it.
# Symmetric, which was Task 1's null hypothesis and what the measurement on val
# activations confirmed; +-1 frame is +-46.4 ms, so with the +-23.2 ms the frame
# grid already costs, the window's reach is the +-70 ms tolerance itself.  Widening
# it would let an instant outside the tolerance borrow a downbeat's evidence.
AGG_LO_FRAMES = -1
AGG_HI_FRAMES = 1

# Keeps logit() finite on a saturated activation without perturbing a real one.
EPS = 1e-6


class PhaseDecision(NamedTuple):
    """One immutable commit: which candidate, when, what position, how sure.

    ``phase`` is the position within the bar's cycle, which is four long at
    ``subdivision = 1`` and eight at 2.  **1 is the downbeat either way**, so
    ``downbeat_times`` needs no subdivision argument; ``bar_phase`` is what maps
    the rest back onto 1..4.
    """

    beat: int             # index into the decoded stream, coasted ones included
    time: float           # song-position seconds -- the candidate's own instant
    phase: int            # 1..BEATS_PER_BAR * subdivision; 1 is the downbeat
    confidence: float     # max-plus posterior of this position against the others
    virtual: bool         # True when the candidate was coasted, not observed


@dataclass(frozen=True)
class PhaseParams:
    """Every knob, in one sweepable record.

    Frozen and comparable so a sweep can use it as a dict key and record it
    verbatim beside the number it produced.
    """

    lag_beats: int = DEFAULT_LAG_BEATS
    subdivision: int = DEFAULT_SUBDIVISION
    flip_penalty: float = DEFAULT_FLIP_PENALTY
    downbeat_ref: float = DEFAULT_DOWNBEAT_REF
    coast_ratio: float = DEFAULT_COAST_RATIO
    max_coast_beats: int = MAX_COAST_BEATS
    tempo_window: int = DEFAULT_TEMPO_WINDOW


def _logit(value: float) -> float:
    clipped = min(max(float(value), EPS), 1.0 - EPS)
    return float(np.log(clipped / (1.0 - clipped)))


# --------------------------------------------------------------------------- #
# Candidate instants and their evidence
# --------------------------------------------------------------------------- #
#
# These three live here rather than in ``downbeat_infer`` because in the runtime
# the aggregation *is* decode-path work: the engine receives an activation stream
# and a beat stream and has to put one on the other itself.  The sidecar's stored
# per-beat scores are a cache of the ``subdivision = 1`` case; anything denser has
# to be aggregated at decode time, and doing that must not require torch.


def candidate_grid(beat_times, subdivision: int = DEFAULT_SUBDIVISION) -> np.ndarray:
    """The instants a downbeat may be placed at, from a beat stream.

    ``subdivision = 1`` is the beat stream itself.  ``2`` interleaves the midpoint
    of every consecutive pair, so ``n`` beats yield ``2n - 1`` candidates and the
    bar becomes an eight-state cycle.  Higher subdivisions are refused rather than
    silently approximated: the corpus is 4/4 and the measured failure mode is a
    half-beat lock, so a third would be a different claim about the music.
    """
    times = np.asarray(beat_times, dtype=np.float64).reshape(-1)
    if int(subdivision) == 1:
        return times
    if int(subdivision) != 2:
        raise ValueError(
            f"subdivision must be 1 or 2, got {subdivision} -- see the module "
            f"docstring; anything else is an unmeasured claim about the metre")
    if times.size < 2:
        return times
    dense = np.empty(2 * times.size - 1, dtype=np.float64)
    dense[0::2] = times
    dense[1::2] = 0.5 * (times[:-1] + times[1:])
    return dense


def nearest_frames(instants, n_frames: int, frame_sec: float,
                   t0: float) -> np.ndarray:
    """Activation frame whose stamp is nearest each instant; ``-1`` past the end.

    Task 1's convention exactly, including its correction: an instant *before* the
    first frame's stamp is clamped to frame 0 -- frame 0 pools the audio in
    ``(t0 - frame_sec, t0]``, which is where it lives -- so ``-1`` means past the
    end and nothing else.  Filter on it; never clip it.
    """
    index = np.rint((np.asarray(instants, dtype=np.float64) - t0) / frame_sec)
    index = np.maximum(index.astype(np.int64), 0)
    return np.where(index >= int(n_frames), -1, index)


def aggregate_at_beats(activation, instants, frame_sec: float, t0: float, *,
                       lo: int = AGG_LO_FRAMES, hi: int = AGG_HI_FRAMES) -> tuple:
    """``(scores, counts)`` -- the activation peak in a window around each instant.

    The **maximum**, not the mean: aggregation exists to absorb the beat stream's
    timing jitter, and a mean over the window dilutes a sharp peak in proportion
    to how well the head localised it -- penalising exactly the behaviour a
    70 ms-sigma target was built to produce.

    ``scores`` is NaN wherever the window holds no frame at all (an instant past
    the end of the activation), which the decoder reads as *no evidence* rather
    than as evidence of no downbeat.  ``counts`` records how many frames each
    score was taken over, so an instant scored off a clipped window at a track's
    edge is visible instead of looking like every other one.
    """
    activation = np.asarray(activation, dtype=np.float64)
    n_frames = len(activation)
    frames = nearest_frames(instants, n_frames, frame_sec, t0)

    scores = np.full(len(frames), np.nan, dtype=np.float64)
    counts = np.zeros(len(frames), dtype=np.int32)
    for index, frame in enumerate(frames):
        if frame < 0:
            continue
        start = max(0, int(frame) + int(lo))
        end = min(n_frames, int(frame) + int(hi) + 1)
        if end <= start:                               # pragma: no cover
            continue
        counts[index] = end - start
        scores[index] = activation[start:end].max()
    return scores, counts


# --------------------------------------------------------------------------- #
# The decoder
# --------------------------------------------------------------------------- #


class BarPhaseHMM:
    """Beat-rate cyclic phase decoder that commits at a fixed lag and never revises.

    ``push`` one beat at a time and it returns the decisions that became final
    because of it -- the runtime shape.  ``decode`` runs a whole cached track
    through that same path, so an offline sweep and the live engine are provably
    the same decoder rather than two implementations that agree today.
    """

    def __init__(self, params: PhaseParams | None = None) -> None:
        params = params or PhaseParams()
        if int(params.lag_beats) < 0:
            raise ValueError(f"lag_beats must be >= 0, got {params.lag_beats}")
        if int(params.subdivision) not in (1, 2):
            raise ValueError(
                f"subdivision must be 1 or 2, got {params.subdivision}")
        if float(params.flip_penalty) < 0.0:
            raise ValueError(
                f"flip_penalty must be >= 0, got {params.flip_penalty} -- a "
                f"negative penalty pays the decoder to lose the bar")
        if not 0.0 < float(params.downbeat_ref) < 1.0:
            raise ValueError(
                f"downbeat_ref must lie in (0, 1), got {params.downbeat_ref}")
        if float(params.coast_ratio) <= 1.0:
            raise ValueError(
                f"coast_ratio must exceed 1, got {params.coast_ratio} -- at or "
                f"below 1 every ordinary beat would be read as a dropout")

        self.params = params
        self._ref_logit = _logit(params.downbeat_ref)
        # Positions in one bar of the candidate grid: four at subdivision 1, eight
        # at subdivision 2 (each beat plus the half-beat after it).  Position 1 is
        # the downbeat either way.
        self.cycle = BEATS_PER_BAR * int(params.subdivision)

        # Row p is "what does it cost to be at position q next".  Unnormalised,
        # and legitimately so: every row has the same shape, so the omitted
        # normaliser is one shared constant per step and Viterbi cannot see it.
        advance = (np.arange(self.cycle) + 1) % self.cycle
        self._transition = np.full((self.cycle, self.cycle),
                                   -float(params.flip_penalty), dtype=np.float64)
        self._transition[np.arange(self.cycle), advance] = 0.0
        self.reset()

    # -- streaming ---------------------------------------------------------- #

    def reset(self) -> None:
        """Forget the track.  A reset decoder decodes exactly like a fresh one."""
        # `_delta` scores the oldest UNCOMMITTED beat, already conditioned on
        # every commit before it; `_pending` is the look-ahead behind it.
        self._delta = None
        self._frontier: tuple | None = None      # (emission, time, virtual) of it
        self._pending: deque = deque()           # (emission, time, virtual), in order
        self._committed = 0
        # Phase of the last commit -- the only thing the next beat inherits.
        # Its path score is deliberately not carried: every step normalises the
        # trellis by its own maximum, so an additive constant cancels exactly.
        self._tail: int | None = None
        self._last_time = None
        self._periods: deque = deque(maxlen=int(self.params.tempo_window))

    def push(self, time: float, score: float) -> list:
        """Advance to the next observed beat; return the decisions now final.

        ``score`` is the aggregated downbeat activation at this instant, or NaN
        for a beat with no usable evidence (past the end of the mel, or on a
        stretch no window voted on).  Any gap since the previous beat is coasted
        first, so the returned decisions can include interpolated beats.
        """
        time = float(time)
        if self._last_time is not None and time <= self._last_time:
            raise ValueError(
                f"beat times must be strictly increasing: {time} follows "
                f"{self._last_time}")

        coasted, period = self._plan_gap(time)
        decisions: list = []
        for moment in coasted:
            decisions.extend(self._observe(moment, np.nan, virtual=True))
        decisions.extend(self._observe(time, score, virtual=False))
        if period is not None:
            # One entry per beat interval actually traversed, so the estimate
            # stays a median of *beat periods* rather than of arrival gaps.
            self._periods.extend([period] * (len(coasted) + 1))
        self._last_time = time
        return decisions

    def flush(self) -> list:
        """Force out every beat still inside the lag window.  Idempotent."""
        decisions: list = []
        while self._delta is not None:
            decisions.append(self._commit_one())
        return decisions

    def decode(self, beat_times, beat_scores) -> list:
        """Decode a whole cached track through the streaming path.

        Not a second batch implementation: an offline sweep that disagreed with
        the runtime by one line would be measuring the wrong decoder.
        """
        times = np.asarray(beat_times, dtype=np.float64).reshape(-1)
        scores = np.asarray(beat_scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(times):
            raise ValueError(
                f"got {len(scores)} scores for {len(times)} beats -- the "
                f"aggregation and the beat stream have come apart")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("beat times must be strictly increasing")

        self.reset()
        decisions: list = []
        for time, score in zip(times, scores):
            decisions.extend(self.push(time, score))
        decisions.extend(self.flush())
        return decisions

    # -- internals ---------------------------------------------------------- #

    def _plan_gap(self, time: float) -> tuple:
        """``([interpolated times], beat period to record)`` for one arrival.

        The gap is measured against the *running* period rather than a global
        tempo so a track that rides its tempo still coasts correctly, and the
        interpolation divides the observed gap rather than stepping forward by
        the estimate -- stepping would drift the last coasted beat away from the
        real one that closes the gap.

        A period of ``None`` means "this interval is not a beat period": it is
        returned for a discontinuity, so a 10-minute silence cannot be folded
        into the tempo estimate that has to survive it.
        """
        if self._last_time is None:
            return [], None
        gap = time - self._last_time
        if not self._periods:
            return [], gap                      # the first interval seeds the tempo
        period = float(np.median(self._periods))
        if period <= 0.0 or gap < self.params.coast_ratio * period:
            return [], gap
        missing = int(round(gap / period)) - 1
        if missing < 1:
            return [], gap
        if missing > int(self.params.max_coast_beats):
            # A discontinuity, not a dropout.  Emit nothing and let the phase
            # re-lock; the stale tempo would only mistrack the next gap.
            self._periods.clear()
            return [], None
        step = gap / (missing + 1)
        return [self._last_time + step * (index + 1) for index in range(missing)], step

    def _emission(self, score: float) -> np.ndarray:
        """Log-evidence this beat lends each of the four phases.

        Only phase 1 has an acoustic signature, so only phase 1 carries a term:
        the score's log-odds against ``downbeat_ref``.  A NaN leaves the row flat
        -- no evidence is not evidence of no downbeat, and the cyclic prior is
        then what carries the phase through the gap.
        """
        emission = np.zeros(self.cycle, dtype=np.float64)
        if np.isfinite(score):
            emission[0] = _logit(score) - self._ref_logit
        return emission

    def _observe(self, time: float, score: float, *, virtual: bool) -> list:
        """Take one beat into the trellis and emit whatever that makes final."""
        emission = self._emission(score)
        if self._delta is None:
            # Uniform over the four phases: an aubio stream starts wherever the
            # analysis started, which says nothing at all about the bar.
            seed = (np.zeros(self.cycle) if self._tail is None
                    else self._transition[self._tail])
            self._delta = seed + emission
            self._delta -= self._delta.max()
            self._frontier = (emission, float(time), bool(virtual))
        else:
            self._pending.append((emission, float(time), bool(virtual)))

        decisions: list = []
        while (self._delta is not None
               and len(self._pending) >= int(self.params.lag_beats)):
            decisions.append(self._commit_one())
        return decisions

    def _look_ahead(self, seed: np.ndarray) -> np.ndarray:
        """Best score of a path per (phase here, phase at the look-ahead head).

        ``table.max(axis=1)`` is then exactly "how well can the track do if this
        beat is phase p", which is the quantity both the decision and the
        confidence are read off.  ``lag`` operations on a 4x4 array.
        """
        table = np.full((self.cycle, self.cycle), -np.inf, dtype=np.float64)
        np.fill_diagonal(table, seed)
        for emission, _time, _virtual in self._pending:
            table = (table[:, :, None] + self._transition).max(axis=1) + emission
        return table.max(axis=1)

    def _commit_one(self) -> PhaseDecision:
        """Freeze the oldest uncommitted beat against the buffered look-ahead.

        The *decision* maximises over every path consistent with what has already
        been emitted (``_delta`` carries that constraint), which is what makes the
        commit immutable.  The *confidence* deliberately does not: it re-scores
        the same window from this beat's own evidence alone, with no memory of
        the committed prefix.  Conditioning it would make it useless -- given a
        committed phase the next one is near-certain under any flip penalty, so a
        track the model understands nothing about would report near-1 confidence
        forever, which is the exact opposite of what the engine's fall-back to
        beat-snapping needs to read.  Unconditioned, a blind track sits at 1/4 and
        a half-bar-ambiguous one at 1/2, which is the honest statement.
        """
        score = self._look_ahead(self._delta)
        phase = int(np.argmax(score))

        local = self._look_ahead(self._frontier[0])
        weights = np.exp(local - local.max())
        confidence = float(weights[phase] / weights.sum())

        _emission, time, virtual = self._frontier
        decision = PhaseDecision(self._committed, time, phase + 1, confidence, virtual)
        self._committed += 1
        self._tail = phase

        if self._pending:
            emission, time, virtual = self._pending.popleft()
            # The next frontier inherits the constraint: only paths through the
            # phase just committed exist from here on.
            self._delta = self._transition[phase] + emission
            self._delta -= self._delta.max()
            self._frontier = (emission, time, virtual)
        else:
            self._delta = None
            self._frontier = None
        return decision


# --------------------------------------------------------------------------- #
# Reading a decode
# --------------------------------------------------------------------------- #


def downbeat_times(decisions) -> np.ndarray:
    """The phase-1 instants -- the predicted bar grid.

    Continuous beat instants, not frame centres: the decoder never quantises to
    the 46.44 ms mel grid, which is worth separating from the phase model's value
    when the two are scored together (task-2 §7c).
    """
    return np.array([d.time for d in decisions if d.phase == 1], dtype=np.float64)


def phase_flips(decisions, subdivision: int = DEFAULT_SUBDIVISION) -> int:
    """Candidates whose position did not advance -- the plan's stability number.

    ``subdivision`` has to be the one the decode ran at, because it decides how
    long the cycle is and therefore what "advance" means.  Passing the wrong one
    reports every candidate as a flip on an eight-state decode, which is loud
    rather than subtle, but it is still worth stating.
    """
    cycle = BEATS_PER_BAR * int(subdivision)
    flips = 0
    for previous, current in zip(decisions, decisions[1:]):
        if current.phase != previous.phase % cycle + 1:
            flips += 1
    return flips


def bar_phase(phase: int, subdivision: int = DEFAULT_SUBDIVISION) -> int:
    """Cycle position -> bar phase in 1..4, or 0 for an interstitial candidate.

    At ``subdivision = 1`` this is the identity.  At 2 the odd positions are the
    beats (1, 3, 5, 7 -> bar phases 1, 2, 3, 4) and the even ones are the
    half-beats between them, which no bar phase names.  Written down once here
    because "phase accuracy" is a reported metric and everyone deriving the
    mapping separately is how two reports come to disagree.
    """
    step = int(subdivision)
    return (phase - 1) // step + 1 if (phase - 1) % step == 0 else 0


def decode_track(sidecar_npz, condition: str, params: PhaseParams | None = None,
                 *, refine: bool = False) -> list:
    """One track, one input condition, end to end.

    ``condition`` selects which beat stream drives the decode -- ``aubio`` is the
    live condition the gates bind to, ``expert`` the diagnostic upper bound.  At
    ``subdivision = 2`` the candidate grid is built here and its evidence is
    aggregated off the sidecar's stored activation curve, so a half-beat decode
    needs no second inference pass.  Pure numpy over a cached array, which is what
    makes a decoder parameter sweep cost seconds.

    ``refine`` is a *separate* effect and is off by default: it moves each emitted
    instant to the interpolated peak of the activation.  Task 4 has to be able to
    attribute a gain to the candidate grid, to the phase model, or to
    de-quantisation, and it cannot do that if the three arrive blended.
    """
    params = params or PhaseParams()
    path = Path(sidecar_npz)
    with np.load(path) as archive:
        time_key = f"{condition}_beat_time"
        if time_key not in archive:
            raise KeyError(
                f"{path.name} carries no '{condition}' beat stream; it has "
                f"{sorted(k[:-10] for k in archive if k.endswith('_beat_time'))}")
        beats = np.asarray(archive[time_key], dtype=np.float64)
        activation = np.asarray(archive["activation"], dtype=np.float64)
        frame_sec = float(archive["frame_sec"])
        t0 = float(archive["t0"])
        cached = np.asarray(archive[f"{condition}_beat_score"], dtype=np.float64)

    if int(params.subdivision) == 1:
        # The stored scores are this exact aggregation; reuse them rather than
        # recompute, so the sidecar and the decode cannot drift apart unnoticed.
        times, scores = beats, cached
    else:
        times = candidate_grid(beats, params.subdivision)
        scores, _counts = aggregate_at_beats(activation, times, frame_sec, t0)

    decisions = BarPhaseHMM(params).decode(times, scores)
    if refine:
        decisions = refine_instants(decisions, activation, frame_sec, t0)
    return decisions


def refine_instants(decisions, activation, frame_sec: float, t0: float, *,
                    lo: int = AGG_LO_FRAMES, hi: int = AGG_HI_FRAMES) -> list:
    """Move each committed instant to the interpolated peak of the activation.

    Parabolic interpolation through the winning frame and its two neighbours --
    the standard sub-sample peak estimate -- clamped to the aggregation window so
    a refinement can never move an instant further than the evidence that chose
    it.  It exists because the head's peaks are quantised to the 46.44 ms mel grid
    and a third of the +-70 ms budget is spent there before the decoder acts.

    **Deliberately a post-process, not part of the decode.**  It changes the
    emitted time and nothing else -- not a phase, not a flip, not a confidence --
    so its contribution can be measured by running the same decode twice.
    """
    activation = np.asarray(activation, dtype=np.float64)
    n_frames = len(activation)
    frames = nearest_frames([d.time for d in decisions], n_frames, frame_sec, t0)
    span = np.arange(int(lo), int(hi) + 1)

    refined: list = []
    for decision, frame in zip(decisions, frames):
        if frame < 0:
            refined.append(decision)
            continue
        window = np.clip(int(frame) + span, 0, n_frames - 1)
        peak = int(window[int(np.argmax(activation[window]))])
        left = activation[max(peak - 1, 0)]
        centre = activation[peak]
        right = activation[min(peak + 1, n_frames - 1)]
        curvature = left - 2.0 * centre + right
        shift = 0.5 * (left - right) / curvature if curvature < 0.0 else 0.0
        moment = t0 + (peak + float(np.clip(shift, -0.5, 0.5))) * frame_sec
        refined.append(decision._replace(
            time=float(np.clip(moment, decision.time + lo * frame_sec,
                               decision.time + hi * frame_sec))))
    return refined
