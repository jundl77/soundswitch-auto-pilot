"""Bar-phase decoder: beat instants + downbeat activation in, a bar grid out.

The downbeat head answers "does this instant sound like the start of a bar" and
Task 2 measured exactly how far that gets you: **94.6 % of its peaks are
beat-locked and 52 % of its false positives land on beat 3**.  The head has
solved *is this a beat* and not *which beat*, and no better activation fixes
that, because in four-on-the-floor beat 3 is acoustically the same event as beat
1.  What separates them is the *sequence*: a track cannot alternate 1-3-1-3
without paying for a phase flip over and over.  So this module is a 4-state
cyclic HMM at beat rate, and the flip penalty is the knob that turns a ranking
into a grid.

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
light show cannot un-fire a strobe, so a decision ``lag_beats`` behind the
arrival head is final.  The section decoder gets that by reading a backtrace and
then setting every disagreeing trellis state to -inf; **that construction is
wrong for a four-state cyclic chain and it fails silently.**  Viterbi keeps one
best path per state, so with four states and a near-permutation transition graph
the -inf prune leaves exactly *one* live state -- which fixes not just the
committed beat but every beat between it and the head, i.e. it throws the
look-ahead away and turns the decoder greedy after its first commit, while still
producing perfectly plausible output.  (The section decoder is safe from this
because its 48 ``(class, dwell)`` states share ancestors; nothing about the
pattern generalises.)

So the commit here is exact instead: the trellis carries ``delta`` for the oldest
*uncommitted* beat, and a commit rolls a 4x4 ``(frontier phase, head phase)``
table forward over the buffered look-ahead.  That gives the maximum over all
paths that agree with everything already emitted -- immutability by construction
rather than by pruning -- with the full lag of look-ahead on every decision, and
it hands back the phase posterior the confidence output needs as a by-product.

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
# **Provisional.**  Measured on a 20-track val sanity check, where 12-16 is a
# plateau (expert-driven F1 0.859-0.876 against a 0.605 naive-picking floor on
# the same activations) with a sharp fall either side -- but on 20 tracks one
# track's global bar offset moves the micro-averaged F1 by ~5 points, so this is
# a defensible starting point and not a tuned value.  It is the primary sweep
# axis and Task 4 owns the tuning, on all 215 val tracks and on the aubio-driven
# numbers the gates bind to.
DEFAULT_FLIP_PENALTY = 14.0

# The sigmoid's own midpoint, the same neutral reference the section decoder uses
# for its equally uncalibrated boundary score.  The analytic de-weighting point is
# ``pos_weight / (1 + pos_weight)`` (0.903 for this checkpoint); see the module
# docstring for why the choice is second order.
DEFAULT_DOWNBEAT_REF = 0.5

# One bar at 128 BPM is 1.9 s, inside the runtime's 2.5 s look-ahead, and the bar
# grid has to be committed *before* the section decoder can quantise to it.
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

# Keeps logit() finite on a saturated activation without perturbing a real one.
EPS = 1e-6


class PhaseDecision(NamedTuple):
    """One immutable commit: which beat, when, what phase, how sure."""

    beat: int             # index into the decoded stream, virtual beats included
    time: float           # song-position seconds -- the beat's own instant
    phase: int            # 1..BEATS_PER_BAR; 1 is the downbeat
    confidence: float     # max-plus posterior of this phase against the other three
    virtual: bool         # True when the beat was coasted, not observed


@dataclass(frozen=True)
class PhaseParams:
    """Every knob, in one sweepable record.

    Frozen and comparable so a sweep can use it as a dict key and record it
    verbatim beside the number it produced.
    """

    lag_beats: int = DEFAULT_LAG_BEATS
    flip_penalty: float = DEFAULT_FLIP_PENALTY
    downbeat_ref: float = DEFAULT_DOWNBEAT_REF
    coast_ratio: float = DEFAULT_COAST_RATIO
    max_coast_beats: int = MAX_COAST_BEATS
    tempo_window: int = DEFAULT_TEMPO_WINDOW


def _logit(value: float) -> float:
    clipped = min(max(float(value), EPS), 1.0 - EPS)
    return float(np.log(clipped / (1.0 - clipped)))


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

        # Row p is "what does it cost to be in phase q at the next beat".
        # Unnormalised, and legitimately so: every row has the same shape, so the
        # omitted normaliser is one shared constant per step and Viterbi's argmax
        # cannot see it.
        advance = (np.arange(BEATS_PER_BAR) + 1) % BEATS_PER_BAR
        self._transition = np.full((BEATS_PER_BAR, BEATS_PER_BAR),
                                   -float(params.flip_penalty), dtype=np.float64)
        self._transition[np.arange(BEATS_PER_BAR), advance] = 0.0
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
        self._tail: tuple | None = None          # (phase, score) of the last commit
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
        emission = np.zeros(BEATS_PER_BAR, dtype=np.float64)
        if np.isfinite(score):
            emission[0] = _logit(score) - self._ref_logit
        return emission

    def _observe(self, time: float, score: float, *, virtual: bool) -> list:
        """Take one beat into the trellis and emit whatever that makes final."""
        emission = self._emission(score)
        if self._delta is None:
            # Uniform over the four phases: an aubio stream starts wherever the
            # analysis started, which says nothing at all about the bar.
            seed = np.zeros(BEATS_PER_BAR) if self._tail is None else (
                self._tail[1] + self._transition[self._tail[0]])
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
        table = np.full((BEATS_PER_BAR, BEATS_PER_BAR), -np.inf, dtype=np.float64)
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
        self._tail = (phase, float(self._delta[phase]))

        if self._pending:
            emission, time, virtual = self._pending.popleft()
            # The next frontier inherits the constraint: only paths through the
            # phase just committed exist from here on.
            self._delta = self._tail[1] + self._transition[phase] + emission
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


def phase_flips(decisions) -> int:
    """Beats whose phase did not advance from the previous one -- the stability
    number the plan's acceptance gate reads."""
    flips = 0
    for previous, current in zip(decisions, decisions[1:]):
        if current.phase != previous.phase % BEATS_PER_BAR + 1:
            flips += 1
    return flips


def decode_track(sidecar_npz, condition: str,
                 params: PhaseParams | None = None) -> list:
    """One track, one input condition, end to end.

    ``condition`` selects which beat stream in the sidecar drives the decode --
    ``aubio`` is the live condition the gates bind to, ``expert`` the diagnostic
    upper bound.  Pure numpy over a cached array, which is what makes a decoder
    parameter sweep cost seconds.
    """
    with np.load(Path(sidecar_npz)) as archive:
        time_key = f"{condition}_beat_time"
        if time_key not in archive:
            raise KeyError(
                f"{Path(sidecar_npz).name} carries no '{condition}' beat stream; "
                f"it has {sorted(k[:-10] for k in archive if k.endswith('_beat_time'))}")
        times = np.asarray(archive[time_key], dtype=np.float64)
        scores = np.asarray(archive[f"{condition}_beat_score"], dtype=np.float64)
    return BarPhaseHMM(params).decode(times, scores)
