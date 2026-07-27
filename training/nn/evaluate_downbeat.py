"""The downbeat verdict: what the live condition can reach, and what it costs.

Three questions, in the order the plan asks them, because the second and third
are only readable in the light of the first.

**1. What does aubio's beat stream do to the bar grid?**  The decoder can only
place a downbeat where its candidate grid has an instant, so the live condition
is bounded by aubio before any model or decoder is involved.  ``alignment_row``
measures the stream against the annotator's (offset, jitter, missed and extra
beats, tempo ratio) and ``reach_labels`` splits the *unreachable* downbeats into
named causes -- outside the stream, a dropout, a tempo mismatch, a steady
off-grid lock, or an unsteady one.  That split is the evidence any decision about
this component rests on: a shortfall caused by aubio dropping beats and a
shortfall caused by aubio locking a quarter beat off have different fixes and
neither of them is "tune the decoder harder".

**2. How well does the decoder do inside that bound?**  ``score_downbeats`` is
the plan's F1@+-70 ms through the committed matcher, ``phase_scores`` is bar-phase
accuracy against the annotated grid, and stability gets *two* numbers because the
obvious one is wrong: ``phase_flips`` at subdivision 2 counts every beat aubio
inserts or drops, which is a property of the input, not of the grid.  So the two
reported are ``beat_anchored_flips`` (did the bar position advance by exactly one
beat between consecutive real beats) and ``interval_deviation`` (do the emitted
downbeats keep a steady spacing).  The first is the plan's unit -- flips per
track -- and is what the gate is read against; the second is the grid-direct
measurement and is reported beside it.

**3. Does a predicted grid still carry a show?**  ``ablation_rows`` decodes
sections on the *predicted* bar grid and on the expert one with everything else
held fixed, so the boundary-F1 and flicker deltas are attributable to the grid
alone.  That is the go/no-go number for live bar-snapping, and it is the only
number here that speaks the audience's language rather than the metric's.

**Every matcher is imported.**  ``match_events``, ``prf``, the decoder, the
candidate grid and the section chain's own ``bar_observations`` are used as
shipped; a verdict measured by a second implementation of its metrics is a
verdict about the two implementations.  The one thing this module defines for
itself is the *grid* it hands the section decoder, and it builds that by the same
rule ``decoder.bar_grid`` uses on the annotated one.

**The test split is read once**, by ``--verdict --split test``, and only against a
config file that already exists on disk -- so the choice provably predates the
read.  Every other mode is guarded on split *membership*, not on a flag.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .downbeat_decoder import (
    BEATS_PER_BAR,
    BarPhaseHMM,
    PhaseParams,
    bar_phase,
    candidate_grid,
    downbeat_times,
)
from .downbeat_infer import SIDECAR_DIR
from .downbeat_train import MODEL_VERSION, TOLERANCE_SEC, match_events, prf

MODELS_DIR = "models"
CONFIG_FILE = "downbeat_decoder_config.json"
ALIGNMENT_FILE = "downbeat_alignment_{split}.json"
EVAL_FILE = "downbeat_eval_{split}.json"
ABLATION_FILE = "downbeat_ablation_{split}.json"
SPLITS_FILE = "splits.json"

# Bars of look-ahead the runtime can afford: the section decoder commits three
# bars behind, and the show's own delay is 2.5 s, so a bar-grid commit that costs
# more than ~2 s of its own pushes the chain past the 8 s budget the section spec
# set.  Longer lags are still measured -- the lag curve is the evidence for
# whether the budget should move -- but a selected config has to fit.
LOOK_AHEAD_BUDGET_BEATS = 4

# A predicted bar interval this far from the track's own running median is a
# grid instability rather than a tempo ride.  15 % of a bar at 128 BPM is 280 ms
# -- far larger than any real tempo change between adjacent bars, far smaller
# than the half-bar (50 %) or beat (25 %) errors a lost phase produces.
INTERVAL_DEVIATION = 0.15
INTERVAL_WINDOW = 9

# Residual (in beats) below which a stream's off-grid offset counts as *steady*.
# The distinction is the actionable one: a steady offset is a lock, which a
# different candidate grid could capture; an unsteady one is jitter, which only a
# better beat tracker fixes.
LOCK_IQR_BEATS = 0.06

# Local period agreement, after folding onto the nearest half/double octave: a
# stream running at exactly half or double the annotated tempo still puts beats
# on the annotated instants, so it is not a mismatch.  Anything that does not
# fold onto one of those within 2 % is tracking a different pulse.
TEMPO_TOLERANCE = 0.02
TEMPO_MULTIPLES = (0.5, 1.0, 2.0)

# Phase-confidence cut points for the spec's beat-snap fall-back.  0.25 is chance
# on a four-position cycle and 0.125 on eight, so the low end is "no information
# at all" and the high end is where a grid would actually be trusted.
CONFIDENCE_THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.9)

REACH_LABELS = ("beat", "midpoint", "coast", "no_coverage", "dropout",
                "tempo_mismatch", "fraction_lock", "jitter")
REACHED = ("beat", "midpoint", "coast")

# Where in the beat an unreachable downbeat sits.  The bins are 1/20 of a beat so
# a quarter-beat lock (0.25) and a triplet feel (0.33) land in different ones --
# the two have different fixes and a histogram that merged them would hide that.
RESIDUAL_BINS = tuple(round(0.05 * step, 2) for step in range(11))

CONDITIONS = ("aubio", "expert")


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #


def lag_for(look_ahead_beats: int, subdivision: int) -> int:
    """Candidates of lag that buy ``look_ahead_beats`` beats of wall clock.

    The one place allowed to know that ``lag_beats`` counts *candidates*: at
    subdivision 2 a lag of 4 is two beats, not four, so a sweep that varies the
    grid at a fixed lag is measuring the lag and calling it the grid.
    """
    return int(look_ahead_beats) * int(subdivision)


def rolling_median(values, window: int = INTERVAL_WINDOW) -> np.ndarray:
    """Centred running median, edge-clamped, same length as the input.

    Running rather than global so a track that rides its tempo is measured
    against what it is doing *there* -- a global median would report the ride
    itself as instability.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    window = max(1, int(window)) | 1                      # centring needs it odd
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=1)


def nearest_index(query, reference) -> np.ndarray:
    """Index of the nearest ``reference`` entry for each ``query``; -1 if empty."""
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        return np.full(query.size, -1, dtype=np.int64)
    right = np.searchsorted(reference, query)
    left = np.clip(right - 1, 0, reference.size - 1)
    right = np.clip(right, 0, reference.size - 1)
    take_left = np.abs(query - reference[left]) <= np.abs(reference[right] - query)
    return np.where(take_left, left, right)


def nearest_offset(query, reference) -> np.ndarray:
    """Signed ``query - nearest(reference)``; NaN where there is no reference.

    NaN rather than 0: a track whose beat stream is empty is not a track that is
    perfectly aligned, and the two must not average together.
    """
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        return np.full(query.size, np.nan, dtype=np.float64)
    return query - reference[nearest_index(query, reference)]


def fold_to_beats(offset_sec, period_sec) -> np.ndarray:
    """Offsets in seconds -> position within the beat, in ``(-0.5, +0.5]``.

    Half a beat late and half a beat early are the same place, so the interval is
    closed at ``+0.5``: a half-beat lock must read as one number rather than
    splitting into two whose median is zero.
    """
    offset = np.asarray(offset_sec, dtype=np.float64).reshape(-1)
    period = np.asarray(period_sec, dtype=np.float64).reshape(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beats = np.where(period > 0, offset / np.where(period > 0, period, 1.0), np.nan)
    folded = beats - np.floor(beats + 0.5)
    return np.where(folded == -0.5, 0.5, folded)


def local_periods(times, window: int = INTERVAL_WINDOW) -> np.ndarray:
    """Running beat period at each instant (one entry per instant).

    A median of adjacent intervals: local enough to follow a tempo ride and
    unmoved by a dropout, which is what a *gap* test needs.  It is not precise
    enough to compare two tempi against a few-percent tolerance -- that is what
    ``pulse_period`` is for.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size < 2:
        return np.full(times.size, np.nan, dtype=np.float64)
    intervals = rolling_median(np.diff(times), window)
    return np.concatenate([intervals[:1], intervals])


def pulse_period(times, window: int = 2 * INTERVAL_WINDOW - 1,
                 gap_factor: float = 1.5) -> np.ndarray:
    """Running *pulse* period: elapsed time over beats elapsed, dropouts counted.

    Two properties ``local_periods`` does not have, both needed to compare a
    stream's tempo against the annotator's at a few-percent tolerance:

    * **The timing noise telescopes.**  Summing a window of intervals is the
      difference of its two endpoints, so per-beat jitter divides by the window
      length instead of contributing in full.  A stream that wobbles +-100 ms per
      beat still reports its tempo to about a percent.
    * **A dropout is counted, not averaged.**  Each interval is credited with the
      whole number of beats it plausibly spans, so a gap of four missing beats
      adds four to the denominator rather than reading as a tempo change.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size < 3:
        return np.full(times.size, np.nan, dtype=np.float64)
    intervals = np.diff(times)
    scale = rolling_median(intervals, INTERVAL_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        steps = np.where(scale > 0, np.rint(intervals / scale), 1.0)
    steps = np.maximum(np.nan_to_num(steps, nan=1.0), 1.0)
    # A gap this large is a discontinuity rather than a dropout; crediting it
    # with its implied beats would fold a track boundary into the tempo.
    keep = intervals <= gap_factor * scale * steps
    window = max(1, int(window)) | 1
    half = window // 2
    padded_time = np.pad(np.where(keep, intervals, 0.0), half, mode="edge")
    padded_steps = np.pad(np.where(keep, steps, 0.0), half, mode="edge")
    kernel = np.ones(window)
    elapsed = np.convolve(padded_time, kernel, mode="valid")
    beats = np.convolve(padded_steps, kernel, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        period = np.where(beats > 0, elapsed / np.where(beats > 0, beats, 1.0), np.nan)
    return np.concatenate([period[:1], period])


def _iqr(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 75) - np.percentile(values, 25))


# --------------------------------------------------------------------------- #
# 1. Alignment: what aubio does to the grid
# --------------------------------------------------------------------------- #


def alignment_row(aubio, expert, downbeats, *,
                  tolerance: float = TOLERANCE_SEC) -> dict:
    """One track's aubio-vs-expert beat alignment.

    ``median_abs_phase`` is the *lock* -- how far off a beat the stream sits, in
    beats -- and ``phase_iqr`` is the *jitter* around it.  Reporting both is what
    separates "aubio is half a beat off, steadily" (a state a denser grid can
    occupy) from "aubio is all over the place" (which nothing downstream fixes).
    """
    aubio = np.asarray(aubio, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64).reshape(-1)
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)

    offsets = nearest_offset(aubio, expert)
    periods = local_periods(expert)
    matched = nearest_index(aubio, expert)
    period_at = periods[matched] if expert.size and aubio.size else np.zeros(0)
    phases = fold_to_beats(offsets, period_at) if aubio.size else np.zeros(0)

    reverse = nearest_offset(expert, aubio)
    to_downbeats = nearest_offset(downbeats, aubio)

    def share(values) -> float:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0
        return float(np.mean(np.abs(finite) <= tolerance))

    return {
        "n_aubio": int(aubio.size),
        "n_expert": int(expert.size),
        "n_downbeats": int(downbeats.size),
        "median_offset_sec": float(np.nanmedian(offsets)) if aubio.size else float("nan"),
        "median_abs_phase": float(np.nanmedian(np.abs(phases))) if aubio.size else float("nan"),
        "phase_iqr": _iqr(np.abs(phases)) if aubio.size else float("nan"),
        "aubio_on_grid": share(offsets),
        "expert_covered": share(reverse),
        "downbeat_on_beats": share(to_downbeats),
        "median_ibi_aubio": float(np.median(np.diff(aubio))) if aubio.size > 1 else float("nan"),
        "median_ibi_expert": float(np.median(np.diff(expert))) if expert.size > 1 else float("nan"),
        "ibi_ratio": (float(np.median(np.diff(aubio)) / np.median(np.diff(expert)))
                      if aubio.size > 1 and expert.size > 1 else float("nan")),
        "extra_beats": int(aubio.size) - int(expert.size),
    }


def _tempo_residual(aubio_period: float, expert_period: float) -> float:
    """Relative period error after folding onto the nearest half/double octave.

    A stream at exactly half or double the annotated tempo still has beats *on*
    the annotated instants, so it is not tracking a different pulse and must not
    be counted as one.  What this catches is a pulse that fits no octave of the
    grid: a drift, a triplet feel, a mistracked tempo.
    """
    if not (np.isfinite(aubio_period) and np.isfinite(expert_period)) or expert_period <= 0:
        return float("nan")
    ratio = aubio_period / expert_period
    return float(min(abs(ratio / multiple - 1.0) for multiple in TEMPO_MULTIPLES))


def decoder_instants(beat_times, params: PhaseParams | None = None) -> np.ndarray:
    """Every instant the decoder would place a decision on, coasting included.

    Measured *through the decoder* -- a NaN-scored decode emits exactly the
    candidates it would have decoded, with the coasted ones in place -- rather
    than by a second implementation of the coasting rule.  Two different things
    are read off this: which downbeats are reachable at all, and how many
    candidates a bar's worth of music actually produces.
    """
    params = params or PhaseParams(subdivision=2, lag_beats=lag_for(1, 2))
    dense = candidate_grid(beat_times, params.subdivision)
    if dense.size == 0:
        return dense
    decisions = BarPhaseHMM(params).decode(dense, np.full(dense.size, np.nan))
    return np.asarray([d.time for d in decisions], dtype=np.float64)


def bar_rate_ratio(beat_times, downbeats, params: PhaseParams | None = None) -> float:
    """Candidates per bar, over the cycle length -- 1.0 is a correctly paced grid.

    **The second ceiling, and the one nobody looked for.**  A cyclic decoder that
    never flips emits exactly one downbeat per cycle, so the *rate* of the
    emitted bar grid is set by the rate of the candidate stream, not by the
    music.  If aubio produces more candidates per bar than the cycle is long --
    by inserting beats, or by the decoder coasting extra ones through a gap --
    the grid runs fast and the surplus downbeats are false positives no phase
    model can retract.  Above 1 this bounds precision at ``coverage / ratio`` for
    a flip-free decode, which is what the plan's stability gate demands.
    """
    params = params or PhaseParams(subdivision=2, lag_beats=lag_for(1, 2))
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    if downbeats.size == 0:
        return float("nan")
    cycle = BEATS_PER_BAR * int(params.subdivision)
    return float(decoder_instants(beat_times, params).size / (cycle * downbeats.size))


def reach_labels(aubio, downbeats, expert, *, tolerance: float = TOLERANCE_SEC,
                 params: PhaseParams | None = None) -> list:
    """Why each annotated downbeat is, or is not, reachable from this beat stream.

    One label per downbeat, from ``REACH_LABELS``, tested in a fixed order so the
    categories are mutually exclusive and a total is a partition rather than a
    tally of overlapping conditions:

    ``beat`` / ``midpoint``   reachable on the shipped candidate grid
    ``coast``                 reachable only because the decoder fills the gap
    ``no_coverage``           outside the beat stream entirely
    ``dropout``               inside a gap the decoder will not fill
    ``tempo_mismatch``        the local pulse fits no octave of the annotated one
    ``fraction_lock``         a STEADY off-grid offset -- a lock a different grid
                              could capture
    ``jitter``                an UNSTEADY offset -- only a better tracker helps

    The coast row is measured through the decoder itself (a NaN-scored decode
    emits exactly the candidates it would have decoded), never by a second
    implementation of the coasting rule.
    """
    params = params or PhaseParams(subdivision=2, lag_beats=lag_for(1, 2))
    aubio = np.asarray(aubio, dtype=np.float64).reshape(-1)
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64).reshape(-1)
    if downbeats.size == 0:
        return []
    if aubio.size == 0:
        return ["no_coverage"] * downbeats.size

    dense = candidate_grid(aubio, params.subdivision)
    coasted = decoder_instants(aubio, params)

    to_beat = np.abs(nearest_offset(downbeats, aubio))
    to_dense = np.abs(nearest_offset(downbeats, dense))
    to_coast = np.abs(nearest_offset(downbeats, coasted))

    aubio_periods = local_periods(aubio)
    aubio_pulse = pulse_period(aubio)
    expert_pulse = pulse_period(expert) if expert.size > 2 else None
    near = nearest_index(downbeats, aubio)
    # Signed residual of every aubio beat against the annotated grid, in beats:
    # its local spread is what separates a lock from jitter.
    if expert.size:
        beat_phase = fold_to_beats(nearest_offset(aubio, expert),
                                   local_periods(expert)[nearest_index(aubio, expert)])
    else:
        beat_phase = np.full(aubio.size, np.nan)

    labels: list = []
    for index, moment in enumerate(downbeats):
        if to_beat[index] <= tolerance:
            labels.append("beat")
            continue
        if to_dense[index] <= tolerance:
            labels.append("midpoint")
            continue
        if to_coast[index] <= tolerance:
            labels.append("coast")
            continue

        anchor = int(near[index])
        period = float(aubio_periods[anchor])
        if (moment < aubio[0] - 0.5 * period) or (moment > aubio[-1] + 0.5 * period):
            labels.append("no_coverage")
            continue

        after = int(np.searchsorted(aubio, moment))
        gap = (aubio[after] - aubio[after - 1]
               if 0 < after < aubio.size else float("nan"))
        if np.isfinite(gap) and np.isfinite(period) and period > 0 \
                and gap >= params.coast_ratio * period:
            labels.append("dropout")
            continue

        expert_period = (float(expert_pulse[int(nearest_index([moment], expert)[0])])
                         if expert_pulse is not None else float("nan"))
        residual = _tempo_residual(float(aubio_pulse[anchor]), expert_period)
        if np.isfinite(residual) and residual > TEMPO_TOLERANCE:
            labels.append("tempo_mismatch")
            continue

        lo = max(0, anchor - BEATS_PER_BAR * 2)
        hi = min(aubio.size, anchor + BEATS_PER_BAR * 2 + 1)
        labels.append("fraction_lock" if _iqr(beat_phase[lo:hi]) <= LOCK_IQR_BEATS
                      else "jitter")
    return labels


def subdivided_grid(beat_times, subdivision: int) -> np.ndarray:
    """A uniformly subdivided beat stream -- **for ceiling analysis only**.

    ``candidate_grid`` is what decodes, and it refuses anything past 2 on purpose
    (a third or a quarter is a claim about the metre that nothing has measured).
    This function exists to answer the different question the owner package
    needs: *if* the grid were denser, how much of the shortfall would become
    reachable at all?  A bound, never a result -- nothing here decodes on it, and
    a denser grid also multiplies the ways to be wrong, which a ceiling cannot
    see.  Pinned equal to ``candidate_grid`` at subdivision 2 by test.
    """
    times = np.asarray(beat_times, dtype=np.float64).reshape(-1)
    subdivision = int(subdivision)
    if subdivision <= 1 or times.size < 2:
        return times
    steps = np.arange(subdivision, dtype=np.float64) / subdivision
    dense = times[:-1, None] + steps[None, :] * np.diff(times)[:, None]
    return np.append(dense.reshape(-1), times[-1])


def grid_ceiling(aubio, downbeats, subdivision: int,
                 tolerance: float = TOLERANCE_SEC) -> float:
    """Share of annotated downbeats a subdivided beat stream can even reach."""
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    if downbeats.size == 0:
        return 0.0
    grid = subdivided_grid(aubio, subdivision)
    if grid.size == 0:
        return 0.0
    return float(np.mean(np.abs(nearest_offset(downbeats, grid)) <= tolerance))


def downbeat_residuals(aubio, downbeats, expert) -> np.ndarray:
    """Where in the beat each annotated downbeat falls, relative to aubio's grid.

    ``|position|`` in beats, folded into ``[0, 0.5]``: 0 is on an aubio beat,
    0.5 is exactly between two.  The shape of this distribution is what says
    whether the unreachable downbeats are a quarter-beat lock, a triplet feel or
    a spread -- three different upstream problems that a single "not reachable"
    count cannot tell apart.
    """
    aubio = np.asarray(aubio, dtype=np.float64).reshape(-1)
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64).reshape(-1)
    if downbeats.size == 0:
        return np.zeros(0, dtype=np.float64)
    if aubio.size == 0 or expert.size < 2:
        return np.full(downbeats.size, np.nan, dtype=np.float64)
    periods = local_periods(expert)[nearest_index(downbeats, expert)]
    return np.abs(fold_to_beats(nearest_offset(downbeats, aubio), periods))


# --------------------------------------------------------------------------- #
# 2. Scoring a decode
# --------------------------------------------------------------------------- #


def score_downbeats(predicted, truth, tolerance: float = TOLERANCE_SEC) -> dict:
    """Downbeat P/R/F1 at the plan's tolerance, through the committed matcher."""
    return prf(*match_events(np.asarray(predicted, dtype=np.float64),
                             np.asarray(truth, dtype=np.float64), tolerance))


def confidence_sweep(decisions, truth, thresholds=CONFIDENCE_THRESHOLDS,
                     tolerance: float = TOLERANCE_SEC) -> dict:
    """P/R/F1 of the emitted grid when only confident downbeats are kept.

    The spec's fall-back -- "bar-snap when the grid is sure, beat-snap when it is
    not" -- is exactly this filter, so this is the measurement that says whether
    the fall-back is worth building: if confidence carries information, precision
    rises as the threshold does, and the retained share says how much of the show
    would still get bars.  Scored through the committed matcher on the filtered
    prediction set rather than by a per-instant correctness flag, so the numbers
    are comparable with every other F1 in this report.
    """
    times = np.asarray([d.time for d in decisions if d.phase == 1], dtype=np.float64)
    confidence = np.asarray([d.confidence for d in decisions if d.phase == 1],
                            dtype=np.float64)
    rows: dict = {}
    for threshold in thresholds:
        kept = times[confidence >= threshold]
        score = score_downbeats(kept, truth, tolerance)
        rows[float(threshold)] = {**{key: score[key] for key in ("tp", "fp", "fn")},
                                  "kept": int(kept.size), "total": int(times.size)}
    return rows


def phase_scores(decisions, subdivision: int, expert_times, expert_phases,
                 tolerance: float = TOLERANCE_SEC) -> dict:
    """Bar-phase accuracy against the annotated grid, with its own coverage.

    Scored per *annotated beat*, because that is what carries a truth phase.  A
    beat with no committed candidate within the tolerance is **uncovered**, not
    wrong -- and the coverage is reported beside the accuracy rather than folded
    into it, since on the live condition the two move in opposite directions.

    A candidate committed to an interstitial half-beat position has no bar phase
    at all (``bar_phase`` returns 0).  It is counted **wrong**, not skipped:
    excluding it would flatter a decoder that locked onto aubio's off-beats.
    """
    expert_times = np.asarray(expert_times, dtype=np.float64).reshape(-1)
    expert_phases = np.asarray(expert_phases, dtype=np.int64).reshape(-1)
    times = np.asarray([d.time for d in decisions], dtype=np.float64)
    phases = np.asarray([d.phase for d in decisions], dtype=np.int64)

    correct = covered = interstitial = 0
    if times.size and expert_times.size:
        index = nearest_index(expert_times, times)
        near = np.abs(times[index] - expert_times) <= tolerance
        mapped = np.asarray([bar_phase(int(p), subdivision) for p in phases[index]],
                            dtype=np.int64)
        covered = int(np.count_nonzero(near))
        interstitial = int(np.count_nonzero(near & (mapped == 0)))
        correct = int(np.count_nonzero(near & (mapped == expert_phases)))
    total = int(expert_times.size)
    return {
        "correct": correct,
        "covered": covered,
        "total": total,
        "interstitial": interstitial,
        "accuracy": correct / covered if covered else 0.0,
        "coverage": covered / total if total else 0.0,
    }


def interval_deviation(downbeats, *, deviation: float = INTERVAL_DEVIATION,
                       window: int = INTERVAL_WINDOW) -> dict:
    """Grid-direct stability: bar intervals that jump away from the running one.

    The metric review recommended, and the reason is worth keeping: at
    subdivision 2 ``phase_flips`` counts every candidate aubio inserts or drops,
    so a *perfectly steady* bar grid over a noisy beat stream can read as
    hundreds of flips.  This reads the emitted grid instead of its input.
    """
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    if downbeats.size < 3:
        return {"events": 0, "intervals": max(int(downbeats.size) - 1, 0),
                "minutes": 0.0, "per_minute": 0.0}
    intervals = np.diff(downbeats)
    running = rolling_median(intervals, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(running > 0, np.abs(intervals - running) / running, 0.0)
    minutes = float(downbeats[-1] - downbeats[0]) / 60.0
    events = int(np.count_nonzero(relative > deviation))
    return {
        "events": events,
        "intervals": int(intervals.size),
        "minutes": minutes,
        "per_minute": events / minutes if minutes > 0 else 0.0,
    }


def beat_anchored_flips(decisions, subdivision: int) -> dict:
    """Did the bar position advance by exactly one beat between real beats?

    The plan's stability unit -- flips per track -- restricted to the candidates
    that are actual beats, which is what makes it comparable across subdivisions.
    At subdivision 1 it is ``phase_flips`` (proved by test); at 2 it ignores the
    interstitial candidates, so a stream locked half a beat off the music reads
    as *stable*, which it is: it produces a perfectly steady bar grid.

    A pair with a coasted candidate between its members is a **break**, not a
    flip.  Coasting is the decoder's answer to aubio dropping beats; counting the
    phase it walks through as instability would charge the decoder for its input.
    """
    cycle = BEATS_PER_BAR * int(subdivision)
    records: list = []
    grid_index = 0
    coasted_since = False
    for decision in decisions:
        if decision.virtual:
            coasted_since = True
            continue
        if grid_index % int(subdivision) == 0:
            records.append((int(decision.phase), coasted_since))
            coasted_since = False
        grid_index += 1

    flips = pairs = breaks = 0
    for (previous, _), (current, broken) in zip(records, records[1:]):
        if broken:
            breaks += 1
            continue
        pairs += 1
        if (current - previous) % cycle != int(subdivision) % cycle:
            flips += 1
    return {"flips": flips, "pairs": pairs, "breaks": breaks,
            "beats": len(records)}


# --------------------------------------------------------------------------- #
# 3. The show ablation's grid adapter
# --------------------------------------------------------------------------- #


def edges_from_downbeats(downbeats) -> np.ndarray:
    """Predicted downbeats -> bar edges, by ``decoder.bar_grid``'s own rule.

    ``B`` bars need ``B + 1`` edges and a decoded grid ends on a downbeat, so the
    last bar is closed at the median interval -- the same closing rule the
    annotated grid gets, because the ablation must differ in the *grid* and in
    nothing else.
    """
    edges = np.unique(np.asarray(downbeats, dtype=np.float64).reshape(-1))
    if edges.size < 2:
        raise RuntimeError(
            f"{edges.size} predicted downbeats -- there is no bar grid to decode "
            f"on (the section decoder runs at bar rate by design)")
    return np.append(edges, edges[-1] + float(np.median(np.diff(edges))))


# --------------------------------------------------------------------------- #
# Provenance and split hygiene
# --------------------------------------------------------------------------- #


def config_fingerprint(params: PhaseParams, condition: str, *,
                       refine: bool) -> dict:
    """A stable identity for "which decoder produced this number".

    Both halves matter: the sha is what a report quotes and what a frozen config
    is checked against, and the JSON beside it is what a reader can actually
    audit.  A hash nobody can invert is provenance theatre.
    """
    payload = {**{key: (float(value) if isinstance(value, float) else value)
                  for key, value in asdict(params).items()},
               "condition": str(condition), "refine": bool(refine),
               "tolerance_sec": float(TOLERANCE_SEC)}
    document = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {"config": document,
            "sha256": hashlib.sha256(document.encode("utf-8")).hexdigest()}


def split_guard(data_dir, ids, split: str = "val") -> list:
    """Refuse ids outside ``split`` -- membership, not the flag that selected it.

    Task 3's lesson: a guard that only covers the default path is not a guard,
    because an explicit id list walks straight past a flag-level check.
    """
    path = Path(data_dir) / SPLITS_FILE
    if not path.exists():
        raise RuntimeError(f"no splits at {path} -- it is never regenerated implicitly")
    document = json.loads(path.read_text(encoding="utf-8"))
    if split not in document:
        raise RuntimeError(f"{path} has no {split!r} split (has {sorted(document)})")
    allowed = {str(i) for i in document[split]}
    ids = [str(i) for i in ids]
    outside = [i for i in ids if i not in allowed]
    if outside:
        raise RuntimeError(
            f"{len(outside)} of the {len(ids)} requested ids are not in {split} "
            f"({outside[:5]}{' ...' if len(outside) > 5 else ''}) -- this mode "
            f"reads annotated truth to choose a value, which is a tuning read")
    return ids


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Corpus plumbing
# --------------------------------------------------------------------------- #


def model_dir(data_dir) -> Path:
    return Path(data_dir) / MODELS_DIR / MODEL_VERSION


def sidecar_path(data_dir, youtube_id: str) -> Path:
    return Path(data_dir) / SIDECAR_DIR / f"{youtube_id}.npz"


def load_truth(data_dir, ids) -> dict:
    """``{youtube_id: (beat_times, beat_phases, downbeat_times)}`` from the grids.

    Imported lazily: ``downbeat_dataset`` pulls the training dataset and with it
    torch, and every pure helper above must stay importable without either.
    """
    from .downbeat_dataset import load_beat_grids

    grids, missing = load_beat_grids(Path(data_dir), ids)
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(list(ids))} tracks have no beat grid "
            f"({missing[:5]}) -- truth is not optional")
    return {youtube_id: (np.asarray(grid.times, dtype=np.float64),
                         np.asarray(grid.phases, dtype=np.int64),
                         np.asarray(grid.downbeat_times, dtype=np.float64))
            for youtube_id, grid in grids.items()}


def read_sidecar(path) -> dict:
    """The arrays a decode and an alignment need, read once per track."""
    with np.load(Path(path)) as archive:
        data = {"activation": np.asarray(archive["activation"], dtype=np.float64),
                "frame_sec": float(archive["frame_sec"]),
                "t0": float(archive["t0"]),
                "model_sha": str(archive["model_sha"])}
        for condition in CONDITIONS:
            data[f"{condition}_beat_time"] = np.asarray(
                archive[f"{condition}_beat_time"], dtype=np.float64)
            data[f"{condition}_beat_score"] = np.asarray(
                archive[f"{condition}_beat_score"], dtype=np.float64)
    return data


def refine_decisions(decisions, sidecar: dict) -> list:
    """De-quantise emitted instants against the stored activation curve.

    Its own function so the sweep can score refined and unrefined *from one
    decode* without a second code path deciding what refinement means -- the
    plan requires the two to be attributed separately, and separate attribution
    of two things computed by two implementations is not attribution.
    """
    from .downbeat_decoder import refine_instants

    return refine_instants(decisions, sidecar["activation"], sidecar["frame_sec"],
                           sidecar["t0"])


def decode_evidence(sidecar: dict, condition: str, params: PhaseParams, *,
                    refine: bool = False) -> list:
    """``decode_track`` off arrays already in memory.

    Identical by construction -- the subdivision-1 branch reuses the cached
    per-beat scores and the subdivision-2 branch re-aggregates off the stored
    curve, exactly as the shipped function does -- and pinned to it by a test on
    a real sidecar, because "identical by construction" is what every drifted
    copy said about itself.
    """
    from .downbeat_decoder import aggregate_at_beats

    beats = sidecar[f"{condition}_beat_time"]
    if int(params.subdivision) == 1:
        times, scores = beats, sidecar[f"{condition}_beat_score"]
    else:
        times = candidate_grid(beats, params.subdivision)
        scores, _counts = aggregate_at_beats(
            sidecar["activation"], times, sidecar["frame_sec"], sidecar["t0"])
    decisions = BarPhaseHMM(params).decode(times, scores)
    return refine_decisions(decisions, sidecar) if refine else decisions


def score_decisions(decisions, truth: tuple, subdivision: int) -> dict:
    """Every metric this module reports, for one decode."""
    beat_times, beat_phases, downbeats = truth
    predicted = downbeat_times(decisions)
    score = score_downbeats(predicted, downbeats)
    return {
        **{key: score[key] for key in ("precision", "recall", "f1", "tp", "fp", "fn")},
        "phase": phase_scores(decisions, subdivision, beat_times, beat_phases),
        "stability": beat_anchored_flips(decisions, subdivision),
        "interval": interval_deviation(predicted),
        "confidence": confidence_sweep(decisions, downbeats),
        "n_predicted": int(predicted.size),
        "n_truth": int(downbeats.size),
    }


def evaluate_track(sidecar: dict, truth: tuple, condition: str,
                   params: PhaseParams, *, refine: bool = False) -> dict:
    """One track, one config, decode and score."""
    return score_decisions(decode_evidence(sidecar, condition, params, refine=refine),
                           truth, params.subdivision)


def aggregate_rows(rows: dict) -> dict:
    """Micro totals plus the per-track distribution.

    Micro because a nine-minute track carries more bars than a ninety-second one
    and the corpus verdict counts bars; the medians beside it because a micro
    number cannot say whether a corpus is uniformly mediocre or bimodal, and this
    one is bimodal.
    """
    values = list(rows.values())
    if not values:
        return {}
    tp = sum(row["tp"] for row in values)
    fp = sum(row["fp"] for row in values)
    fn = sum(row["fn"] for row in values)
    per_track_f1 = np.asarray([row["f1"] for row in values], dtype=np.float64)
    flips = np.asarray([row["stability"]["flips"] for row in values], dtype=np.float64)
    deviation = np.asarray([row["interval"]["per_minute"] for row in values],
                           dtype=np.float64)
    phase_correct = sum(row["phase"]["correct"] for row in values)
    phase_covered = sum(row["phase"]["covered"] for row in values)
    phase_total = sum(row["phase"]["total"] for row in values)
    interstitial = sum(row["phase"]["interstitial"] for row in values)
    events = sum(row["interval"]["events"] for row in values)
    minutes = sum(row["interval"]["minutes"] for row in values)
    return {
        "tracks": len(values),
        **prf(tp, fp, fn),
        "f1_median": float(np.median(per_track_f1)),
        "f1_mean": float(np.mean(per_track_f1)),
        "phase_accuracy": phase_correct / phase_covered if phase_covered else 0.0,
        "phase_coverage": phase_covered / phase_total if phase_total else 0.0,
        # How often a real beat was committed to a half-beat position, which has
        # no bar phase at all.  Reported because it is the *mechanism* behind a
        # low phase accuracy on the half-beat grid, not a second symptom of it.
        "phase_interstitial_share": interstitial / phase_covered if phase_covered else 0.0,
        # The emitted-to-annotated downbeat ratio: the direct read on whether a
        # precision shortfall is wrong placement or simply too many bars.
        "predicted_per_truth": ((tp + fp) / (tp + fn)) if (tp + fn) else 0.0,
        "flips_median": float(np.median(flips)),
        "flips_mean": float(np.mean(flips)),
        "flips_max": float(np.max(flips)),
        "flips_le_1_share": float(np.mean(flips <= 1.0)),
        "interval_dev_per_min_median": float(np.median(deviation)),
        "interval_dev_per_min_micro": events / minutes if minutes else 0.0,
        "confidence": {
            str(threshold): {
                **prf(sum(row["confidence"][threshold]["tp"] for row in values),
                      sum(row["confidence"][threshold]["fp"] for row in values),
                      sum(row["confidence"][threshold]["fn"] for row in values)),
                "kept_share": (sum(row["confidence"][threshold]["kept"] for row in values)
                               / max(sum(row["confidence"][threshold]["total"]
                                         for row in values), 1)),
            }
            for threshold in values[0]["confidence"]
        },
    }


def _track_job(args) -> tuple:
    """Every config for one track, off one sidecar read and one decode each.

    Parallelism is per *track* rather than per config, which is what makes the
    numbers independent of the worker count: no track's row is ever split across
    processes, and a decode is a pure function of the arrays it is handed.
    """
    youtube_id, path, truth, specs = args
    sidecar = read_sidecar(path)
    rows: list = []
    for condition, params in specs:
        decisions = decode_evidence(sidecar, condition, params)
        rows.append(score_decisions(decisions, truth, params.subdivision))
        rows.append(score_decisions(refine_decisions(decisions, sidecar), truth,
                                    params.subdivision))
    return youtube_id, rows


def evaluate_configs(data_dir, ids, truth: dict, specs, *,
                     workers: int = 1) -> list:
    """``[{youtube_id: row}]`` -- two entries per spec, refine off then on."""
    data_dir = Path(data_dir)
    specs = list(specs)
    jobs = [(youtube_id, sidecar_path(data_dir, youtube_id), truth[youtube_id],
             specs) for youtube_id in ids]
    collected: dict = {}
    if workers <= 1:
        for job in jobs:
            youtube_id, rows = _track_job(job)
            collected[youtube_id] = rows
    else:
        with futures.ProcessPoolExecutor(max_workers=int(workers)) as pool:
            for youtube_id, rows in pool.map(_track_job, jobs, chunksize=1):
                collected[youtube_id] = rows
    return [{youtube_id: collected[youtube_id][index] for youtube_id in ids}
            for index in range(2 * len(specs))]


def evaluate_ids(data_dir, ids, truth: dict, condition: str, params: PhaseParams,
                 *, refine: bool = False, workers: int = 1) -> dict:
    """One config over a set of tracks; ``{youtube_id: row}``, id order."""
    both = evaluate_configs(data_dir, ids, truth, [(condition, params)],
                            workers=workers)
    return both[1 if refine else 0]


# --------------------------------------------------------------------------- #
# CLI modes
# --------------------------------------------------------------------------- #


def run_alignment(data_dir, ids, truth: dict) -> dict:
    """The corpus-wide aubio-vs-expert analysis and the residual decomposition."""
    data_dir = Path(data_dir)
    rows: dict = {}
    reach: dict = {label: 0 for label in REACH_LABELS}
    per_track_reach: dict = {}
    per_track_ceiling: dict = {}
    per_track_rate: dict = {}
    residuals: dict = {label: [] for label in REACH_LABELS}
    for youtube_id in ids:
        sidecar = read_sidecar(sidecar_path(data_dir, youtube_id))
        beat_times, _phases, downbeats = truth[youtube_id]
        aubio = sidecar["aubio_beat_time"]
        rows[youtube_id] = alignment_row(aubio, beat_times, downbeats)
        labels = reach_labels(aubio, downbeats, beat_times)
        counts = {label: labels.count(label) for label in REACH_LABELS}
        per_track_reach[youtube_id] = counts
        for label, count in counts.items():
            reach[label] += count
        found = sum(counts[label] for label in REACHED)
        per_track_ceiling[youtube_id] = {
            "beats": counts["beat"] / len(labels) if labels else 0.0,
            "grid": (counts["beat"] + counts["midpoint"]) / len(labels) if labels else 0.0,
            "decoder": found / len(labels) if labels else 0.0,
            # Bounds only: nothing decodes on these grids today (see subdivided_grid).
            "quarter_bound": grid_ceiling(aubio, downbeats, 4),
            "eighth_bound": grid_ceiling(aubio, downbeats, 8),
        }
        per_track_rate[youtube_id] = {
            "half_beat_grid": bar_rate_ratio(
                aubio, downbeats, PhaseParams(subdivision=2, lag_beats=lag_for(1, 2))),
            "beat_grid": bar_rate_ratio(
                aubio, downbeats, PhaseParams(subdivision=1, lag_beats=lag_for(1, 1))),
        }
        for label, residual in zip(labels, downbeat_residuals(aubio, downbeats, beat_times)):
            if np.isfinite(residual):
                residuals[label].append(float(residual))
    total = sum(reach.values())
    summary = {
        "tracks": len(rows),
        "downbeats": total,
        "reach_counts": reach,
        "reach_share": {label: (count / total if total else 0.0)
                        for label, count in reach.items()},
        "ceiling_beats": sum(reach[label] for label in ("beat",)) / total if total else 0.0,
        "ceiling_grid": sum(reach[label] for label in ("beat", "midpoint")) / total
                        if total else 0.0,
        "ceiling_decoder": sum(reach[label] for label in REACHED) / total if total else 0.0,
    }
    for key in ("median_abs_phase", "phase_iqr", "aubio_on_grid", "expert_covered",
                "downbeat_on_beats", "ibi_ratio", "median_offset_sec"):
        values = np.asarray([row[key] for row in rows.values()], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"{key}_median"] = float(np.median(finite)) if finite.size else float("nan")
    for key in ("beats", "grid", "decoder", "quarter_bound", "eighth_bound"):
        values = np.asarray([row[key] for row in per_track_ceiling.values()],
                            dtype=np.float64)
        summary[f"ceiling_{key}_micro"] = float(np.average(
            values, weights=[row["n_downbeats"] for row in rows.values()]))
        summary[f"ceiling_{key}_deciles"] = [
            float(np.percentile(values, share)) for share in range(0, 101, 10)]
        summary[f"ceiling_{key}_tracks_at_85"] = int(np.count_nonzero(values >= 0.85))
    downbeat_total = sum(row["n_downbeats"] for row in rows.values())
    for key in ("half_beat_grid", "beat_grid"):
        values = np.asarray([row[key] for row in per_track_rate.values()],
                            dtype=np.float64)
        weights = np.asarray([row["n_downbeats"] for row in rows.values()],
                             dtype=np.float64)
        micro = float(np.sum(values * weights) / downbeat_total) if downbeat_total else float("nan")
        summary[f"bar_rate_{key}_micro"] = micro
        summary[f"bar_rate_{key}_median"] = float(np.median(values))
        # What a flip-free decode can reach given BOTH ceilings: it can only place
        # a downbeat where a candidate is (coverage) and it emits one per cycle
        # (rate), so F1 <= 2 * coverage / (1 + rate).
        coverage = (summary["ceiling_decoder"] if key == "half_beat_grid"
                    else summary["ceiling_beats"])
        summary[f"f1_ceiling_{key}"] = 2.0 * coverage / (1.0 + micro)
    summary["residual_histogram"] = {
        label: np.histogram(values, bins=RESIDUAL_BINS)[0].tolist() if values else []
        for label, values in residuals.items()}
    summary["residual_median"] = {
        label: (float(np.median(values)) if values else float("nan"))
        for label, values in residuals.items()}
    summary["residual_bins"] = list(RESIDUAL_BINS)
    return {"summary": summary, "per_track": rows, "per_track_reach": per_track_reach,
            "per_track_ceiling": per_track_ceiling, "per_track_rate": per_track_rate}


def sweep_rows(data_dir, ids, truth: dict, specs, *, workers: int = 1) -> list:
    """Every ``(condition, params)`` spec, aggregated, refinement off and on.

    Refinement is always reported as its own row off the *same* decode: the plan
    requires de-quantisation to be attributable separately from the phase model,
    and two rows that shared no decode could differ for a second reason.
    """
    specs = list(specs)
    per_config = evaluate_configs(data_dir, ids, truth, specs, workers=workers)
    rows: list = []
    for index, (condition, params) in enumerate(specs):
        for offset, refine in ((0, False), (1, True)):
            rows.append({
                "condition": condition,
                "params": {key: (float(v) if isinstance(v, float) else v)
                           for key, v in asdict(params).items()},
                "look_ahead_beats": params.lag_beats / params.subdivision,
                "refine": refine,
                **config_fingerprint(params, condition, refine=refine),
                "aggregate": aggregate_rows(per_config[2 * index + offset]),
            })
    return rows


def naive_grids(data_dir, ids) -> dict:
    """Bars from every fourth aubio beat -- the show's null hypothesis.

    The engine could do this today with no model at all: take the beat stream,
    call every fourth one a bar line.  It is wrong about *phase* by construction
    (it starts wherever aubio started) but it is right about *rate* whenever
    aubio is, so it is the baseline any claim about the downbeat model's value to
    a show has to clear.  A number without its null is a decoration.
    """
    data_dir = Path(data_dir)
    return {youtube_id: read_sidecar(sidecar_path(data_dir, youtube_id))
                        ["aubio_beat_time"][::BEATS_PER_BAR]
            for youtube_id in ids}


def ablation_rows(data_dir, ids, predicted: dict, *, section_dir: str,
                  models_subdir: str, naive: dict | None = None) -> dict:
    """Section decoding on the predicted bar grid vs the expert one.

    Everything except the grid is held fixed: the same posterior sidecars, the
    same priors, the same frozen decoder config, the same tracks, the same
    scoring functions.  The delta is therefore attributable to the grid, which is
    the only claim the plan asks this to support.  ``naive`` adds the third
    column that says whether the model earned its place at all.
    """
    from .decoder import DecodeParams, bar_grid
    from .evaluate_v1 import (
        DEFAULT_SPACE,
        POSTERIORS_DIR,
        TrackInputs,
        bar_observations,
        evaluate_config,
        load_decoder_config,
    )
    from .priors import Priors
    from build_training_table import TABLE_FILE
    from evaluate_against_labels import TOLERANCES_SEC as SECTION_TOLERANCES_SEC
    from evaluate_against_labels import load_tracks
    from raveform_fetch_annotations import BEATS_DIR, annotations_dir

    data_dir = Path(data_dir)
    models = data_dir / MODELS_DIR / models_subdir
    params = load_decoder_config(models / "decoder_config.json")
    priors = Priors.load(models / "priors.json")
    by_youtube_id = {t.track_id.split(".", 1)[-1]: t
                     for t in load_tracks(data_dir / TABLE_FILE)}
    beats_dir = annotations_dir(data_dir) / BEATS_DIR
    posteriors_dir = data_dir / (section_dir or POSTERIORS_DIR)

    expert_inputs: list = []
    predicted_inputs: list = []
    naive_inputs: list = []
    skipped: list = []
    for youtube_id in ids:
        track = by_youtube_id.get(youtube_id)
        sidecar = posteriors_dir / f"{youtube_id}.npz"
        if track is None or not sidecar.exists():
            skipped.append({"youtube_id": youtube_id,
                            "reason": "no table rows" if track is None
                                      else "no posterior sidecar"})
            continue
        beat_csv = beats_dir / f"{track.track_id}.beat.csv"
        if not beat_csv.exists():
            skipped.append({"youtube_id": youtube_id, "reason": "no beat grid"})
            continue
        try:
            columns = [(bar_grid(beat_csv), expert_inputs),
                       (edges_from_downbeats(predicted[youtube_id]), predicted_inputs)]
            if naive is not None:
                columns.append((edges_from_downbeats(naive[youtube_id]), naive_inputs))
        except RuntimeError as error:
            skipped.append({"youtube_id": youtube_id, "reason": str(error)})
            continue
        for edges, bucket in columns:
            posteriors, boundary = bar_observations(
                sidecar, edges, min_coverage=params.min_coverage,
                boundary_tolerance_sec=params.boundary_tolerance_sec)
            bucket.append(TrackInputs(
                track_id=track.track_id, youtube_id=youtube_id, edges=edges,
                posteriors=posteriors, boundary=boundary, times=track.times,
                labels=track.labels, intents=track.intents))

    def column(inputs) -> dict:
        result = evaluate_config(inputs, priors, params, space=DEFAULT_SPACE)
        score = result["score"]
        # The bar count is reported because it is the mechanism: a grid with
        # twice the downbeats gives the section decoder twice the bars, so its
        # 3-bar lag and its bar-counted duration priors are quietly re-scaled.
        bars = int(sum(item.edges.size - 1 for item in inputs))
        boundary = {}
        for tolerance in SECTION_TOLERANCES_SEC:
            precision, recall, f1 = score.boundary_prf("class", tolerance)
            boundary[f"{tolerance}"] = {"precision": precision, "recall": recall,
                                        "f1": f1}
        return {
            "tracks": score.tracks,
            "bars": bars,
            "macro_f1": float(score.macro_f1),
            "boundary": boundary,
            "flicker_per_audience_minute": {
                f"{tolerance}": float(rate)
                for tolerance, rate in score.flicker_per_minute["class"].items()},
            "undecoded_share": float(score.no_intent_sec / score.exposure_sec
                                     if score.exposure_sec else 0.0),
        }

    expert_column = column(expert_inputs)
    predicted_column = column(predicted_inputs)
    return {
        "section_chain": models_subdir,
        **({"naive_grid": column(naive_inputs)} if naive is not None else {}),
        "posteriors_dir": str(posteriors_dir.name),
        "decoder_config_sha256": file_sha256(models / "decoder_config.json"),
        "priors_sha256": file_sha256(models / "priors.json"),
        "skipped": skipped,
        "expert_grid": expert_column,
        "predicted_grid": predicted_column,
        "delta": {
            "bars_ratio": (predicted_column["bars"] / expert_column["bars"]
                           if expert_column["bars"] else float("nan")),
            "macro_f1": predicted_column["macro_f1"] - expert_column["macro_f1"],
            "boundary_f1": {
                key: predicted_column["boundary"][key]["f1"]
                     - expert_column["boundary"][key]["f1"]
                for key in expert_column["boundary"]},
            "flicker_per_audience_minute": {
                key: predicted_column["flicker_per_audience_minute"][key]
                     - expert_column["flicker_per_audience_minute"][key]
                for key in expert_column["flicker_per_audience_minute"]},
        },
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "training" / "data" / "raveform"


def read_split(data_dir, split: str) -> list:
    path = Path(data_dir) / SPLITS_FILE
    document = json.loads(path.read_text(encoding="utf-8"))
    return [str(i) for i in document[split]]


def sweep_grid(look_ahead, penalties, subdivisions) -> list:
    """The committed sweep grid, in wall-clock look-ahead rather than in lag.

    The axis is *beats of look-ahead*; the lag in candidates is derived from it
    and the subdivision.  Stated this way round because the other way round is
    the confound Task 3 measured: at a fixed lag, doubling the grid halves the
    look-ahead, and the sweep then reads a lag effect as a grid effect.
    """
    return [PhaseParams(lag_beats=lag_for(beats_ahead, subdivision),
                        subdivision=subdivision, flip_penalty=float(penalty))
            for subdivision in subdivisions
            for beats_ahead in look_ahead
            for penalty in penalties]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Downbeat tracking verdict: alignment, val sweep, one test read")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--align", action="store_true",
                      help="corpus aubio-vs-expert alignment + residual decomposition")
    mode.add_argument("--sweep", action="store_true",
                      help="the val lever sweep (flip penalty x look-ahead x refine)")
    mode.add_argument("--freeze", action="store_true",
                      help="write the chosen config, hashed, BEFORE any test read")
    mode.add_argument("--verdict", action="store_true",
                      help="score the frozen config; --split test is the one read")
    parser.add_argument("--flip-penalty", type=float, default=None)
    parser.add_argument("--look-ahead", type=int, default=LOOK_AHEAD_BUDGET_BEATS)
    parser.add_argument("--subdivision", type=int, default=2)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--penalties", type=float, nargs="*",
                        default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0])
    parser.add_argument("--look-aheads", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--subdivisions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--role", default="headline",
                        help="what this frozen config is FOR, recorded in the file")
    parser.add_argument("--config", default=None,
                        help="where the frozen config lives (--freeze writes it, "
                             "--verdict reads it)")
    parser.add_argument("--ablate", action="store_true",
                        help="with --verdict: the show ablation on the same tracks")
    # Comma-separated and zipped: the ablation can run against more than one
    # generation of the section chain inside ONE test read, which is the only way
    # to report a robustness check without spending a second read on it.
    parser.add_argument("--section-dir", default="posteriors")
    parser.add_argument("--section-models", default="v1")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    ids = args.ids if args.ids is not None else read_split(data_dir, args.split)
    if args.limit:
        ids = ids[:args.limit]

    if args.align or args.sweep:
        # Both read annotated truth to choose a value.  Val, by membership.
        ids = split_guard(data_dir, ids, "val")
    truth = load_truth(data_dir, ids)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()

    if args.align:
        payload = {"generated_at": stamp, "split": "val", **run_alignment(data_dir, ids, truth)}
        out = Path(args.out) if args.out else model_dir(data_dir) / ALIGNMENT_FILE.format(split="val")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["summary"], indent=2))
        return 0

    if args.sweep:
        specs = [(condition, params) for condition in CONDITIONS
                 for params in sweep_grid(args.look_aheads, args.penalties,
                                          args.subdivisions)]
        rows = sweep_rows(data_dir, ids, truth, specs, workers=args.workers)
        payload = {"generated_at": stamp, "split": "val", "tracks": len(ids),
                   "rows": rows}
        out = Path(args.out) if args.out else model_dir(data_dir) / "downbeat_sweep_val.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for row in sorted(rows, key=lambda r: -r["aggregate"]["f1"])[:12]:
            print(f"{row['condition']:6s} sub {row['params']['subdivision']} "
                  f"look-ahead {row['look_ahead_beats']:.0f} beats "
                  f"flip {row['params']['flip_penalty']:5.1f} "
                  f"refine {int(row['refine'])} -> F1 {row['aggregate']['f1']:.4f} "
                  f"P {row['aggregate']['precision']:.3f} R {row['aggregate']['recall']:.3f} "
                  f"flips med {row['aggregate']['flips_median']:.0f}")
        return 0

    if args.freeze:
        if args.flip_penalty is None:
            raise SystemExit("--freeze needs --flip-penalty: the config is a choice")
        params = PhaseParams(lag_beats=lag_for(args.look_ahead, args.subdivision),
                             subdivision=args.subdivision,
                             flip_penalty=float(args.flip_penalty))
        payload = {"frozen_at": stamp,
                   # The role is written into the file, not decided afterwards:
                   # a pre-registration that does not say which config is the
                   # headline is not a pre-registration.
                   "role": args.role,
                   "chosen_on": "val", "condition": "aubio",
                   "refine": bool(args.refine),
                   "look_ahead_beats": args.look_ahead,
                   **config_fingerprint(params, "aubio", refine=args.refine)}
        out = Path(args.config) if args.config else model_dir(data_dir) / CONFIG_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    config_path = Path(args.config) if args.config else model_dir(data_dir) / CONFIG_FILE
    if not config_path.exists():
        raise SystemExit(
            f"no frozen config at {config_path} -- run --freeze first, so the "
            f"choice provably predates the read")
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    chosen = json.loads(frozen["config"])
    params = PhaseParams(**{key: value for key, value in chosen.items()
                            if key not in ("condition", "refine", "tolerance_sec")})
    refine = bool(chosen["refine"])

    rows = {condition: evaluate_ids(data_dir, ids, truth, condition, params,
                                    refine=refine, workers=args.workers)
            for condition in CONDITIONS}
    payload = {
        "generated_at": stamp,
        "split": args.split,
        "tracks": len(ids),
        "frozen_config": frozen,
        "frozen_config_sha256": file_sha256(config_path),
        "model_sha": read_sidecar(sidecar_path(data_dir, ids[0]))["model_sha"],
        "conditions": {condition: aggregate_rows(row) for condition, row in rows.items()},
        "per_track": {condition: row for condition, row in rows.items()},
    }
    if args.ablate:
        predicted = {}
        for youtube_id in ids:
            sidecar = read_sidecar(sidecar_path(data_dir, youtube_id))
            predicted[youtube_id] = downbeat_times(
                decode_evidence(sidecar, "aubio", params, refine=refine))
        chains = list(zip(args.section_dir.split(","),
                          args.section_models.split(",")))
        naive = naive_grids(data_dir, ids)
        payload["ablation"] = {
            models_subdir: ablation_rows(data_dir, ids, predicted,
                                         section_dir=section_dir,
                                         models_subdir=models_subdir, naive=naive)
            for section_dir, models_subdir in chains}
    out = Path(args.out) if args.out else model_dir(data_dir) / EVAL_FILE.format(split=args.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"split": args.split, "tracks": len(ids),
                      "conditions": payload["conditions"],
                      **({"ablation": {chain: {"tracks": row["expert_grid"]["tracks"],
                                               **row["delta"]}
                                       for chain, row in payload["ablation"].items()}}
                         if args.ablate else {})}, indent=2))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
