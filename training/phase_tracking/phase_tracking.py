"""Live bar-PHASE trackers: grid builders, the slip inventory, and the truth.

1d priced the shipping fallback -- bars every 4 madmom online beats from the
first detected beat -- at 0.4966 crispness@0.5 against 0.6362 for the same rule
on annotated beats.  The whole -0.1396 is booked by phase slips: a missed or
spurious beat rotates a counted grid for the rest of the track.

Everything here builds a bar grid from the LIVE beat stream, causally, with
corrections applied FORWARD ONLY -- a correction that arrives N beats late
re-times future bar lines and never rewrites one already emitted.
"""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np

MATCH_TOL_SEC = 0.07


def annotation_beats(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (np.asarray([float(r["time"]) for r in rows], dtype=np.float64),
            np.asarray([int(r["downbeat"]) for r in rows], dtype=np.int64))


def expert_phase(times: np.ndarray, positions: np.ndarray):
    expert = np.asarray(sorted(t for t, p in zip(times, positions) if p == 1))
    for phase in range(4):
        candidate = times[phase::4]
        if candidate.shape == expert.shape and np.allclose(candidate, expert,
                                                           atol=1e-6):
            return phase
    return None


def match(madmom: np.ndarray, expert: np.ndarray, tol: float = MATCH_TOL_SEC):
    """``(j_indices, i_indices)`` -- one-to-one nearest matches within ``tol``.

    Lifted verbatim from ``task1c_beat_slip`` so this artifact and 1c/1d report
    one measurement rather than three of them.
    """
    if madmom.size == 0 or expert.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    idx = np.searchsorted(expert, madmom)
    idx = np.clip(idx, 1, len(expert) - 1)
    left, right = expert[idx - 1], expert[idx]
    nearest = np.where(np.abs(madmom - left) <= np.abs(madmom - right),
                       idx - 1, idx)
    delta = np.abs(madmom - expert[nearest])
    ok = delta <= tol
    js, ins, ds = np.nonzero(ok)[0], nearest[ok], delta[ok]
    best = {}
    for j, i, d in zip(js, ins, ds):
        if i not in best or d < best[i][1]:
            best[i] = (j, d)
    pairs = sorted((j, i) for i, (j, _) in best.items())
    return (np.asarray([p[0] for p in pairs], np.int64),
            np.asarray([p[1] for p in pairs], np.int64))


def required_phase(js: np.ndarray, isx: np.ndarray, phase: int) -> np.ndarray:
    return (js - (isx - phase)) % 4


def truth_phase_track(n_beats: int, js: np.ndarray, required: np.ndarray):
    """Per live beat, the counting phase that would put bars on real downbeats.

    A step function: the value observed at the last matched beat at or before
    ``j``, carried forward.  Beats before the first match get the first value,
    which is the only defensible extension and is flagged by ``known``.
    """
    out = np.full(n_beats, -1, dtype=np.int64)
    known = np.zeros(n_beats, dtype=bool)
    if js.size == 0:
        return out, known
    out[:] = required[0]
    for k in range(js.size):
        lo = js[k]
        hi = js[k + 1] if k + 1 < js.size else n_beats
        out[lo:hi] = required[k]
        known[lo:hi] = True
    known[:js[0]] = False
    return out, known


# --------------------------------------------------------------------------- #
# bar lines from a per-beat bar-position sequence
# --------------------------------------------------------------------------- #
def close_grid(edges: np.ndarray) -> np.ndarray:
    """``bar_grid``'s own closing rule: B bars need B+1 edges."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        raise RuntimeError("fewer than two bar edges")
    return np.append(edges, edges[-1] + float(np.median(np.diff(edges))))


def edges_from_positions(beats: np.ndarray, positions: np.ndarray,
                         advances: np.ndarray | None = None,
                         interpolate: bool = False) -> np.ndarray:
    """Bar lines: the beats at which the running count wraps to 0.

    ``interpolate`` additionally emits a line inside an interval the count
    stepped OVER -- a beat the tracker believes was dropped, which carried the
    bar line.  The emitted timestamp is in the past by at most one beat when the
    step is detected, which the 13.66 s show delay absorbs.
    """
    lines = []
    for j in range(len(beats)):
        if j == 0:
            if positions[0] == 0:
                lines.append(beats[0])
            continue
        step = 1 if advances is None else int(advances[j])
        if interpolate and step >= 2:
            start = int(positions[j - 1])
            span = beats[j] - beats[j - 1]
            for k in range(1, step):
                if (start + k) % 4 == 0:
                    lines.append(beats[j - 1] + span * (k / step))
        if positions[j] == 0 and positions[j - 1] != 0:
            lines.append(beats[j])
    return np.asarray(lines, dtype=np.float64)


# --------------------------------------------------------------------------- #
# candidate A -- interval-signature slip repair, no model input at all
# --------------------------------------------------------------------------- #
def interval_ratios(beats: np.ndarray, *, window: int = 8, min_history: int = 3,
                    clean_lo: float = 0.75, clean_hi: float = 1.35) -> np.ndarray:
    """Each interval over the running median of the recent CLEAN ones.

    Split out from the repair because the ratios depend on the history window
    alone -- the repair thresholds cannot move them -- so a sweep over those
    thresholds pays the running median once per window instead of once per
    configuration.
    """
    n = len(beats)
    ratios = np.full(n, np.nan, dtype=np.float64)
    history: deque = deque(maxlen=window)
    for j in range(1, n):
        span = beats[j] - beats[j - 1]
        period = float(np.median(history)) if len(history) >= min_history else 0.0
        if period > 0.0:
            ratios[j] = span / period
        if period <= 0.0 or clean_lo <= span / period <= clean_hi:
            history.append(span)
    return ratios


def repair_from_ratios(ratios: np.ndarray, *, del_lo: float = 1.5,
                       ins_hi: float = 0.65, pair_lo: float = 0.70,
                       pair_hi: float = 1.35, max_advance: int = 8):
    """``(positions, advances)`` -- how many bar positions each interval crosses.

    A dropped beat leaves one ~double interval, so the count advances by
    ``round(r)``.  A spurious beat splits one interval into two ~halves, so the
    count holds on the beat that closes the pair.  Both are decidable at the
    beat that ends the signature, i.e. with zero look-ahead.
    """
    n = len(ratios)
    positions = np.zeros(n, dtype=np.int64)
    advances = np.zeros(n, dtype=np.int64)
    for j in range(1, n):
        r = ratios[j]
        step = 1
        if np.isfinite(r):
            previous = ratios[j - 1]
            if r >= del_lo:
                step = int(min(max(round(r), 1), max_advance))
            elif (r <= ins_hi and np.isfinite(previous) and previous <= ins_hi
                    and pair_lo <= previous + r <= pair_hi):
                step = 0
        positions[j] = (positions[j - 1] + step) % 4
        advances[j] = step
    return positions, advances


def interval_repair(beats: np.ndarray, *, window: int = 8, min_history: int = 3,
                    del_lo: float = 1.5, ins_hi: float = 0.65,
                    pair_lo: float = 0.70, pair_hi: float = 1.35,
                    clean_lo: float = 0.75, clean_hi: float = 1.35,
                    max_advance: int = 8):
    ratios = interval_ratios(beats, window=window, min_history=min_history,
                             clean_lo=clean_lo, clean_hi=clean_hi)
    positions, advances = repair_from_ratios(
        ratios, del_lo=del_lo, ins_hi=ins_hi, pair_lo=pair_lo, pair_hi=pair_hi,
        max_advance=max_advance)
    return positions, advances, ratios


# --------------------------------------------------------------------------- #
# candidate B -- a 4-state fixed-lag phase filter
# --------------------------------------------------------------------------- #
def advance_log_weights(ratios: np.ndarray, *, sigma: float, eta: float,
                        max_advance: int) -> np.ndarray:
    """``[n, max_advance+1]`` log weights for "this interval spans ``a`` beats".

    ``a = 0`` is the second half of a split interval, and it is scored on the
    PAIR -- this interval near a half AND the two together near one whole.  A
    lone half-interval is evidence of nothing: both halves of a split look
    identical, so a weight read from one interval alone gives the same answer
    twice and the count either misses the insertion or over-corrects it.  That
    is a structural inability to express an insertion, not a tuning miss.
    """
    n = len(ratios)
    out = np.zeros((n, max_advance + 1), dtype=np.float64)
    centres = np.arange(max_advance + 1, dtype=np.float64)
    prior = np.full(max_advance + 1, np.log(eta), dtype=np.float64)
    prior[1] = 0.0
    for a in range(2, max_advance + 1):
        prior[a] = np.log(eta) * (a - 1)
    variance = 2.0 * sigma * sigma
    for j in range(n):
        r = ratios[j]
        if not np.isfinite(r):
            out[j] = -np.inf
            out[j, 1] = 0.0
            continue
        out[j] = prior - (r - centres) ** 2 / variance
        previous = ratios[j - 1] if j else np.nan
        if np.isfinite(previous):
            out[j, 0] = prior[0] - ((r - 0.5) ** 2
                                    + (previous + r - 1.0) ** 2) / variance
        else:
            out[j, 0] = -np.inf
    return out


def flat_log_weights(n: int, *, slip_rate: float, max_advance: int) -> np.ndarray:
    """The interval-blind ablation: a fixed slip probability, no ratio read."""
    out = np.full((n, max_advance + 1), -np.inf, dtype=np.float64)
    out[:, 1] = np.log(1.0 - 2.0 * slip_rate)
    out[:, 0] = np.log(slip_rate)
    out[:, 2] = np.log(slip_rate)
    return out


def causal_z(values: np.ndarray, warmup: int) -> np.ndarray:
    """Running standardisation over past samples only.  NaN reads score 0."""
    out = np.zeros(len(values), dtype=np.float64)
    count = 0
    mean = 0.0
    m2 = 0.0
    for j, v in enumerate(values):
        if count >= warmup and m2 > 0.0:
            out[j] = (v - mean) / np.sqrt(m2 / count) if np.isfinite(v) else 0.0
        if np.isfinite(v):
            count += 1
            delta = v - mean
            mean += delta / count
            m2 += delta * (v - mean)
    return out


def fixed_lag_viterbi(log_transition: np.ndarray, log_emission: np.ndarray,
                      log_start: np.ndarray, lag_beats: int) -> np.ndarray:
    """Per beat, the state decided ``lag_beats`` later by backtracking.

    The project's own decoder shape: a forward Viterbi with backpointers, read
    ``lag_beats`` steps behind the running best path.  Beats inside the final
    lag are decided by the last available backtrack.
    """
    n_steps, n_advance = log_transition.shape
    n = n_steps + 1
    delta = log_start + log_emission[0]
    back = np.zeros((n, 4), dtype=np.int64)
    decided = np.full(n, -1, dtype=np.int64)
    residue = np.arange(n_advance) % 4
    best_by_residue = np.full((n_steps, 4), -np.inf, dtype=np.float64)
    for r in range(4):
        columns = log_transition[:, residue == r]
        if columns.size:
            best_by_residue[:, r] = columns.max(axis=1)
    rotation = (np.arange(4)[:, None] - np.arange(4)[None, :]) % 4
    index = np.arange(4)
    for j in range(1, n):
        candidate = delta[None, :] + best_by_residue[j - 1][rotation]
        back[j] = np.argmax(candidate, axis=1)
        delta = candidate[index, back[j]] + log_emission[j]
        delta -= delta.max()
        if j >= lag_beats:
            state = int(np.argmax(delta))
            for k in range(j, j - lag_beats, -1):
                state = int(back[k, state])
            decided[j - lag_beats] = state
    state = int(np.argmax(delta))
    for k in range(n - 1, max(n - 1 - lag_beats, -1), -1):
        if decided[k] < 0:
            decided[k] = state
        state = int(back[k, state])
    return decided


def apply_forward_only(base_positions: np.ndarray, decided: np.ndarray,
                       lag_beats: int):
    """Emit on the running count; rotate it when a late decision disagrees.

    The committed past is immutable: at beat ``j`` the line is emitted from the
    offset in force, and only then does the decision for beat ``j - lag`` move
    the offset for everything after ``j``.

    The disagreement is measured against what the CURRENT offset would put at
    beat ``k``, never against the value committed there at the time.  Comparing
    against history re-detects one disagreement on every beat of the lag window
    and rotates once per beat -- a single slip then compounds into ``lag``
    rotations, which is thrashing dressed as tracking.
    """
    n = len(base_positions)
    committed = np.zeros(n, dtype=np.int64)
    offset = 0
    corrections = []
    for j in range(n):
        committed[j] = (base_positions[j] + offset) % 4
        k = j - lag_beats
        if k >= 0 and decided[k] >= 0:
            target = (base_positions[k] + offset) % 4
            if decided[k] != target:
                offset = (offset + decided[k] - target) % 4
                corrections.append(j)
    return committed, corrections


# Where the first DETECTED beat actually sits in the bar, measured over the 215
# val tracks: madmom's online warmup normally costs the first annotated beat, so
# the first thing it emits is bar position 1 on 147 of 215 tracks and position 0
# on only 51.  The shipping fallback assumes position 0, which is why it is on
# the correct phase for a median 0.7 % of a track's beats.
LIVE_START_PRIOR = (51.0 / 215.0, 147.0 / 215.0, 6.0 / 215.0, 11.0 / 215.0)


def phase_filter(beats: np.ndarray, boundary_samples: np.ndarray | None, *,
                 lag_beats: int = 8, sigma: float = 0.22, eta: float = 0.02,
                 max_advance: int = 4, beta: float = 0.0, warmup: int = 16,
                 slip_rate: float | None = None, start_prior=LIVE_START_PRIOR,
                 anchor: int = 0, base: tuple | None = None,
                 ratios: np.ndarray | None = None):
    """``(committed positions, advances, corrections, decided)``.

    ``slip_rate`` not None selects the interval-blind ablation.  ``beta`` 0.0
    selects the boundary-blind one.  ``base`` supplies candidate A's repaired
    count as the thing the corrections rotate (candidate C); without it the
    count is a plain +1 per beat, which is what ships.

    ``start_prior`` is over the bar position of the FIRST DETECTED beat, not
    over the annotated grid's phase.  Those are different questions and the
    annotated one (84 % on phase 0) is the wrong answer to this one.
    """
    n = len(beats)
    if ratios is None:
        ratios = interval_ratios(beats)
    if slip_rate is None:
        log_transition = advance_log_weights(ratios[1:], sigma=sigma, eta=eta,
                                             max_advance=max_advance)
    else:
        log_transition = flat_log_weights(n - 1, slip_rate=slip_rate,
                                          max_advance=max_advance)
    log_emission = np.zeros((n, 4), dtype=np.float64)
    if beta != 0.0 and boundary_samples is not None:
        log_emission[:, 0] = beta * causal_z(boundary_samples, warmup)
    log_start = np.log(np.asarray(start_prior, dtype=np.float64))
    decided = fixed_lag_viterbi(log_transition, log_emission, log_start,
                                lag_beats)
    if base is None:
        base_positions = (np.arange(n, dtype=np.int64) + anchor) % 4
        advances = np.ones(n, dtype=np.int64)
        advances[0] = 0
    else:
        base_positions, advances = base
        base_positions = (base_positions + anchor) % 4
    committed, corrections = apply_forward_only(base_positions, decided,
                                                lag_beats)
    return committed, advances, corrections, decided


# --------------------------------------------------------------------------- #
# the oracle -- perfect phase knowledge, delivered ``lag_beats`` late
# --------------------------------------------------------------------------- #
def oracle_positions(n: int, truth: np.ndarray, known: np.ndarray,
                     lag_beats: int) -> np.ndarray:
    """What a tracker that is never wrong, only late, would commit."""
    committed = np.zeros(n, dtype=np.int64)
    phase = 0
    for j in range(n):
        k = j - lag_beats
        if k >= 0 and known[k]:
            phase = int(truth[k])
        committed[j] = (j - phase) % 4
    return committed
