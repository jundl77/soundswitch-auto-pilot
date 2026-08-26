"""Per-beat downbeat evidence fused into the cyclic bar-phase HMM, online.

The port of the #330 operating point (cc_fuse_score.py in the campaign dir,
W2.5 T0.6 lag2 beta0.25, interval-ratio slip process, ±60 ms aggregation,
forward-only corrections): tracker chunks become per-beat votes, votes become
rotated position evidence on the first beat at or after their availability
instant, and a fixed-lag Viterbi over four bar positions decides each beat two
beats late.  The committed count is base-plus-offset — exactly
``phase_tracking.apply_forward_only`` — so with no evidence the offset never
moves and the grid IS the counting grid.  The trellis mathematics is
``phase_tracking``'s own (`advance_log_weights` is imported, not copied), and
the equivalence of this online form to the offline reference is pinned by
tests/test_bar_phase_equivalence.py.

Births mirror the decoder's: a cold start takes the measured first-beat prior,
a beat-gap re-anchor takes the coin-toss uniform prior, and a tracker outage
(shed, or a dead tracker) suspends the trellis so the count continues from the
position it holds — today's counting rule exactly — with a resume rebirth that
carries that position and lets evidence refuse it forward-only.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

_PHASE_TRACKING_DIR = str(Path(__file__).resolve().parents[2] / "training"
                          / "phase_tracking")
if _PHASE_TRACKING_DIR not in sys.path:
    sys.path.insert(0, _PHASE_TRACKING_DIR)

from phase_tracking import (LIVE_START_PRIOR,  # noqa: E402
                            advance_log_weights)

from lib.analyser.bar_tracker import FPS  # noqa: E402

# The #330/#332 operating point (c2a_causal/RESULTS.md).
GATE_T = 0.6
BETA = 0.25
GAMMA = 0.03
LAG_BEATS = 2
AGG_HALFWIDTH = 3
SIGMA = 0.22
ETA = 0.02
MAX_ADVANCE = 4

_POSITIONS = 4
_FRAME_STORE = 4096
_UNIFORM_PRIOR = (0.25, 0.25, 0.25, 0.25)

_RATIO_WINDOW = 8
_RATIO_MIN_HISTORY = 3
_RATIO_CLEAN_LO = 0.75
_RATIO_CLEAN_HI = 1.35


def _logit(x: float) -> float:
    return float(np.log(x / (1.0 - x)))


class _IntervalRatios:
    """phase_tracking.interval_ratios, one beat at a time."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._history: deque = deque(maxlen=_RATIO_WINDOW)
        self._last: float | None = None

    def push(self, at_sec: float) -> float:
        if self._last is None:
            self._last = at_sec
            return float("nan")
        span = at_sec - self._last
        self._last = at_sec
        period = (float(np.median(self._history))
                  if len(self._history) >= _RATIO_MIN_HISTORY else 0.0)
        ratio = span / period if period > 0.0 else float("nan")
        if period <= 0.0 or _RATIO_CLEAN_LO <= span / period <= _RATIO_CLEAN_HI:
            self._history.append(span)
        return ratio


class _Trellis:
    """phase_tracking.fixed_lag_viterbi, one step at a time."""

    def __init__(self, prior, lag: int) -> None:
        self._log_start = np.log(np.asarray(prior, dtype=np.float64))
        self._lag = int(lag)
        self._delta: np.ndarray | None = None
        self._back: deque = deque(maxlen=self._lag)
        self._step = -1
        rotation = (np.arange(_POSITIONS)[:, None]
                    - np.arange(_POSITIONS)[None, :]) % _POSITIONS
        self._rotation = rotation
        self._residue = np.arange(MAX_ADVANCE + 1) % _POSITIONS

    def push(self, emission_row: np.ndarray,
             transition_row: np.ndarray | None):
        """Advance one beat; return the state decided for step - lag, or None."""
        self._step += 1
        if self._delta is None:
            self._delta = self._log_start + emission_row
        else:
            best_by_residue = np.full(_POSITIONS, -np.inf)
            for residue in range(_POSITIONS):
                columns = transition_row[self._residue == residue]
                if columns.size:
                    best_by_residue[residue] = columns.max()
            candidate = self._delta[None, :] + best_by_residue[self._rotation]
            back_row = np.argmax(candidate, axis=1)
            self._back.append(back_row)
            self._delta = (candidate[np.arange(_POSITIONS), back_row]
                           + emission_row)
            self._delta -= self._delta.max()
        if self._step < self._lag:
            return None
        state = int(np.argmax(self._delta))
        for row in reversed(list(self._back)[-self._lag:] or []):
            state = int(row[state])
        return state


class BarPhaseFusion:
    def __init__(self, tracker=None, watchdog=None) -> None:
        self._tracker = tracker
        self._watchdog = watchdog
        self._ratios = _IntervalRatios()
        self._logit_store = np.zeros(_FRAME_STORE, dtype=np.float32)
        self._avail_store = np.full(_FRAME_STORE, np.nan)
        self._frame_id = np.full(_FRAME_STORE, -1, dtype=np.int64)
        self._frontier = 0
        self.corrections = 0
        self.reset()

    def reset(self) -> None:
        self._birth(LIVE_START_PRIOR)
        self._ratios.reset()
        self._avail_store[:] = np.nan
        self._frame_id[:] = -1
        self._frontier = 0

    def reanchor(self, at_sec: float) -> int:
        """The gap-ending beat: uniform prior — its true position is a coin toss."""
        self._birth(_UNIFORM_PRIOR)
        self._ratios.reset()
        return self.push_beat(at_sec)

    def _birth(self, prior) -> None:
        self._count = 0
        self._offset = 0
        self._prev_ratio = float("nan")
        self._pending: deque = deque()
        self._votes: list = []
        self._active = self._evidence_live()
        self._trellis = (_Trellis(prior, LAG_BEATS) if self._active else None)
        self._trellis_start = 0

    def _evidence_live(self) -> bool:
        if self._tracker is None or not getattr(self._tracker, "alive", False):
            return False
        if self._watchdog is None:
            return True
        from lib.analyser.drift_watchdog import ShedLevel

        return self._watchdog.level is ShedLevel.NONE

    def _suspend(self) -> None:
        self._trellis = None
        self._pending.clear()
        self._votes.clear()
        self._active = False

    def _resume(self) -> None:
        held = (self._count + self._offset) % _POSITIONS
        prior = np.full(_POSITIONS, 1e-12)
        prior[held] = 1.0
        self._trellis = _Trellis(prior, LAG_BEATS)
        self._trellis_start = self._count
        self._prev_ratio = float("nan")
        self._active = True

    def push_chunk(self, chunk) -> None:
        frames = np.arange(chunk.frame_lo, chunk.frame_hi)
        slots = frames % _FRAME_STORE
        self._logit_store[slots] = np.asarray(chunk.db_logits,
                                              dtype=np.float32)
        self._avail_store[slots] = float(chunk.avail_sec)
        self._frame_id[slots] = frames
        self._frontier = max(self._frontier, int(chunk.frame_hi))
        self._resolve_pending()

    def _resolve_pending(self) -> None:
        while self._pending:
            local, at_sec, frame = self._pending[0]
            if self._frontier < frame + AGG_HALFWIDTH + 1:
                return
            self._pending.popleft()
            vote = self._vote_for(local, frame)
            if vote is not None:
                self._votes.append(vote)

    def _vote_for(self, local: int, frame: int):
        lo = max(0, frame - AGG_HALFWIDTH)
        hi = frame + AGG_HALFWIDTH + 1
        frames = np.arange(lo, hi)
        slots = frames % _FRAME_STORE
        held = (self._frame_id[slots] == frames) \
            & np.isfinite(self._avail_store[slots])
        if not held.any():
            return None
        logits = np.clip(self._logit_store[slots[held]].astype(np.float64),
                         -30.0, 30.0)
        p = float((1.0 / (1.0 + np.exp(-logits))).max())
        if p < GATE_T:
            return None
        strength = _logit(min(p, 1.0 - GAMMA)) - _logit(GATE_T)
        return (local, strength, float(self._avail_store[slots[held]].max()))

    def push_beat(self, at_sec: float) -> int:
        at_sec = float(at_sec)
        live = self._evidence_live()
        if live and not self._active:
            self._resume()
        elif not live and self._active:
            self._suspend()

        ratio = self._ratios.push(at_sec)
        local = self._count - self._trellis_start
        committed = (self._count + self._offset) % _POSITIONS

        if self._active:
            self._resolve_pending()
            emission = np.zeros(_POSITIONS, dtype=np.float64)
            kept = []
            for source, strength, avail_t in self._votes:
                if avail_t <= at_sec:
                    emission[(local - source) % _POSITIONS] += BETA * strength
                else:
                    kept.append((source, strength, avail_t))
            self._votes = kept

            transition = None
            if local > 0:
                transition = advance_log_weights(
                    np.array([self._prev_ratio, ratio]), sigma=SIGMA, eta=ETA,
                    max_advance=MAX_ADVANCE)[1]
            decided = self._trellis.push(emission, transition)
            if decided is not None:
                target = (self._trellis_start + local - LAG_BEATS
                          + self._offset) % _POSITIONS
                if decided != target:
                    self._offset = (self._offset + decided - target) \
                        % _POSITIONS
                    self.corrections += 1
            frame = int(np.round(at_sec * FPS))
            self._pending.append((local, at_sec, frame))

        self._prev_ratio = ratio
        self._count += 1
        return committed
