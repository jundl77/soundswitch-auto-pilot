"""Bar-phase decoder: beat instants + downbeat activation in, a bar grid out."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

BEATS_PER_BAR = 4

# Expert-grid val sweep, 215 tracks: F1 0.7063 at 14 against a 0.5592 naive floor.
DEFAULT_FLIP_PENALTY = 14.0

DEFAULT_DOWNBEAT_REF = 0.5
DEFAULT_LAG_BEATS = 4
DEFAULT_COAST_RATIO = 1.5
MAX_COAST_BEATS = 4 * BEATS_PER_BAR
DEFAULT_TEMPO_WINDOW = 2 * BEATS_PER_BAR
DEFAULT_SUBDIVISION = 1

# +-1 frame is +-46.4 ms; with the frame grid's own +-23.2 ms that is the +-70 ms tolerance.
AGG_LO_FRAMES = -1
AGG_HI_FRAMES = 1

EPS = 1e-6


class PhaseDecision(NamedTuple):
    beat: int
    time: float
    phase: int
    confidence: float
    virtual: bool


@dataclass(frozen=True)
class PhaseParams:
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


def candidate_grid(beat_times, subdivision: int = DEFAULT_SUBDIVISION) -> np.ndarray:
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
    """Activation frame whose stamp is nearest each instant; ``-1`` past the end."""
    index = np.rint((np.asarray(instants, dtype=np.float64) - t0) / frame_sec)
    index = np.maximum(index.astype(np.int64), 0)
    return np.where(index >= int(n_frames), -1, index)


def aggregate_at_beats(activation, instants, frame_sec: float, t0: float, *,
                       lo: int = AGG_LO_FRAMES, hi: int = AGG_HI_FRAMES) -> tuple:
    """``(scores, counts)``; a score is NaN where no frame covers the instant."""
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


class BarPhaseHMM:
    """Cyclic phase decoder that commits at a fixed lag and never revises."""

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
        self.cycle = BEATS_PER_BAR * int(params.subdivision)

        advance = (np.arange(self.cycle) + 1) % self.cycle
        self._transition = np.full((self.cycle, self.cycle),
                                   -float(params.flip_penalty), dtype=np.float64)
        self._transition[np.arange(self.cycle), advance] = 0.0
        self.reset()

    def reset(self) -> None:
        self._frontier_score = None
        self._frontier_candidate: tuple | None = None
        self._pending: deque = deque()
        self._committed = 0
        self._committed_phase: int | None = None
        self._last_time = None
        self._periods: deque = deque(maxlen=int(self.params.tempo_window))
        self._coast_streak = 0

    def push(self, time: float, score: float) -> list:
        time = float(time)
        if self._last_time is not None and time <= self._last_time:
            raise ValueError(
                f"beat times must be strictly increasing: {time} follows "
                f"{self._last_time}")

        coasted, period = self._plan_gap(time)
        self._coast_streak = self._coast_streak + 1 if coasted else 0
        decisions: list = []
        for moment in coasted:
            decisions.extend(self._observe(moment, np.nan, virtual=True))
        decisions.extend(self._observe(time, score, virtual=False))
        if period is not None:
            self._periods.extend([period] * (len(coasted) + 1))
        self._last_time = time
        return decisions

    def flush(self) -> list:
        decisions: list = []
        while self._frontier_score is not None:
            decisions.append(self._commit_one())
        return decisions

    def decode(self, beat_times, beat_scores) -> list:
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

    def _tempo_estimate_is_stale(self) -> bool:
        return self._coast_streak >= int(self.params.max_coast_beats)

    def _plan_gap(self, time: float) -> tuple:
        """``([interpolated times], period)``; a ``None`` period is not a beat period."""
        if self._last_time is None:
            return [], None
        gap = time - self._last_time
        if self._tempo_estimate_is_stale():
            self._periods.clear()
            return [], gap
        if not self._periods:
            return [], gap
        period = float(np.median(self._periods))
        if period <= 0.0 or gap < self.params.coast_ratio * period:
            return [], gap
        missing = int(round(gap / period)) - 1
        if missing < 1:
            return [], gap
        if missing > int(self.params.max_coast_beats):
            self._periods.clear()
            return [], None
        step = gap / (missing + 1)
        return [self._last_time + step * (index + 1) for index in range(missing)], step

    def _emission(self, score: float) -> np.ndarray:
        emission = np.zeros(self.cycle, dtype=np.float64)
        if np.isfinite(score):
            emission[0] = _logit(score) - self._ref_logit
        return emission

    def _observe(self, time: float, score: float, *, virtual: bool) -> list:
        emission = self._emission(score)
        if self._frontier_score is None:
            seed = (np.zeros(self.cycle) if self._committed_phase is None
                    else self._transition[self._committed_phase])
            self._frontier_score = seed + emission
            self._frontier_score -= self._frontier_score.max()
            self._frontier_candidate = (emission, float(time), bool(virtual))
        else:
            self._pending.append((emission, float(time), bool(virtual)))

        decisions: list = []
        while (self._frontier_score is not None
               and len(self._pending) >= int(self.params.lag_beats)):
            decisions.append(self._commit_one())
        return decisions

    def _look_ahead(self, seed: np.ndarray) -> np.ndarray:
        path_score = np.full((self.cycle, self.cycle), -np.inf, dtype=np.float64)
        np.fill_diagonal(path_score, seed)
        for emission, _time, _virtual in self._pending:
            path_score = ((path_score[:, :, None] + self._transition).max(axis=1)
                          + emission)
        return path_score.max(axis=1)

    def _commit_one(self) -> PhaseDecision:
        conditioned_score = self._look_ahead(self._frontier_score)
        phase = int(np.argmax(conditioned_score))

        unconditioned_score = self._look_ahead(self._frontier_candidate[0])
        phase_posterior = np.exp(unconditioned_score - unconditioned_score.max())
        confidence = float(phase_posterior[phase] / phase_posterior.sum())

        _emission, time, virtual = self._frontier_candidate
        decision = PhaseDecision(self._committed, time, phase + 1, confidence, virtual)
        self._committed += 1
        self._committed_phase = phase

        if self._pending:
            emission, time, virtual = self._pending.popleft()
            self._frontier_score = self._transition[phase] + emission
            self._frontier_score -= self._frontier_score.max()
            self._frontier_candidate = (emission, time, virtual)
        else:
            self._frontier_score = None
            self._frontier_candidate = None
        return decision


def downbeat_times(decisions) -> np.ndarray:
    return np.array([d.time for d in decisions if d.phase == 1], dtype=np.float64)


def phase_flips(decisions, subdivision: int = DEFAULT_SUBDIVISION) -> int:
    cycle = BEATS_PER_BAR * int(subdivision)
    flips = 0
    for previous, current in zip(decisions, decisions[1:]):
        if current.phase != previous.phase % cycle + 1:
            flips += 1
    return flips


def bar_phase(phase: int, subdivision: int = DEFAULT_SUBDIVISION) -> int:
    """Cycle position -> bar phase in 1..4, or 0 for an interstitial candidate."""
    step = int(subdivision)
    return (phase - 1) // step + 1 if (phase - 1) % step == 0 else 0


def decode_track(sidecar_npz, condition: str, params: PhaseParams | None = None,
                 *, refine: bool = False) -> list:
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
        times, scores = beats, cached
    else:
        times = candidate_grid(beats, params.subdivision)
        scores, _counts = aggregate_at_beats(activation, times, frame_sec, t0)

    decisions = BarPhaseHMM(params).decode(times, scores)
    if refine:
        decisions = refine_instants(decisions, activation, frame_sec, t0)
    return decisions


def _parabolic_peak_shift(left: float, centre: float, right: float) -> float:
    curvature = left - 2.0 * centre + right
    return 0.5 * (left - right) / curvature if curvature < 0.0 else 0.0


def refine_instants(decisions, activation, frame_sec: float, t0: float, *,
                    lo: int = AGG_LO_FRAMES, hi: int = AGG_HI_FRAMES) -> list:
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
        shift = _parabolic_peak_shift(activation[max(peak - 1, 0)],
                                      activation[peak],
                                      activation[min(peak + 1, n_frames - 1)])
        moment = t0 + (peak + float(np.clip(shift, -0.5, 0.5))) * frame_sec
        refined.append(decision._replace(
            time=float(np.clip(moment, decision.time + lo * frame_sec,
                               decision.time + hi * frame_sec))))
    return refined
