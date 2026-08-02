"""Fixed-lag HSMM committer over bars; the live engine imports this, not a copy."""
from __future__ import annotations

import csv
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .priors import Priors

# ~5.6 s at the corpus median bar of 1.875 s, inside the 8 s look-ahead budget.
DEFAULT_LAG_BARS = 3

DEFAULT_PRIOR_STRENGTH = 0.0
DEFAULT_DROP_MISS_COST = 1.0
DEFAULT_BOUNDARY_REF = 0.5
DEFAULT_BOUNDARY_WEIGHT = 2.0
DEFAULT_BOUNDARY_TOLERANCE_SEC = 0.5
DEFAULT_MIN_COVERAGE = 2
DEFAULT_OUTRO_ESCAPE = 0.0
DEFAULT_TEMPERATURE = 1.0
EPS = 1e-12


class Decision(NamedTuple):
    bar: int
    class_index: int
    label: str


@dataclass(frozen=True)
class DecodeParams:
    lag_bars: int = DEFAULT_LAG_BARS
    class_prior_division: bool = True
    prior_strength: float = DEFAULT_PRIOR_STRENGTH
    drop_miss_cost: float = DEFAULT_DROP_MISS_COST
    boundary_weight: float = DEFAULT_BOUNDARY_WEIGHT
    boundary_ref: float = DEFAULT_BOUNDARY_REF
    boundary_tolerance_sec: float = DEFAULT_BOUNDARY_TOLERANCE_SEC
    min_coverage: int = DEFAULT_MIN_COVERAGE
    floor_scale: float = 1.0
    floor_bars: tuple | None = None
    outro_escape: float = DEFAULT_OUTRO_ESCAPE
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        if self.floor_bars is not None and not isinstance(self.floor_bars, tuple):
            object.__setattr__(self, "floor_bars", tuple(self.floor_bars))


SHIPPING_DECODER_CONFIG = Path(__file__).with_name("decoder_config.json")


def load_decoder_config(path) -> DecodeParams:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    chosen = document.get("chosen", document)
    known = {field.name for field in dataclasses.fields(DecodeParams)}
    unknown = sorted(set(chosen) - known)
    if unknown:
        raise ValueError(
            f"{path}: decoder config names {', '.join(unknown)}, which "
            f"DecodeParams does not have -- dropping them would decode with a "
            f"different decoder than the one this config was measured on. "
            f"Known knobs: {', '.join(sorted(known))}")
    return DecodeParams(**chosen)


class FixedLagViterbi:
    def __init__(self, priors: Priors, lag_bars: int = DEFAULT_LAG_BARS,
                 class_prior_division: bool = True,
                 drop_miss_cost: float = DEFAULT_DROP_MISS_COST, *,
                 prior_strength: float = DEFAULT_PRIOR_STRENGTH,
                 boundary_weight: float = DEFAULT_BOUNDARY_WEIGHT,
                 boundary_ref: float = DEFAULT_BOUNDARY_REF,
                 floor_scale: float = 1.0,
                 floor_bars=None,
                 outro_escape: float = DEFAULT_OUTRO_ESCAPE) -> None:
        if int(lag_bars) < 0:
            raise ValueError(f"lag_bars must be >= 0, got {lag_bars}")
        if float(floor_scale) <= 0.0:
            raise ValueError(f"floor_scale must be > 0, got {floor_scale}")
        if float(drop_miss_cost) <= 0.0:
            raise ValueError(f"drop_miss_cost must be > 0, got {drop_miss_cost}")
        if not 0.0 <= float(outro_escape) < 0.5:
            raise ValueError(
                f"outro_escape must lie in [0, 0.5) -- got {outro_escape}; it is "
                f"a per-bar probability and TWO targets are opened, so the stay "
                f"probability is 1 - 2 * escape")

        self.priors = priors
        self.classes = tuple(priors.classes)
        self.lag_bars = int(lag_bars)
        self.class_prior_division = bool(class_prior_division)
        self.prior_strength = float(prior_strength)
        self.drop_miss_cost = float(drop_miss_cost)
        self.boundary_weight = float(boundary_weight)
        self.boundary_ref = float(boundary_ref)
        self.floor_scale = float(floor_scale)
        self.outro_escape = float(outro_escape)
        self.floor_bars = tuple(floor_bars) if floor_bars is not None else None

        hazard = np.asarray(priors.hazard, dtype=np.float64)
        if not np.all((hazard > 0.0) & (hazard < 1.0)):
            raise ValueError(
                f"every class hazard must lie in (0, 1) -- got {hazard.tolist()}; "
                f"0 makes a class unleavable and 1 caps every run at its floor")

        if self.floor_bars is not None:
            if len(self.floor_bars) != len(self.classes):
                raise ValueError(
                    f"floor_bars has {len(self.floor_bars)} entries for "
                    f"{len(self.classes)} classes {self.classes} -- a floor "
                    f"vector silently misaligned by one would apply drop's floor "
                    f"to breakdown and never raise")
            self._floors = np.maximum(
                1, np.asarray(self.floor_bars, dtype=np.int64))
        else:
            self._floors = np.maximum(
                1, np.rint(np.asarray(priors.floor_bars, dtype=np.float64)
                           * self.floor_scale).astype(np.int64))
        self._emission_bonus = self._class_bonus()
        self._entry_bonus = self._commit_bonus()
        self._build_states()
        self.reset()

    def _build_states(self) -> None:
        floors = self._floors
        self._state_class = np.concatenate(
            [np.full(int(floor), class_index, dtype=np.int64)
             for class_index, floor in enumerate(floors)])
        class_offsets = np.concatenate([[0], np.cumsum(floors)])
        self._entry_state = class_offsets[:-1].copy()
        self._final_state = class_offsets[1:] - 1
        n_states = int(class_offsets[-1])

        log_transition = self.priors.log_transition
        hazard = np.clip(np.asarray(self.priors.hazard, dtype=np.float64), EPS, 1.0)
        with np.errstate(divide="ignore"):
            log_stay = np.log(1.0 - hazard)
            log_leave = np.log(hazard)

        transition = np.full((n_states, n_states), -np.inf, dtype=np.float64)
        switch = np.zeros((n_states, n_states), dtype=bool)
        for state in range(n_states):
            class_index = int(self._state_class[state])
            saturated = state == self._final_state[class_index]
            if not saturated:
                transition[state, state + 1] = 0.0
            else:
                transition[state, state] = log_stay[class_index]
                for next_class in range(len(self.classes)):
                    if next_class == class_index:
                        continue
                    target = int(self._entry_state[next_class])
                    transition[state, target] = (
                        log_leave[class_index]
                        + log_transition[class_index, next_class]
                        + self._entry_bonus[next_class])
                    switch[state, target] = True

        self._apply_outro_escape(transition, switch)

        self._transition = transition
        self._switch = switch
        self._log_initial = np.full(n_states, -np.inf, dtype=np.float64)
        self._log_initial[self._entry_state] = (self.priors.log_initial
                                                + self._entry_bonus)

    ESCAPE_TARGETS = ("breakdown", "drop")

    def _apply_outro_escape(self, transition: np.ndarray, switch: np.ndarray) -> None:
        if self.outro_escape <= 0.0 or "outro" not in self.classes:
            return
        source_class = self.classes.index("outro")
        source = int(self._final_state[source_class])
        targets = [self.classes.index(name) for name in self.ESCAPE_TARGETS
                   if name in self.classes]
        if not targets:
            return

        stay = 1.0 - self.outro_escape * len(targets)
        transition[source, source] = math.log(stay) if stay > 0.0 else -np.inf
        for target_class in targets:
            entry = int(self._entry_state[target_class])
            transition[source, entry] = (math.log(self.outro_escape)
                                         + self._entry_bonus[target_class])
            switch[source, entry] = True

    def _class_bonus(self) -> np.ndarray:
        bonus = np.zeros(len(self.classes), dtype=np.float64)
        if self.class_prior_division and self.prior_strength:
            prior = np.clip(np.asarray(self.priors.class_prior, dtype=np.float64),
                            EPS, None)
            bonus -= self.prior_strength * np.log(prior)
        return bonus

    def _commit_bonus(self) -> np.ndarray:
        bonus = np.zeros(len(self.classes), dtype=np.float64)
        if "drop" in self.classes and self.drop_miss_cost != 1.0:
            bonus[self.classes.index("drop")] += np.log(self.drop_miss_cost)
        return bonus

    def reset(self) -> None:
        self._delta = None
        self._psi: list = [None] * (self.lag_bars + 1)
        self._bars = 0
        self._next_commit = 0

    @property
    def backtrace_rows(self) -> int:
        return sum(1 for entry in self._psi if entry is not None)

    def _remember(self, bar: int, back: np.ndarray) -> None:
        self._psi[bar % len(self._psi)] = (bar, back)

    def _recall(self, bar: int) -> np.ndarray:
        entry = self._psi[bar % len(self._psi)]
        if entry is None or entry[0] != bar:
            raise RuntimeError(
                f"bar {bar}'s backtrace has been evicted from a ring of "
                f"{len(self._psi)} -- the commit rule reached further back than "
                f"the fixed lag allows, which means a decision was about to be "
                f"read off whatever the modulo landed on")
        return entry[1]

    def push(self, posterior, boundary=None) -> list:
        emission = self._emission(posterior)
        bar = self._bars

        if bar == 0:
            self._delta = self._log_initial + emission
            self._remember(bar, np.full(len(emission), -1, dtype=np.int64))
        else:
            transition = self._transition
            boundary_hazard = self._switch_bonus(boundary)
            if boundary_hazard:
                transition = transition + boundary_hazard * self._switch
            score = self._delta[:, None] + transition
            predecessor = score.argmax(axis=0)
            self._delta = score[predecessor, np.arange(score.shape[1])] + emission
            self._remember(bar, predecessor)

        self._bars += 1
        return self._commit_due()

    def flush(self) -> list:
        if self._delta is None or self._next_commit >= self._bars:
            return []
        return self._commit_through(self._bars - 1)

    def decode(self, posteriors, boundary=None) -> list:
        self.reset()
        posteriors = np.asarray(posteriors, dtype=np.float64)
        if posteriors.ndim != 2 or posteriors.shape[1] != len(self.classes):
            raise ValueError(
                f"posteriors must be [bars, {len(self.classes)}], got "
                f"{posteriors.shape}")
        scores = None if boundary is None else np.asarray(boundary, dtype=np.float64)
        if scores is not None and len(scores) != len(posteriors):
            raise ValueError(
                f"boundary has {len(scores)} entries for {len(posteriors)} bars")

        decisions: list = []
        for bar, row in enumerate(posteriors):
            decisions.extend(self.push(row, None if scores is None else scores[bar]))
        decisions.extend(self.flush())
        return decisions

    def _emission(self, posterior) -> np.ndarray:
        n_states = len(self._state_class)
        if posterior is None:
            return np.zeros(n_states, dtype=np.float64)
        row = np.asarray(posterior, dtype=np.float64).reshape(-1)
        if row.size != len(self.classes):
            raise ValueError(
                f"posterior must have {len(self.classes)} entries, got {row.size}")
        if np.any(row < 0.0):
            raise ValueError(f"posterior has negative entries: {row.tolist()}")
        total = row.sum()
        if not np.isfinite(total) or total <= 0.0:
            return np.zeros(n_states, dtype=np.float64)
        scores = np.log(row / total + EPS) + self._emission_bonus
        return scores[self._state_class]

    def _switch_bonus(self, boundary) -> float:
        if boundary is None or not self.boundary_weight:
            return 0.0
        score = float(boundary)
        if not np.isfinite(score):
            return 0.0
        return self.boundary_weight * (score - self.boundary_ref)

    def _commit_due(self) -> list:
        target = self._bars - 1 - self.lag_bars
        if target < self._next_commit:
            return []
        return self._commit_through(target)

    def _commit_through(self, target: int) -> list:
        ancestors = self._ancestors(self._bars - 1, self._next_commit)
        best_state = int(np.argmax(self._delta))
        decisions: list = []
        for bar in range(self._next_commit, target + 1):
            state = int(ancestors[bar - self._next_commit][best_state])
            index = int(self._state_class[state])
            decisions.append(Decision(bar, index, self.classes[index]))

        final_ancestor = ancestors[target - self._next_commit]
        committed_class = int(self._state_class[final_ancestor[best_state]])
        self._delta = np.where(
            self._state_class[final_ancestor] == committed_class,
            self._delta, -np.inf)
        self._next_commit = target + 1
        return decisions

    def _ancestors(self, frm: int, to: int) -> list:
        ancestor = np.arange(len(self._state_class), dtype=np.int64)
        chain = [ancestor]
        for bar in range(frm, to, -1):
            ancestor = self._recall(bar)[ancestor]
            chain.append(ancestor)
        return chain[::-1]


def segments(decisions) -> list:
    spans: list = []
    for decision in decisions:
        if spans and spans[-1][2] == decision.label and spans[-1][1] == decision.bar:
            spans[-1][1] = decision.bar + 1
        else:
            spans.append([decision.bar, decision.bar + 1, decision.label])
    return [(start, end, label) for start, end, label in spans]


def bar_grid(beat_csv_path) -> np.ndarray:
    path = Path(beat_csv_path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        downbeats = [float(row["time"]) for row in csv.DictReader(handle)
                     if int(row["downbeat"]) == 1]
    if len(downbeats) < 2:
        raise RuntimeError(
            f"{path}: fewer than two downbeat rows -- there is no bar grid to "
            f"decode on (the decoder runs at bar rate by design)")
    edges = np.asarray(sorted(downbeats), dtype=np.float64)
    median_bar_sec = float(np.median(np.diff(edges)))
    return np.append(edges, edges[-1] + median_bar_sec)


def temper(label_post: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if temperature == 1.0:
        return label_post
    powered = np.power(np.clip(label_post, EPS, None), 1.0 / temperature)
    return powered / powered.sum(axis=-1, keepdims=True)


def bar_observations(posterior_npz, edges, *, min_coverage: int = DEFAULT_MIN_COVERAGE,
                     boundary_tolerance_sec: float = DEFAULT_BOUNDARY_TOLERANCE_SEC,
                     temperature: float = DEFAULT_TEMPERATURE) -> tuple:
    with np.load(posterior_npz) as archive:
        label_post = np.asarray(archive["label_post"], dtype=np.float64)
        boundary = np.asarray(archive["boundary"], dtype=np.float64)
        coverage = np.asarray(archive["coverage"], dtype=np.int64)
        frame_sec = float(archive["frame_sec"])
        label_frame_sec = float(archive["label_frame_sec"])
        label_t0 = float(archive["label_t0"])
        label_pool = int(archive["label_pool"])
        t0 = float(archive["t0"])

    label_post = temper(label_post, temperature)

    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("need at least two bar edges to define one bar")
    n_bars = edges.size - 1

    label_times = label_t0 + np.arange(len(label_post)) * label_frame_sec
    label_ok = coverage[::label_pool][:len(label_post)] >= int(min_coverage)
    frame_times = t0 + np.arange(len(boundary)) * frame_sec
    frame_ok = coverage[:len(boundary)] >= int(min_coverage)

    if not label_ok.any():
        raise RuntimeError(
            f"{posterior_npz}: no frame has coverage >= {min_coverage} "
            f"(max is {int(coverage.max()) if coverage.size else 0}) -- every "
            f"bar would decode from the duration prior alone. Either these "
            f"sidecars were written by a single-pass model (coverage 1) and the "
            f"config wants overlap-averaged ones, or the two were mispaired")

    posteriors = np.full((n_bars, label_post.shape[1]), np.nan, dtype=np.float64)
    scores = np.full(n_bars, np.nan, dtype=np.float64)

    lo = np.searchsorted(label_times, edges[:-1], side="left")
    hi = np.searchsorted(label_times, edges[1:], side="left")
    for bar in range(n_bars):
        usable = label_ok[lo[bar]:hi[bar]]
        if usable.any():
            posteriors[bar] = label_post[lo[bar]:hi[bar]][usable].mean(axis=0)

    downbeats = edges[:-1]
    starts = np.searchsorted(frame_times, downbeats - boundary_tolerance_sec, "left")
    ends = np.searchsorted(frame_times, downbeats + boundary_tolerance_sec, "right")
    for bar in range(n_bars):
        usable = frame_ok[starts[bar]:ends[bar]]
        if usable.any():
            scores[bar] = boundary[starts[bar]:ends[bar]][usable].max()
    return posteriors, scores


def decode_track(posterior_npz, beat_csv, params: DecodeParams | None = None, *,
                 priors: Priors) -> list:
    params = params or DecodeParams()
    edges = bar_grid(beat_csv)
    posteriors, boundary = bar_observations(
        posterior_npz, edges, min_coverage=params.min_coverage,
        boundary_tolerance_sec=params.boundary_tolerance_sec,
        temperature=params.temperature)

    decoder = FixedLagViterbi(
        priors, params.lag_bars,
        class_prior_division=params.class_prior_division,
        drop_miss_cost=params.drop_miss_cost,
        prior_strength=params.prior_strength,
        boundary_weight=params.boundary_weight,
        boundary_ref=params.boundary_ref,
        floor_scale=params.floor_scale,
        floor_bars=params.floor_bars,
        outro_escape=params.outro_escape)
    decisions = decoder.decode(posteriors, boundary)
    return [(float(edges[d.bar]), d.label) for d in decisions]
