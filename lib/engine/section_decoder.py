"""The live bar grid, and the committer that runs on it."""
from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np

# The offline decoder is imported, not copied, so a sweep measures the runtime.
_TRAINING_DIR = str(Path(__file__).resolve().parents[2] / "training")
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from nn.decoder import (DecodeParams, FixedLagViterbi,  # noqa: E402
                        SHIPPING_DECODER_CONFIG, load_decoder_config, temper)
from nn.priors import Priors  # noqa: E402

__all__ = ["BEATS_PER_BAR", "NOMINAL_BAR_SEC", "BarDecision", "BarObservation",
           "SectionDecoder", "DecodeParams", "Priors", "SHIPPING_DECODER_CONFIG",
           "load_decoder_config"]

BEATS_PER_BAR = 4

# Corpus median bar, pooled over 47,278 bars of the 215 val tracks; 127.00 BPM in 4/4.
NOMINAL_BAR_SEC = 1.8898

_ORPHAN_CELL_WINDOW_SEC = 10.0
_MAX_PENDING_CELLS = 256
_BEAT_GAP_SEC = 4.0
_EDGE_RETAIN_BARS = 64
_BAR_MEDIAN_BARS = 25

# madmom's online warm-up costs the first annotated beat, so the first beat the
# runtime sees is bar position 1 on 147 of 215 val tracks and 0 on 51.  A beat
# gap carries no such evidence -- the true position of the beat that ends one is
# measurably flat -- so it opens a bar instead.  A gap in the *feature* stage
# stops neither madmom nor the count, so that path keeps the position it holds
# and only the grid restarts.  models/phase_b/phase_tracking/phase_tracking_gate.json.
_FIRST_BEAT_BAR_POSITION = 1
_RE_ANCHOR_BAR_POSITION = 0


class BarDecision(NamedTuple):
    bar: int
    label: str
    start_sec: float


class BarObservation(NamedTuple):
    bar: int
    start_sec: float
    end_sec: float
    posterior: np.ndarray | None
    boundary: float


class SectionDecoder:
    def __init__(self, priors: Priors, params: DecodeParams | None = None, *,
                 feature_latency_sec: float = 0.0) -> None:
        self.params = params or DecodeParams()
        if self.params.min_coverage > 1:
            raise ValueError(
                f"this config wants frames covered by {self.params.min_coverage} "
                f"windows; the live stage runs ONE pass per cell, so every cell "
                f"has coverage 1 and every bar would decode from the duration "
                f"prior with nothing to say so")
        self.feature_latency_sec = float(feature_latency_sec)
        self._decoder = FixedLagViterbi(
            priors, self.params.lag_bars,
            class_prior_division=self.params.class_prior_division,
            drop_miss_cost=self.params.drop_miss_cost,
            prior_strength=self.params.prior_strength,
            boundary_weight=self.params.boundary_weight,
            boundary_ref=self.params.boundary_ref,
            floor_scale=self.params.floor_scale,
            floor_bars=self.params.floor_bars,
            outro_escape=self.params.outro_escape)
        self._n_classes = len(priors.classes)
        self.recent_observations: deque = deque(
            maxlen=self.params.lag_bars + 2)
        self.reset()

    def reset(self, *, cold_start: bool = True) -> None:
        self._decoder.reset()
        self._edges: deque = deque(maxlen=max(_EDGE_RETAIN_BARS,
                                              self.params.lag_bars + 4))
        self._edge_base: int = 0
        self._bar_offset: int = 0
        if cold_start:
            self._bar_position: int = _FIRST_BEAT_BAR_POSITION
            self._last_beat_sec: float | None = None
        self._cells: deque = deque()
        self._newest_cell_sec: float = -np.inf
        self._next_bar: int = 0
        self.recent_observations.clear()

    @property
    def bar_edges(self) -> list:
        return list(self._edges)

    def _edge(self, bar: int) -> float:
        if not self._have_edge(bar):
            raise KeyError(f"bar {bar}'s line is no longer held; the grid keeps "
                           f"{self._edges.maxlen} and is at {self._edge_base}")
        return self._edges[bar - self._edge_base]

    def _have_edge(self, bar: int) -> bool:
        return 0 <= bar - self._edge_base < len(self._edges)

    def _append_edge(self, at_sec: float) -> None:
        if len(self._edges) == self._edges.maxlen:
            self._edge_base += 1
        self._edges.append(float(at_sec))

    @property
    def classes(self) -> tuple:
        return tuple(self._decoder.classes)

    @property
    def bars_pushed(self) -> int:
        return self._next_bar

    @property
    def pending_cells(self) -> int:
        return len(self._cells)

    @property
    def backtrace_rows(self) -> int:
        return self._decoder.backtrace_rows

    @property
    def bar_sec(self) -> float:
        if len(self._edges) < 2:
            return NOMINAL_BAR_SEC
        recent = list(self._edges)[-_BAR_MEDIAN_BARS:]
        return float(np.median(np.diff(np.asarray(recent))))

    @property
    def chain_latency_sec(self) -> float:
        return self.feature_latency_sec + \
            (self.params.lag_bars + 1) * self.bar_sec

    def push_beat(self, at_sec: float) -> list:
        at_sec = float(at_sec)
        if self._re_anchoring(at_sec):
            self._re_anchor(at_sec)
        elif self._bar_position == 0:
            self._append_edge(at_sec)
        self._bar_position = (self._bar_position + 1) % BEATS_PER_BAR
        self._last_beat_sec = at_sec
        return self._advance()

    def _re_anchoring(self, at_sec: float) -> bool:
        return (self._last_beat_sec is not None
                and at_sec - self._last_beat_sec > _BEAT_GAP_SEC)

    def _re_anchor(self, at_sec: float) -> None:
        logging.warning(
            f'[decoder] {at_sec - self._last_beat_sec:.1f}s without a beat — '
            f're-anchoring the bar grid at {at_sec:.2f}s and dropping '
            f'{len(self._cells)} pending cells')
        self._append_edge(at_sec)
        self._bar_position = _RE_ANCHOR_BAR_POSITION
        self._cells.clear()
        self._restart_committer_at(self._edge_base + len(self._edges) - 1)

    def push_posterior(self, at_sec: float, posterior, boundary: float) -> list:
        row = np.asarray(posterior, dtype=np.float64).reshape(-1)
        if row.size != self._n_classes:
            raise ValueError(f"a cell carries {row.size} classes, the priors "
                             f"name {self._n_classes}")
        self._cells.append((float(at_sec), row, float(boundary)))
        self._newest_cell_sec = float(at_sec)
        return self._advance()

    def _advance(self) -> list:
        self._skip_bars_the_grid_no_longer_holds()
        decisions: list = []
        while self._observable(self._next_bar):
            observation = self._observe(self._next_bar)
            self.recent_observations.append(observation)
            self._next_bar += 1
            for decision in self._decoder.push(observation.posterior,
                                               observation.boundary):
                bar = decision.bar + self._bar_offset
                decisions.append(BarDecision(bar, decision.label,
                                             self._edge(bar)))
        self._prune()
        return decisions

    def _skip_bars_the_grid_no_longer_holds(self) -> None:
        oldest = max(0, self._next_bar - self.params.lag_bars - 1)
        if oldest >= self._edge_base:
            return
        logging.warning(
            f'[decoder] the beat grid has run {self._edge_base - oldest} bars '
            f'past the oldest line the committer can still name (cursor '
            f'{self._next_bar}, grid holds from {self._edge_base}) — '
            f'restarting the committer')
        self._restart_committer_at(max(self._next_bar, self._edge_base))

    def _restart_committer_at(self, bar: int) -> None:
        self._next_bar = bar
        self._bar_offset = bar
        self._decoder.reset()
        self.recent_observations.clear()

    def _observable(self, bar: int) -> bool:
        if not self._have_edge(bar + 1):
            return False
        reach = max(self._edge(bar + 1),
                    self._edge(bar) + self.params.boundary_tolerance_sec)
        return self._newest_cell_sec >= reach

    def _observe(self, bar: int) -> BarObservation:
        lo, hi = self._edge(bar), self._edge(bar + 1)
        tolerance = self.params.boundary_tolerance_sec
        rows = [row for when, row, _ in self._cells if lo <= when < hi]
        scores = [score for when, _, score in self._cells
                  if lo - tolerance <= when <= lo + tolerance]
        posterior = None
        if rows:
            posterior = temper(np.asarray(rows, dtype=np.float64),
                               self.params.temperature).mean(axis=0)
        return BarObservation(bar, lo, hi, posterior,
                              max(scores) if scores else float("nan"))

    def _prune(self) -> None:
        if self._have_edge(self._next_bar):
            floor = self._edge(self._next_bar) - self.params.boundary_tolerance_sec
        else:
            floor = self._newest_cell_sec - _ORPHAN_CELL_WINDOW_SEC
        while self._cells and self._cells[0][0] < floor:
            self._cells.popleft()
        while len(self._cells) > _MAX_PENDING_CELLS:
            self._cells.popleft()
