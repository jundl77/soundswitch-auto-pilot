"""The live bar grid, and the committer that runs on it.

The decoder is the only component permitted to say what the lights are doing.
It is `training/nn/decoder.py`'s ``FixedLagViterbi`` -- imported, not copied,
because an offline sweep that disagreed with the runtime by one line would be
measuring the wrong decoder -- fed from two live streams that arrive at
completely different times:

* **beats**, essentially as the audio does, which build the bar grid;
* **posterior cells**, ~8 s behind it, which are what a bar is decoded from.

So a bar is *formed* long before it can be *observed*, and this module's whole
job is to keep those two facts apart.

**A bar is four beats from the first detected beat, and that is measured.**
Task 1c showed the production beat stream does not carry bar phase -- a median
two phase slips per track, and an oracle frozen phase covers only ~66 % of one
-- so every estimator loses to counting from the first beat (#157/#158).  Task
1d then decoded on the real stream and priced it: -0.1396 crispness@0.5 s
against the expert grid, and **all** of it placement rather than
classification.  The class decisions are nearly grid-invariant; they land at a
displaced instant.  That is the known cost of shipping without a downbeat
tracker, not a defect of this file.

**Observation assembly is `bar_observations`' semantics on a stream.**  Same
half-open bar, same per-cell tempering before the average, same max-within-
tolerance read of the boundary at the bar LINE.  It is ported rather than
imported because the offline function reads an npz sidecar off disk, which a
show does not have.  Two things it does not need to port: the sidecar's
``coverage`` array (the live stage runs one pass per cell, so coverage is 1 --
which is exactly what the shipped config's ``min_coverage`` asks for, and a
config asking for more is refused rather than silently decoding from the
duration prior), and the sidecar's boundary duplication onto a half-rate frame
grid (the head emits one score per cell either way).

**Nothing here grows with the length of a set, and that is now true of every
part of it.**  The trellis backtrace is a ring of ``lag_bars + 1`` in the
decoder itself; the bar grid keeps only the recent edges (which is also the
window its tempo is read over); and the cells waiting for a bar are pruned to
what the next bar's own boundary window can still reach -- with a hard cap
under that, because the pruning floor stops moving whenever beats do.

**Beats can stop while audio keeps arriving**, which is not a song boundary and
is not silence: heavy sidechain compression, a beatless passage, crowd noise
between sets.  The grid re-anchors at the next beat rather than closing a bar
across the gap, because averaging minutes of audio into one bar produces a
confident decision about a section that was never played.
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np

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

# The corpus median bar, pooled over 47,278 bars of the 215 val tracks (Task
# 1a); 127.00 BPM in 4/4.  Used only until the track has two bar lines of its
# own -- the queue needs a delay from the first command, and a stand-in with
# provenance beats a guess.
NOMINAL_BAR_SEC = 1.8898

# How long a cell may sit unattached to any bar, before the first beat and
# whenever the beat stream stops: the pruning floor is the next bar's opening
# line, which stops moving exactly when beats do.
_ORPHAN_CELL_WINDOW_SEC = 10.0

# The last resort under both: ~24 s of cells at the shipped rate.  A window in
# seconds is the right rule and a count is what makes "nothing grows" a property
# of the code rather than of the tempo.
_MAX_PENDING_CELLS = 256

# A beat stream that goes quiet for longer than this is a gap, not a slow bar.
# Above the engine's 2.5 s beat-absence timer, so a re-anchor accompanies an
# ATMOSPHERIC rather than pre-empting one, and far above any single-beat dropout
# (a 60 BPM track is one beat a second).
_BEAT_GAP_SEC = 4.0

# Bar lines kept.  A decision's stamp reaches back as far as the beat stream
# runs ahead of the cell stream (a whole chain latency) plus the commit lag, so
# this is sized for a very fast grid rather than for the corpus median -- 64
# bars is 16 s even at a quarter-second bar, and it is 512 bytes.
_EDGE_RETAIN_BARS = 64

# The window the tempo is read over.  A median over the whole set stops moving
# after a tempo change -- the new tempo would have to own more than half of
# everything played, which in one continuous mix never happens.
_BAR_MEDIAN_BARS = 25


class BarDecision(NamedTuple):
    """One immutable commit, stamped on the bar line it starts on."""

    bar: int
    label: str
    start_sec: float


class BarObservation(NamedTuple):
    """What one bar was decoded from.  ``posterior`` is None for no evidence."""

    bar: int
    start_sec: float
    end_sec: float
    posterior: np.ndarray | None
    boundary: float


class SectionDecoder:
    """Beats and posterior cells in, immutable per-bar decisions out."""

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

    # -- state -------------------------------------------------------------- #

    def reset(self) -> None:
        """Forget the song.  A reset decoder decodes like a fresh one."""
        self._decoder.reset()
        self._edges: deque = deque(maxlen=max(_EDGE_RETAIN_BARS,
                                              self.params.lag_bars + 4))
        self._edge_base: int = 0
        # Trellis bar + offset = grid bar.  Non-zero only after a skip, which is
        # the one thing that restarts the committer without restarting the grid.
        self._bar_offset: int = 0
        self._beats: int = 0
        self._last_beat_sec: float | None = None
        self._cells: deque = deque()
        self._newest_cell_sec: float = -np.inf
        self._next_bar: int = 0
        self.recent_observations.clear()

    @property
    def bar_edges(self) -> list:
        """The edges still held -- the recent window, not the whole set."""
        return list(self._edges)

    def _edge(self, bar: int) -> float:
        """A bar line, by absolute bar number.

        Raises on an evicted one rather than returning a neighbour: the ring is
        sized so this cannot happen, and if it ever does, every stamp after it
        would be quietly wrong.  Same rule the trellis ring already follows.
        """
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
        """The committer's own class order -- never a second copy of it."""
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
        """The recent median bar, or the corpus median until there is one.

        Recent, not cumulative: a set is one continuous mix with no song
        boundary to reset anything, so a median over the whole history stops
        tracking the tempo the moment the DJ changes it -- the new tempo has to
        own more than half of everything played.  The window is the same edges
        the grid keeps.
        """
        if len(self._edges) < 2:
            return NOMINAL_BAR_SEC
        recent = list(self._edges)[-_BAR_MEDIAN_BARS:]
        return float(np.median(np.diff(np.asarray(recent))))

    @property
    def chain_latency_sec(self) -> float:
        """Audio -> committed decision, Task 1a's own delay model.

        ``feature_latency + (lag_bars + 1) * bar``: bar b's observation needs
        bar b to finish, and the commit lands ``lag_bars`` bars after that.  The
        decoder's half is proportional to bar length -- 12.11 s to 16.37 s
        across the val corpus at lag 2 -- which is why the show measures it per
        track instead of taking the median.
        """
        return self.feature_latency_sec + \
            (self.params.lag_bars + 1) * self.bar_sec

    # -- the two input streams ---------------------------------------------- #

    def push_beat(self, at_sec: float) -> list:
        """One detected beat, in the same time base the cells are stamped in."""
        at_sec = float(at_sec)
        if self._re_anchoring(at_sec):
            self._re_anchor(at_sec)
        elif self._beats % BEATS_PER_BAR == 0:
            self._append_edge(at_sec)
        self._beats += 1
        self._last_beat_sec = at_sec
        return self._advance()

    def _re_anchoring(self, at_sec: float) -> bool:
        return (self._last_beat_sec is not None
                and at_sec - self._last_beat_sec > _BEAT_GAP_SEC)

    def _re_anchor(self, at_sec: float) -> None:
        """Start a bar here rather than close one across the gap (#157's rule
        for the first beat, applied to the first beat after a silence).

        The open bar is dropped rather than stretched: its span would cover
        every cell of the gap, and one averaged observation over minutes of
        audio is a confident decision about a section nobody played.  It cannot
        have been observed yet -- a bar needs its closing edge -- so nothing
        already committed is disturbed.
        """
        logging.warning(
            f'[decoder] {at_sec - self._last_beat_sec:.1f}s without a beat — '
            f're-anchoring the bar grid at {at_sec:.2f}s and dropping '
            f'{len(self._cells)} pending cells')
        if self._edges and self._edge_base + len(self._edges) - 1 >= self._next_bar:
            self._edges.pop()
        self._append_edge(at_sec)
        self._beats = 0
        self._cells.clear()

    def push_posterior(self, at_sec: float, posterior, boundary: float) -> list:
        """One label cell: its END stamp, its class distribution, its hazard."""
        row = np.asarray(posterior, dtype=np.float64).reshape(-1)
        if row.size != self._n_classes:
            raise ValueError(f"a cell carries {row.size} classes, the priors "
                             f"name {self._n_classes}")
        self._cells.append((float(at_sec), row, float(boundary)))
        self._newest_cell_sec = float(at_sec)
        return self._advance()

    # -- internals ---------------------------------------------------------- #

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
        """Bars whose opening line has aged out can never be assembled.

        Beats arrive as the audio does and cells a whole chain latency behind
        them, so the grid is sized for that -- but if the feature stage stops
        while beats keep coming (a shed that outlasts the window), the grid runs
        away from the decode cursor and the bars in between have lost their
        cells as well as their lines.  Skipping them is the only honest answer,
        and the committer restarts because a trellis cannot span a hole.
        """
        if self._next_bar >= self._edge_base:
            return
        logging.warning(
            f'[decoder] the cell stream is {self._edge_base - self._next_bar} '
            f'bars behind the beat grid — bars {self._next_bar}..'
            f'{self._edge_base - 1} lost their lines and are skipped')
        self._next_bar = self._edge_base
        self._bar_offset = self._edge_base
        self._decoder.reset()
        self.recent_observations.clear()

    def _observable(self, bar: int) -> bool:
        """Both bar lines known, and every cell either of them can read seen.

        The closing edge is not enough on its own: the boundary is read within
        a tolerance of the OPENING line, and at a fast enough tempo that window
        reaches past the bar's own end.
        """
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
        """Drop every cell no future bar can still read.

        The floor is the next bar's opening line, which exists as soon as one
        beat has landed -- and which stops moving exactly when the beat stream
        does, so the seconds window and the hard cap under it are not the
        before-the-first-beat corner they look like.  Both are live paths on a
        stream that can lose beats while audio keeps arriving.
        """
        if self._have_edge(self._next_bar):
            floor = self._edge(self._next_bar) - self.params.boundary_tolerance_sec
        else:
            floor = self._newest_cell_sec - _ORPHAN_CELL_WINDOW_SEC
        while self._cells and self._cells[0][0] < floor:
            self._cells.popleft()
        while len(self._cells) > _MAX_PENDING_CELLS:
            self._cells.popleft()
