"""The rebirth principle: never charge minimum-duration for time the grid did not witness.

The live bar grid re-anchors when the beat stream stops for longer than
``BEAT_GAP_SEC``, and the runtime's re-anchor restarts the committer -- which
resets the HSMM to the corpus's START-OF-TRACK prior.  A decoder that is reborn
34 minutes into a set therefore believes it is at bar zero of an imaginary
track: it can commit ``intro`` (unreachable by any fitted transition), and it
must traverse a whole duration floor before it may leave, charging a minimum
duration for time it was not alive to witness.

Three coupled changes, each independently switchable so the val sweep can
attribute the result:

R1 carry     -- a rebirth starts from the class that was committed, not the prior.
R2 preaged   -- initial mass sits on FINAL duration states, so the run being
                joined is treated as already old.  Transitions observed AFTER
                the birth pay their floors in full.
R3 preroll   -- one virtual predecessor bar, so a boundary landing on the
                grid's own first bar has a transition to apply its bonus to.

This module owns the grid and the committer's lifecycle only.  Every number the
decoder produces comes from ``FixedLagViterbi`` unchanged; the policy is applied
by rewriting the initial distribution at birth, which is the one thing the
principle is about.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

BEATS_PER_BAR = 4
BEAT_GAP_SEC = 4.0
FIRST_BEAT_BAR_POSITION = 1
RE_ANCHOR_BAR_POSITION = 0

# evaluate_against_labels' NO_INTENT; the gate asserts the two still agree.
UNDECODED = ""


class LiveGrid(NamedTuple):
    edges: np.ndarray
    re_anchors: tuple
    orphans: tuple


class Rebirth(NamedTuple):
    carry: bool = False
    preaged: bool = False
    preroll: bool = False


SHIPPED = Rebirth(False, False, False)
R1 = Rebirth(True, False, False)
R2 = Rebirth(False, True, False)
R3 = Rebirth(False, False, True)
R1_R2 = Rebirth(True, True, False)
FULL = Rebirth(True, True, True)

ARMS = {"shipped": SHIPPED, "R1_carry": R1, "R2_preaged": R2, "R3_preroll": R3,
        "R1_R2": R1_R2, "full_R1_R2_R3": FULL}


def live_grid(beats, *, anchor: int = FIRST_BEAT_BAR_POSITION,
              gap_sec: float = BEAT_GAP_SEC,
              re_anchor_position: int = RE_ANCHOR_BAR_POSITION) -> LiveGrid:
    """``SectionDecoder.push_beat``'s grid, reproduced offline.

    ``orphans`` are the bars the runtime never observes: a re-anchor sets the
    cursor to the line it just drew, so the bar spanning the gap is skipped
    rather than decoded from a posterior that covers several seconds of silence.
    """
    lines: list = []
    re_anchors: list = []
    position = int(anchor)
    last: float | None = None
    for value in beats:
        at_sec = float(value)
        if last is not None and at_sec - last > gap_sec:
            lines.append(at_sec)
            re_anchors.append(len(lines) - 1)
            position = int(re_anchor_position)
        elif position == 0:
            lines.append(at_sec)
        position = (position + 1) % BEATS_PER_BAR
        last = at_sec
    edges = np.asarray(lines, dtype=np.float64)
    if edges.size < 2:
        raise RuntimeError(f"{edges.size} bar lines -- there is no grid to decode on")
    closed = np.append(edges, edges[-1] + float(np.median(np.diff(edges))))
    return LiveGrid(closed, tuple(re_anchors),
                    tuple(bar - 1 for bar in re_anchors if bar >= 1))


class LiveCommitter:
    """One ``FixedLagViterbi`` across a track's whole lifecycle of births."""

    def __init__(self, viterbi, policy: Rebirth = SHIPPED, *,
                 carry_override: int | None = None) -> None:
        self._viterbi = viterbi
        self._policy = policy
        self._override = carry_override
        self._base = 0
        self._offset = 0
        self._carried: int | None = None
        self.births: list = []

    @property
    def carried(self) -> int | None:
        return self._carried

    def birth(self, first_bar: int) -> None:
        viterbi = self._viterbi
        viterbi.reset()
        carried = self._carried if self._policy.carry else None
        if carried is not None and self._override is not None:
            carried = self._override
        if carried is not None and not np.isfinite(viterbi.priors.log_initial[carried]):
            logging.warning(
                f'[rebirth] the priors give {viterbi.classes[carried]!r} no way '
                f'to start a track, so a birth cannot carry it -- spreading the '
                f'start-of-track prior instead')
            carried = None
        states = viterbi._final_state if self._policy.preaged else viterbi._entry_state
        initial = np.full(len(viterbi._state_class), -np.inf, dtype=np.float64)
        if carried is None:
            initial[states] = viterbi.priors.log_initial + viterbi._entry_bonus
        else:
            initial[states[carried]] = (viterbi.priors.log_initial[carried]
                                        + viterbi._entry_bonus[carried])
        viterbi._log_initial = initial
        self._base = int(first_bar)
        self._offset = 0
        self.births.append({"bar": int(first_bar),
                            "carried": None if carried is None
                            else viterbi.classes[carried]})
        if self._policy.preroll:
            viterbi.push(None, None)
            self._offset = 1

    def _emit(self, decisions) -> list:
        out = []
        for decision in decisions:
            index = decision.bar - self._offset
            if index < 0:
                continue
            out.append((self._base + index, decision.label))
            self._carried = decision.class_index
        return out

    def push(self, posterior, boundary) -> list:
        return self._emit(self._viterbi.push(posterior, boundary))

    def flush(self) -> list:
        return self._emit(self._viterbi.flush())


def decode_live(grid: LiveGrid, posteriors, boundary, viterbi,
                policy: Rebirth = SHIPPED, *, carry_override: int | None = None) -> tuple:
    """``(label per bar, birth records)``; ``UNDECODED`` where nothing committed.

    A bar is undecoded for exactly the two reasons the runtime produces one: it
    was orphaned by a re-anchor, or it was still inside the lag window when the
    committer was reborn and its backtrace went with it.
    """
    n_bars = len(grid.edges) - 1
    labels = [UNDECODED] * n_bars
    committer = LiveCommitter(viterbi, policy, carry_override=carry_override)
    committer.birth(0)
    re_anchors = set(grid.re_anchors)
    orphans = set(grid.orphans)
    for bar in range(n_bars):
        if bar in re_anchors:
            committer.birth(bar)
        if bar in orphans:
            continue
        for real_bar, label in committer.push(posteriors[bar], boundary[bar]):
            labels[real_bar] = label
    for real_bar, label in committer.flush():
        labels[real_bar] = label
    return tuple(labels), committer.births


def runs(labels) -> list:
    """``[(first_bar, end_bar, label)]`` including the undecoded stretches."""
    spans: list = []
    for bar, label in enumerate(labels):
        if spans and spans[-1][2] == label and spans[-1][1] == bar:
            spans[-1][1] = bar + 1
        else:
            spans.append([bar, bar + 1, label])
    return [(start, end, label) for start, end, label in spans]
