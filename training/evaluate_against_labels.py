#!/usr/bin/env python
"""Score the CURRENT rule classifier against the Raveform expert labels.

Reads the label-aligned per-beat table built by ``build_training_table.py`` and
answers one question: *when the annotator says the track is in a drop, what are
the lights doing?*  It replaces the plumbing-only evaluator in
``simulate/evaluator.py`` (which asks only "did anything happen at all") with
musical ground truth, and its output is the baseline every future model must
beat::

    <data-dir>/baseline_eval.json   every number below, machine readable
    stdout                          the same numbers as a pasteable report

What is measured
----------------

**(a) Time-weighted confusion, intent x label.**  Rows are the six
``LightIntent`` values the engine can commit, columns are the annotator's
labels; cells are seconds of show.  The matrix is deliberately NOT square --
the intent vocabulary and the label vocabulary are different alphabets, and
flattening one into the other before looking at it hides exactly the failures
worth seeing.

**(b) Macro-F1 in two spaces.**  ``v1`` (5 classes) is primary: it is the space
the neural classifier trains on, so its numbers are the ones a model will be
compared against.  ``canonical`` (7 classes) is diagnostic -- it keeps
``cooldown`` and ``altoutro`` separate, which is where GROOVE actually lives.

**(c) Boundary-F1** at three tolerances: state-change instants against label
boundary instants, overall and broken out per boundary type (``-> drop`` is the
show-critical one).

**(d) Flicker rate** -- state changes that are NOT near any real boundary, per
audience-minute.  Continuity is the product metric: a classifier that is right
on average but twitches every four seconds is unusable on a dance floor.

(c) and (d) are reported for TWO change streams -- the intent stream (every
lighting change; the owner's continuity metric) and the class stream (changes
of label class only; the only fair comparand for a model that predicts label
classes and therefore cannot express DROP -> PEAK).  See ``STREAM_ORDER``.

**(e) Worst-15 songs** with their confusion rows, so failures can be listened to.

Design decisions worth knowing before reading a number
------------------------------------------------------

*Both event streams are quantised the same way.*  The table is per beat, so a
label boundary is only observable as "the first beat that carries the new
label", and an intent change as "the first beat that carries the new intent".
Using the SAME estimator on both sides means the quantisation cancels exactly
when a change is correctly timed: a perfectly placed intent change scores 0.0 s
of error, not half a beat.  The residual uncertainty is one beat period
(~0.47 s on this corpus), which is why the strict +/-0.5 s tolerance sits at the
resolution floor and should be read as "within a beat", not "sample accurate".

*Time weighting, not beat counting.*  Each beat carries the time until the next
beat, so the matrix reads in minutes of show.  A dropout (beats lost to
sidechain compression) would otherwise smear one classification over a silent
stretch, so a beat's weight is capped at ``MAX_GAP_FACTOR`` x the track's own
median beat interval and the discarded excess is reported.

*ATMOSPHERIC is scored against intro OR outro.*  An intent describes a moment,
not a position in the arrangement; the same quiet bar is ``intro`` at the start
of a track and ``outro`` at the end, and the classifier has no way to tell.  So
ATMOSPHERIC is credited if EITHER matches, and when it is wrong its false
positive is split evenly across the classes it claimed (no single class absorbs
the blame for an ambiguous prediction).  Its confusion row is printed separately.

*Beats with no committed intent are not a class.*  Before the engine's first
commit there is no prediction to score; those beats are excluded from every cell
and their share is reported instead.  Scoring them as errors would blame the
classifier for the engine's start-up, and scoring them as correct would flatter
it.

Usage::

    uv run python training/evaluate_against_labels.py \\
        --data-dir C:\\Users\\Julian\\Projects\\soundswitch-auto-pilot\\training\\data\\raveform
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import dataclasses
import datetime
import gzip
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from build_training_table import (  # noqa: E402  (needs the path inserts above)
    CANONICAL_ORDER,
    NO_INTENT,
    TABLE_FILE,
    V1_ORDER,
)

# --------------------------------------------------------------------------- #
# The mapping -- the one dict the owner iterates on
# --------------------------------------------------------------------------- #

# Intent -> the annotator label(s) that intent is CORRECT for, per space.
#
# A tuple with more than one member means the intent is genuinely ambiguous
# about which label it claims: it is credited against whichever of them is
# actually there, and its false positive is split across them when neither is.
#
#   atmospheric  quiet with no beat.  intro and outro are the same sound in
#                different places, and an intent cannot know track position.
#                (`altoutro` is the same structural role as `outro`; it survives
#                only in the canonical space.)
#   groove       steady mid-energy dance-floor loop.  Its semantic home is
#                `cooldown` -- the post-drop groove that keeps the floor moving.
#                v1 merges `cooldown` into `breakdown`, so in v1 groove and
#                breakdown are correct for the same class and become
#                indistinguishable; that merge is the NN spec's, not ours.
#   drop/peak    both mean "maximum arrangement"; the annotator has one word.
INTENT_TO_LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    #  intent          v1 (primary, 5-class)          canonical (diagnostic, 7-class)
    "atmospheric": {"v1": ("intro", "outro"), "canonical": ("intro", "outro", "altoutro")},
    "breakdown":   {"v1": ("breakdown",),     "canonical": ("breakdown",)},
    "groove":      {"v1": ("breakdown",),     "canonical": ("cooldown",)},
    "buildup":     {"v1": ("buildup",),       "canonical": ("buildup",)},
    "drop":        {"v1": ("drop",),          "canonical": ("drop",)},
    "peak":        {"v1": ("drop",),          "canonical": ("drop",)},
}

INTENT_ORDER = ("atmospheric", "breakdown", "groove", "buildup", "drop", "peak")


@dataclasses.dataclass(frozen=True)
class SpaceSpec:
    name: str
    column: str          # the training-table column holding this space's labels
    labels: tuple[str, ...]
    caption: str


SPACES: dict[str, SpaceSpec] = {
    "v1": SpaceSpec("v1", "label_v1", V1_ORDER,
                    "primary -- the space the NN classifier trains on"),
    "canonical": SpaceSpec("canonical", "label_canonical", CANONICAL_ORDER,
                           "diagnostic -- keeps cooldown and altoutro separate"),
}

# Boundary tolerances.  +/-2 s and +/-4 s are the plan's; +/-0.5 s is the NN
# spec's strict tier and sits at this table's beat-quantisation floor.
TOLERANCES_SEC = (0.5, 2.0, 4.0)

# The tolerance the headline flicker number and the worst-song ranking use.
PRIMARY_TOLERANCE_SEC = 2.0

# Two change streams, because "how often did it change" has two honest answers
# and they are not interchangeable:
#
#   intent  the committed LightIntent stream.  Every change here is a lighting
#           change -- the engine re-picks an effect on each one -- so this is
#           the show as an audience experiences it, and it is the owner's
#           continuity metric.
#   class   the same stream mapped into the label space FIRST, then
#           differenced, so a DROP -> PEAK or GROOVE -> BREAKDOWN move (same
#           label class, different lights) is not counted.  A model predicting
#           in `label_v1` emits a class stream by construction, so this is the
#           only stream a model may be compared against.
#
# The intent stream is a superset of the class stream: on this corpus 15.5% of
# intent changes leave the v1 class unchanged.  Quoting an intent-stream flicker
# rate at a model that can only produce class changes overstates the model's
# advantage; quoting a class-stream rate at the owner understates what the room
# sees.  Both are reported everywhere.
STREAM_ORDER = ("intent", "class")

# A beat may claim at most this multiple of its track's median beat interval.
# Beyond it the pipeline has lost beats, and extrapolating one classification
# across the hole would invent show time nobody observed.
MAX_GAP_FACTOR = 4.0

OUT_FILE = "baseline_eval.json"
WORST_SONGS = 15


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class TrackBeats:
    """One track's beats: parallel arrays, ordered by song time."""
    track_id: str
    times: tuple[float, ...]
    intents: tuple[str, ...]
    labels: dict[str, tuple[str, ...]]      # space name -> per-beat labels


def load_tracks(table_path: Path) -> list[TrackBeats]:
    """Read ``training_table.csv.gz`` into per-track beat arrays, order preserved."""
    times: dict[str, list[float]] = collections.OrderedDict()
    intents: dict[str, list[str]] = collections.defaultdict(list)
    labels: dict[str, dict[str, list[str]]] = {
        space: collections.defaultdict(list) for space in SPACES
    }
    with gzip.open(table_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            track_id = row["track_id"]
            times.setdefault(track_id, []).append(float(row["t_song"]))
            intents[track_id].append(row["intent_at_beat"])
            for space, spec in SPACES.items():
                labels[space][track_id].append(row[spec.column])
    return [
        TrackBeats(
            track_id=track_id,
            times=tuple(beat_times),
            intents=tuple(intents[track_id]),
            labels={space: tuple(labels[space][track_id]) for space in SPACES},
        )
        for track_id, beat_times in times.items()
    ]


# --------------------------------------------------------------------------- #
# Time weighting
# --------------------------------------------------------------------------- #


def beat_weights(times) -> tuple[list[float], float]:
    """Seconds of show each beat accounts for, plus the seconds discarded.

    A beat owns the time until the next beat.  The last beat owns one median
    beat interval -- the track does not stop dead on its final detected beat,
    but nothing is known past it either.  Gaps longer than ``MAX_GAP_FACTOR``
    medians are beat dropouts, not slow music: the excess is dropped and
    returned so a run can prove the clamp is not quietly rewriting the corpus.
    """
    count = len(times)
    if count == 0:
        return [], 0.0
    if count == 1:
        return [0.0], 0.0
    gaps = [max(0.0, times[i + 1] - times[i]) for i in range(count - 1)]
    median = statistics.median(gaps)
    cap = MAX_GAP_FACTOR * median
    weights: list[float] = []
    clamped = 0.0
    for gap in gaps:
        if cap > 0.0 and gap > cap:
            clamped += gap - cap
            weights.append(cap)
        else:
            weights.append(gap)
    weights.append(median)
    return weights, clamped


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


def intent_changes(times, intents) -> list[tuple[float, str]]:
    """``(t, new_intent)`` for every committed-intent change, in song time.

    Beats with no committed intent are removed BEFORE differencing, so the
    engine's first commit -- a rise out of "nothing decided yet" -- is not
    reported as a musical change.
    """
    committed = [(t, intent) for t, intent in zip(times, intents) if intent != NO_INTENT]
    return [
        (committed[i][0], committed[i][1])
        for i in range(1, len(committed))
        if committed[i][1] != committed[i - 1][1]
    ]


def class_changes(times, intents, space: str) -> list[tuple[float, tuple]]:
    """``(t, claimed_labels)`` for every change of the CLAIMED LABEL CLASS.

    The intent is mapped into the label space before differencing, so a move
    between two intents that claim the same class -- DROP to PEAK, or GROOVE to
    BREAKDOWN in v1 -- is not a change here even though the lights changed.
    This is the stream a model predicting in the label space can produce, and
    therefore the only fair comparand for one.
    """
    claims = [(t, INTENT_TO_LABELS[intent][space])
              for t, intent in zip(times, intents) if intent != NO_INTENT]
    return [claims[i] for i in range(1, len(claims))
            if claims[i][1] != claims[i - 1][1]]


def label_boundaries(times, labels) -> list[tuple[float, str]]:
    """``(t, new_label)`` for every label change, stamped like intent changes.

    Same estimator as ``intent_changes`` on purpose: both are "the first beat
    carrying the new value", so beat quantisation cancels when the classifier
    is right.
    """
    return [
        (times[i], labels[i])
        for i in range(1, len(labels))
        if labels[i] != labels[i - 1]
    ]


def match_events(truth, pred, tolerance: float) -> int:
    """Maximum number of truth instants pairable 1:1 with a prediction.

    Walking truths in time order and taking the earliest still-available
    prediction inside the window is optimal here: the graph is convex (each
    prediction's admissible truths form a contiguous run), so no earlier truth
    can profit from a prediction a later one needs.
    """
    truth_sorted = sorted(truth)
    pred_sorted = sorted(pred)
    matched = 0
    index = 0
    for instant in truth_sorted:
        while index < len(pred_sorted) and pred_sorted[index] < instant - tolerance:
            index += 1                              # too early for this and every later truth
        if index < len(pred_sorted) and pred_sorted[index] <= instant + tolerance:
            matched += 1
            index += 1
    return matched


def typed_predictions(changes, truth_by_label: dict) -> dict:
    """Split a change stream into one bucket per claimed class -- a PARTITION.

    A change into an unambiguous class goes to that class.  A change into an
    ambiguous one (ATMOSPHERIC claims intro AND outro) is still ONE prediction
    and must be counted once, or its two typed rows each carry a copy of it and
    both precision denominators inflate.  It is credited to whichever of its
    claimed classes it is nearest a real boundary of -- the same "correct if
    either matches" reading the confusion matrix uses -- with the claim order as
    a deterministic tie-break when neither class has a boundary at all.

    The alternative, splitting the credit 0.5/0.5, makes ``matched`` (a count of
    pairs) and ``n_pred`` (a sum of fractions) incommensurable: one ambiguous
    change matching one boundary would score precision 2.0.

    Guarantees ``sum(len(bucket)) == len(changes)``, which is asserted by the
    tests and visible in the report as the typed rows summing to the overall.
    """
    buckets: dict = {label: [] for label in truth_by_label}
    for instant, claimed in changes:
        if len(claimed) == 1:
            buckets[claimed[0]].append(instant)
            continue
        best, best_distance = claimed[0], float("inf")
        for label in claimed:
            boundaries = truth_by_label.get(label) or []
            distance = min((abs(t - instant) for t in boundaries), default=float("inf"))
            if distance < best_distance:
                best, best_distance = label, distance
        buckets[best].append(instant)
    return buckets


def flicker_instants(pred, truth, tolerance: float) -> list[float]:
    """Predicted changes with NO truth boundary within ``tolerance``.

    Proximity, not matching: two changes straddling one real boundary are a
    correctly-placed decision made twice, which costs boundary precision but is
    not what an audience perceives as flicker.  What they perceive is a change
    with no musical reason anywhere near it, which is what this counts.
    """
    truth_sorted = sorted(truth)
    loose: list[float] = []
    for instant in pred:
        index = bisect.bisect_left(truth_sorted, instant)
        near = False
        for candidate in truth_sorted[max(0, index - 1): index + 1]:
            if abs(candidate - instant) <= tolerance:
                near = True
                break
        if not near:
            loose.append(instant)
    return loose


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall, F1 -- 0.0 rather than NaN when a term is undefined."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _empty_boundary(labels) -> dict:
    return {
        stream: {
            tolerance: {
                "overall": {"n_truth": 0, "n_pred": 0, "matched": 0},
                "by_type": {label: {"n_truth": 0, "n_pred": 0, "matched": 0}
                            for label in labels},
            }
            for tolerance in TOLERANCES_SEC
        }
        for stream in STREAM_ORDER
    }


def _empty_flicker() -> dict:
    return {stream: {tolerance: 0 for tolerance in TOLERANCES_SEC}
            for stream in STREAM_ORDER}


@dataclasses.dataclass
class Score:
    """Raw accumulators for one track or a whole corpus; metrics are derived.

    Everything here is additive, so a corpus score is the element-wise sum of
    its songs (``aggregate``) and cannot drift from the per-song numbers
    printed beside it.
    """
    space: str
    track_id: str = "(corpus)"
    tracks: int = 1
    rows: int = 0
    confusion: dict = dataclasses.field(default_factory=dict)   # intent -> label -> sec
    counts: dict = dataclasses.field(default_factory=dict)      # label -> [tp, fp, fn]
    no_intent_by_label: dict = dataclasses.field(default_factory=dict)
    no_intent_sec: float = 0.0
    no_intent_rows: int = 0
    # Where the unpredicted beats sit relative to the committed ones.  An
    # INTERIOR gap is the dangerous one: `intent_changes` closes over it, so a
    # different intent either side would read as one change instead of a
    # commit, a silence and a re-commit.  Zero on this corpus -- if it ever
    # stops being zero, the change stream needs a gap rule.
    no_intent_leading: int = 0
    no_intent_interior: int = 0
    no_intent_trailing: int = 0
    scored_sec: float = 0.0
    weight_total_sec: float = 0.0
    clamped_sec: float = 0.0
    observed_intents: set = dataclasses.field(default_factory=set)
    boundary: dict = dataclasses.field(default_factory=dict)
    flicker: dict = dataclasses.field(default_factory=dict)

    # -- derived ---------------------------------------------------------- #

    @property
    def labels(self) -> tuple[str, ...]:
        return SPACES[self.space].labels

    @property
    def exposure_sec(self) -> float:
        return self.weight_total_sec

    @property
    def accuracy(self) -> float:
        correct = sum(count[0] for count in self.counts.values())
        return correct / self.scored_sec if self.scored_sec > 0 else 0.0

    @property
    def macro_classes(self) -> tuple[str, ...]:
        """Classes that were present in the truth or claimed by a prediction."""
        return tuple(
            label for label in self.labels
            if (self.counts[label][0] + self.counts[label][2]) > 0
            or (self.counts[label][0] + self.counts[label][1]) > 0
        )

    @property
    def expressible_classes(self) -> tuple[str, ...]:
        """Macro classes some OBSERVED intent is able to claim."""
        reachable = {
            label
            for intent in self.observed_intents
            for label in INTENT_TO_LABELS[intent][self.space]
        }
        return tuple(label for label in self.macro_classes if label in reachable)

    def f1(self, label: str) -> float:
        return prf(*self.counts[label])[2]

    def _macro(self, classes) -> float:
        return sum(self.f1(label) for label in classes) / len(classes) if classes else 0.0

    @property
    def macro_f1(self) -> float:
        return self._macro(self.macro_classes)

    @property
    def macro_f1_expressible(self) -> float:
        return self._macro(self.expressible_classes)

    @property
    def flicker_per_minute(self) -> dict:
        """Loose changes per audience-minute, per stream.

        The denominator is total evaluated show time, NOT the time that carried
        a prediction: a change is loose relative to the whole show, and using
        the scored time would make the rate creep up as unpredicted beats grow.
        """
        minutes = self.exposure_sec / 60.0
        return {
            stream: {
                tolerance: (count / minutes if minutes > 0 else 0.0)
                for tolerance, count in per_tolerance.items()
            }
            for stream, per_tolerance in self.flicker.items()
        }

    def boundary_prf(self, stream: str, tolerance: float, kind: str = "overall",
                     label: str | None = None) -> tuple[float, float, float]:
        cell = (self.boundary[stream][tolerance]["overall"] if kind == "overall"
                else self.boundary[stream][tolerance]["by_type"][label])
        return prf(cell["matched"],
                   cell["n_pred"] - cell["matched"],
                   cell["n_truth"] - cell["matched"])


def score_track(track: TrackBeats, space: str) -> Score:
    """Score one track in one label space."""
    spec = SPACES[space]
    labels = track.labels[space]
    weights, clamped = beat_weights(track.times)

    score = Score(
        space=space,
        track_id=track.track_id,
        rows=len(track.times),
        confusion={intent: {label: 0.0 for label in spec.labels}
                   for intent in INTENT_ORDER},
        counts={label: [0.0, 0.0, 0.0] for label in spec.labels},
        no_intent_by_label={label: 0.0 for label in spec.labels},
        clamped_sec=clamped,
        boundary=_empty_boundary(spec.labels),
        flicker=_empty_flicker(),
    )

    for intent, label, weight in zip(track.intents, labels, weights):
        if label not in score.counts:
            raise ValueError(f"unknown label {label!r} in space {space!r} "
                             f"(track {track.track_id})")
        score.weight_total_sec += weight
        if intent == NO_INTENT:
            score.no_intent_sec += weight
            score.no_intent_rows += 1
            score.no_intent_by_label[label] += weight
            continue
        if intent not in INTENT_TO_LABELS:
            raise ValueError(f"unknown intent {intent!r} (track {track.track_id})")
        score.observed_intents.add(intent)
        score.scored_sec += weight
        score.confusion[intent][label] += weight
        claimed = INTENT_TO_LABELS[intent][space]
        if label in claimed:
            score.counts[label][0] += weight                    # tp
        else:
            score.counts[label][2] += weight                    # fn
            share = weight / len(claimed)
            for target in claimed:
                score.counts[target][1] += share                # fp, split if ambiguous

    committed = [index for index, intent in enumerate(track.intents)
                 if intent != NO_INTENT]
    first, last = (committed[0], committed[-1]) if committed else (len(track.intents), -1)
    for index, intent in enumerate(track.intents):
        if intent != NO_INTENT:
            continue
        if index < first:
            score.no_intent_leading += 1
        elif index > last:
            score.no_intent_trailing += 1
        else:
            score.no_intent_interior += 1

    truth = label_boundaries(track.times, labels)
    truth_times = [t for t, _ in truth]
    truth_by_label = {label: [t for t, new in truth if new == label]
                      for label in spec.labels}

    # Both streams carry (instant, claimed labels), so the typed split below is
    # written once.  The intent stream maps AFTER differencing, the class stream
    # maps BEFORE it -- that difference is the whole point (see STREAM_ORDER).
    streams = {
        "intent": [(t, INTENT_TO_LABELS[intent][space])
                   for t, intent in intent_changes(track.times, track.intents)],
        "class": class_changes(track.times, track.intents, space),
    }

    for stream, changes in streams.items():
        change_times = [t for t, _ in changes]
        buckets = typed_predictions(changes, truth_by_label)
        for tolerance in TOLERANCES_SEC:
            cell = score.boundary[stream][tolerance]
            cell["overall"] = {
                "n_truth": len(truth_times),
                "n_pred": len(change_times),
                "matched": match_events(truth_times, change_times, tolerance),
            }
            for label in spec.labels:
                cell["by_type"][label] = {
                    "n_truth": len(truth_by_label[label]),
                    "n_pred": len(buckets[label]),
                    "matched": match_events(truth_by_label[label],
                                            buckets[label], tolerance),
                }
            score.flicker[stream][tolerance] = len(
                flicker_instants(change_times, truth_times, tolerance))
    return score


def aggregate(scores: list[Score]) -> Score:
    """Element-wise sum of per-track scores; metrics re-derive from the sum."""
    if not scores:
        raise ValueError("nothing to aggregate")
    space = scores[0].space
    spec = SPACES[space]
    total = Score(
        space=space,
        tracks=len(scores),
        confusion={intent: {label: 0.0 for label in spec.labels}
                   for intent in INTENT_ORDER},
        counts={label: [0.0, 0.0, 0.0] for label in spec.labels},
        no_intent_by_label={label: 0.0 for label in spec.labels},
        boundary=_empty_boundary(spec.labels),
        flicker=_empty_flicker(),
    )
    for score in scores:
        if score.space != space:
            raise ValueError("cannot aggregate across label spaces")
        total.rows += score.rows
        total.no_intent_sec += score.no_intent_sec
        total.no_intent_rows += score.no_intent_rows
        total.no_intent_leading += score.no_intent_leading
        total.no_intent_interior += score.no_intent_interior
        total.no_intent_trailing += score.no_intent_trailing
        total.scored_sec += score.scored_sec
        total.weight_total_sec += score.weight_total_sec
        total.clamped_sec += score.clamped_sec
        total.observed_intents |= score.observed_intents
        for intent, row in score.confusion.items():
            for label, seconds in row.items():
                total.confusion[intent][label] += seconds
        for label, count in score.counts.items():
            for index in range(3):
                total.counts[label][index] += count[index]
        for label, seconds in score.no_intent_by_label.items():
            total.no_intent_by_label[label] += seconds
        for stream in STREAM_ORDER:
            for tolerance in TOLERANCES_SEC:
                for key in ("n_truth", "n_pred", "matched"):
                    total.boundary[stream][tolerance]["overall"][key] += \
                        score.boundary[stream][tolerance]["overall"][key]
                    for label in spec.labels:
                        total.boundary[stream][tolerance]["by_type"][label][key] += \
                            score.boundary[stream][tolerance]["by_type"][label][key]
                total.flicker[stream][tolerance] += score.flicker[stream][tolerance]
    return total


# --------------------------------------------------------------------------- #
# Result assembly (JSON-ready)
# --------------------------------------------------------------------------- #


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _class_table(score: Score) -> dict:
    table = {}
    for label in score.labels:
        tp, fp, fn = score.counts[label]
        precision, recall, f1 = prf(tp, fp, fn)
        table[label] = {
            "support_sec": _round(tp + fn, 3),
            "predicted_sec": _round(tp + fp, 3),
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        }
    return table


def _boundary_table(score: Score, stream: str) -> dict:
    out: dict = {}
    for tolerance in TOLERANCES_SEC:
        precision, recall, f1 = score.boundary_prf(stream, tolerance)
        cell = score.boundary[stream][tolerance]["overall"]
        entry = {
            "overall": {**cell, "precision": _round(precision),
                        "recall": _round(recall), "f1": _round(f1)},
            "by_type": {},
        }
        for label in score.labels:
            typed = score.boundary[stream][tolerance]["by_type"][label]
            tprecision, trecall, tf1 = score.boundary_prf(
                stream, tolerance, "type", label)
            entry["by_type"][label] = {
                **typed, "precision": _round(tprecision),
                "recall": _round(trecall), "f1": _round(tf1),
            }
        out[f"{tolerance}"] = entry
    return out


def _stream_block(score: Score, stream: str) -> dict:
    return {
        "changes_total":
            score.boundary[stream][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"],
        "boundary": _boundary_table(score, stream),
        "flicker": {f"{tolerance}": count
                    for tolerance, count in score.flicker[stream].items()},
        "flicker_per_audience_minute": {
            f"{tolerance}": _round(rate, 4)
            for tolerance, rate in score.flicker_per_minute[stream].items()
        },
    }


def best_achievable_macro_f1(support: dict, unreachable) -> float:
    """Highest macro-F1 an always-committing system with this vocabulary can reach.

    The naive bound -- reachable classes divided by all classes -- is not
    attainable, and quoting it as a target is misleading: the truth mass sitting
    in the unreachable classes does not disappear when the system cannot name
    them.  Every one of those seconds is still predicted as SOMETHING, so it
    lands as a false positive on a reachable class and drags that class's
    precision below 1.0.

    So: assume every reachable class is otherwise perfect (recall 1.0) and
    allocate the unreachable mass ``U`` across them to do the least damage.
    A class of support ``S`` carrying extra mass ``x`` scores ``2S / (2S + x)``.
    That is CONVEX in ``x``, so the mean of them is convex, and a convex
    function over the simplex ``{x >= 0, sum(x) == U}`` takes its maximum at a
    VERTEX -- all of the mass on one class -- never spread across them.
    (Spreading is the *worst* allocation; the interior stationary point of the
    Lagrangian is the minimum, which is exactly the trap a "water-filling"
    reading falls into.  The property test pins this.)

    Concentrating the damage on the largest class is therefore optimal, and a
    reachable class with no ground truth at all absorbs everything for free --
    its F1 is zero whatever happens to it.
    """
    classes = list(support)
    total = len(classes)
    if total == 0:
        return 0.0
    reachable = [label for label in classes if label not in unreachable]
    scoring = [label for label in reachable if support[label] > 0.0]
    extra = sum(max(support[label], 0.0) for label in unreachable)
    if not scoring:
        return 0.0
    if extra <= 0.0 or len(scoring) < len(reachable):
        # Nothing to absorb, or a zero-support class to dump it all on.
        return len(scoring) / total
    biggest = max(support[label] for label in scoring)
    damaged = 2 * biggest / (2 * biggest + extra)
    return (len(scoring) - 1 + damaged) / total


def _structural(score: Score) -> dict:
    """What the intent vocabulary CANNOT say, separated from what it gets wrong."""
    spec = SPACES[score.space]
    reachable = {
        label
        for intent in score.observed_intents
        for label in INTENT_TO_LABELS[intent][score.space]
    }
    unreachable = [label for label in spec.labels if label not in reachable]
    truth_sec = {label: score.counts[label][0] + score.counts[label][2]
                 for label in spec.labels}
    total_truth = sum(truth_sec.values())
    unreachable_sec = sum(truth_sec[label] for label in unreachable)
    silent_intents = [intent for intent in INTENT_ORDER
                      if intent not in score.observed_intents]
    return {
        "observed_intents": sorted(score.observed_intents),
        "never_observed_intents": silent_intents,
        "unreachable_classes": unreachable,
        "unreachable_label_sec": _round(unreachable_sec, 3),
        "unreachable_label_share": _round(
            unreachable_sec / total_truth if total_truth > 0 else 0.0),
        # A bound that CANNOT BE EXCEEDED, not a target: it ignores where the
        # unreachable mass has to go.
        "macro_f1_upper_bound": _round(
            (len(spec.labels) - len(unreachable)) / len(spec.labels)
            if spec.labels else 0.0),
        # The number to actually aim a comparison at.
        "macro_f1_best_achievable": _round(
            best_achievable_macro_f1(truth_sec, unreachable)),
    }


def _song_entry(score: Score) -> dict:
    return {
        "track_id": score.track_id,
        "rows": score.rows,
        "exposure_sec": _round(score.exposure_sec, 3),
        "macro_f1": _round(score.macro_f1),
        "macro_classes": list(score.macro_classes),
        "accuracy": _round(score.accuracy),
        "f1": {label: _round(score.f1(label)) for label in score.macro_classes},
        "boundary_f1": {
            stream: {f"{tolerance}": _round(score.boundary_prf(stream, tolerance)[2])
                     for tolerance in TOLERANCES_SEC}
            for stream in STREAM_ORDER
        },
        "flicker_per_audience_minute": {
            stream: {f"{tolerance}": _round(rate, 3)
                     for tolerance, rate in per_tolerance.items()}
            for stream, per_tolerance in score.flicker_per_minute.items()
        },
        "changes": {
            stream: score.boundary[stream][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"]
            for stream in STREAM_ORDER
        },
        "label_boundaries":
            score.boundary["intent"][PRIMARY_TOLERANCE_SEC]["overall"]["n_truth"],
        "confusion_sec": {
            intent: {label: _round(seconds, 3)
                     for label, seconds in row.items() if seconds > 0}
            for intent, row in score.confusion.items()
            if any(seconds > 0 for seconds in row.values())
        },
    }


def evaluate(tracks: list[TrackBeats], worst: int = WORST_SONGS) -> dict:
    """Score every track in every space and assemble the JSON-ready result."""
    result: dict = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                                .replace(microsecond=0).isoformat(),
        "tolerances_sec": list(TOLERANCES_SEC),
        "primary_tolerance_sec": PRIMARY_TOLERANCE_SEC,
        "max_gap_factor": MAX_GAP_FACTOR,
        "intent_to_labels": {intent: {space: list(labels)
                                      for space, labels in per_space.items()}
                             for intent, per_space in INTENT_TO_LABELS.items()},
        "spaces": {},
    }

    corpus_by_space: dict[str, Score] = {}
    beat_intervals: list[float] = []
    for track in tracks:
        beat_intervals.extend(
            track.times[i + 1] - track.times[i] for i in range(len(track.times) - 1))

    for space in SPACES:
        songs = [score_track(track, space) for track in tracks]
        corpus = corpus_by_space[space] = aggregate(songs)
        ranked = sorted(songs, key=lambda score: (score.macro_f1, score.track_id))
        per_song = [_song_entry(score) for score in songs]
        result["spaces"][space] = {
            "space": space,
            "caption": SPACES[space].caption,
            "labels": list(SPACES[space].labels),
            "macro_f1": _round(corpus.macro_f1),
            "macro_classes": list(corpus.macro_classes),
            "macro_f1_expressible": _round(corpus.macro_f1_expressible),
            "expressible_classes": list(corpus.expressible_classes),
            "accuracy": _round(corpus.accuracy),
            "per_class": _class_table(corpus),
            "confusion_sec": {
                intent: {label: _round(seconds, 3) for label, seconds in row.items()}
                for intent, row in corpus.confusion.items()
            },
            "no_intent_by_label_sec": {
                label: _round(seconds, 3)
                for label, seconds in corpus.no_intent_by_label.items()
            },
            "streams": {stream: _stream_block(corpus, stream)
                        for stream in STREAM_ORDER},
            "label_boundaries_total":
                corpus.boundary["intent"][PRIMARY_TOLERANCE_SEC]["overall"]["n_truth"],
            "structural": _structural(corpus),
            "worst_songs": [_song_entry(score) for score in ranked[:worst]],
            "per_song": per_song,
        }

    # Coverage is a property of the table, not of a label space: every space
    # sees the same beats and the same weights, so any of them can report it.
    reference = corpus_by_space["v1"]
    result["coverage"] = {
        "tracks": len(tracks),
        "rows": reference.rows,
        "exposure_sec": _round(reference.exposure_sec, 3),
        "exposure_min": _round(reference.exposure_sec / 60.0, 3),
        "scored_sec": _round(reference.scored_sec, 3),
        "no_intent_sec": _round(reference.no_intent_sec, 3),
        "no_intent_rows": reference.no_intent_rows,
        "no_intent_leading_rows": reference.no_intent_leading,
        "no_intent_interior_rows": reference.no_intent_interior,
        "no_intent_trailing_rows": reference.no_intent_trailing,
        "no_intent_share": _round(
            reference.no_intent_sec / reference.weight_total_sec
            if reference.weight_total_sec > 0 else 0.0),
        "clamped_sec": _round(reference.clamped_sec, 3),
        "median_beat_interval_sec": _round(
            statistics.median(beat_intervals) if beat_intervals else 0.0),
    }
    return result


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

WIDTH = 100


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _heading(text: str, char: str = "=") -> list[str]:
    return [_rule(char), f"  {text}", _rule(char)]


def _matrix_block(space_result: dict, no_intent: dict, unit: float,
                  suffix: str) -> list[str]:
    labels = space_result["labels"]
    confusion = space_result["confusion_sec"]
    header = f'  {"intent":<12}' + "".join(f"{label:>11}" for label in labels) \
        + f'{"total":>11}'
    lines = [header, "  " + "-" * (len(header) - 2)]
    for intent in INTENT_ORDER:
        row = confusion[intent]
        total = sum(row.values())
        mark = " *" if total == 0 else "  "
        cells = "".join(f"{row[label] / unit:>11.1f}" for label in labels)
        lines.append(f'  {intent:<12}{cells}{total / unit:>11.1f}{mark}')
    lines.append("  " + "-" * (len(header) - 2))
    no_intent_total = sum(no_intent.values())
    lines.append(f'  {"(no intent)":<12}'
                 + "".join(f"{no_intent[label] / unit:>11.1f}" for label in labels)
                 + f"{no_intent_total / unit:>11.1f}"
                 + "  excluded from every score")
    column_total = {
        label: sum(confusion[intent][label] for intent in INTENT_ORDER)
        for label in labels
    }
    lines.append(f'  {"label total":<12}'
                 + "".join(f"{column_total[label] / unit:>11.1f}" for label in labels)
                 + f"{sum(column_total.values()) / unit:>11.1f}")
    lines.append(f"  values in {suffix}; '*' marks an intent the engine never committed")
    return lines


def _row_percent_block(space_result: dict) -> list[str]:
    labels = space_result["labels"]
    confusion = space_result["confusion_sec"]
    header = f'  {"intent":<12}' + "".join(f"{label:>11}" for label in labels)
    lines = [header, "  " + "-" * (len(header) - 2)]
    for intent in INTENT_ORDER:
        row = confusion[intent]
        total = sum(row.values())
        if total <= 0:
            lines.append(f'  {intent:<12}' + f'{"(never committed)":>{11 * len(labels)}}')
            continue
        cells = "".join(f"{100.0 * row[label] / total:>11.1f}" for label in labels)
        lines.append(f'  {intent:<12}{cells}')
    lines.append("  each row sums to 100%: where the lights were, the annotator said ...")
    return lines


def _per_class_block(space_result: dict) -> list[str]:
    lines = [
        f'  {"class":<12}{"support":>10}{"share":>8}{"predicted":>11}'
        f'{"precision":>11}{"recall":>9}{"f1":>8}',
        "  " + "-" * 67,
    ]
    per_class = space_result["per_class"]
    total_support = sum(entry["support_sec"] for entry in per_class.values()) or 1.0
    for label in space_result["labels"]:
        entry = per_class[label]
        lines.append(
            f'  {label:<12}{entry["support_sec"] / 60.0:>9.1f}m'
            f'{100.0 * entry["support_sec"] / total_support:>7.1f}%'
            f'{entry["predicted_sec"] / 60.0:>10.1f}m'
            f'{entry["precision"]:>11.3f}{entry["recall"]:>9.3f}{entry["f1"]:>8.3f}'
        )
    lines.append("  " + "-" * 67)
    lines.append(f'  macro-F1 over {len(space_result["macro_classes"])} classes'
                 f'{space_result["macro_f1"]:>44.3f}')
    lines.append(f'  macro-F1 over the {len(space_result["expressible_classes"])} '
                 f'classes the observed intents can express'
                 f'{space_result["macro_f1_expressible"]:>17.3f}')
    lines.append(f'  time-weighted accuracy{space_result["accuracy"]:>45.3f}')
    drop = per_class["drop"]
    lines.append("")
    lines.append(f'  DROP specifically -- the show-critical class: '
                 f'recall {drop["recall"]:.3f} '
                 f'(the lights are in DROP/PEAK for {100.0 * drop["recall"]:.1f}% of')
    lines.append(f'  real drop time), precision {drop["precision"]:.3f} '
                 f'(they are wrong {100.0 * (1 - drop["precision"]):.1f}% of the time '
                 f'they claim it),')
    lines.append(f'  F1 {drop["f1"]:.3f} on {drop["support_sec"] / 60.0:.1f} min of '
                 f'ground truth.')
    return lines


def _boundary_block(space_result: dict) -> list[str]:
    lines = [
        "  Two streams, two questions.  Use the INTENT stream to ask how the show",
        "  behaved: every one of those changes re-picked a lighting effect.  Use the",
        "  CLASS stream to compare against a model that predicts label classes -- it",
        "  cannot express a DROP -> PEAK move, so scoring it against intent-stream",
        "  numbers would credit it for changes it is structurally unable to make.",
        "",
        f'  {"stream":<8}{"boundary set":<16}{"tol":>7}{"truth":>9}{"pred":>9}'
        f'{"matched":>9}{"precision":>11}{"recall":>9}{"f1":>8}',
        "  " + "-" * 84,
    ]
    for stream in STREAM_ORDER:
        for tolerance in TOLERANCES_SEC:
            cell = space_result["streams"][stream]["boundary"][f"{tolerance}"]["overall"]
            lines.append(
                f'  {stream:<8}{"all boundaries":<16}{f"+/-{tolerance}s":>7}'
                f'{cell["n_truth"]:>9}{cell["n_pred"]:>9}{cell["matched"]:>9}'
                f'{cell["precision"]:>11.3f}{cell["recall"]:>9.3f}{cell["f1"]:>8.3f}')
    lines.append("  " + "-" * 84)
    for label in space_result["labels"]:
        for stream in STREAM_ORDER:
            for tolerance in TOLERANCES_SEC:
                cell = (space_result["streams"][stream]["boundary"]
                        [f"{tolerance}"]["by_type"][label])
                lines.append(
                    f'  {stream:<8}{"-> " + label:<16}{f"+/-{tolerance}s":>7}'
                    f'{cell["n_truth"]:>9}{cell["n_pred"]:>9}{cell["matched"]:>9}'
                    f'{cell["precision"]:>11.3f}{cell["recall"]:>9.3f}'
                    f'{cell["f1"]:>8.3f}')
        lines.append("")
    lines.append("  typed rows pair boundaries INTO a class with changes into a state that")
    lines.append("  claims that class; '-> drop' is the show-critical row.  A change that")
    lines.append("  claims two classes is credited to one of them (the nearer boundary), so")
    lines.append("  the typed 'pred' column partitions the stream and sums to the overall.")
    return lines


def _flicker_block(space_result: dict) -> list[str]:
    lines = [
        f'  {"stream":<8}{"tolerance":<11}{"loose changes":>15}{"of that stream":>17}'
        f'{"per audience-minute":>22}',
        "  " + "-" * 73,
    ]
    for stream in STREAM_ORDER:
        block = space_result["streams"][stream]
        total_changes = block["changes_total"] or 1
        for tolerance in TOLERANCES_SEC:
            key = f"{tolerance}"
            lines.append(
                f'  {stream:<8}{f"+/-{tolerance}s":<11}{block["flicker"][key]:>15}'
                f'{100.0 * block["flicker"][key] / total_changes:>16.1f}%'
                f'{block["flicker_per_audience_minute"][key]:>22.2f}')
        per_song = sorted(song["flicker_per_audience_minute"][stream]
                          [f"{PRIMARY_TOLERANCE_SEC}"]
                          for song in space_result["per_song"])
        if per_song:
            def _quantile(fraction: float) -> float:
                return per_song[min(len(per_song) - 1, int(fraction * len(per_song)))]
            lines.append(f'           per-song at +/-{PRIMARY_TOLERANCE_SEC}s: '
                         f'median {_quantile(0.5):.2f}, p90 {_quantile(0.9):.2f}, '
                         f'max {per_song[-1]:.2f} changes/min '
                         f'({block["changes_total"]} changes total)')
        lines.append("")
    lines.append(f'  against {space_result["label_boundaries_total"]} real label '
                 f'boundaries.  Flicker asks "was there any musical reason NEAR this')
    lines.append('  change" (proximity); boundary precision additionally punishes a')
    lines.append('  correctly-placed decision made twice.  An audience notices the first.')
    return lines


def _structural_block(space_result: dict) -> list[str]:
    structural = space_result["structural"]
    unreachable = structural["unreachable_classes"]
    silent = structural["never_observed_intents"]
    lines = [
        f'  intents the engine committed : {", ".join(structural["observed_intents"])}',
        f'  intents NEVER committed      : '
        f'{", ".join(silent) if silent else "(none)"}',
        f'  label classes therefore out of reach: '
        f'{", ".join(unreachable) if unreachable else "(none)"}',
        f'  ground truth sitting in those classes: '
        f'{structural["unreachable_label_sec"] / 60.0:.1f} min '
        f'({100.0 * structural["unreachable_label_share"]:.1f}% of labelled time)',
        f'  macro-F1 CANNOT EXCEED               : '
        f'{structural["macro_f1_upper_bound"]:.3f}  (reachable classes / all classes)',
        f'  best macro-F1 actually achievable    : '
        f'{structural["macro_f1_best_achievable"]:.3f}  (the unreachable time still '
        f'has to be',
        "                                                predicted as something, so it "
        "lands as",
        "                                                false positives on the "
        "reachable classes)",
    ]
    if silent:
        lines += [
            "",
            "  This is a STRUCTURAL result, not a tuning error.  Recall on those",
            "  classes is zero because the engine has no state that claims them --",
            "  no threshold change can move it.  Read their rows as 'cannot say',",
            "  and compare the macro-F1 against the achievable figure above -- the",
            "  upper bound is a wall, not a target, and nothing can sit on it.",
        ]
    return lines


def space_result_space(space_result: dict) -> str:
    """Recover a space name from its result block (labels are unique per space)."""
    for name, spec in SPACES.items():
        if list(spec.labels) == space_result["labels"]:
            return name
    raise ValueError("unrecognised label space")


def _worst_block(space_result: dict) -> list[str]:
    space = space_result_space(space_result)
    lines = [
        f'  {"track_id":<22}{"macro-F1":>9}{"acc":>7}{"bF1@2s":>8}'
        f'{"flick/min":>11}{"chg i/c":>10}{"bounds":>8}{"min":>7}   '
        f'(boundary/flicker: intent stream)',
        "  " + "-" * 83,
    ]
    tol = f"{PRIMARY_TOLERANCE_SEC}"
    for song in space_result["worst_songs"]:
        changes = f'{song["changes"]["intent"]}/{song["changes"]["class"]}'
        lines.append(
            f'  {song["track_id"]:<22}{song["macro_f1"]:>9.3f}{song["accuracy"]:>7.3f}'
            f'{song["boundary_f1"]["intent"][tol]:>8.3f}'
            f'{song["flicker_per_audience_minute"]["intent"][tol]:>11.2f}'
            f'{changes:>10}{song["label_boundaries"]:>8}'
            f'{song["exposure_sec"] / 60.0:>7.1f}')
        cells = []
        for intent, row in song["confusion_sec"].items():
            for label, seconds in row.items():
                mark = "=" if label in INTENT_TO_LABELS[intent][space] else ">"
                cells.append((seconds, f"{intent}{mark}{label} {seconds / 60.0:.1f}"))
        cells.sort(key=lambda cell: -cell[0])
        line = "     "
        for _, text in cells:
            if len(line) + len(text) + 2 > WIDTH:
                lines.append(line)
                line = "     "
            line += "  " + text
        lines.append(line)
    lines.append("  every non-zero confusion cell, minutes, largest first; "
                 "'=' correct, '>' wrong")
    return lines


def render_report(result: dict) -> str:
    coverage = result["coverage"]
    lines: list[str] = []
    lines += _heading("BASELINE EVALUATION -- rule classifier vs Raveform expert labels")
    lines += [
        f'  generated            : {result["generated_at"]}',
        f'  tracks               : {coverage["tracks"]}',
        f'  labelled beats (rows): {coverage["rows"]}',
        f'  evaluated show time  : {coverage["exposure_min"]:.1f} min '
        f'({coverage["exposure_sec"] / 3600.0:.2f} h)',
        f'  median beat interval : {coverage["median_beat_interval_sec"]:.3f} s '
        f'-- the quantisation floor of every boundary number below',
        f'  beats with no intent : {coverage["no_intent_rows"]} rows / '
        f'{coverage["no_intent_sec"]:.1f} s '
        f'({100.0 * coverage["no_intent_share"]:.3f}% of show time), excluded as '
        f'"no prediction"',
        f'                         leading {coverage["no_intent_leading_rows"]}, '
        f'interior {coverage["no_intent_interior_rows"]}, '
        f'trailing {coverage["no_intent_trailing_rows"]} '
        f'-- interior gaps would bridge the change stream',
        f'  time discarded by the dropout clamp: {coverage["clamped_sec"]:.1f} s '
        f'({100.0 * coverage["clamped_sec"] / max(coverage["exposure_sec"], 1e-9):.4f}%)',
        "",
        "  mapping used (intent -> labels it is correct for):",
    ]
    for intent in INTENT_ORDER:
        per_space = result["intent_to_labels"][intent]
        lines.append(f'    {intent:<12} v1: {"|".join(per_space["v1"]):<14} '
                     f'canonical: {"|".join(per_space["canonical"])}')
    lines.append("")

    for space in ("v1", "canonical"):
        space_result = result["spaces"][space]
        lines += _heading(f'{space.upper()} SPACE ({space_result["caption"]})')
        lines += ["", "  CONFUSION MATRIX -- intent (rows) x label (columns), minutes", ""]
        lines += _matrix_block(space_result, space_result["no_intent_by_label_sec"],
                               60.0, "minutes of show")
        lines += ["", "  CONFUSION MATRIX -- row-normalised, %", ""]
        lines += _row_percent_block(space_result)
        lines += ["", "  PER-CLASS PRECISION / RECALL / F1 (time-weighted)", ""]
        lines += _per_class_block(space_result)
        lines += ["", "  ATMOSPHERIC / STRUCTURAL COVERAGE", ""]
        lines += _structural_block(space_result)
        lines += ["", "  BOUNDARY-F1 -- state-change instants vs label boundaries", ""]
        lines += _boundary_block(space_result)
        lines += ["", "  FLICKER -- state changes with no boundary nearby", ""]
        lines += _flicker_block(space_result)
        lines += ["", f'  WORST {len(space_result["worst_songs"])} SONGS by macro-F1', ""]
        lines += _worst_block(space_result)
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_data_dir() -> Path:
    return REPO_ROOT / "training" / "data" / "raveform"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
    except Exception:                                           # pragma: no cover
        return "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp.replace(path)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="corpus root holding training_table.csv.gz "
                             "(default: %(default)s)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"result JSON (default: <data-dir>/{OUT_FILE})")
    parser.add_argument("--worst", type=int, default=WORST_SONGS,
                        help="how many worst songs to list (default: %(default)s)")
    parser.add_argument("--report", type=Path, default=None,
                        help="also write the printed report to this file")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    table_path = data_dir / TABLE_FILE
    if not table_path.exists():
        print(f"no training table at {table_path}", file=sys.stderr)
        return 2

    tracks = load_tracks(table_path)
    result = evaluate(tracks, worst=args.worst)
    result["table"] = {
        "path": str(table_path),
        "sha256": file_sha256(table_path),
        "bytes": table_path.stat().st_size,
    }
    result["git_sha"] = git_sha()

    out_path = args.out or (data_dir / OUT_FILE)
    write_json(out_path, result)

    report = render_report(result)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)
    print(f"result json: {out_path}")
    if args.report:
        args.report.write_text(report, encoding="utf-8", newline="\n")
        print(f"report text: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
