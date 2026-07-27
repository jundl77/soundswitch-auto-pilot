#!/usr/bin/env python
"""Score the decoded NN chain against the Raveform labels, beside the rule classifier.

This is the verdict run.  Everything before it -- dataset, CRNN, ONNX export,
posterior sidecars, priors, fixed-lag Viterbi -- exists to produce a decision
stream; this module asks the only question that matters about one: **is it
better than what ships today, measured the same way, on the same tracks?**

It answers on the VAL split by default, and on the TEST split exactly once --
the selection-clean verdict.  ``--split`` is how that read happens (``sweep.py``
refuses ``test`` outright; this module cannot, because the verdict needs it), so
the discipline lives in the workflow rather than in the code: after a test read,
a bad result is a NEW versioned model with its own single read, never a re-tune
scored against the same split.  Both verdicts are written side by side --
``eval_val.json`` is the tuned reading, ``eval_test.json`` the clean one.

How the comparison is kept honest
---------------------------------

*The metrics are not reimplemented.*  Every number comes from
``evaluate_against_labels`` -- the same ``Score`` accumulators, the same
``match_events``, ``flicker_instants``, ``beat_weights`` and ``prf`` that
produced the committed rule baseline.  What this module adds is one accumulation
loop, because two things genuinely differ (below) and only those two.

*Undecoded is not wrong.*  The decoder runs on a bar grid, so it says nothing
about ``[0, first_downbeat)`` -- median 0.107 s, max 0.466 s on this corpus --
or about the sub-bar tail past the final bar line.  Those beats are marked with
the evaluator's own ``NO_INTENT`` sentinel, which it already excludes from every
cell and reports separately.  Counting them as misdecodes would be a systematic
charge against ``intro`` and ``outro`` specifically, since those are the classes
that own the ends of a track, and it would be a charge for the annotation
grid's origin rather than for anything the network did.

*Two of the five classes are not actually contested, and the report says so.*
ATMOSPHERIC -- the only intent covering ``intro`` and ``outro`` -- fires from a
beat-ABSENCE timer, while the training table carries one row per DETECTED BEAT.
The rows that exist are therefore exactly the rows where that trigger cannot be
active, so the baseline's zero on those two classes is a property of the
beat-indexed harness, not a measured limit of its vocabulary.  Those classes
carry ~64 % of the headline delta, so ``expressible_comparison`` publishes the
restricted macro-F1 over the contested classes as the model-vs-model number, and
per-track win counts are reported in BOTH readings -- only the restricted one
can go negative.

*The claim map is the fairness argument, and it is reported both ways.*  The
rule classifier's ATMOSPHERIC is credited against ``intro`` OR ``outro`` because
an intent describes a moment and cannot know its position in the arrangement.
A network predicting ``label_v1`` is not under that handicap, so its primary
score uses IDENTITY claims: predicting ``intro`` over ``outro`` is a miss.  That
is a strictly higher bar than the baseline is held to, so the same decode is
ALSO scored with rule-equivalent claims -- intro/outro collapsed back into one
ambiguous class, exactly the handicap ATMOSPHERIC carries -- and both appear in
the report.  If the NN only wins under one of them, the report says so instead
of picking the flattering one.

*The class stream is the comparand.*  A model predicting label classes cannot
express DROP -> PEAK, so quoting the rule classifier's intent-stream flicker
against it would overstate the model.  ``STREAM_ORDER`` exists for exactly this
and the side-by-side table reads the ``class`` stream on both sides.  Under
identity claims the NN's two streams are provably the same stream; the report
records that rather than printing one number twice.

Usage::

    uv run python -m training.nn.evaluate_v1 --data-dir <data> [--split val]
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from .decoder import DecodeParams, FixedLagViterbi, bar_grid, bar_observations
from .priors import MODEL_VERSION, MODELS_DIR, PRIORS_FILE, Priors

from build_training_table import NO_INTENT, TABLE_FILE  # noqa: E402  (nn/__init__ sets the path)
from evaluate_against_labels import (  # noqa: E402
    INTENT_ORDER,
    INTENT_TO_LABELS,
    PRIMARY_TOLERANCE_SEC,
    SPACES,
    TOLERANCES_SEC,
    Score,
    TrackBeats,
    _empty_boundary,
    _empty_flicker,
    _structural,
    aggregate,
    beat_weights,
    file_sha256,
    flicker_instants,
    git_sha,
    intent_changes,
    label_boundaries,
    load_tracks,
    match_events,
    prf,
    score_track,
    typed_predictions,
)
from evaluate_against_labels import write_json as _write_json  # noqa: E402
from raveform_fetch_annotations import BEATS_DIR, annotations_dir  # noqa: E402

# The evaluator's "no prediction at this beat" sentinel, reused rather than
# reinvented: it already means "exclude from every cell and report the share",
# which is exactly what an undecoded bar-grid edge is.
UNDECODED = NO_INTENT

SPLITS_FILE = "splits.json"
POSTERIORS_DIR = "posteriors"
# Duplicated from export_onnx rather than imported: that module builds torch
# modules at import time, and the decode-and-score path is torch-free by
# contract.  One filename is a cheaper coupling than dragging torch in.
MODEL_FILE = "model.onnx"
# Named after the split it scored, so a ``--split test`` run without ``--out``
# cannot silently overwrite the val verdict with test numbers -- the two
# readings answer different questions and must both survive on disk.
EVAL_FILE = "eval_{split}.json"
DECODER_CONFIG_FILE = "decoder_config.json"

# The primary space.  ``canonical`` is not scored here: the network's vocabulary
# IS ``label_v1``, so a canonical score would be measuring the label merge
# rather than the model.
DEFAULT_SPACE = "v1"


# --------------------------------------------------------------------------- #
# Claim maps
# --------------------------------------------------------------------------- #


def identity_claims(space: str = DEFAULT_SPACE) -> dict:
    """Each predicted class claims exactly itself -- the primary scoring.

    The network names the class; there is no ambiguity to forgive.
    """
    return {label: (label,) for label in SPACES[space].labels}


def rule_equivalent_claims(space: str = DEFAULT_SPACE) -> dict:
    """The network handed the rule classifier's exact handicap, for comparison.

    ATMOSPHERIC claims ``intro`` OR ``outro`` and is credited against whichever
    is there.  Giving the NN the same ambiguity answers the one question a
    side-by-side table cannot answer on its own: how much of the gap is the
    model, and how much is the two sides being scored under different rules.

    Note the second-order effect, which is the point of doing it properly rather
    than just relabelling the confusion matrix: two classes that make the same
    claim are not distinguishable in the CLASS stream either, so an intro ->
    outro switch stops being a class change here exactly as it does for the rule
    classifier.  Boundary and flicker move with it.
    """
    ambiguous = INTENT_TO_LABELS["atmospheric"][space]
    return {label: (ambiguous if label in ambiguous else (label,))
            for label in SPACES[space].labels}


def _claim_key(claims: dict, label: str) -> str:
    """A stable identity for "what class does this prediction claim".

    Differencing THIS instead of the raw prediction is what turns
    ``intent_changes`` into the evaluator's ``class_changes``: the claim is
    formed before the difference, so two predictions making the same claim are
    not a change.  Reusing the evaluator's own differencer for both streams
    keeps the quantisation identical on both sides of the table.
    """
    return "\x00".join(claims[label])


# --------------------------------------------------------------------------- #
# Bar decisions -> beat grid
# --------------------------------------------------------------------------- #


def beat_classes(beat_times, edges, bar_labels) -> tuple:
    """The decoded class at every beat; ``UNDECODED`` outside the bar grid.

    The decoder commits per bar and the evaluator scores per beat, so this is
    the only place the two clocks meet.  A beat exactly on a bar line belongs to
    the bar it opens (the decision takes effect there), and a beat outside
    ``[edges[0], edges[n_bars])`` -- the pre-downbeat head or the sub-bar tail --
    has no decision at all rather than the nearest one.
    """
    labels = tuple(bar_labels)
    edges = np.asarray(edges, dtype=np.float64)
    if len(labels) + 1 > edges.size:
        raise ValueError(
            f"{len(labels)} bar labels need {len(labels) + 1} edges, got {edges.size}")
    if not labels:
        return tuple(UNDECODED for _ in beat_times)
    index = np.searchsorted(edges[:len(labels) + 1],
                            np.asarray(beat_times, dtype=np.float64), side="right") - 1
    return tuple(labels[i] if 0 <= i < len(labels) else UNDECODED for i in index)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_predicted(track: TrackBeats, space: str, predicted, *,
                    claims: dict | None = None) -> Score:
    """Score a per-beat predicted-class stream into the evaluator's own ``Score``.

    Deliberately parallel to ``score_track`` rather than a call to it: that
    function reads the fixed intent vocabulary out of a module global, and the
    two things this needs to vary -- the claim map and the "no prediction"
    sentinel -- are exactly what it hard-codes.  Everything downstream of the
    accumulation (``match_events``, ``flicker_instants``, ``typed_predictions``,
    ``beat_weights``, ``prf``, and every derived property on ``Score``) is the
    imported original, so the two sides of the table cannot drift apart in the
    arithmetic.  The result aggregates with ``aggregate`` unchanged.

    Two fields on the returned ``Score`` describe the INTENT vocabulary and are
    therefore meaningless here, and are left empty rather than faked:
    ``confusion`` (the intent-keyed skeleton ``aggregate`` needs to sum -- the
    network's confusion is square, predicted class x truth label, and
    ``class_confusion`` builds it) and ``observed_intents``, which drives
    ``expressible_classes`` and ``_structural``.  Those two answer "what can the
    engine's vocabulary not say", and the answer for a model whose vocabulary IS
    the label space is "nothing" -- so the report calls ``_structural`` on the
    rule side only, and compares vocabularies through ``expressible_comparison``
    instead.
    """
    spec = SPACES[space]
    claims = claims or identity_claims(space)
    labels = track.labels[space]
    predicted = tuple(predicted)
    if len(predicted) != len(track.times):
        raise ValueError(
            f"{track.track_id}: {len(predicted)} predictions for "
            f"{len(track.times)} beats")
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

    for prediction, label, weight in zip(predicted, labels, weights):
        if label not in score.counts:
            raise ValueError(f"unknown label {label!r} in space {space!r} "
                             f"(track {track.track_id})")
        score.weight_total_sec += weight
        if prediction == UNDECODED:
            score.no_intent_sec += weight
            score.no_intent_rows += 1
            score.no_intent_by_label[label] += weight
            continue
        if prediction not in claims:
            raise ValueError(f"unknown predicted class {prediction!r} in space "
                             f"{space!r} (track {track.track_id})")
        score.scored_sec += weight
        claimed = claims[prediction]
        if label in claimed:
            score.counts[label][0] += weight                    # tp
        else:
            score.counts[label][2] += weight                    # fn
            share = weight / len(claimed)
            for target in claimed:
                score.counts[target][1] += share                # fp, split if ambiguous

    decoded = [index for index, prediction in enumerate(predicted)
               if prediction != UNDECODED]
    first, last = (decoded[0], decoded[-1]) if decoded else (len(predicted), -1)
    for index, prediction in enumerate(predicted):
        if prediction != UNDECODED:
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

    # Same two streams the rule classifier is measured on, produced by the same
    # differencer.  The intent stream differences the prediction and maps after;
    # the class stream maps to the claim FIRST and differences that.  Under
    # identity claims the two coincide -- which is the honest statement about a
    # model that predicts label classes -- and under rule-equivalent claims they
    # separate exactly where ATMOSPHERIC's ambiguity separates them.
    keys = tuple(UNDECODED if p == UNDECODED else _claim_key(claims, p)
                 for p in predicted)
    key_to_claim = {_claim_key(claims, label): claims[label] for label in claims}
    streams = {
        "intent": [(t, claims[prediction])
                   for t, prediction in intent_changes(track.times, predicted)],
        "class": [(t, key_to_claim[key])
                  for t, key in intent_changes(track.times, keys)],
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


def class_confusion(track: TrackBeats, space: str, predicted) -> dict:
    """``predicted class -> truth label -> seconds``, the square matrix.

    Time-weighted like the rule classifier's confusion so the two read in the
    same unit (seconds of show), but square, because the network's vocabulary
    and the annotator's are the same alphabet.
    """
    labels = track.labels[space]
    weights, _clamped = beat_weights(track.times)
    matrix = {predicted_label: {label: 0.0 for label in SPACES[space].labels}
              for predicted_label in SPACES[space].labels}
    for prediction, label, weight in zip(predicted, labels, weights):
        if prediction != UNDECODED:
            matrix[prediction][label] += weight
    return matrix


def merge_confusion(total: dict, addition: dict) -> dict:
    for predicted_label, row in addition.items():
        for label, seconds in row.items():
            total[predicted_label][label] += seconds
    return total


# --------------------------------------------------------------------------- #
# Cached decode inputs
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class TrackInputs:
    """Everything a sweep config needs about one track, decoded from disk once.

    ``bar_observations`` depends only on ``min_coverage`` and
    ``boundary_tolerance_sec``, so a cache built at one setting is valid for
    every config that shares them -- which is what makes a config cost
    milliseconds instead of a sidecar read.  ``sweep`` groups configs by that
    pair rather than assuming it, so the cache cannot silently go stale.
    """

    track_id: str
    youtube_id: str
    edges: np.ndarray
    posteriors: np.ndarray
    boundary: np.ndarray
    times: tuple
    labels: dict
    intents: tuple

    def as_track_beats(self) -> TrackBeats:
        return TrackBeats(track_id=self.track_id, times=self.times,
                          intents=self.intents, labels=self.labels)


def build_decoder(priors: Priors, params: DecodeParams) -> FixedLagViterbi:
    """One decoder for a whole config: the trellis is built once, not per track."""
    return FixedLagViterbi(
        priors, params.lag_bars,
        class_prior_division=params.class_prior_division,
        drop_miss_cost=params.drop_miss_cost,
        prior_strength=params.prior_strength,
        boundary_weight=params.boundary_weight,
        boundary_ref=params.boundary_ref,
        floor_scale=params.floor_scale)


def decode_bars(inputs: TrackInputs, decoder: FixedLagViterbi) -> tuple:
    """The committed class per bar.  ``decode`` resets, so instances are reusable."""
    return tuple(decision.label
                 for decision in decoder.decode(inputs.posteriors, inputs.boundary))


def decode_beats(inputs: TrackInputs, decoder: FixedLagViterbi) -> tuple:
    """The committed class per beat, ``UNDECODED`` off the bar grid."""
    return beat_classes(inputs.times, inputs.edges, decode_bars(inputs, decoder))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "training" / "data" / "raveform"


def split_ids(data_dir: Path, split: str) -> list:
    """The split's youtube ids, in the frozen file's order.

    Reads ``splits.json`` and nothing else -- in particular it never derives a
    split, so a missing file is an error rather than a quiet reshuffle.
    """
    path = Path(data_dir) / SPLITS_FILE
    if not path.exists():
        raise RuntimeError(f"no splits at {path} -- Task 1 writes it and it is "
                           f"never regenerated implicitly")
    document = json.loads(path.read_text(encoding="utf-8"))
    if split not in document:
        raise RuntimeError(f"{path} has no {split!r} split (has {sorted(document)})")
    return list(document[split])


def read_ids_file(path) -> list:
    """An explicit youtube-id list: a JSON array, or one id per line.

    Exists so a run over a hand-picked subset -- comparing two generations on
    exactly the tracks one of them was already judged on, say -- is reachable
    from the CLI instead of from a script that lives in someone's scratchpad and
    cannot be re-run by a reader of the artifact.  Blank lines and ``#``
    comments are ignored so a list can say why it exists.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"{path} is empty")
    if text[0] in "[{":
        document = json.loads(text)
        ids = document if isinstance(document, list) else (
            document.get("ids") or document.get("youtube_ids"))
        if not isinstance(ids, list):
            raise RuntimeError(
                f"{path} is JSON but carries no id list (expected an array, or "
                f"an object with 'ids' or 'youtube_ids')")
    else:
        ids = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    ids = [str(i) for i in ids if str(i).strip()]
    if not ids:
        raise RuntimeError(f"{path} lists no ids")
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise RuntimeError(f"{path} repeats {len(duplicates)} id(s): {duplicates[:5]} "
                           f"-- a track scored twice would be weighted twice")
    return ids


def sidecar_model_sha(path) -> str | None:
    """The graph sha a posterior sidecar records, or ``None`` if it records none.

    Read on its own rather than out of ``bar_observations`` because the identity
    check has to happen BEFORE the expensive aggregation, and because a sidecar
    predating the stamp must read as "unknown" rather than raise -- the caller
    decides whether unknown is fatal.
    """
    try:
        with np.load(path) as archive:
            return str(archive["model_sha"])
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def load_inputs(data_dir, ids, *, min_coverage: int = DecodeParams().min_coverage,
                boundary_tolerance_sec: float = DecodeParams().boundary_tolerance_sec,
                table_path: Path | None = None,
                posteriors_dir: Path | None = None,
                model_sha: str | None = None,
                allow_missing: bool = False) -> tuple:
    """``(inputs, skipped)`` for the given youtube ids, in id order.

    A track missing any of its three inputs -- table rows, posterior sidecar,
    beat grid -- **raises**.  It would otherwise drop out of *both* columns at
    once, which keeps them comparable to each other while quietly making them
    incomparable to the split they claim to cover: the header would still say
    123 tracks.  A verdict measured over a silently different track set than it
    names is worse than no verdict, so the default is fail-loud and
    ``allow_missing`` is the deliberate, recorded override.

    ``model_sha`` closes the reader half of the identity check.  The writer
    stamps each sidecar with the graph that produced it (``infer.sidecar_is_current``
    refuses to reuse another model's answers), but the generation and the sidecar
    directory are independent arguments here, so a caller can name one
    generation's priors and decoder while pointing at another's posteriors.
    That combination loads cleanly, decodes, and writes a provenance block naming
    a chain that never ran.  Given the expected sha, every sidecar must carry it
    or the whole run is refused -- a mismatch is never a per-track skip, because
    a partial answer to "which model is this" is not an answer.
    """
    data_dir = Path(data_dir)
    table_path = Path(table_path) if table_path else data_dir / TABLE_FILE
    # ``load_tracks`` keys by ``track_id`` ("NNNN.<youtube_id>"), which is also
    # the beat CSV stem; youtube ids are [A-Za-z0-9_-]{11} so the split is exact.
    by_youtube_id = {t.track_id.split(".", 1)[-1]: t for t in load_tracks(table_path)}
    beats_dir = annotations_dir(data_dir) / BEATS_DIR
    posteriors_dir = Path(posteriors_dir) if posteriors_dir else data_dir / POSTERIORS_DIR

    inputs: list = []
    skipped: list = []
    wrong_model: list = []
    for youtube_id in ids:
        track = by_youtube_id.get(youtube_id)
        sidecar = posteriors_dir / f"{youtube_id}.npz"
        if track is None:
            skipped.append({"youtube_id": youtube_id, "reason": "no table rows"})
            continue
        beat_csv = beats_dir / f"{track.track_id}.beat.csv"
        if not sidecar.exists():
            skipped.append({"youtube_id": youtube_id, "reason": "no posterior sidecar"})
            continue
        if model_sha is not None:
            found = sidecar_model_sha(sidecar)
            if found != model_sha:
                wrong_model.append((youtube_id, found))
                continue
        if not beat_csv.exists():
            skipped.append({"youtube_id": youtube_id, "reason": "no beat grid"})
            continue
        try:
            edges = bar_grid(beat_csv)
        except RuntimeError as error:
            skipped.append({"youtube_id": youtube_id, "reason": str(error)})
            continue
        posteriors, boundary = bar_observations(
            sidecar, edges, min_coverage=min_coverage,
            boundary_tolerance_sec=boundary_tolerance_sec)
        inputs.append(TrackInputs(
            track_id=track.track_id, youtube_id=youtube_id, edges=edges,
            posteriors=posteriors, boundary=boundary, times=track.times,
            labels=track.labels, intents=track.intents))
    if wrong_model:
        shown = ", ".join(f"{youtube_id} ({(found or 'unstamped')[:12]})"
                          for youtube_id, found in wrong_model[:10])
        raise RuntimeError(
            f"{len(wrong_model)} of {len(list(ids))} sidecars in {posteriors_dir} "
            f"were written by a different model than the one named: expected "
            f"{model_sha[:12]}, found {shown}"
            f"{' ...' if len(wrong_model) > 10 else ''} -- regenerate the sidecars "
            f"for this model, or point --posteriors-dir at the ones it wrote"
        )
    if skipped and not allow_missing:
        detail = ", ".join(f"{item['youtube_id']} ({item['reason']})"
                           for item in skipped[:10])
        raise RuntimeError(
            f"{len(skipped)} of {len(list(ids))} requested tracks are missing "
            f"inputs: {detail}{' ...' if len(skipped) > 10 else ''} -- rebuild "
            f"the sidecars, or pass allow_missing=True to score a deliberately "
            f"partial split (the artifact records what was dropped)")
    return inputs, skipped


# --------------------------------------------------------------------------- #
# Running a config
# --------------------------------------------------------------------------- #


def evaluate_config(inputs, priors: Priors, params: DecodeParams, *,
                    space: str = DEFAULT_SPACE, claims: dict | None = None,
                    with_confusion: bool = False) -> dict:
    """Decode every track under one config and aggregate the scores.

    The returned dict carries the two numbers the selection rule reads
    (``macro_f1``, ``flicker_per_min``) at the top level so a sweep never has to
    reach into a ``Score``, plus the ``Score`` itself for the report.
    """
    decoder = build_decoder(priors, params)
    scores: list = []
    matrix = ({predicted: {label: 0.0 for label in SPACES[space].labels}
               for predicted in SPACES[space].labels} if with_confusion else None)
    for item in inputs:
        predicted = decode_beats(item, decoder)
        track = item.as_track_beats()
        scores.append(score_predicted(track, space, predicted, claims=claims))
        if matrix is not None:
            merge_confusion(matrix, class_confusion(track, space, predicted))
    total = aggregate(scores)
    return {
        "params": params,
        "macro_f1": total.macro_f1,
        "flicker_per_min": total.flicker_per_minute["class"][PRIMARY_TOLERANCE_SEC],
        "score": total,
        "per_track": scores,
        "confusion": matrix,
    }


def rule_baseline(inputs, space: str = DEFAULT_SPACE) -> dict:
    """The shipping classifier scored on the SAME tracks, by its own rules.

    ``intent_at_beat`` in the training table is the committed intent the live
    engine produced for that beat, so this is the deployed system's timeline --
    not a re-simulation of it -- and ``score_track`` is the function that
    produced the committed corpus baseline.
    """
    scores = [score_track(item.as_track_beats(), space) for item in inputs]
    total = aggregate(scores)
    return {
        "macro_f1": total.macro_f1,
        "flicker_per_min": total.flicker_per_minute["class"][PRIMARY_TOLERANCE_SEC],
        "score": total,
        "per_track": scores,
    }


# --------------------------------------------------------------------------- #
# The side-by-side table
# --------------------------------------------------------------------------- #


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _side(score: Score, stream: str) -> dict:
    """One column of the table: everything the plan asks to compare."""
    boundary = {}
    for tolerance in TOLERANCES_SEC:
        precision, recall, f1 = score.boundary_prf(stream, tolerance)
        drop_p, drop_r, drop_f1 = score.boundary_prf(stream, tolerance, "type", "drop")
        boundary[f"{tolerance}"] = {
            "precision": _round(precision), "recall": _round(recall), "f1": _round(f1),
            "to_drop": {"precision": _round(drop_p), "recall": _round(drop_r),
                        "f1": _round(drop_f1),
                        **score.boundary[stream][tolerance]["by_type"]["drop"]},
            **score.boundary[stream][tolerance]["overall"],
        }
    drop_precision, drop_recall, drop_f1 = prf(*score.counts["drop"])
    return {
        "tracks": score.tracks,
        "exposure_sec": _round(score.exposure_sec, 3),
        "scored_sec": _round(score.scored_sec, 3),
        "undecoded_sec": _round(score.no_intent_sec, 3),
        "undecoded_share": _round(score.no_intent_sec / score.exposure_sec
                                  if score.exposure_sec else 0.0),
        "undecoded_leading_beats": score.no_intent_leading,
        "undecoded_interior_beats": score.no_intent_interior,
        "undecoded_trailing_beats": score.no_intent_trailing,
        "accuracy": _round(score.accuracy),
        "macro_f1": _round(score.macro_f1),
        "macro_classes": list(score.macro_classes),
        "per_class_f1": {label: _round(score.f1(label)) for label in score.labels},
        "per_class": {
            label: {"precision": _round(prf(*score.counts[label])[0]),
                    "recall": _round(prf(*score.counts[label])[1]),
                    "f1": _round(prf(*score.counts[label])[2]),
                    "support_sec": _round(score.counts[label][0]
                                          + score.counts[label][2], 3)}
            for label in score.labels
        },
        "drop": {"precision": _round(drop_precision), "recall": _round(drop_recall),
                 "f1": _round(drop_f1)},
        "boundary": boundary,
        "changes": score.boundary[stream][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"],
        "flicker_per_audience_minute": {
            f"{tolerance}": _round(rate, 4)
            for tolerance, rate in score.flicker_per_minute[stream].items()
        },
    }


def side_by_side(nn_score: Score, rule_score: Score, *, stream: str = "class") -> dict:
    """NN column, rule column, and the deltas -- one dict, no prose."""
    nn = _side(nn_score, stream)
    rule = _side(rule_score, stream)
    return {
        "stream": stream,
        "nn": nn,
        "rule": rule,
        "delta": {
            "macro_f1": _round(nn["macro_f1"] - rule["macro_f1"]),
            "accuracy": _round(nn["accuracy"] - rule["accuracy"]),
            "drop_recall": _round(nn["drop"]["recall"] - rule["drop"]["recall"]),
            "drop_precision": _round(nn["drop"]["precision"] - rule["drop"]["precision"]),
            "boundary_f1": {
                f"{tolerance}": _round(nn["boundary"][f"{tolerance}"]["f1"]
                                       - rule["boundary"][f"{tolerance}"]["f1"])
                for tolerance in TOLERANCES_SEC
            },
            "flicker_per_audience_minute": {
                f"{tolerance}": _round(
                    nn["flicker_per_audience_minute"][f"{tolerance}"]
                    - rule["flicker_per_audience_minute"][f"{tolerance}"], 4)
                for tolerance in TOLERANCES_SEC
            },
        },
    }


def restricted_macro_f1(score: Score, classes) -> float:
    """Macro-F1 over a GIVEN class set, so two vocabularies can be compared fairly."""
    classes = tuple(classes)
    if not classes:
        return 0.0
    return sum(score.f1(label) for label in classes) / len(classes)


def expressible_comparison(nn_score: Score, rule_score: Score) -> dict:
    """The comparison restricted to the classes both sides can actually contest.

    **This is the honest comparable core, not a footnote.**  The rule classifier
    scores exactly zero on ``intro`` and ``outro``, and the tempting reading --
    "its vocabulary cannot express them" -- misattributes a property of the
    HARNESS to the classifier.  ATMOSPHERIC, the intent that covers both, fires
    from a beat-ABSENCE timer; the training table carries one row per DETECTED
    BEAT.  So the only rows that exist are rows where a beat was just found,
    which is precisely where the beat-absence trigger cannot be active.  Zero
    atmospheric rows in 652,766 is what that construction guarantees, not a
    measurement of the engine's vocabulary.

    The consequence is quantitative and large: ``intro`` and ``outro`` carry
    ~0.26 of the network's ~0.60 macro-F1 and ~64 % of the headline delta, and
    the baseline is structurally unable to score on either. So the full
    five-class macro-F1 is still reported -- a beat-indexed evaluation is a fair
    description of what the *deployed system* does at beats, and the network's
    intro/outro capability is real added value -- but it must never be quoted as
    though the two sides contested those classes. This restricted number, over
    the classes the baseline can actually reach, is the model-vs-model
    comparison, and it is the one a per-track claim has to be read from.
    """
    classes = list(rule_score.expressible_classes)
    return {
        "classes": classes,
        "unreachable_for_rule": [label for label in rule_score.labels
                                 if label not in classes],
        "nn_macro_f1": _round(restricted_macro_f1(nn_score, classes)),
        "rule_macro_f1": _round(restricted_macro_f1(rule_score, classes)),
        "delta": _round(restricted_macro_f1(nn_score, classes)
                        - restricted_macro_f1(rule_score, classes)),
    }


def streams_identical(score: Score) -> bool:
    """True when the intent and class streams are the same stream.

    Under identity claims they always are, because every prediction claims a
    distinct singleton -- so differencing before or after the map is the same
    operation.  Asserted rather than assumed so the report can state it.
    """
    return all(score.boundary["intent"][tolerance] == score.boundary["class"][tolerance]
               and score.flicker["intent"][tolerance] == score.flicker["class"][tolerance]
               for tolerance in TOLERANCES_SEC)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _per_track_deltas(nn_scores, rule_scores, restricted_classes) -> list:
    """Per-track NN-minus-rule macro-F1, in BOTH readings, ascending by the full one.

    The restricted delta is the one that can actually go negative.  The full
    reading gives the rule classifier a zero on every class the beat-indexed
    table structurally prevents it from claiming, so "the NN wins on every
    track" in that reading is close to a tautology and must not be quoted as
    evidence that the win is universal.  Carrying both per track is what lets
    the report say which claim is which.
    """
    by_id = {score.track_id: score for score in rule_scores}
    rows = []
    for score in nn_scores:
        other = by_id.get(score.track_id)
        if other is None:
            continue
        nn_restricted = restricted_macro_f1(score, restricted_classes)
        rule_restricted = restricted_macro_f1(other, restricted_classes)
        rows.append({
            "track_id": score.track_id,
            "nn_macro_f1": _round(score.macro_f1),
            "rule_macro_f1": _round(other.macro_f1),
            "delta": _round(score.macro_f1 - other.macro_f1),
            "nn_restricted_macro_f1": _round(nn_restricted),
            "rule_restricted_macro_f1": _round(rule_restricted),
            "restricted_delta": _round(nn_restricted - rule_restricted),
            "exposure_sec": _round(score.exposure_sec, 3),
        })
    rows.sort(key=lambda row: (row["delta"], row["track_id"]))
    return rows


def _head_to_head(rows, key: str = "delta") -> dict:
    """How the win is distributed across tracks, not just its average.

    A corpus mean can be carried by a minority of tracks while the rest regress,
    and a lighting rig is experienced one track at a time -- so "on how many
    tracks is this actually better" is a different and more useful question than
    "is the mean higher".
    """
    deltas = [row[key] for row in rows]
    return {
        "tracks": len(rows),
        "nn_better": sum(1 for value in deltas if value > 0),
        "rule_better": sum(1 for value in deltas if value < 0),
        "tied": sum(1 for value in deltas if value == 0),
        "min_delta": _round(min(deltas)) if deltas else 0.0,
        "median_delta": _round(sorted(deltas)[len(deltas) // 2]) if deltas else 0.0,
        "max_delta": _round(max(deltas)) if deltas else 0.0,
    }


def build_report(inputs, priors: Priors, params: DecodeParams, *,
                 space: str = DEFAULT_SPACE, split: str = "val",
                 skipped: list | None = None, provenance: dict | None = None) -> dict:
    """The whole val verdict: both claim maps, both columns, the deltas."""
    strict = evaluate_config(inputs, priors, params, space=space,
                             claims=identity_claims(space), with_confusion=True)
    lenient = evaluate_config(inputs, priors, params, space=space,
                              claims=rule_equivalent_claims(space))
    rule = rule_baseline(inputs, space)
    restricted_classes = list(rule["score"].expressible_classes)
    deltas = _per_track_deltas(strict["per_track"], rule["per_track"],
                               restricted_classes)
    by_restricted = sorted(deltas, key=lambda row: (row["restricted_delta"],
                                                    row["track_id"]))
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split": split,
        "space": space,
        "tracks": len(inputs),
        "skipped": list(skipped or []),
        "decoder_config": dataclasses.asdict(params),
        "provenance": dict(provenance or {}),
        "claims": {
            "primary": "identity -- the network names the class, so intro over "
                       "outro is a miss",
            "secondary": "rule-equivalent -- intro/outro collapsed into one "
                         "ambiguous class, the exact handicap ATMOSPHERIC carries",
        },
        "nn_streams_identical": streams_identical(strict["score"]),
        "expressible_comparison": expressible_comparison(strict["score"],
                                                         rule["score"]),
        "rule_structural": _structural(rule["score"]),
        "primary": side_by_side(strict["score"], rule["score"], stream="class"),
        "rule_equivalent_claims": side_by_side(lenient["score"], rule["score"],
                                               stream="class"),
        "rule_intent_stream": _side(rule["score"], "intent"),
        "confusion_sec": {
            predicted: {label: _round(seconds, 3)
                        for label, seconds in row.items() if seconds > 0}
            for predicted, row in strict["confusion"].items()
            if any(seconds > 0 for seconds in row.values())
        },
        "head_to_head": {
            "full": _head_to_head(deltas, "delta"),
            "restricted": _head_to_head(deltas, "restricted_delta"),
            "restricted_classes": restricted_classes,
            "note": "the full reading hands the rule classifier a structural "
                    "zero on the classes the beat-indexed table prevents it "
                    "from claiming, so only the restricted reading can go "
                    "negative -- quote that one for 'is this better per track'",
        },
        "worst_tracks": deltas[:10],
        "worst_tracks_restricted": by_restricted[:10],
    }


def render(report: dict) -> str:
    """The table, as text, for a terminal and a PR description."""
    primary = report["primary"]
    nn, rule = primary["nn"], primary["rule"]
    lines = [
        f"NN v1 decoded vs rule classifier -- {report['split']} split, "
        f"{report['tracks']} tracks, space {report['space']}, class stream",
        "=" * 78,
        f"{'metric':<34}{'NN':>13}{'rule':>13}{'delta':>13}",
        "-" * 78,
    ]

    def row(name, left, right, digits=4):
        lines.append(f"{name:<34}{left:>13.{digits}f}{right:>13.{digits}f}"
                     f"{left - right:>+13.{digits}f}")

    row("macro-F1 (v1)", nn["macro_f1"], rule["macro_f1"])
    row("accuracy (time-weighted)", nn["accuracy"], rule["accuracy"])
    lines.append("-" * 78)
    for label in nn["per_class_f1"]:
        row(f"  F1 {label}", nn["per_class_f1"][label], rule["per_class_f1"][label])
    lines.append("-" * 78)
    for tolerance in TOLERANCES_SEC:
        key = f"{tolerance}"
        row(f"boundary-F1 +/-{tolerance}s",
            nn["boundary"][key]["f1"], rule["boundary"][key]["f1"])
    for tolerance in TOLERANCES_SEC:
        key = f"{tolerance}"
        row(f"  -> drop boundary-F1 +/-{tolerance}s",
            nn["boundary"][key]["to_drop"]["f1"], rule["boundary"][key]["to_drop"]["f1"])
    lines.append("-" * 78)
    row("drop recall", nn["drop"]["recall"], rule["drop"]["recall"])
    row("drop precision", nn["drop"]["precision"], rule["drop"]["precision"])
    lines.append("-" * 78)
    for tolerance in TOLERANCES_SEC:
        key = f"{tolerance}"
        row(f"flicker/min (+/-{tolerance}s)",
            nn["flicker_per_audience_minute"][key],
            rule["flicker_per_audience_minute"][key], digits=3)
    lines.append("-" * 78)
    row("undecoded share of show", nn["undecoded_share"], rule["undecoded_share"])
    lines.append(f"changes committed: NN {nn['changes']}, rule {rule['changes']}")
    for name, head in (("all 5 classes", report["head_to_head"]["full"]),
                       ("restricted", report["head_to_head"]["restricted"])):
        lines.append(
            f"per-track macro-F1 ({name}): NN better on "
            f"{head['nn_better']}/{head['tracks']}, rule better on "
            f"{head['rule_better']}, tied {head['tied']} (delta min "
            f"{head['min_delta']:+.4f}, median {head['median_delta']:+.4f}, "
            f"max {head['max_delta']:+.4f})")
    lines.append("")

    restricted = report["expressible_comparison"]
    structural = report["rule_structural"]
    lines.append("Caveats, as numbers rather than footnotes")
    lines.append("-" * 78)
    lines.append(
        f"{restricted['unreachable_for_rule']} are NOT CONTESTED: ATMOSPHERIC "
        f"fires on beat ABSENCE and this table has one row per detected BEAT, so "
        f"the baseline's zero there is a property of the harness, not of its "
        f"vocabulary. Those classes carry "
        f"{sum(nn['per_class_f1'][c] for c in restricted['unreachable_for_rule']) / len(nn['per_class_f1']):.4f} "
        f"of the NN's {nn['macro_f1']:.4f}.")
    lines.append(
        f"  rule macro-F1 best achievable given that: "
        f"{structural['macro_f1_best_achievable']:.4f} "
        f"(hard upper bound {structural['macro_f1_upper_bound']:.4f})")
    lines.append(
        f"  CONTESTED CORE {restricted['classes']}: "
        f"NN {restricted['nn_macro_f1']:.4f} vs rule "
        f"{restricted['rule_macro_f1']:.4f} "
        f"({restricted['delta']:+.4f}) <- the model-vs-model number")
    lenient = report["rule_equivalent_claims"]["nn"]
    lines.append(
        f"  NN handed the rule's own intro/outro ambiguity: macro-F1 "
        f"{lenient['macro_f1']:.4f}, flicker/min "
        f"{lenient['flicker_per_audience_minute']['2.0']:.3f} "
        f"(primary scoring is stricter: {nn['macro_f1']:.4f})")
    lines.append(
        f"  NN intent stream == class stream: {report['nn_streams_identical']} "
        f"(a label-space model cannot express DROP -> PEAK; the rule's intent-stream "
        f"flicker is {report['rule_intent_stream']['flicker_per_audience_minute']['2.0']:.3f}"
        f"/min)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def load_decoder_config(path: Path) -> DecodeParams:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    chosen = document.get("chosen", document)
    fields = {field.name for field in dataclasses.fields(DecodeParams)}
    return DecodeParams(**{k: v for k, v in chosen.items() if k in fields})


def write_json(path: Path, payload: dict) -> None:
    """The eval pipeline's atomic writer, plus the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)


def artifact_provenance(data_dir: Path, model_version: str = MODEL_VERSION,
                        posteriors_dir: Path | None = None) -> dict:
    """Which model, priors, splits and table produced these numbers.

    A verdict table without this is unfalsifiable: the whole chain is content
    addressed elsewhere, and the eval artifact is the one place a reader can
    check that the numbers came from the exported model rather than a
    checkpoint someone re-trained in between.
    """
    data_dir = Path(data_dir)
    model_dir = data_dir / MODELS_DIR / model_version
    wanted = {
        "model_onnx": model_dir / MODEL_FILE,
        "priors": model_dir / PRIORS_FILE,
        "splits": data_dir / SPLITS_FILE,
        "training_table": data_dir / TABLE_FILE,
    }
    files = {}
    for name, path in wanted.items():
        files[name] = ({"path": str(path), "sha256": file_sha256(path),
                        "bytes": path.stat().st_size} if path.exists()
                       else {"path": str(path), "sha256": None})
    return {"git_sha": git_sha(), "model_version": model_version,
            "posteriors_dir": str(Path(posteriors_dir) if posteriors_dir
                                  else data_dir / POSTERIORS_DIR),
            "files": files}


def default_output_name(split: str, ids_file=None) -> str:
    """The filename a run writes to when ``--out`` was not given.

    ``--split`` does not select anything once ``--ids-file`` is present -- it
    only labels the run -- so both would otherwise land on ``eval_<split>.json``,
    the published verdict for that whole split.  A subset scored over a handful
    of hand-picked tracks would silently replace the artifact the generation is
    judged by, in the model directory, with no flag involved.  The ids file's own
    name distinguishes them, and distinguishes two subsets from each other.
    """
    if ids_file is None:
        return EVAL_FILE.format(split=split)
    return EVAL_FILE.format(split=f"{split}_{Path(ids_file).stem}")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split", default="val",
                        help="split to score; val by default, and the test split "
                             "is read once. With --ids-file this names the run "
                             "rather than selecting it (default: %(default)s)")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="score exactly the youtube ids in this file (JSON "
                             "array, or one per line) instead of a whole split. "
                             "--split then only labels the run, and the default "
                             "output is named after this file so a subset can "
                             "never overwrite the split's published verdict")
    parser.add_argument("--config", type=Path, default=None,
                        help="decoder_config.json from the sweep (default: the "
                             "named generation's, if present)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model-version", default=MODEL_VERSION,
                        help="artifact generation to score: reads priors, the "
                             "decoder config and the graph from "
                             f"<data-dir>/{MODELS_DIR}/<model-version>/ and writes "
                             "the verdict beside them (default: %(default)s)")
    parser.add_argument("--posteriors-dir", type=Path, default=None,
                        help=f"sidecar directory (default: <data-dir>/{POSTERIORS_DIR}); "
                             "a retrain writes its own so the sidecars backing a "
                             "published verdict are never overwritten")
    args = parser.parse_args(argv)

    model_dir = args.data_dir / MODELS_DIR / args.model_version
    config_path = args.config or model_dir / DECODER_CONFIG_FILE
    params = (load_decoder_config(config_path) if Path(config_path).exists()
              else DecodeParams())
    priors = Priors.load(model_dir / PRIORS_FILE)

    graph = model_dir / MODEL_FILE
    if not graph.exists():
        raise RuntimeError(
            f"no exported graph at {graph} -- without it the sidecars cannot be "
            f"checked against the generation this run claims to score")
    model_sha = file_sha256(graph)

    ids = (read_ids_file(args.ids_file) if args.ids_file
           else split_ids(args.data_dir, args.split))
    inputs, skipped = load_inputs(
        args.data_dir, ids, min_coverage=params.min_coverage,
        boundary_tolerance_sec=params.boundary_tolerance_sec,
        posteriors_dir=args.posteriors_dir, model_sha=model_sha)
    if not inputs:
        print(f"no usable tracks in split {args.split!r}", file=sys.stderr)
        return 1

    report = build_report(
        inputs, priors, params, split=args.split, skipped=skipped,
        provenance={"config_source": str(config_path)
                                     if Path(config_path).exists() else "defaults",
                    "requested_tracks": len(ids),
                    # An explicit list is recorded in full: the artifact has to
                    # say which tracks it covers, or a reader holding only the
                    # report cannot reproduce the run it describes.
                    **({"ids_file": str(args.ids_file), "ids": list(ids)}
                       if args.ids_file else {}),
                    "artifacts": artifact_provenance(
                        args.data_dir, args.model_version, args.posteriors_dir)})
    out = args.out or model_dir / default_output_name(args.split, args.ids_file)
    write_json(out, report)
    print(render(report))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
