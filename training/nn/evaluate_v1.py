#!/usr/bin/env python
"""Score the decoded NN chain against the Raveform labels, beside the rule classifier."""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from .decoder import (DecodeParams, FixedLagViterbi, bar_grid, bar_observations,
                      load_decoder_config)
from .priors import MODEL_VERSION, MODELS_DIR, PRIORS_FILE, Priors

from build_training_table import NO_INTENT, TABLE_FILE  # noqa: E402
# Metrics imported, never reimplemented: only the claim map and sentinel differ.
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

UNDECODED = NO_INTENT

SPLITS_FILE = "splits.json"
POSTERIORS_DIR = "posteriors"
# export_onnx builds torch modules at import time; this path is torch-free.
MODEL_FILE = "model.onnx"
EVAL_FILE = "eval_{split}.json"
DECODER_CONFIG_FILE = "decoder_config.json"

DEFAULT_SPACE = "v1"

_TP, _FP, _FN = 0, 1, 2


def identity_claims(space: str = DEFAULT_SPACE) -> dict:
    return {label: (label,) for label in SPACES[space].labels}


def rule_equivalent_claims(space: str = DEFAULT_SPACE) -> dict:
    ambiguous = INTENT_TO_LABELS["atmospheric"][space]
    return {label: (ambiguous if label in ambiguous else (label,))
            for label in SPACES[space].labels}


def _claim_key(claims: dict, label: str) -> str:
    return "\x00".join(claims[label])


def beat_classes(beat_times, edges, bar_labels) -> tuple:
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


def score_predicted(track: TrackBeats, space: str, predicted, *,
                    claims: dict | None = None) -> Score:
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
            score.counts[label][_TP] += weight
        else:
            score.counts[label][_FN] += weight
            share = weight / len(claimed)
            for target in claimed:
                score.counts[target][_FP] += share

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


@dataclasses.dataclass(frozen=True)
class TrackInputs:
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
    return FixedLagViterbi(
        priors, params.lag_bars,
        class_prior_division=params.class_prior_division,
        drop_miss_cost=params.drop_miss_cost,
        prior_strength=params.prior_strength,
        boundary_weight=params.boundary_weight,
        boundary_ref=params.boundary_ref,
        floor_scale=params.floor_scale,
        floor_bars=params.floor_bars,
        outro_escape=params.outro_escape)


def decode_bars(inputs: TrackInputs, decoder: FixedLagViterbi) -> tuple:
    return tuple(decision.label
                 for decision in decoder.decode(inputs.posteriors, inputs.boundary))


def decode_beats(inputs: TrackInputs, decoder: FixedLagViterbi) -> tuple:
    return beat_classes(inputs.times, inputs.edges, decode_bars(inputs, decoder))


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "training" / "data" / "raveform"


def split_ids(data_dir: Path, split: str) -> list:
    path = Path(data_dir) / SPLITS_FILE
    if not path.exists():
        raise RuntimeError(f"no splits at {path} -- Task 1 writes it and it is "
                           f"never regenerated implicitly")
    document = json.loads(path.read_text(encoding="utf-8"))
    if split not in document:
        raise RuntimeError(f"{path} has no {split!r} split (has {sorted(document)})")
    return list(document[split])


def read_ids_file(path) -> list:
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
    try:
        with np.load(path) as archive:
            return str(archive["model_sha"])
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def _youtube_id_of(track_id: str) -> str:
    return track_id.split(".", 1)[-1]


def load_inputs(data_dir, ids, *, min_coverage: int = DecodeParams().min_coverage,
                boundary_tolerance_sec: float = DecodeParams().boundary_tolerance_sec,
                temperature: float = DecodeParams().temperature,
                table_path: Path | None = None,
                posteriors_dir: Path | None = None,
                model_sha: str | None = None,
                allow_missing: bool = False) -> tuple:
    data_dir = Path(data_dir)
    table_path = Path(table_path) if table_path else data_dir / TABLE_FILE
    by_youtube_id = {_youtube_id_of(t.track_id): t for t in load_tracks(table_path)}
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
            boundary_tolerance_sec=boundary_tolerance_sec,
            temperature=temperature)
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


def evaluate_config(inputs, priors: Priors, params: DecodeParams, *,
                    space: str = DEFAULT_SPACE, claims: dict | None = None,
                    with_confusion: bool = False) -> dict:
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
    scores = [score_track(item.as_track_beats(), space) for item in inputs]
    total = aggregate(scores)
    return {
        "macro_f1": total.macro_f1,
        "flicker_per_min": total.flicker_per_minute["class"][PRIMARY_TOLERANCE_SEC],
        "score": total,
        "per_track": scores,
    }


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _side(score: Score, stream: str) -> dict:
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
    classes = tuple(classes)
    if not classes:
        return 0.0
    return sum(score.f1(label) for label in classes) / len(classes)


def expressible_comparison(nn_score: Score, rule_score: Score) -> dict:
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
    return all(score.boundary["intent"][tolerance] == score.boundary["class"][tolerance]
               and score.flicker["intent"][tolerance] == score.flicker["class"][tolerance]
               for tolerance in TOLERANCES_SEC)


def _per_track_deltas(nn_scores, rule_scores, restricted_classes) -> list:
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


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)


def artifact_provenance(data_dir: Path, model_version: str = MODEL_VERSION,
                        posteriors_dir: Path | None = None) -> dict:
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
        temperature=params.temperature,
        posteriors_dir=args.posteriors_dir, model_sha=model_sha)
    if not inputs:
        print(f"no usable tracks in split {args.split!r}", file=sys.stderr)
        return 1

    report = build_report(
        inputs, priors, params, split=args.split, skipped=skipped,
        provenance={"config_source": str(config_path)
                                     if Path(config_path).exists() else "defaults",
                    "requested_tracks": len(ids),
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
