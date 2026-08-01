#!/usr/bin/env python
"""Run the frozen eval set through the simulation and score it against labels.

This is the benchmark.  It replaces the single bundled Generate track -- and the
plumbing-only PASS gate that judged it -- with the ten expert-labeled Raveform
tracks frozen in ``training/eval_set.json``, and it asks a musical question
instead of a plumbing one: *does the show still land on the music?*

Each track goes through the SAME code path production uses -- ``FileAudioClient``
into ``run_fast_simulation`` on the virtual clock -- and the resulting report is
judged two ways:

**Behaviour change.**  ``simulate.evaluator.report_checksum`` over the report.
Identical pipeline + identical audio => identical bytes.  Any change to that
hash means the pipeline now behaves differently; the runner says which track and
exits nonzero.  Nothing else in the suite detects a behaviour change this
cheaply, and on ten tracks of real music it is hard to change anything that
matters without moving one.

**Musical quality.**  The report is joined to the annotation with
``build_training_table.join_track`` -- the same join the training table uses, so
the look-ahead realignment (``realign_intents``) happens here exactly once and
exactly the way the corpus does it -- and scored with
``evaluate_against_labels.score_track`` in the ``v1`` space.  Four numbers are
gated: macro-F1, time-weighted accuracy, boundary-F1 of the intent stream at the
primary tolerance, and flicker per audience-minute.  The first three regress
downward; flicker regresses UPWARD, because it counts changes the audience had
no musical reason for.  Beside them, the recorded COUNT facts (rows,
label boundaries, scored seconds) are compared exactly: they cost nothing and
they move for causes a 0.02 score tolerance absorbs.

**The same question.**  Scores are only comparable against the labels they were
cut over.  The ten tracks' labels are COMMITTED (``training/eval_labels.json``,
cut by ``eval_assets``) and carry the sha256 of the ``annotations/segments.json``
they came out of, so the run proves the slice belongs to the frozen set before
scoring anything.  A machine without the slice falls back to the gitignored
corpus file, which a re-fetch can move, and is checked against the sha the eval
set recorded at freeze time.  Either way the run refuses outright on a mismatch.

Both gates fire together on a deliberate improvement (better scores still move
the checksums), and that is the intended workflow: read the table, decide the
change is wanted, re-cut the baseline in the same commit.

Two modes::

    # cut the baseline (COMMITTED: training/eval_set_baseline.json)
    uv run python training/run_eval_set.py --write-baseline

    # compare against it -- the gate; nonzero on drift or regression
    uv run python training/run_eval_set.py

    # a subset, by track_id or youtube_id (what the integration test runs)
    uv run python training/run_eval_set.py --only 0096.PNpXKsge4xM,0834.NyEKXA7_6z0

A subset run compares only its own tracks and does NOT compare the aggregate: an
aggregate over three tracks is not the ten-track number, and pretending
otherwise would either fail every subset run or gate on nothing.  For the same
reason a subset run REFUSES to overwrite the committed baseline: a three-track
baseline at that path passes by construction and the other seven tracks stop
being gated with nothing to say so.  ``--allow-partial-baseline`` is the
deliberate override; an explicit ``--baseline PATH`` is the experiment.

**Decode caches are kept.**  Every other consumer of the corpus deletes the
``<mp3>.<rate>.npy`` a simulation leaves behind (it is ~7.7x the mp3's size), but
the eval set is ten tracks and it is re-run constantly -- by the integration
suite on every ``uv run pytest``.  Keeping the caches costs under a gigabyte and
removes the decode (several seconds a track) from every run after the first.
This is deliberate, and ``build_training_table`` already honours it: its
``keep_cache`` rule leaves behind any cache that existed before its batch.  They
now land in ``training/eval_audio/`` beside the committed mp3s, where the
repository's ``*.npy`` rule keeps them out of git.

Exit codes: 0 clean, 1 the gate failed, 2 an input is missing or the command was
refused.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    str(REPO_ROOT),
    str(REPO_ROOT / "training"),
    str(REPO_ROOT / "training" / "raveform"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from build_training_table import (  # noqa: E402  (needs the path inserts above)
    _git,
    default_workers,
    join_track,
    load_sections_by_track,
    pipeline_sha,
)
from eval_assets import (  # noqa: E402
    EVAL_AUDIO_DIR,
    EVAL_LABELS_FILE,
    committed_audio_path,
    corpus_audio_path,
    labels_source_sha,
    load_labels,
)
from eval_assets import sections_by_track as sections_from_slice  # noqa: E402
from evaluate_against_labels import (  # noqa: E402
    PRIMARY_TOLERANCE_SEC,
    SPACES,
    TrackBeats,
    aggregate,
    default_data_dir,
    file_sha256,
    score_track,
    write_json,
)
from raveform_fetch_annotations import SEGMENTS_FILE, annotations_dir  # noqa: E402
from select_eval_set import EVAL_SET_FILE, load_eval_set, verify_inputs  # noqa: E402

BASELINE_FILE = REPO_ROOT / "training" / "eval_set_baseline.json"

# The scoring configuration the baseline is cut under.  Recorded in the file so
# a baseline can never be read under different assumptions than it was written.
SPACE = "v1"                    # the space the NN trains on -- the primary one
STREAM = "intent"               # every lighting change: the show as the room sees it
BOUNDARY_TOLERANCE_SEC = PRIMARY_TOLERANCE_SEC

# #141(a).  Boundary-F1 at the tightest tolerance the scorer computes: does the
# change land ON the section change, not merely near it.  It is a headline
# number rather than a diagnostic because it is the axis the shipped decoder was
# SELECTED on -- the frontier re-rank that chose `reduced_plus_floors_x0.75`
# ranked on exactly this, and the 2.0 s lens it replaced hid most of the spread
# it found (post-decoder dwell scored 0.68 at 2.0 s and 0.01 here).  A benchmark
# that cannot see the axis the model was chosen on cannot defend that choice.
CRISPNESS_TOLERANCE_SEC = 0.5

# metric -> the direction that is a REGRESSION.
GUARDED_METRICS = {
    "macro_f1": "down",
    "accuracy": "down",
    "boundary_f1": "down",
    "crispness": "down",
    "flicker_per_min": "up",
}

# How far a number may move before it is a regression.  Runs are deterministic,
# so this is not noise headroom -- it is the size of a score change worth
# stopping a commit for.  F1/accuracy live in [0, 1]; flicker is changes per
# audience-minute and sits near 1-3 on this corpus, so it gets its own slack.
DEFAULT_SCORE_TOLERANCE = 0.02
DEFAULT_FLICKER_TOLERANCE = 0.20

# Facts, not scores: how many beats survived the join, how many boundaries the
# annotation carries, and how much show time was scored.  Compared for EXACT
# equality, because a tolerance on a count means nothing and because they fail
# for causes the four gated scores cannot see -- a join that starts dropping
# beats, or an annotation that gained a section, moves these while macro-F1
# stays comfortably inside its 0.02.  Free: they are already in every row.
COUNT_FACTS = ("rows", "label_boundaries", "exposure_sec")

# The one line a machine with neither copy must see.  The ten mp3s are COMMITTED
# (owner-authorised, under derived names -- see eval_assets), so this is no
# longer an ordinary state of a fresh clone; it means the committed dir was
# pruned and the corpus is not there either.  Still loud, never a silent skip: a
# skipped benchmark that nobody notices is the same as no benchmark.
AUDIO_MISSING_HINT = (
    "eval-set audio missing -- expected the committed copy in "
    f"{EVAL_AUDIO_DIR.relative_to(REPO_ROOT).as_posix()}/ (re-cut it with "
    "training/eval_assets.py --cut) or the corpus mp3 from "
    "training/raveform/raveform_download.py"
)

SCHEMA_VERSION = 1

# Where the corpus is, when it is not where the code is.
DATA_DIR_ENV = "RAVEFORM_DATA_DIR"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


# One definition, in a stdlib-only module a show can import without pulling this
# harness in behind it (`lib/section_chain.py` asks the same question).
from corpus_root import corpus_dir  # noqa: E402,F401


def audio_path(data_dir: Path, youtube_id: str) -> Path:
    """The mp3 this run will read: the committed copy first, else the corpus.

    Committed first because that is the copy every machine has, and because the
    two are byte-identical by construction (``eval_assets.copy_audio`` re-hashes
    every copy), so which one is read cannot move a report checksum.  When
    neither exists the CORPUS path is returned: it is the one a human can act
    on, and it is what the missing-input line prints.
    """
    committed = committed_audio_path(youtube_id)
    if committed.exists():
        return committed
    return corpus_audio_path(Path(data_dir), youtube_id)


def labels_source(data_dir: Path, labels: Path | None = None) -> tuple:
    """``(path, committed)`` -- the section labels this run scores against.

    One resolver for the three things that need the answer (is it there, is it
    the right one, read it), because a run that CHECKED the corpus file and
    SCORED against the committed slice would be checking nothing.
    """
    committed = Path(labels or EVAL_LABELS_FILE)
    if committed.exists():
        return committed, True
    return annotations_dir(Path(data_dir)) / SEGMENTS_FILE, False


def load_sections(data_dir: Path, labels: Path | None = None) -> dict:
    """``track_id -> [(start, end, label)]`` from whichever labels resolved."""
    path, committed = labels_source(data_dir, labels)
    if committed:
        return sections_from_slice(load_labels(path))
    return load_sections_by_track(Path(data_dir))


def select_tracks(document: dict, only: list | None = None) -> list:
    """The eval-set track records to run, in the frozen document's order.

    ``only`` accepts either identifier a human has to hand -- the ``track_id``
    that names the corpus row or the ``youtube_id`` that names the file -- and
    refuses anything outside the frozen set, because a benchmark that silently
    runs a track the baseline never saw is not a benchmark.
    """
    tracks = list(document.get("tracks") or [])
    if only is None:
        return tracks
    wanted = {str(item).strip() for item in only if str(item).strip()}
    picked = [track for track in tracks
              if track["track_id"] in wanted or track["youtube_id"] in wanted]
    known = {track["track_id"] for track in tracks} | {
        track["youtube_id"] for track in tracks}
    unknown = sorted(wanted - known)
    if unknown:
        raise RuntimeError(
            f"not in the eval set: {', '.join(unknown)} "
            f"(the frozen set is {', '.join(track['track_id'] for track in tracks)})"
        )
    return picked


def shortest_track_ids(document: dict, count: int) -> list:
    """The ``count`` shortest tracks, returned in the frozen document's order.

    The integration test names its tracks through this function rather than by
    literal id: the budget it is protecting is wall time, so the selection has
    to follow the durations if the set is ever re-frozen.  Ties break on
    ``track_id`` so the answer is a function of the document and not of the
    order anything happened to be written in.  A library entry point, not a
    flag: ``--only`` already spells any subset a human wants from the shell.
    """
    tracks = list(document.get("tracks") or [])
    by_length = sorted(tracks, key=lambda track: (float(track["duration_sec"]),
                                                  track["track_id"]))
    chosen = {track["track_id"] for track in by_length[:max(0, count)]}
    return [track["track_id"] for track in tracks if track["track_id"] in chosen]


def missing_inputs(data_dir: Path, tracks: list, labels: Path | None = None) -> list:
    """Everything the run needs and does not have, one human line each.

    Reported all at once rather than raising on the first: a machine missing
    both should learn it needs the annotations AND three mp3s from one run, not
    three.  On a fresh clone this is empty -- both are committed.
    """
    problems = []
    segments, committed = labels_source(data_dir, labels)
    if not committed and not segments.exists():
        problems.append(
            f"missing {segments} -- run "
            f"training/raveform/raveform_fetch_annotations.py, or restore the "
            f"committed {EVAL_LABELS_FILE.name}")
    for track in tracks:
        mp3 = audio_path(data_dir, track["youtube_id"])
        if not mp3.exists():
            problems.append(f"{AUDIO_MISSING_HINT}: {track['track_id']} ({mp3})")
    return problems


def verify_ground_truth(document: dict, data_dir: Path,
                        labels: Path | None = None) -> None:
    """Refuse to score against labels the eval set was not frozen against.

    Re-fetching the annotations can move a section boundary under a baseline
    that was cut before the move, and every number in this run would then be
    measuring a different question while the gate reported "MATCHES BASELINE".

    Two ways to be sure, one per source.  The COMMITTED slice cannot move
    behind git's back, so what has to be proved of it is provenance: it records
    the sha256 of the ``segments.json`` it was cut from, and that must be the
    sha the eval set froze against.  The CORPUS file is gitignored and nothing
    in the repository pins it, so it is hashed on every run.

    Fatal rather than a warning, and checked here rather than at the freeze:
    the freeze can survive a corpus that has moved on (that is what freezing
    is for), a *score* cannot.  ``clean_manifest.csv`` is deliberately not
    checked -- it chose which tracks are in the set and grows with every
    download batch, and it feeds no number in this file.
    """
    path, committed = labels_source(data_dir, labels)
    if committed:
        frozen = ((document.get("selected_from") or {}).get("inputs")
                  or {}).get(SEGMENTS_FILE)
        cut_from = labels_source_sha(load_labels(path))
        if not frozen or not cut_from or frozen != cut_from:
            raise RuntimeError(
                f"the eval set's GROUND TRUTH does not match the freeze: "
                f"{Path(path).name} was cut from {SEGMENTS_FILE} "
                f"{str(cut_from)[:12]}..., the eval set froze against "
                f"{str(frozen)[:12]}... -- re-cut the slice "
                f"(training/eval_assets.py --cut) and the baseline together."
            )
        return

    drift = verify_inputs(document, Path(data_dir), only=(SEGMENTS_FILE,))
    if drift:
        raise RuntimeError(
            f"the eval set's GROUND TRUTH has moved since the freeze: "
            f"{'; '.join(drift)} -- every score here would be against labels "
            f"the committed baseline never saw.  Restore {SEGMENTS_FILE}, or "
            f"re-freeze the eval set and re-cut the baseline together."
        )


def same_path(left: Path, right: Path) -> bool:
    """Do two paths name the same file, spelled differently or not yet existing?"""
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:                                             # pragma: no cover
        return Path(left).absolute() == Path(right).absolute()


def partial_baseline_refusal(selected: list, document: dict, baseline_path: Path,
                             allowed: bool = False) -> str | None:
    """Why a subset run must not overwrite the COMMITTED baseline, or ``None``.

    The tripwire must not be allowed to rewrite its own reference to whatever it
    happened to run.  A three-track baseline at the committed path is green by
    construction -- the integration test compares the tracks it ran against the
    tracks in the file, so the other seven simply stop being checked, and
    nothing fails to say so.  A warning is not enough for that: it scrolls past.

    Writing a subset to an EXPLICIT other path stays allowed.  That is an
    experiment, and no gate reads it.
    """
    total = len(document.get("tracks") or [])
    if allowed or len(selected) >= total or not same_path(baseline_path, BASELINE_FILE):
        return None
    ran = ", ".join(track["track_id"] for track in selected) or "(nothing)"
    return (
        f"REFUSING to overwrite the committed baseline with a SUBSET\n"
        f"  {Path(baseline_path)}\n"
        f"  this run covered {len(selected)} of {total} eval-set tracks: {ran}\n"
        f"  The other {total - len(selected)} would stop being gated, silently:\n"
        f"  the benchmark compares the tracks it ran against the tracks in this\n"
        f"  file, so a subset baseline passes by construction.\n"
        f"  --allow-partial-baseline  do it anyway (you are shrinking the benchmark)\n"
        f"  --baseline PATH           write the subset where no gate reads it"
    )


def load_baseline(path: Path) -> dict:
    """Read the committed baseline, refusing anything that is not one."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise RuntimeError(
            f"missing {path} -- run this script with --write-baseline to cut it"
        ) from None
    except ValueError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(document, dict) or not isinstance(document.get("tracks"), dict):
        raise RuntimeError(f"{path} has no 'tracks' map -- it is not a baseline")
    return document


# --------------------------------------------------------------------------- #
# One track
# --------------------------------------------------------------------------- #


class TrackRun(NamedTuple):
    """One simulated + scored track."""

    track_id: str
    youtube_id: str
    entry: dict          # what goes in the baseline
    score: object        # evaluate_against_labels.Score, for the aggregate
    wall_sec: float


def simulate_report(mp3_path: str) -> tuple:
    """Fast-sim one file; returns ``(report, song_sec, wall_sec)``.

    Identical to what ``auto_pilot simulate file`` does and to what
    ``build_training_table`` does per corpus track -- the point of the benchmark
    is that there is no benchmark-only code path.  The DSP imports stay local so
    the pure-logic half of this module (and its unit tests) never pay to load
    the analysis stack.
    """
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import run_fast_simulation

    started = time.monotonic()
    client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, str(mp3_path))
    _client, event_buffer, command_queue = asyncio.run(run_fast_simulation(client))
    report = event_buffer.to_report(command_queue.get_timing_log())
    return report, client.duration_sec, time.monotonic() - started


def score_report(track_id: str, youtube_id: str, report: dict, sections: list):
    """``(Score, rows)`` for one report against one track's annotation.

    The join is ``build_training_table``'s, not a copy of it: intent blocks come
    out of the report stamped in AUDIENCE time by the delayed command queue
    (except the immediate commits, which are already song-stamped), and
    ``realign_intents`` is the one piece of code that knows how to tell those
    apart.  Re-deriving that here would give the benchmark a different notion of
    when the lights changed than the training table has.
    """
    rows, stats = join_track(track_id, youtube_id, report, sections)
    track = TrackBeats(
        track_id=track_id,
        times=tuple(row["t_song"] for row in rows),
        intents=tuple(row["intent_at_beat"] for row in rows),
        labels={name: tuple(row[spec.column] for row in rows)
                for name, spec in SPACES.items()},
    )
    return score_track(track, SPACE), len(rows), stats


def track_metrics(score) -> dict:
    """The five gated numbers, read off a ``Score``."""
    return {
        "macro_f1": round(score.macro_f1, 6),
        "accuracy": round(score.accuracy, 6),
        "boundary_f1": round(
            score.boundary_prf(STREAM, BOUNDARY_TOLERANCE_SEC)[2], 6),
        "crispness": round(
            score.boundary_prf(STREAM, CRISPNESS_TOLERANCE_SEC)[2], 6),
        "flicker_per_min": round(
            score.flicker_per_minute[STREAM][BOUNDARY_TOLERANCE_SEC], 6),
    }


def track_entry(report: dict, score, rows: int, youtube_id: str,
                song_sec: float, stats=None) -> dict:
    """One track's row of the baseline: identity, size, speed-free facts, scores.

    Wall time and x-realtime are deliberately NOT in here.  They are printed on
    every run and they are the whole reason the caches are kept, but they are a
    property of the machine, and a committed file that changes with the CPU load
    of the laptop that wrote it is a file nobody can diff.

    ``late`` is recorded and NOT gated.  It is #154's accepted lateness -- on a
    track slow enough that the chain is older than the playback delay, a
    decision commits as soon as it can rather than on time -- so it is a
    property of the music the benchmark should SHOW, not a regression to stop a
    commit for.  It ships with its denominator for the reason the batch summary
    does: only a block that recorded its own instant can be measured at all.
    """
    from simulate.evaluator import report_checksum

    entry = {
        "youtube_id": youtube_id,
        "checksum": report_checksum(report),
        "beats": len(report.get("beats", [])),
        "rows": rows,
        "song_sec": round(float(song_sec), 3),
        "exposure_sec": round(score.exposure_sec, 3),
        "changes_intent":
            score.boundary["intent"][BOUNDARY_TOLERANCE_SEC]["overall"]["n_pred"],
        "changes_class":
            score.boundary["class"][BOUNDARY_TOLERANCE_SEC]["overall"]["n_pred"],
        "label_boundaries":
            score.boundary["intent"][BOUNDARY_TOLERANCE_SEC]["overall"]["n_truth"],
    }
    if stats is not None:
        entry["late"] = int(stats.intent_blocks_late)
        entry["blocks_measurable"] = int(stats.intent_blocks_song_recorded)
    entry.update(track_metrics(score))
    return entry


class Job(NamedTuple):
    """One track to simulate and score.  Picklable: crosses the process pool.

    Carries only its OWN annotation, not the corpus-wide map: the map is every
    annotated track in the corpus and shipping it to each worker would cost more
    than the simulation saves.
    """

    data_dir: str
    track: dict
    sections: list


def run_job(job: Job) -> TrackRun:
    """Simulate and score one eval-set track.  The pool's work unit."""
    track_id, youtube_id = job.track["track_id"], job.track["youtube_id"]
    report, song_sec, wall_sec = simulate_report(
        audio_path(Path(job.data_dir), youtube_id))
    score, rows, stats = score_report(track_id, youtube_id, report, job.sections)
    return TrackRun(track_id, youtube_id,
                    track_entry(report, score, rows, youtube_id, song_sec, stats),
                    score, wall_sec)


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #


def build_document(eval_set: dict, pipeline_sha_: str, entries: dict,
                   aggregate_metrics: dict,
                   score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
                   flicker_tolerance: float = DEFAULT_FLICKER_TOLERANCE) -> dict:
    """The committed baseline, and the shape every run is compared in.

    No timestamp: two runs of the same pipeline over the same corpus must
    produce a byte-identical file, so that re-cutting the baseline with nothing
    changed is a no-op diff rather than a line of noise.
    """
    return {
        "schema": SCHEMA_VERSION,
        "eval_set": eval_set,
        "pipeline_sha": pipeline_sha_,
        "space": SPACE,
        "stream": STREAM,
        "boundary_tolerance_sec": BOUNDARY_TOLERANCE_SEC,
        "crispness_tolerance_sec": CRISPNESS_TOLERANCE_SEC,
        "gate": {"score_tolerance": score_tolerance,
                 "flicker_tolerance": flicker_tolerance,
                 "metrics": dict(GUARDED_METRICS)},
        "aggregate": aggregate_metrics,
        "tracks": entries,
    }


def eval_set_identity(path: Path, document: dict) -> dict:
    """The frozen set this run is against -- checksummed, so a re-freeze shows."""
    path = Path(path)
    try:
        name = str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        name = path.name
    return {
        "path": name,
        "sha256": file_sha256(path),
        "tracks": len(document.get("tracks") or []),
        "youtube_ids": list(document.get("youtube_ids") or []),
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


class Comparison(NamedTuple):
    """Why a run did or did not match the baseline, split by what it means."""

    desync: list        # the baseline is not about this eval set at all
    unbaselined: list   # a track the baseline has never seen
    ungated: list       # a guarded metric one side does not carry
    checksum_drift: list
    regressions: list
    fact_drift: list    # a recorded count moved: the run measured something else
    subset: bool        # fewer tracks than the baseline: aggregate not compared

    @property
    def failed(self) -> bool:
        return bool(self.desync or self.unbaselined or self.ungated
                    or self.checksum_drift or self.regressions or self.fact_drift)


def _regression(name: str, metric: str, before: float, after: float,
                tolerance: float) -> str | None:
    """One metric of one row, or ``None`` if it is within tolerance."""
    delta = after - before
    if GUARDED_METRICS[metric] == "down":
        if delta >= -tolerance:
            return None
        arrow = "fell"
    else:
        if delta <= tolerance:
            return None
        arrow = "rose"
    return (f"{name}: {metric} {arrow} {before:.4f} -> {after:.4f} "
            f"({delta:+.4f}, tolerance {tolerance:.4f})")


def _fact_drift(name: str, before: dict, after: dict) -> list:
    """Count facts of one row that stopped agreeing, exactly.

    Only compared where BOTH sides carry the fact: the aggregate row is four
    scores and nothing else, and inventing a failure for a field that row was
    never supposed to have would make the tripwire noise.
    """
    return [
        f"{name}: {fact} {before[fact]} -> {after[fact]}"
        for fact in COUNT_FACTS
        if fact in before and fact in after and before[fact] != after[fact]
    ]


def _compare_metrics(name: str, before: dict, after: dict, score_tolerance: float,
                     flicker_tolerance: float) -> tuple:
    """``(ungated, regressions, facts)`` for one row of the table.

    A guarded metric that either side does not carry is a FAILURE, not a skip.
    Skipping it is silent un-gating: an old-schema or hand-edited baseline would
    keep passing while the number it was supposed to protect drifted anywhere it
    liked, and nothing on the way out would say so.
    """
    ungated, regressions = [], []
    for metric in GUARDED_METRICS:
        if metric not in before or metric not in after:
            where = "the baseline" if metric not in before else "this run"
            ungated.append(f"{name}: {metric} is missing from {where} -- it is "
                           f"NOT being gated")
            continue
        tolerance = (flicker_tolerance if metric == "flicker_per_min"
                     else score_tolerance)
        line = _regression(name, metric, float(before[metric]),
                           float(after[metric]), tolerance)
        if line:
            regressions.append(line)
    return ungated, regressions, _fact_drift(name, before, after)


def compare(baseline: dict, current: dict,
            score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
            flicker_tolerance: float = DEFAULT_FLICKER_TOLERANCE) -> Comparison:
    """Judge a run against the committed baseline.

    Five independent failures, kept apart because they mean different things: a
    DESYNC says the baseline describes a different benchmark (re-cut it), an
    UNGATED metric says the baseline cannot judge this run at all, a CHECKSUM
    DRIFT says the pipeline's behaviour moved (look at the scores, then accept
    or fix), a REGRESSION says the show got worse (fix it), and a FACT DRIFT
    says a recorded count moved, i.e. the run scored something other than what
    the baseline scored.
    """
    desync, unbaselined, ungated, drift, regressions, facts = [], [], [], [], [], []

    baseline_sha = (baseline.get("eval_set") or {}).get("sha256")
    current_sha = (current.get("eval_set") or {}).get("sha256")
    if baseline_sha != current_sha:
        desync.append(
            f"the baseline was cut against a different eval set "
            f"({str(baseline_sha)[:12]}... on record, {str(current_sha)[:12]}... "
            f"on disk) -- re-cut it with --write-baseline"
        )

    baseline_tracks = baseline.get("tracks") or {}
    current_tracks = current.get("tracks") or {}
    subset = set(current_tracks) < set(baseline_tracks)

    for track_id in current_tracks:
        before = baseline_tracks.get(track_id)
        after = current_tracks[track_id]
        if before is None:
            unbaselined.append(
                f"{track_id}: no baseline entry -- the eval set grew without the "
                f"baseline being re-cut")
            continue
        if before.get("checksum") != after.get("checksum"):
            drift.append(
                f"{track_id}: report checksum {str(before.get('checksum'))[:12]}..."
                f" -> {str(after.get('checksum'))[:12]}... "
                f"(beats {before.get('beats')} -> {after.get('beats')}, "
                f"intent changes {before.get('changes_intent')} -> "
                f"{after.get('changes_intent')})")
        row_ungated, row_regressions, row_facts = _compare_metrics(
            track_id, before, after, score_tolerance, flicker_tolerance)
        ungated += row_ungated
        regressions += row_regressions
        facts += row_facts

    # A subset run's aggregate is an aggregate of the subset; comparing it to
    # the ten-track number would be comparing two different quantities.
    if not subset:
        row_ungated, row_regressions, row_facts = _compare_metrics(
            "(aggregate)", baseline.get("aggregate") or {},
            current.get("aggregate") or {}, score_tolerance, flicker_tolerance)
        ungated += row_ungated
        regressions += row_regressions
        facts += row_facts

    return Comparison(desync, unbaselined, ungated, drift, regressions, facts,
                      subset)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

WIDTH = 116


def render_table(runs: list, aggregate_entry: dict, total_song: float,
                 total_wall: float, workers: int = 1) -> str:
    """The per-track table every run prints, baseline or not."""
    lines = [
        f'  {"track_id":<20}{"song":>7}{"wall":>7}{"x-rt":>7}{"beats":>7}'
        f'{"rows":>7}{"macroF1":>9}{"acc":>7}{"bF1":>7}{"crisp":>7}'
        f'{"flick/m":>9}{"chg i/c":>10}{"late":>8}  checksum',
        "  " + "-" * (WIDTH - 2),
    ]
    for run in runs:
        entry = run.entry
        speed = entry["song_sec"] / run.wall_sec if run.wall_sec > 0 else 0.0
        changes = f'{entry["changes_intent"]}/{entry["changes_class"]}'
        late = f'{entry.get("late", "-")}/{entry.get("blocks_measurable", "-")}'
        lines.append(
            f'  {run.track_id:<20}{entry["song_sec"] / 60.0:>6.1f}m'
            f'{run.wall_sec:>6.1f}s{speed:>6.0f}x{entry["beats"]:>7}'
            f'{entry["rows"]:>7}{entry["macro_f1"]:>9.3f}{entry["accuracy"]:>7.3f}'
            f'{entry["boundary_f1"]:>7.3f}{entry["crispness"]:>7.3f}'
            f'{entry["flicker_per_min"]:>9.2f}{changes:>10}{late:>8}  '
            f'{entry["checksum"][:12]}'
        )
    lines.append("  " + "-" * (WIDTH - 2))
    speed = total_song / total_wall if total_wall > 0 else 0.0
    lines.append(
        f'  {"(aggregate)":<20}{total_song / 60.0:>6.1f}m{total_wall:>6.1f}s'
        f'{speed:>6.0f}x{"":>7}{"":>7}{aggregate_entry["macro_f1"]:>9.3f}'
        f'{aggregate_entry["accuracy"]:>7.3f}{aggregate_entry["boundary_f1"]:>7.3f}'
        f'{aggregate_entry["crispness"]:>7.3f}'
        f'{aggregate_entry["flicker_per_min"]:>9.2f}'
    )
    lines.append(
        f'  macro-F1 and accuracy in the {SPACE} space; boundary-F1 and flicker on '
        f'the {STREAM} stream at +/-{BOUNDARY_TOLERANCE_SEC}s, crispness at '
        f'+/-{CRISPNESS_TOLERANCE_SEC}s'
    )
    lines.append(
        f'  late = intent blocks committed more than the playback delay after '
        f'the audio they name, of those that can be measured (#154, accepted)'
    )
    if workers > 1 and len(runs) > 1:
        lines.append(
            f'  the aggregate wall is ELAPSED across {min(workers, len(runs))} '
            f'workers, so it is less than the per-track column sums'
        )
    return "\n".join(lines)


def render_comparison(outcome: Comparison, baseline_path: Path) -> str:
    lines = []
    if outcome.desync:
        lines += ["", "  EVAL SET DESYNC"] + [f"    - {line}" for line in outcome.desync]
    if outcome.unbaselined:
        lines += ["", "  NOT IN THE BASELINE"] + [f"    - {line}"
                                                  for line in outcome.unbaselined]
    if outcome.ungated:
        lines += ["", "  NOT GATED (a guarded metric is missing)"]
        lines += [f"    - {line}" for line in outcome.ungated]
    if outcome.checksum_drift:
        lines += ["", "  BEHAVIOUR CHANGED (report checksums moved)"]
        lines += [f"    - {line}" for line in outcome.checksum_drift]
    if outcome.regressions:
        lines += ["", "  REGRESSIONS"] + [f"    - {line}" for line in outcome.regressions]
    if outcome.fact_drift:
        lines += ["", "  MEASURING SOMETHING ELSE (a recorded count moved)"]
        lines += [f"    - {line}" for line in outcome.fact_drift]
    if not outcome.failed:
        scope = "subset" if outcome.subset else "full set"
        lines += ["", f"  MATCHES BASELINE ({scope}): {baseline_path}"]
        return "\n".join(lines)
    lines += [""]
    if (outcome.checksum_drift and not outcome.regressions
            and not outcome.desync and not outcome.ungated
            and not outcome.fact_drift):
        lines += ["  Scores held or improved (no regressions).  A behaviour change",
                  "  fails this gate whichever way the numbers moved -- it is the",
                  "  operator, not the tool, who decides an improvement was meant.",
                  "  If it was, re-cut the baseline in the SAME commit:",
                  "  uv run python training/run_eval_set.py --write-baseline"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_jobs(data_dir: Path, tracks: list, sections_by_track: dict) -> list:
    """One ``Job`` per selected track, in eval-set order."""
    jobs = []
    for track in tracks:
        sections = sections_by_track.get(track["track_id"])
        if sections is None:
            raise RuntimeError(
                f"{track['track_id']} has no annotation in {SEGMENTS_FILE}")
        jobs.append(Job(str(data_dir), track, sections))
    return jobs


def execute(jobs: list, workers: int, quiet: bool = False) -> list:
    """Run every job, results in job order.

    Parallel is safe and changes nothing: each run seeds its own RNG and drives
    its own ``VirtualClock``, so a track's report is a pure function of its
    audio.  The pool exists purely so the integration suite -- which runs three
    of these on every ``uv run pytest`` -- fits its wall-time budget.
    """
    total = len(jobs)

    def announce(index: int, result: TrackRun) -> None:
        if quiet:
            return
        speed = result.entry["song_sec"] / result.wall_sec if result.wall_sec else 0.0
        print(f"  [{index}/{total}] {result.track_id} {result.wall_sec:.1f}s "
              f"({speed:.0f}x realtime)", flush=True)

    if workers <= 1 or total <= 1:
        results = []
        for index, job in enumerate(jobs, start=1):
            result = run_job(job)
            results.append(result)
            announce(index, result)
        return results

    results = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, total)) as pool:
        for index, result in enumerate(pool.map(run_job, jobs, chunksize=1),
                                       start=1):
            results.append(result)
            announce(index, result)
    return results


def run(data_dir: Path, eval_set_path: Path, only: list | None = None,
        workers: int = 1, quiet: bool = False) -> tuple:
    """Simulate and score the selected eval-set tracks.

    Returns ``(document, runs, total_song_sec, total_wall_sec)``.  Raises
    ``RuntimeError`` with a one-line explanation if an input is missing.
    """
    eval_document = load_eval_set(Path(eval_set_path))
    tracks = select_tracks(eval_document, only)
    if not tracks:
        raise RuntimeError("no eval-set tracks selected")

    problems = missing_inputs(data_dir, tracks)
    if problems:
        raise RuntimeError("; ".join(problems))
    verify_ground_truth(eval_document, Path(data_dir))

    jobs = build_jobs(data_dir, tracks, load_sections(data_dir))
    started = time.monotonic()
    runs = execute(jobs, workers, quiet=quiet)
    # Elapsed, not the sum of the tracks': under a pool they overlap, and the
    # aggregate x-realtime has to describe the run a human waited through.
    total_wall = time.monotonic() - started
    total_song = sum(result.entry["song_sec"] for result in runs)

    corpus = aggregate([result.score for result in runs])
    document = build_document(
        eval_set=eval_set_identity(eval_set_path, eval_document),
        pipeline_sha_=pipeline_sha(REPO_ROOT),
        entries={result.track_id: result.entry for result in runs},
        aggregate_metrics=track_metrics(corpus),
    )
    return document, runs, total_song, total_wall


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=corpus_dir(),
                        help=f"corpus root holding audio/ and annotations/; "
                             f"${DATA_DIR_ENV} overrides (default: %(default)s)")
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_FILE,
                        help="the frozen eval set (default: %(default)s)")
    parser.add_argument("--baseline", type=Path, default=BASELINE_FILE,
                        help="the committed baseline (default: %(default)s)")
    parser.add_argument("--write-baseline", action="store_true",
                        help="cut a new baseline instead of comparing against one")
    parser.add_argument("--allow-partial-baseline", action="store_true",
                        help="permit --write-baseline to shrink the COMMITTED "
                             "baseline to the tracks this run covered")
    parser.add_argument("--only", default=None,
                        help="comma-separated track_ids or youtube_ids to run "
                             "(default: the whole frozen set)")
    parser.add_argument("--score-tolerance", type=float,
                        default=DEFAULT_SCORE_TOLERANCE,
                        help="how far F1/accuracy may fall (default: %(default)s)")
    parser.add_argument("--flicker-tolerance", type=float,
                        default=DEFAULT_FLICKER_TOLERANCE,
                        help="how far flicker/min may rise (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="parallel simulations; results are identical either "
                             "way (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                        help="only the table and the verdict")
    args = parser.parse_args(argv)

    try:
        document = load_eval_set(Path(args.eval_set))
        only = None
        if args.only:
            only = [item for item in args.only.split(",") if item.strip()]
        # Refused BEFORE the simulations run: a two-minute run that ends in a
        # refusal teaches the same lesson two minutes later.
        if args.write_baseline:
            refusal = partial_baseline_refusal(
                select_tracks(document, only), document, Path(args.baseline),
                args.allow_partial_baseline)
            if refusal:
                print(refusal, file=sys.stderr)
                return 2
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    try:
        result, runs, total_song, total_wall = run(
            args.data_dir, args.eval_set, only,
            workers=args.workers, quiet=args.quiet)
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    print()
    print(render_table(runs, result["aggregate"], total_song, total_wall,
                       args.workers))

    if args.write_baseline:
        write_json(Path(args.baseline), result)
        print()
        print(f"  baseline written: {args.baseline}")
        print(f"  eval set        : {result['eval_set']['sha256'][:12]}... "
              f"({result['eval_set']['tracks']} tracks)")
        print(f"  pipeline        : {result['pipeline_sha'][:12]}...")
        if len(result["tracks"]) != result["eval_set"]["tracks"]:
            print(f"  WARNING: this baseline covers {len(result['tracks'])} of "
                  f"{result['eval_set']['tracks']} eval-set tracks -- the rest "
                  f"are NOT gated by it")
        return 0

    try:
        baseline = load_baseline(Path(args.baseline))
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    outcome = compare(baseline, result, args.score_tolerance, args.flicker_tolerance)
    print(render_comparison(outcome, args.baseline))
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    # Only when run as a script: under pytest, sys.stdout is the capture object
    # and main() has to stay a plain library call.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
