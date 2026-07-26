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
no musical reason for.

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
otherwise would either fail every subset run or gate on nothing.

**Decode caches are kept.**  Every other consumer of the corpus deletes the
``<mp3>.<rate>.npy`` a simulation leaves behind (it is ~7.7x the mp3's size), but
the eval set is ten tracks and it is re-run constantly -- by the integration
suite on every ``uv run pytest``.  Keeping the caches costs under a gigabyte and
removes the decode (several seconds a track) from every run after the first.
This is deliberate, and ``build_training_table`` already honours it: its
``keep_cache`` rule leaves behind any cache that existed before its batch.

Exit codes: 0 clean, 1 the gate failed, 2 an input is missing.
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
for _path in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from build_training_table import (  # noqa: E402  (needs the path inserts above)
    AUDIO_DIR,
    _git,
    default_workers,
    join_track,
    load_sections_by_track,
    pipeline_sha,
)
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
from select_eval_set import EVAL_SET_FILE, load_eval_set  # noqa: E402

BASELINE_FILE = REPO_ROOT / "training" / "eval_set_baseline.json"

# The scoring configuration the baseline is cut under.  Recorded in the file so
# a baseline can never be read under different assumptions than it was written.
SPACE = "v1"                    # the space the NN trains on -- the primary one
STREAM = "intent"               # every lighting change: the show as the room sees it
BOUNDARY_TOLERANCE_SEC = PRIMARY_TOLERANCE_SEC

# metric -> the direction that is a REGRESSION.
GUARDED_METRICS = {
    "macro_f1": "down",
    "accuracy": "down",
    "boundary_f1": "down",
    "flicker_per_min": "up",
}

# How far a number may move before it is a regression.  Runs are deterministic,
# so this is not noise headroom -- it is the size of a score change worth
# stopping a commit for.  F1/accuracy live in [0, 1]; flicker is changes per
# audience-minute and sits near 1-3 on this corpus, so it gets its own slack.
DEFAULT_SCORE_TOLERANCE = 0.02
DEFAULT_FLICKER_TOLERANCE = 0.20

# The one line a machine without the corpus must see.  Audio is never committed
# (the repo is public), so an absent eval set is an ordinary state of a fresh
# clone -- but it must be loud, not a silent skip: a skipped benchmark that
# nobody notices is the same as no benchmark.
AUDIO_MISSING_HINT = "eval-set audio missing -- run training/raveform_download.py"

SCHEMA_VERSION = 1

# Where the corpus is, when it is not where the code is.
DATA_DIR_ENV = "RAVEFORM_DATA_DIR"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def corpus_dir() -> Path:
    """The corpus root: ``$RAVEFORM_DATA_DIR``, else the repo's, else the main
    worktree's.

    The corpus is gitignored, so there is exactly ONE copy of it on a machine and
    it does not follow ``git worktree add``: branch work in a linked worktree
    still has to reach the audio sitting in the main checkout.  Resolving that
    here -- rather than making every caller pass ``--data-dir`` -- is what lets
    the integration suite run green from any worktree.  The environment variable
    wins for the case this cannot guess (a corpus on another drive).
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).resolve()
    local = default_data_dir()
    if local.exists():
        return local
    # `--git-common-dir` is the main checkout's .git even from a linked worktree.
    common = _git(REPO_ROOT, "rev-parse", "--path-format=absolute",
                  "--git-common-dir")
    if common:
        shared = Path(common).parent / local.relative_to(REPO_ROOT)
        if shared.exists():
            return shared
    return local


def audio_path(data_dir: Path, youtube_id: str) -> Path:
    """Where the downloader parks one eval track's mp3."""
    return Path(data_dir) / AUDIO_DIR / f"{youtube_id}.mp3"


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
    order anything happened to be written in.
    """
    tracks = list(document.get("tracks") or [])
    by_length = sorted(tracks, key=lambda track: (float(track["duration_sec"]),
                                                  track["track_id"]))
    chosen = {track["track_id"] for track in by_length[:max(0, count)]}
    return [track["track_id"] for track in tracks if track["track_id"] in chosen]


def missing_inputs(data_dir: Path, tracks: list) -> list:
    """Everything the run needs and does not have, one human line each.

    Reported all at once rather than raising on the first: a fresh clone should
    learn it needs the annotations AND three mp3s from one run, not three.
    """
    problems = []
    segments = annotations_dir(Path(data_dir)) / SEGMENTS_FILE
    if not segments.exists():
        problems.append(f"missing {segments} -- run "
                        f"training/raveform_fetch_annotations.py")
    for track in tracks:
        mp3 = audio_path(data_dir, track["youtube_id"])
        if not mp3.exists():
            problems.append(f"{AUDIO_MISSING_HINT}: {track['track_id']} ({mp3})")
    return problems


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
    the pure-logic half of this module (and its unit tests) never pay for aubio.
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
    rows, _stats = join_track(track_id, youtube_id, report, sections)
    track = TrackBeats(
        track_id=track_id,
        times=tuple(row["t_song"] for row in rows),
        intents=tuple(row["intent_at_beat"] for row in rows),
        labels={name: tuple(row[spec.column] for row in rows)
                for name, spec in SPACES.items()},
    )
    return score_track(track, SPACE), len(rows)


def track_metrics(score) -> dict:
    """The four gated numbers, read off a ``Score``."""
    return {
        "macro_f1": round(score.macro_f1, 6),
        "accuracy": round(score.accuracy, 6),
        "boundary_f1": round(
            score.boundary_prf(STREAM, BOUNDARY_TOLERANCE_SEC)[2], 6),
        "flicker_per_min": round(
            score.flicker_per_minute[STREAM][BOUNDARY_TOLERANCE_SEC], 6),
    }


def track_entry(report: dict, score, rows: int, youtube_id: str,
                song_sec: float) -> dict:
    """One track's row of the baseline: identity, size, speed-free facts, scores.

    Wall time and x-realtime are deliberately NOT in here.  They are printed on
    every run and they are the whole reason the caches are kept, but they are a
    property of the machine, and a committed file that changes with the CPU load
    of the laptop that wrote it is a file nobody can diff.
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
    score, rows = score_report(track_id, youtube_id, report, job.sections)
    return TrackRun(track_id, youtube_id,
                    track_entry(report, score, rows, youtube_id, song_sec),
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
    checksum_drift: list
    regressions: list
    subset: bool        # fewer tracks than the baseline: aggregate not compared

    @property
    def failed(self) -> bool:
        return bool(self.desync or self.unbaselined
                    or self.checksum_drift or self.regressions)


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


def compare(baseline: dict, current: dict,
            score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
            flicker_tolerance: float = DEFAULT_FLICKER_TOLERANCE) -> Comparison:
    """Judge a run against the committed baseline.

    Three independent failures, kept apart because they mean different things:
    a DESYNC says the baseline describes a different benchmark (re-cut it), a
    CHECKSUM DRIFT says the pipeline's behaviour moved (look at the scores, then
    accept or fix), and a REGRESSION says the show got worse (fix it).
    """
    desync, unbaselined, drift, regressions = [], [], [], []

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
        for metric in GUARDED_METRICS:
            if metric not in before or metric not in after:
                continue
            tolerance = (flicker_tolerance if metric == "flicker_per_min"
                         else score_tolerance)
            line = _regression(track_id, metric, float(before[metric]),
                               float(after[metric]), tolerance)
            if line:
                regressions.append(line)

    # A subset run's aggregate is an aggregate of the subset; comparing it to
    # the ten-track number would be comparing two different quantities.
    if not subset:
        before_all = baseline.get("aggregate") or {}
        after_all = current.get("aggregate") or {}
        for metric in GUARDED_METRICS:
            if metric not in before_all or metric not in after_all:
                continue
            tolerance = (flicker_tolerance if metric == "flicker_per_min"
                         else score_tolerance)
            line = _regression("(aggregate)", metric, float(before_all[metric]),
                               float(after_all[metric]), tolerance)
            if line:
                regressions.append(line)

    return Comparison(desync, unbaselined, drift, regressions, subset)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

WIDTH = 100


def render_table(runs: list, aggregate_entry: dict, total_song: float,
                 total_wall: float, workers: int = 1) -> str:
    """The per-track table every run prints, baseline or not."""
    lines = [
        f'  {"track_id":<20}{"song":>7}{"wall":>7}{"x-rt":>7}{"beats":>7}'
        f'{"rows":>7}{"macroF1":>9}{"acc":>7}{"bF1":>7}{"flick/m":>9}'
        f'{"chg i/c":>10}  checksum',
        "  " + "-" * (WIDTH - 2),
    ]
    for run in runs:
        entry = run.entry
        speed = entry["song_sec"] / run.wall_sec if run.wall_sec > 0 else 0.0
        changes = f'{entry["changes_intent"]}/{entry["changes_class"]}'
        lines.append(
            f'  {run.track_id:<20}{entry["song_sec"] / 60.0:>6.1f}m'
            f'{run.wall_sec:>6.1f}s{speed:>6.0f}x{entry["beats"]:>7}'
            f'{entry["rows"]:>7}{entry["macro_f1"]:>9.3f}{entry["accuracy"]:>7.3f}'
            f'{entry["boundary_f1"]:>7.3f}{entry["flicker_per_min"]:>9.2f}'
            f'{changes:>10}  {entry["checksum"][:12]}'
        )
    lines.append("  " + "-" * (WIDTH - 2))
    speed = total_song / total_wall if total_wall > 0 else 0.0
    lines.append(
        f'  {"(aggregate)":<20}{total_song / 60.0:>6.1f}m{total_wall:>6.1f}s'
        f'{speed:>6.0f}x{"":>7}{"":>7}{aggregate_entry["macro_f1"]:>9.3f}'
        f'{aggregate_entry["accuracy"]:>7.3f}{aggregate_entry["boundary_f1"]:>7.3f}'
        f'{aggregate_entry["flicker_per_min"]:>9.2f}'
    )
    lines.append(
        f'  macro-F1 and accuracy in the {SPACE} space; boundary-F1 and flicker on '
        f'the {STREAM} stream at +/-{BOUNDARY_TOLERANCE_SEC}s'
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
    if outcome.checksum_drift:
        lines += ["", "  BEHAVIOUR CHANGED (report checksums moved)"]
        lines += [f"    - {line}" for line in outcome.checksum_drift]
    if outcome.regressions:
        lines += ["", "  REGRESSIONS"] + [f"    - {line}" for line in outcome.regressions]
    if not outcome.failed:
        scope = "subset" if outcome.subset else "full set"
        lines += ["", f"  MATCHES BASELINE ({scope}): {baseline_path}"]
        return "\n".join(lines)
    lines += [""]
    if outcome.checksum_drift and not outcome.regressions and not outcome.desync:
        lines += ["  Scores held.  If the behaviour change is wanted, re-cut the",
                  "  baseline in the same commit: run with --write-baseline."]
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

    jobs = build_jobs(data_dir, tracks, load_sections_by_track(Path(data_dir)))
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
    parser.add_argument("--only", default=None,
                        help="comma-separated track_ids or youtube_ids to run "
                             "(default: the whole frozen set)")
    parser.add_argument("--shortest", type=int, default=None,
                        help="run the N shortest tracks of the set (a subset run)")
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

    only = None
    if args.only:
        only = [item for item in args.only.split(",") if item.strip()]
    if args.shortest is not None:
        try:
            document = load_eval_set(Path(args.eval_set))
        except RuntimeError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2
        shortest = shortest_track_ids(document, args.shortest)
        only = shortest if only is None else [*only, *shortest]

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
            print("  WARNING: this baseline covers a SUBSET of the eval set; the "
                  "gate will report the rest as unbaselined")
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
