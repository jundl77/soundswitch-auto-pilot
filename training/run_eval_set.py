#!/usr/bin/env python
"""Run the frozen eval set through the simulation and score it against labels."""

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
    LABEL_COLUMN,
    LEGACY_V1,
    PRIMARY_TOLERANCE_SEC,
    RAW9,
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

# Every granularity the run measures and prints.
REPORTED_SPACES = (RAW9, LEGACY_V1)

# The one the committed baseline gates, and the ONE CONSTANT phase 2 flips to
# RAW9.  It stays on the retired fold until then because moving it changes the
# ground truth every gated number is measured against, and re-cutting the
# baseline needs simulations.  Only the LABEL side is held still by the fold
# view; the prediction side moves with the engine like it always did.
GATED_SPACE = LEGACY_V1

STREAM = "intent"
BOUNDARY_TOLERANCE_SEC = PRIMARY_TOLERANCE_SEC

# The tolerance the shipped decoder was selected on (#141(a)).
CRISPNESS_TOLERANCE_SEC = 0.5

REGRESSES_DOWN = "down"
REGRESSES_UP = "up"
GUARDED_METRICS = {
    "macro_f1": REGRESSES_DOWN,
    "accuracy": REGRESSES_DOWN,
    "boundary_f1": REGRESSES_DOWN,
    "crispness": REGRESSES_DOWN,
    "flicker_per_min": REGRESSES_UP,
}

# Not noise headroom: runs are deterministic. The size of a change worth stopping a commit for.
DEFAULT_SCORE_TOLERANCE = 0.02
DEFAULT_FLICKER_TOLERANCE = 0.20

COUNT_FACTS = ("rows", "label_boundaries", "exposure_sec", "beats",
               "changes_intent", "late",
               "silence_leading", "silence_interior", "silence_trailing")

AUDIO_MISSING_HINT = (
    "eval-set audio missing -- expected the committed copy in "
    f"{EVAL_AUDIO_DIR.relative_to(REPO_ROOT).as_posix()}/ (re-cut it with "
    "training/eval_assets.py --cut) or the corpus mp3 from "
    "training/raveform/raveform_download.py"
)

SCHEMA_VERSION = 1

DATA_DIR_ENV = "RAVEFORM_DATA_DIR"

from corpus_root import corpus_dir  # noqa: E402,F401


def audio_path(data_dir: Path, youtube_id: str) -> Path:
    committed = committed_audio_path(youtube_id)
    if committed.exists():
        return committed
    return corpus_audio_path(Path(data_dir), youtube_id)


def labels_source(data_dir: Path, labels: Path | None = None) -> tuple:
    committed = Path(labels or EVAL_LABELS_FILE)
    if committed.exists():
        return committed, True
    return annotations_dir(Path(data_dir)) / SEGMENTS_FILE, False


def load_sections(data_dir: Path, labels: Path | None = None) -> dict:
    path, committed = labels_source(data_dir, labels)
    if committed:
        return sections_from_slice(load_labels(path))
    return load_sections_by_track(Path(data_dir), include_hand=False)


def select_tracks(document: dict, only: list | None = None) -> list:
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
    tracks = list(document.get("tracks") or [])
    by_length = sorted(tracks, key=lambda track: (float(track["duration_sec"]),
                                                  track["track_id"]))
    chosen = {track["track_id"] for track in by_length[:max(0, count)]}
    return [track["track_id"] for track in tracks if track["track_id"] in chosen]


def missing_inputs(data_dir: Path, tracks: list, labels: Path | None = None) -> list:
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


def missing_model(data_dir: Path) -> str | None:
    from lib import section_chain

    if section_chain.artifacts_present(data_dir):
        return None
    return ("the shipped model is not on this machine: "
            f"{', '.join(section_chain.artifacts(data_dir).missing())}\n"
            "the benchmark scores a neural show; without the model every track "
            "runs the degradation state, so its checksums and scores describe "
            "nothing.  This is a missing download, not a regression -- do not "
            "re-cut the baseline from here.")


def verify_ground_truth(document: dict, data_dir: Path,
                        labels: Path | None = None) -> None:
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
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:                                             # pragma: no cover
        return Path(left).absolute() == Path(right).absolute()


def partial_baseline_refusal(selected: list, document: dict, baseline_path: Path,
                             allowed: bool = False) -> str | None:
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


class TrackRun(NamedTuple):
    track_id: str
    youtube_id: str
    entry: dict
    scores: dict
    wall_sec: float


def simulate_report(mp3_path: str) -> tuple:
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import run_fast_simulation

    started = time.monotonic()
    client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, str(mp3_path))
    _client, event_buffer, command_queue = asyncio.run(run_fast_simulation(client))
    report = event_buffer.to_report(command_queue.get_timing_log())
    return report, client.duration_sec, time.monotonic() - started


def score_report(track_id: str, youtube_id: str, report: dict, sections: list):
    rows, stats = join_track(track_id, youtube_id, report, sections)
    track = TrackBeats(
        track_id=track_id,
        times=tuple(row["t_song"] for row in rows),
        intents=tuple(row["intent_at_beat"] for row in rows),
        labels={name: tuple(spec.view(row[LABEL_COLUMN]) for row in rows)
                for name, spec in SPACES.items()},
    )
    return ({space: score_track(track, space) for space in REPORTED_SPACES},
            len(rows), stats)


def track_metrics(score) -> dict:
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


def space_facts(score) -> dict:
    """The counts that depend on which vocabulary the labels were read in.

    A coarser vocabulary merges classes, so it sees fewer label boundaries and
    fewer class changes over the same beats.  Everything else a row records --
    beats, rows, exposure, lateness, blackouts, intent changes -- is a fact
    about the run rather than about the reading.
    """
    return {
        "label_boundaries":
            score.boundary["intent"][BOUNDARY_TOLERANCE_SEC]["overall"]["n_truth"],
        "changes_class":
            score.boundary["class"][BOUNDARY_TOLERANCE_SEC]["overall"]["n_pred"],
    }


def space_block(score) -> dict:
    return {**track_metrics(score), **space_facts(score)}


def track_entry(report: dict, scores: dict, rows: int, youtube_id: str,
                song_sec: float, stats=None) -> dict:
    from simulate.evaluator import report_checksum

    gated = scores[GATED_SPACE]
    entry = {
        "youtube_id": youtube_id,
        "checksum": report_checksum(report),
        "beats": len(report.get("beats", [])),
        "rows": rows,
        "song_sec": round(float(song_sec), 3),
        "exposure_sec": round(gated.exposure_sec, 3),
        "changes_intent":
            gated.boundary["intent"][BOUNDARY_TOLERANCE_SEC]["overall"]["n_pred"],
    }
    if stats is not None:
        entry["late"] = int(stats.intent_blocks_late)
        entry["blocks_measurable"] = int(stats.intent_blocks_song_recorded)
        entry["silence_leading"] = int(stats.silence_blocks_leading)
        entry["silence_interior"] = int(stats.silence_blocks_interior)
        entry["silence_trailing"] = int(stats.silence_blocks_trailing)
    entry.update(space_block(gated))
    entry["spaces"] = {space: space_block(scores[space])
                       for space in REPORTED_SPACES}
    return entry


class Job(NamedTuple):
    data_dir: str
    track: dict
    sections: list


def run_job(job: Job) -> TrackRun:
    track_id, youtube_id = job.track["track_id"], job.track["youtube_id"]
    report, song_sec, wall_sec = simulate_report(
        audio_path(Path(job.data_dir), youtube_id))
    scores, rows, stats = score_report(track_id, youtube_id, report, job.sections)
    return TrackRun(track_id, youtube_id,
                    track_entry(report, scores, rows, youtube_id, song_sec, stats),
                    scores, wall_sec)


def build_document(eval_set: dict, pipeline_sha_: str, entries: dict,
                   aggregate_metrics: dict, aggregate_spaces: dict | None = None,
                   score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
                   flicker_tolerance: float = DEFAULT_FLICKER_TOLERANCE) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "eval_set": eval_set,
        "pipeline_sha": pipeline_sha_,
        "space": GATED_SPACE,
        "reported_spaces": list(REPORTED_SPACES),
        "stream": STREAM,
        "boundary_tolerance_sec": BOUNDARY_TOLERANCE_SEC,
        "crispness_tolerance_sec": CRISPNESS_TOLERANCE_SEC,
        "gate": {"score_tolerance": score_tolerance,
                 "flicker_tolerance": flicker_tolerance,
                 "metrics": dict(GUARDED_METRICS)},
        "aggregate": {**aggregate_metrics,
                      "spaces": dict(aggregate_spaces or {})},
        "tracks": entries,
    }


def eval_set_identity(path: Path, document: dict) -> dict:
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


class Comparison(NamedTuple):
    desync: list
    unbaselined: list
    ungated: list
    checksum_drift: list
    regressions: list
    fact_drift: list
    subset: bool

    @property
    def failed(self) -> bool:
        return bool(self.desync or self.unbaselined or self.ungated
                    or self.checksum_drift or self.regressions or self.fact_drift)


def _regression(name: str, metric: str, before: float, after: float,
                tolerance: float) -> str | None:
    delta = after - before
    if GUARDED_METRICS[metric] == REGRESSES_DOWN:
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
    if not any(fact in before or fact in after for fact in COUNT_FACTS):
        return []
    drift = []
    for fact in COUNT_FACTS:
        if fact not in before or fact not in after:
            where = "the baseline" if fact not in before else "this run"
            drift.append(f"{name}: {fact} is missing from {where} -- it is "
                         f"NOT being compared")
        elif before[fact] != after[fact]:
            drift.append(f"{name}: {fact} {before[fact]} -> {after[fact]}")
    return drift


def _compare_metrics(name: str, before: dict, after: dict, score_tolerance: float,
                     flicker_tolerance: float) -> tuple:
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

    if not subset:
        row_ungated, row_regressions, row_facts = _compare_metrics(
            "(aggregate)", baseline.get("aggregate") or {},
            current.get("aggregate") or {}, score_tolerance, flicker_tolerance)
        ungated += row_ungated
        regressions += row_regressions
        facts += row_facts

    return Comparison(desync, unbaselined, ungated, drift, regressions, facts,
                      subset)


WIDTH = 116


def render_table(runs: list, aggregate_entry: dict, total_song: float,
                 total_wall: float, workers: int = 1) -> str:
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
        f'  macro-F1 and accuracy in the {GATED_SPACE} space; boundary-F1 and '
        f'flicker on the {STREAM} stream at +/-{BOUNDARY_TOLERANCE_SEC}s, '
        f'crispness at +/-{CRISPNESS_TOLERANCE_SEC}s'
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
    lines += render_spaces(runs, aggregate_entry)
    return "\n".join(lines)


def render_spaces(runs: list, aggregate_entry: dict) -> list:
    """The same run read at every granularity -- one gated, the rest reported."""
    lines = ["", f'  every granularity ({GATED_SPACE} is the gated one)',
             f'  {"track_id":<20}' + "".join(
                 f'{space + " macroF1":>19}{"acc":>7}{"bF1":>7}{"flick/m":>9}{"bnd":>6}'
                 for space in REPORTED_SPACES),
             "  " + "-" * (20 + 48 * len(REPORTED_SPACES))]
    aggregate_spaces = aggregate_entry.get("spaces") or {}
    rows = [(run.track_id, run.entry.get("spaces") or {}) for run in runs]
    if aggregate_spaces:
        rows.append(("(aggregate)", aggregate_spaces))
    for track_id, blocks in rows:
        cells = ""
        for space in REPORTED_SPACES:
            block = blocks.get(space)
            if block is None:
                cells += f'{"-":>48}'
                continue
            cells += (f'{block["macro_f1"]:>19.3f}{block["accuracy"]:>7.3f}'
                      f'{block["boundary_f1"]:>7.3f}'
                      f'{block["flicker_per_min"]:>9.2f}'
                      f'{block["label_boundaries"]:>6}')
        lines.append(f'  {track_id:<20}{cells}')
    lines.append('  bnd = label boundaries the vocabulary can see; a coarser '
                 'reading merges classes and sees fewer')
    return lines


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


def build_jobs(data_dir: Path, tracks: list, sections_by_track: dict) -> list:
    jobs = []
    for track in tracks:
        sections = sections_by_track.get(track["track_id"])
        if sections is None:
            raise RuntimeError(
                f"{track['track_id']} has no annotation in {SEGMENTS_FILE}")
        jobs.append(Job(str(data_dir), track, sections))
    return jobs


def execute(jobs: list, workers: int, quiet: bool = False) -> list:
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
    total_wall = time.monotonic() - started
    total_song = sum(result.entry["song_sec"] for result in runs)

    corpus = {space: aggregate([result.scores[space] for result in runs])
              for space in REPORTED_SPACES}
    document = build_document(
        eval_set=eval_set_identity(eval_set_path, eval_document),
        pipeline_sha_=pipeline_sha(REPO_ROOT),
        entries={result.track_id: result.entry for result in runs},
        aggregate_metrics=track_metrics(corpus[GATED_SPACE]),
        aggregate_spaces={space: space_block(corpus[space])
                          for space in REPORTED_SPACES},
    )
    return document, runs, total_song, total_wall


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel simulations; the bytes are identical "
                             "either way, but a cold run holds several GB of "
                             "VRAM per worker and oversubscribing one card "
                             "wedges rather than slows (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                        help="only the table and the verdict")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        document = load_eval_set(Path(args.eval_set))
        only = None
        if args.only:
            only = [item for item in args.only.split(",") if item.strip()]
        absent = missing_model(Path(args.data_dir))
        if absent:
            print(absent, file=sys.stderr)
            return 2
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
    # Not at module scope: under pytest sys.stdout is the capture object.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
