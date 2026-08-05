#!/usr/bin/env python
"""Batch fast-sim over the clean corpus -> label-aligned per-beat training table."""

from __future__ import annotations

import argparse
import bisect
import collections
import concurrent.futures
import csv
import datetime
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    str(REPO_ROOT),
    str(REPO_ROOT / "training"),
    str(REPO_ROOT / "training" / "raveform"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from build_clean_manifest import (  # noqa: E402
    CLEAN_MANIFEST_FILE,
    MIN_AGE_SEC,
    STATUS_OK,
    is_settled,
)
from raveform_fetch_annotations import (  # noqa: E402
    load_all_tracks,
    load_tracks,
    parse_sections,
)
from raveform_manifest import section_runs  # noqa: E402

from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE  # noqa: E402
from lib.label_space import DROPPED_LABELS, SECTION_LABELS  # noqa: E402
from lib.engine.event_buffer import CLASSIFIER_TRIGGER, SILENCE_TRIGGER  # noqa: E402

from corpus_root import default_data_dir  # noqa: E402,F401

TABLE_FILE = "training_table.csv.gz"
META_FILE = "training_table.meta.json"
REPORTS_DIR = "reports"
FEATURES_DIR = "features"
AUDIO_DIR = "audio"

CACHE_VERSION = 1
_REPRODUCIBLE_GZIP_MTIME = 0

CONTINUOUS_COLUMNS = (
    "bpm",
    "rms",
)

TABLE_HEADER = (
    "track_id",
    "youtube_id",
    "t_song",
    "bpm",
    "rms",
    "intent_at_beat",
    "label",
    "bar_position_unknown",
) + tuple(f"{column}_z" for column in CONTINUOUS_COLUMNS)

NO_INTENT = ""

BAR_POSITION_UNKNOWN = 1

MEL_BANDS = 40
POOL_BUFFERS = 8

MEL_EXPORTER_VERSION = 1
MEL_EXPORTER_KEY = "exporter_version"
_UNSTAMPED_SIDECAR_GENERATION = 1

ANALYSER_RESET_SEC = 900.0


class Timeline:
    """Half-open ``[start, end)`` spans with an O(log n) point lookup; later span wins."""

    def __init__(self, spans: list) -> None:
        ordered = sorted(spans, key=lambda span: span[0])
        self._starts = [span[0] for span in ordered]
        self._ends = [span[1] for span in ordered]
        self._values = [span[2] for span in ordered]

    def at(self, t: float):
        index = bisect.bisect_right(self._starts, t) - 1
        if index < 0:
            return None
        return self._values[index] if t < self._ends[index] else None


def _clamped_spans(sections: list) -> list:
    return [
        (float(start), max(float(start), float(end)), str(label))
        for start, end, label in sections
    ]


def label_coverage(sections: list) -> list:
    """Labelled spans, per published section -- never per merged run.

    A merged run's span can swallow a dropped sentinel sitting between two
    members, and that time belongs to neither neighbour.
    """
    return [
        span for span in _clamped_spans(sections) if span[2] not in DROPPED_LABELS
    ]


def dropped_coverage(sections: list) -> list:
    return [span for span in _clamped_spans(sections) if span[2] in DROPPED_LABELS]


def labeled_bounds(sections: list) -> tuple:
    runs = section_runs(list(sections))
    if not runs:
        return None, None
    first_start = float(runs[0][0])
    last_end = max(max(float(start), float(end)) for start, end, _label, _dur in runs)
    return first_start, last_end


def song_time_intents(blocks: list, look_ahead_sec: float,
                      default_end: float | None = None) -> list:
    spans = []
    for block in blocks:
        start = float(block["t"]) - look_ahead_sec
        raw_end = block.get("end", default_end)
        end = float("inf") if raw_end is None else float(raw_end) - look_ahead_sec
        spans.append((start, max(start, end), str(block["intent"])))
    return spans


_QUEUE_STAMP_TOLERANCE_BUFFERS = 1.5
_QUEUE_STAMP_TOLERANCE_SEC = _QUEUE_STAMP_TOLERANCE_BUFFERS * BUFFER_SIZE / SAMPLE_RATE
_STAMP_EPS = 1e-9


class IntentAlignment(NamedTuple):
    blocks: int
    song_stamped: int
    clamped_tail: int
    song_recorded: int = 0
    late: int = 0


def _nearest_gap(sorted_times: list, value: float) -> float:
    if not sorted_times:
        return float("inf")
    index = bisect.bisect_left(sorted_times, value)
    best = float("inf")
    for candidate in (index - 1, index):
        if 0 <= candidate < len(sorted_times):
            best = min(best, abs(sorted_times[candidate] - value))
    return best


def realign_intents(blocks: list, look_ahead_sec: float, beat_times: list,
                    duration_sec: float | None = None,
                    tolerance: float = _QUEUE_STAMP_TOLERANCE_SEC) -> tuple:
    """Intent blocks -> song-time spans: ``song_t`` if recorded, else the queue shift."""
    if not blocks:
        return [], IntentAlignment(0, 0, 0, 0)

    starts: list = []
    shifts: list = []
    song_stamped = clamped_tail = song_recorded = 0
    for block in blocks:
        t = float(block["t"])
        is_clamped = (duration_sec is not None
                      and abs(t - float(duration_sec)) <= _STAMP_EPS)
        if block.get("song_t") is not None:
            starts.append(float(block["song_t"]))
            song_recorded += 1
            clamped_tail += 1 if is_clamped else 0
            shifts.append(t - starts[-1])
            continue
        explained_by_queue = (
            look_ahead_sec <= 0
            or not beat_times
            or is_clamped
            or _nearest_gap(beat_times, t - look_ahead_sec) <= tolerance
        )
        if explained_by_queue:
            starts.append(t - look_ahead_sec)
            clamped_tail += 1 if is_clamped else 0
        else:
            starts.append(t)
            song_stamped += 1
        shifts.append(t - starts[-1])

    if beat_times and not blocks[0].get("song_t"):
        starts[0] = max(starts[0], beat_times[0])

    spans = []
    for index, block in enumerate(blocks):
        if index + 1 < len(blocks):
            end = starts[index + 1]
        else:
            raw_end = block.get("end", duration_sec)
            end = (float("inf") if raw_end is None
                   else float(raw_end) - shifts[index])
        spans.append((starts[index], max(starts[index], end), str(block["intent"])))
    late = sum(1 for shift in shifts if shift > look_ahead_sec + tolerance)
    return spans, IntentAlignment(len(blocks), song_stamped, clamped_tail,
                                  song_recorded, late)


LEADING, INTERIOR, TRAILING = "leading", "interior", "trailing"


class SilenceBlocks(NamedTuple):
    leading: int
    interior: int
    trailing: int


def silence_triggered(blocks: list) -> list:
    return [str(block.get("trigger", CLASSIFIER_TRIGGER)) == SILENCE_TRIGGER
            for block in blocks]


def drop_silence_spans(spans: list, silence: list) -> list:
    """An operator blackout is not a classification claim; the span it covers is
    left as a hole so those beats read as uncommitted rather than as the intent
    that preceded them."""
    return [span for span, excluded in zip(spans, silence) if not excluded]


def silence_position(span, first_start, last_end) -> str:
    start, end, _intent = span
    if last_end is None or start >= last_end:
        return TRAILING
    if first_start is not None and end <= first_start:
        return LEADING
    return INTERIOR


def count_silence_positions(spans: list, silence: list,
                            first_start, last_end) -> SilenceBlocks:
    counts = collections.Counter(
        silence_position(span, first_start, last_end)
        for span, excluded in zip(spans, silence) if excluded)
    return SilenceBlocks(counts[LEADING], counts[INTERIOR], counts[TRAILING])


def zscores(values: list) -> list:
    """Population z-scores; all-zero for a feature that never moves."""
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    spread = float(array.std())
    if spread <= 0.0:
        return [0.0] * len(values)
    return ((array - array.mean()) / spread).tolist()


class JoinStats(NamedTuple):
    beats_total: int
    beats_kept: int
    dropped_leading: int
    dropped_gap: int
    dropped_trailing: int
    dropped_in_dropped_section: int
    beats_without_intent: int
    intent_blocks_song_stamped: int
    intent_blocks_song_recorded: int
    intent_blocks_late: int
    intent_reattributed: int
    silence_blocks_leading: int
    silence_blocks_interior: int
    silence_blocks_trailing: int


def join_track(track_id: str, youtube_id_: str, report: dict, sections: list) -> tuple:
    """One track's report + annotation -> ``(rows, JoinStats)``; pure, no I/O."""
    beats = sorted(report.get("beats", []), key=lambda record: float(record["t"]))
    coverage = Timeline(label_coverage(sections))
    sentinels = Timeline(dropped_coverage(sections))
    first_start, last_end = labeled_bounds(sections)

    look_ahead_sec = float(report.get("metrics", {}).get("look_ahead_sec", 0.0))
    blocks = report.get("intents", [])
    duration_sec = report.get("duration_sec")
    beat_times = [float(record["t"]) for record in beats]
    spans, alignment = realign_intents(blocks, look_ahead_sec, beat_times, duration_sec)
    silence = silence_triggered(blocks)
    silence_blocks = count_silence_positions(spans, silence, first_start, last_end)
    intents = Timeline(drop_silence_spans(spans, silence))
    naive_intents = Timeline(drop_silence_spans(
        song_time_intents(blocks, look_ahead_sec, duration_sec), silence))

    rows: list = []
    leading = gap = trailing = in_dropped = without_intent = reattributed = 0

    for record in beats:
        t = float(record["t"])
        label = coverage.at(t)
        if label is None:
            if first_start is not None and t < first_start:
                leading += 1
            elif last_end is not None and t >= last_end:
                trailing += 1
            else:
                gap += 1
            if sentinels.at(t) is not None:
                in_dropped += 1
            continue

        intent = intents.at(t)
        if (naive_intents.at(t) or NO_INTENT) != (intent or NO_INTENT):
            reattributed += 1
        if intent is None:
            intent = NO_INTENT
            without_intent += 1

        rows.append({
            "track_id": track_id,
            "youtube_id": youtube_id_,
            "t_song": t,
            "bpm": float(record.get("bpm", 0.0)),
            "rms": float(record.get("rms", 0.0)),
            "intent_at_beat": intent,
            "label": label,
            "bar_position_unknown": BAR_POSITION_UNKNOWN,
        })

    _add_zscores(rows)
    stats = JoinStats(
        beats_total=len(beats),
        beats_kept=len(rows),
        dropped_leading=leading,
        dropped_gap=gap,
        dropped_trailing=trailing,
        dropped_in_dropped_section=in_dropped,
        beats_without_intent=without_intent,
        intent_blocks_song_stamped=alignment.song_stamped,
        intent_blocks_song_recorded=alignment.song_recorded,
        intent_blocks_late=alignment.late,
        intent_reattributed=reattributed,
        silence_blocks_leading=silence_blocks.leading,
        silence_blocks_interior=silence_blocks.interior,
        silence_blocks_trailing=silence_blocks.trailing,
    )
    return rows, stats


def _add_zscores(rows: list) -> None:
    for column in CONTINUOUS_COLUMNS:
        for row, value in zip(rows, zscores([row[column] for row in rows])):
            row[f"{column}_z"] = value


def _field(value) -> str:
    """One CSV cell.  Fixed-width floats so the file is byte-stable."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int)):
        return str(int(value))
    return f"{float(value):.6f}"


def format_row(row: dict) -> list:
    return [_field(row[column]) for column in TABLE_HEADER]


def write_feature_sidecar(path, mel: np.ndarray, frame_sec: float, t0: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "wb") as handle:
            np.savez_compressed(
                handle,
                mel=mel.astype(np.float32, copy=False),
                frame_sec=np.float64(frame_sec),
                t0=np.float64(t0),
                sample_rate=np.int32(SAMPLE_RATE),
                pool_buffers=np.int32(POOL_BUFFERS),
                **{MEL_EXPORTER_KEY: np.int32(MEL_EXPORTER_VERSION)},
            )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def sidecar_generation(path: Path) -> int:
    try:
        with np.load(path) as archive:
            if MEL_EXPORTER_KEY in archive.files:
                return int(archive[MEL_EXPORTER_KEY])
    except (OSError, ValueError, EOFError, KeyError):
        pass
    return _UNSTAMPED_SIDECAR_GENERATION


class SimJob(NamedTuple):
    track_id: str
    youtube_id: str
    mp3_path: str
    report_path: str
    sidecar_path: str
    preexisting: tuple
    pipeline_sha: str
    mp3_size: int
    mp3_mtime: float


class SimResult(NamedTuple):
    track_id: str
    ok: bool
    detail: str
    beats: int
    frames: int
    sidecar_bytes: int
    wall_sec: float


def decode_cache_path(mp3_path: str) -> str:
    return f"{mp3_path}.{SAMPLE_RATE}.npy"


def derived_cache_paths(mp3_path: str) -> tuple:
    from simulate.cell_cache import sidecar_path
    from simulate.fake_audio_client import FileAudioClient

    return (decode_cache_path(mp3_path),
            str(sidecar_path(mp3_path, FileAudioClient.decode_path)))


def _write_json_gz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6,
                               filename="", mtime=_REPRODUCIBLE_GZIP_MTIME) as compressed:
                compressed.write(
                    json.dumps(payload, separators=(",", ":"), sort_keys=True)
                    .encode("utf-8")
                )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def report_path(data_dir: Path, youtube_id: str) -> Path:
    return data_dir / REPORTS_DIR / f"{youtube_id}.json.gz"


def report_envelope(job: SimJob, report: dict) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "track_id": job.track_id,
        "youtube_id": job.youtube_id,
        "pipeline_sha": job.pipeline_sha,
        "mp3_size": job.mp3_size,
        "mp3_mtime": job.mp3_mtime,
        "report": report,
    }


def cache_is_fresh(envelope: dict, pipeline_sha_: str,
                   mp3_size: int, mp3_mtime: float) -> bool:
    if not isinstance(envelope, dict):
        return False
    return (
        envelope.get("cache_version") == CACHE_VERSION
        and envelope.get("pipeline_sha") == pipeline_sha_
        and envelope.get("mp3_size") == mp3_size
        and envelope.get("mp3_mtime") == mp3_mtime
        and isinstance(envelope.get("report"), dict)
    )


def simulate_track(job: SimJob) -> SimResult:
    """Run one track through the fast sim.  Never raises; always drops its caches."""
    import asyncio

    started = time.monotonic()
    try:
        from simulate.fake_audio_client import FileAudioClient
        from simulate.runner import run_fast_simulation

        client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, job.mp3_path)
        _client, event_buffer, command_queue = asyncio.run(run_fast_simulation(client))
        report = event_buffer.to_report(command_queue.get_timing_log())
        _write_json_gz(Path(job.report_path), report_envelope(job, report))

        return SimResult(job.track_id, True, "", len(report.get("beats", [])),
                         0, 0, time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001
        return SimResult(job.track_id, False, f"{type(exc).__name__}: {exc}"[:300],
                         0, 0, 0, time.monotonic() - started)
    finally:
        for path in derived_cache_paths(job.mp3_path):
            if path in job.preexisting:
                continue
            try:
                os.unlink(path)
            except OSError:
                pass


def load_ok_rows(data_dir: Path) -> list:
    path = data_dir / CLEAN_MANIFEST_FILE
    if not path.exists():
        raise RuntimeError(
            f"missing {path} -- run training/raveform/build_clean_manifest.py first"
        )
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == STATUS_OK]
    rows, rejected = _reject_past_the_analyser_reset(rows)
    for track_id, seconds in rejected:
        print(f"  SKIP {track_id}: {seconds / 60.0:.1f} min reaches MusicAnalyser's "
              f"{ANALYSER_RESET_SEC / 60.0:.0f}-minute self-reset -- its beats and "
              f"its mel sidecar would no longer describe the same audio", flush=True)
    if not rows:
        raise RuntimeError(f"no ok rows in {path} -- nothing to build from")
    rows.sort(key=lambda row: row["track_id"])
    return rows


def _reject_past_the_analyser_reset(rows: list) -> tuple:
    kept, rejected = [], []
    for row in rows:
        try:
            seconds = float(row.get("decoded_duration_sec") or 0.0)
        except ValueError:
            seconds = 0.0
        if seconds >= ANALYSER_RESET_SEC:
            rejected.append((row["track_id"], seconds))
        else:
            kept.append(row)
    return kept, rejected


def load_sections_by_track(data_dir: Path, include_hand: bool = True) -> dict:
    tracks = load_all_tracks(data_dir) if include_hand else load_tracks(data_dir)
    return {str(track["key"]): parse_sections(track) for track in tracks}


def select_jobs(rows: list, data_dir: Path, force: bool = False,
                min_age_sec: float = MIN_AGE_SEC,
                preexisting_caches: set | None = None,
                sha: str | None = None) -> tuple:
    """``(jobs, counts)`` -- which tracks still need simulating, and why."""
    now = time.time()
    preexisting_caches = preexisting_caches or set()
    sha = pipeline_sha() if sha is None else sha
    jobs: list = []
    counts: collections.Counter = collections.Counter()

    for row in rows:
        mp3 = Path(row["mp3_path"])
        cached = report_path(data_dir, row["youtube_id"])
        sidecar = data_dir / FEATURES_DIR / f"{row['youtube_id']}.npz"
        try:
            stat = mp3.stat()
        except OSError:
            counts["missing_audio"] += 1
            continue

        if force:
            reason = "miss_forced"
        else:
            reason = _cache_miss_reason(cached, sidecar, sha, stat.st_size, stat.st_mtime)
            if reason is None:
                counts["hit"] += 1
                continue

        if not is_settled(mp3, now, min_age_sec):
            counts["too_recent"] += 1
            continue

        counts[reason] += 1
        jobs.append(SimJob(
            row["track_id"], row["youtube_id"], str(mp3),
            str(cached), str(sidecar),
            preexisting=tuple(path for path in derived_cache_paths(str(mp3))
                              if path in preexisting_caches),
            pipeline_sha=sha, mp3_size=stat.st_size, mp3_mtime=stat.st_mtime,
        ))
    jobs.sort(key=lambda job: job.track_id)
    return jobs, counts


def _cache_miss_reason(cached: Path, sidecar: Path, sha: str,
                       mp3_size: int, mp3_mtime: float) -> str | None:
    """``None`` when the cache may be used, else the counter name for the miss."""
    if not cached.exists():
        return "miss_new"
    try:
        envelope = _read_json_gz(cached)
    except (OSError, ValueError, EOFError):
        return "miss_unreadable"
    if cache_is_fresh(envelope, sha, mp3_size, mp3_mtime):
        return None
    if envelope.get("pipeline_sha") != sha:
        return "miss_pipeline_changed"
    if (envelope.get("mp3_size") != mp3_size
            or envelope.get("mp3_mtime") != mp3_mtime):
        return "miss_audio_changed"
    return "miss_stale_format"


def find_caches(data_dir: Path) -> set:
    audio_dir = data_dir / AUDIO_DIR
    if not audio_dir.exists():
        return set()
    return {str(path) for path in audio_dir.glob("*.npy")} | {
        str(path) for path in audio_dir.glob("*.mertcells.npz")}


_CORES_RESERVED_FOR_OS = 2


def default_workers() -> int:
    return max(1, (os.cpu_count() or 4) - _CORES_RESERVED_FOR_OS)


def run_simulations(jobs: list, workers: int, progress_every: int = 10) -> list:
    if not jobs:
        return []
    results = []
    started = time.time()
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            results.append(simulate_track(job))
            _print_progress(index, len(jobs), started, progress_every)
        return results

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(pool.map(simulate_track, jobs, chunksize=1),
                                           start=1):
                results.append(result)
                _print_progress(index, len(jobs), started, progress_every)
    except concurrent.futures.process.BrokenProcessPool as exc:
        print(f"  WARNING: worker pool broke after {len(results)}/{len(jobs)} "
              f"track(s): {exc}.  Re-run to continue -- cached reports are kept.",
              flush=True)
    return results


def _print_progress(done: int, total: int, started: float, every: int) -> None:
    if not every or (done % every and done != total):
        return
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    print(f"  simulated {done}/{total}  {elapsed / 60:.1f} min elapsed  "
          f"{rate * 60:.1f} tracks/min  ~{remaining / 60:.1f} min left", flush=True)


class TableStats(NamedTuple):
    tracks: int
    rows: int
    labels: collections.Counter
    intents: collections.Counter
    dropped: collections.Counter
    look_ahead_sec: set
    skipped: list
    missing_reports: list
    missing_sidecars: list


def build_table(data_dir: Path, rows: list, sections_by_track: dict) -> TableStats:
    path = data_dir / TABLE_FILE
    tmp = path.with_suffix(path.suffix + ".part")
    tracks = row_count = 0
    labels = collections.Counter()
    intents = collections.Counter()
    dropped = collections.Counter()
    look_ahead = set()
    skipped: list = []
    missing_reports: list = []
    missing_sidecars: list = []

    try:
        with open(tmp, "wb") as raw_file, \
                gzip.GzipFile(fileobj=raw_file, mode="wb", compresslevel=6,
                              filename="", mtime=_REPRODUCIBLE_GZIP_MTIME) as compressed, \
                io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(TABLE_HEADER)
            for row in rows:
                track_id = row["track_id"]
                cached = report_path(data_dir, row["youtube_id"])
                if not cached.exists():
                    missing_reports.append(track_id)
                    continue
                mel = data_dir / FEATURES_DIR / f"{row['youtube_id']}.npz"
                if (not mel.exists()
                        or sidecar_generation(mel) != MEL_EXPORTER_VERSION):
                    missing_sidecars.append(track_id)
                    continue
                sections = sections_by_track.get(track_id)
                if sections is None:
                    skipped.append((track_id, "no annotation record"))
                    continue
                try:
                    report = _read_json_gz(cached)["report"]
                except (OSError, ValueError, EOFError, KeyError, TypeError) as exc:
                    skipped.append((track_id, f"unreadable report: {exc}"))
                    continue

                joined, stats = join_track(track_id, row["youtube_id"], report, sections)
                look_ahead.add(float(report.get("metrics", {}).get("look_ahead_sec", 0.0)))
                tracks += 1
                row_count += len(joined)
                dropped["beats_total"] += stats.beats_total
                dropped["kept"] += stats.beats_kept
                dropped["leading"] += stats.dropped_leading
                dropped["gap"] += stats.dropped_gap
                dropped["trailing"] += stats.dropped_trailing
                dropped["in_dropped_section"] += stats.dropped_in_dropped_section
                dropped["without_intent"] += stats.beats_without_intent
                dropped["intent_blocks_song_stamped"] += stats.intent_blocks_song_stamped
                dropped["intent_blocks_song_recorded"] += \
                    stats.intent_blocks_song_recorded
                dropped["intent_blocks_late"] += stats.intent_blocks_late
                dropped["intent_reattributed"] += stats.intent_reattributed
                dropped["silence_blocks_leading"] += stats.silence_blocks_leading
                dropped["silence_blocks_interior"] += stats.silence_blocks_interior
                dropped["silence_blocks_trailing"] += stats.silence_blocks_trailing
                for joined_row in joined:
                    labels[joined_row["label"]] += 1
                    intents[joined_row["intent_at_beat"] or "(none)"] += 1
                    writer.writerow(format_row(joined_row))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return TableStats(tracks, row_count, labels, intents, dropped,
                      look_ahead, skipped, missing_reports, missing_sidecars)


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    return _git(repo_root, "rev-parse", "HEAD") or "unknown"


# .py only: a document beside the code cannot change what the simulation produces.
_PIPELINE_PATHSPEC = (":(glob)lib/**/*.py", ":(glob)simulate/**/*.py")


def pipeline_sha(repo_root: Path = REPO_ROOT) -> str:
    sha = _git(repo_root, "log", "-1", "--format=%H", "--", *_PIPELINE_PATHSPEC)
    if not sha:
        return "unknown"
    dirty = _git(repo_root, "status", "--porcelain", "--", *_PIPELINE_PATHSPEC)
    if not dirty:
        return sha
    diff = _git(repo_root, "diff", "HEAD", "--", *_PIPELINE_PATHSPEC) or ""
    digest = hashlib.sha256(f"{dirty}\n{diff}".encode("utf-8")).hexdigest()
    return f"{sha}+dirty.{digest[:12]}"


def sidecar_stats(data_dir: Path) -> tuple:
    features = data_dir / FEATURES_DIR
    if not features.exists():
        return 0, 0
    paths = list(features.glob("*.npz"))
    return len(paths), sum(path.stat().st_size for path in paths)


def write_meta(data_dir: Path, stats: TableStats, failures: list,
               elapsed_sec: float, cache_counts: collections.Counter | None = None,
               sha: str | None = None) -> Path:
    count, total_bytes = sidecar_stats(data_dir)
    meta = {
        "built_at": datetime.datetime.now(datetime.timezone.utc)
                            .replace(microsecond=0).isoformat(),
        "git_sha": git_sha(),
        "pipeline_sha": pipeline_sha() if sha is None else sha,
        "report_cache": dict(sorted((cache_counts or {}).items())),
        "build_wall_sec": round(elapsed_sec, 1),
        "schema": list(TABLE_HEADER),
        "tracks": stats.tracks,
        "rows": stats.rows,
        "look_ahead_sec": sorted(stats.look_ahead_sec),
        "class_histogram": _ordered_counts(stats.labels, SECTION_LABELS),
        "intent_histogram": _ordered_counts(stats.intents, ()),
        "dropped_beats": dict(sorted(stats.dropped.items())),
        "features": {
            "dir": FEATURES_DIR,
            "tracks": count,
            "total_bytes": total_bytes,
            "mel_bands": MEL_BANDS,
            "pool_buffers": POOL_BUFFERS,
            "frame_sec": POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE,
        },
        "failed_tracks": [{"track_id": t, "detail": d} for t, d in failures],
        "skipped_tracks": [{"track_id": t, "detail": d} for t, d in stats.skipped],
        "missing_reports": stats.missing_reports,
        "missing_sidecars": stats.missing_sidecars,
    }
    path = data_dir / META_FILE
    _write_json_pretty(path, meta)
    return path


def _write_json_pretty(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _ordered_counts(counter: collections.Counter, order: tuple) -> dict:
    known = [label for label in order if label in counter]
    extra = sorted(set(counter) - set(order), key=lambda label: (-counter[label], label))
    return {label: counter[label] for label in known + extra}


def print_report(stats: TableStats, results: list, table_path: Path,
                 meta_path: Path, data_dir: Path, elapsed: float,
                 new_caches: set, cache_counts: collections.Counter | None = None) -> None:
    failures = [result for result in results if not result.ok]
    count, total_bytes = sidecar_stats(data_dir)
    dropped = stats.dropped

    if cache_counts:
        print()
        print("report cache")
        print(f"  hits (no sim, no decode): {cache_counts['hit']}")
        misses = {key: value for key, value in sorted(cache_counts.items())
                  if key.startswith("miss_")}
        print(f"  misses                  : {sum(misses.values())}"
              + (f"  ({', '.join(f'{k[5:]} {v}' for k, v in misses.items())})"
                 if misses else ""))
        for key, caption in (("too_recent", "too recent to touch"),
                             ("missing_audio", "audio file missing")):
            if cache_counts[key]:
                print(f"  {caption:<24}: {cache_counts[key]}")

    print()
    print("training table")
    print(f"  tracks joined         : {stats.tracks}")
    print(f"  rows (labeled beats)  : {stats.rows}")
    print(f"  beats seen            : {dropped['beats_total']}")
    print(f"  dropped leading       : {dropped['leading']}  (before the first section)")
    print(f"  dropped gap           : {dropped['gap']}  (unlabeled interior)")
    print(f"  dropped trailing      : {dropped['trailing']}  (past the last section)")
    print(f"     of which in a dropped 'end' sentinel: {dropped['in_dropped_section']}")
    print(f"  rows without an intent: {dropped['without_intent']}")
    print(f"  look_ahead_sec        : {sorted(stats.look_ahead_sec)}")
    print(f"  song-stamped intent blocks realigned: "
          f"{dropped['intent_blocks_song_stamped']}  "
          f"(rows re-attributed: {dropped['intent_reattributed']})")
    print(f"  intent blocks committed late: {dropped['intent_blocks_late']} of "
          f"{dropped['intent_blocks_song_recorded']} eligible  "
          f"(#154's accepted lateness -- the chain was older than the "
          f"playback delay; only a block that RECORDED its instant can be "
          f"measured, so a zero denominator means the reports predate it)")
    print(f"  tracks with no cached report : {len(stats.missing_reports)}"
          + (f"  {stats.missing_reports[:10]}" if stats.missing_reports else ""))
    print(f"  tracks skipped, no sidecar   : {len(stats.missing_sidecars)}"
          + (f"  {stats.missing_sidecars[:10]}" if stats.missing_sidecars else ""))

    print()
    print("class histogram")
    _print_histogram(stats.labels, stats.rows)
    print()
    print("committed intent at beat")
    _print_histogram(stats.intents, stats.rows)

    print()
    print("mel feature sidecars")
    print(f"  tracks    : {count}")
    print(f"  total size: {total_bytes / 2**20:.1f} MiB")
    print(f"  frame     : {MEL_BANDS} bands, "
          f"{POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE * 1000:.2f} ms hop")

    if failures:
        print()
        print(f"failed tracks ({len(failures)}):")
        for result in failures[:40]:
            print(f"  {result.track_id:<20} {result.detail}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more (see {META_FILE})")
    if stats.skipped:
        print()
        print(f"skipped at join time ({len(stats.skipped)}):")
        for track_id, detail in stats.skipped[:40]:
            print(f"  {track_id:<20} {detail}")

    print()
    print(f"decode caches left in {data_dir / AUDIO_DIR}: "
          f"{len(new_caches)} new" + (f" -- {sorted(new_caches)}" if new_caches else ""))
    print(f"wall time: {elapsed / 60:.1f} min")
    print()
    print(f"table: {table_path}  ({table_path.stat().st_size / 2**20:.1f} MiB)")
    print(f"meta : {meta_path}")


def _print_histogram(counter: collections.Counter, total: int) -> None:
    for label, count in counter.most_common():
        share = 100.0 * count / total if total else 0.0
        print(f"  {label or '(none)':<14}{count:>10}{share:>8.1f}%")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir(),
        help="corpus root; reads clean_manifest.csv + annotations/, writes "
             "training_table.csv.gz, reports/ and features/ (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=default_workers(),
        help="parallel simulation workers (default: %(default)s = cpu_count - 2)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="simulate at most N tracks (smoke run; 0 = no limit)",
    )
    parser.add_argument(
        "--min-age-sec", type=float, default=MIN_AGE_SEC,
        help="skip audio written more recently than this -- the corpus "
             "downloader may still be running (default: %(default)s)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="ignore the report cache and re-simulate every track",
    )
    parser.add_argument(
        "--table-only", action="store_true",
        help="skip the simulations and re-join the cached reports on disk",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    started = time.time()
    sha = pipeline_sha()
    print("label-aligned training table")
    print(f"data dir : {data_dir}")
    print(f"repo     : {git_sha()}")
    print(f"pipeline : {sha}  (lib/ + simulate/ -- the report cache key)")

    rows = load_ok_rows(data_dir)
    sections_by_track = load_sections_by_track(data_dir)
    print(f"clean manifest: {len(rows)} ok track(s); "
          f"{len(sections_by_track)} annotated track(s)")

    preexisting_caches = find_caches(data_dir)
    if preexisting_caches:
        print(f"NOTE: {len(preexisting_caches)} decode cache(s) predate this run "
              f"and will be left in place")

    results: list = []
    cache_counts: collections.Counter = collections.Counter()
    if args.table_only:
        print("NOTE: --table-only -- no simulations, re-joining cached reports")
    else:
        jobs, cache_counts = select_jobs(
            rows, data_dir, force=args.force, min_age_sec=args.min_age_sec,
            preexisting_caches=preexisting_caches, sha=sha,
        )
        if args.limit:
            jobs = jobs[: args.limit]
            print(f"NOTE: --limit {args.limit} -- simulating a subset only")
        print(f"stage A: {len(jobs)} to simulate, {cache_counts['hit']} cache hit(s)",
              flush=True)
        if jobs:
            print(f"  {args.workers} worker(s)", flush=True)
            results = run_simulations(jobs, workers=args.workers)

    print("stage B: joining beats to labels ...", flush=True)
    stats = build_table(data_dir, rows, sections_by_track)
    elapsed = time.time() - started
    failures = [(result.track_id, result.detail) for result in results if not result.ok]
    meta_path = write_meta(data_dir, stats, failures, elapsed, cache_counts, sha)

    new_caches = find_caches(data_dir) - preexisting_caches
    print_report(stats, results, data_dir / TABLE_FILE, meta_path, data_dir,
                 elapsed, new_caches, cache_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
