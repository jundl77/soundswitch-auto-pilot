#!/usr/bin/env python
"""Cleanliness gate over the raveform corpus: decode every downloaded track, classify it."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raveform_manifest import (  # noqa: E402
    MANIFEST_FILE,
    build_manifest_rows,
)
from raveform_fetch_annotations import load_all_tracks  # noqa: E402

AUDIO_DIR = "audio"
CLEAN_MANIFEST_FILE = "clean_manifest.csv"
CLEAN_MANIFEST_HEADER = (
    "track_id",
    "youtube_id",
    "mp3_path",
    "ffprobe_duration_sec",
    "decoded_duration_sec",
    "annotation_duration_sec",
    "status",
    "detail",
)

STATUS_OK = "ok"
STATUS_MISMATCH = "duration_mismatch"
STATUS_CORRUPT = "corrupt"
STATUS_ORDER = (STATUS_OK, STATUS_MISMATCH, STATUS_CORRUPT)

MIN_AGE_SEC = 60.0

ABS_TOLERANCE_SEC = 10.0
REL_TOLERANCE = 0.03

DEFAULT_WORKERS = 8

FFMPEG_TIMEOUT_SEC = 300


class ManifestRow(NamedTuple):
    track_id: str
    youtube_id: str
    annotation_duration_sec: float


class TrackJob(NamedTuple):
    track_id: str
    youtube_id: str
    mp3_path: str
    annotation_duration_sec: float


class CheckResult(NamedTuple):
    track_id: str
    youtube_id: str
    mp3_path: str
    ffprobe_duration_sec: float | None
    decoded_duration_sec: float | None
    annotation_duration_sec: float
    status: str
    detail: str


def duration_tolerance(annotation_duration_sec: float) -> float:
    return max(ABS_TOLERANCE_SEC, REL_TOLERANCE * abs(annotation_duration_sec))


def _unusable(duration_sec: float | None) -> bool:
    return (
        duration_sec is None
        or not math.isfinite(duration_sec)
        or duration_sec <= 0.0
    )


def classify(
    decode_error: str,
    decoded_duration_sec: float | None,
    header_duration_sec: float | None,
    annotation_duration_sec: float,
) -> tuple[str, str]:
    if decode_error:
        return STATUS_CORRUPT, decode_error
    if _unusable(decoded_duration_sec):
        return STATUS_CORRUPT, f"decoder produced no audio (decoded {decoded_duration_sec!r})"
    if _unusable(header_duration_sec):
        return STATUS_CORRUPT, f"ffprobe reported no usable duration ({header_duration_sec!r})"

    tolerance = duration_tolerance(annotation_duration_sec)

    shortfall = decoded_duration_sec - header_duration_sec
    if abs(shortfall) > tolerance:
        return (
            STATUS_CORRUPT,
            f"truncated: header claims {header_duration_sec:.3f} s but the decoder "
            f"produced {decoded_duration_sec:.3f} s ({shortfall:+.3f} s, "
            f"tolerance {tolerance:.3f} s)",
        )

    delta = decoded_duration_sec - annotation_duration_sec
    if abs(delta) > tolerance:
        return (
            STATUS_MISMATCH,
            f"decoded {decoded_duration_sec:.3f} s vs annotation "
            f"{annotation_duration_sec:.3f} s (delta {delta:+.3f} s, "
            f"tolerance {tolerance:.3f} s)",
        )
    return STATUS_OK, ""


def _run(command: list) -> subprocess.CompletedProcess:
    # utf-8/replace, not the locale: ffmpeg echoes the offending track's tag text,
    # and one non-cp1252 byte would raise UnicodeDecodeError inside a pool worker.
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=FFMPEG_TIMEOUT_SEC,
    )


def decode(path: str) -> tuple[str, float | None]:
    # An mp3 truncated on a frame boundary decodes clean and still probes at its
    # full header length, so only the decoder's own emitted duration detects it.
    # `-progress -` writes that to stdout; the human `-stats` line would land on
    # stderr and break the "empty stderr means clean" test.
    try:
        proc = _run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-progress", "-",
                "-i", path, "-f", "null", "-",
            ]
        )
    except subprocess.TimeoutExpired:
        return f"ffmpeg timed out after {FFMPEG_TIMEOUT_SEC} s", None
    except OSError as exc:
        return f"ffmpeg could not run: {exc}", None

    complaint = proc.stderr.strip()
    if proc.returncode != 0:
        return complaint or f"ffmpeg exited {proc.returncode}", None
    return complaint, _decoded_seconds(proc.stdout)


def _decoded_seconds(progress_output: str) -> float | None:
    microseconds = None
    for line in progress_output.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "out_time_us":
            try:
                microseconds = int(value.strip())
            except ValueError:
                continue
    return None if microseconds is None else microseconds / 1e6


def probe_duration(path: str) -> float | None:
    try:
        proc = _run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nokey=1:noprint_wrappers=1",
                path,
            ]
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def check_track(job: TrackJob) -> CheckResult:
    error, decoded = decode(job.mp3_path)
    header = probe_duration(job.mp3_path)
    status, detail = classify(error, decoded, header, job.annotation_duration_sec)
    return CheckResult(
        job.track_id,
        job.youtube_id,
        job.mp3_path,
        header,
        decoded,
        job.annotation_duration_sec,
        status,
        _first_line(detail),
    )


def _first_line(text: str, limit: int = 200) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line[:limit]


def audio_path(data_dir: Path, youtube_id: str) -> Path:
    return data_dir / AUDIO_DIR / f"{youtube_id}.mp3"


def load_manifest_rows(data_dir: Path) -> list:
    path = data_dir / MANIFEST_FILE
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows = [
                ManifestRow(row["track_id"], row["youtube_id"], float(row["total_sec"]))
                for row in csv.DictReader(handle)
            ]
    else:
        rows = [
            ManifestRow(track_id, youtube_id, float(total_sec))
            for track_id, youtube_id, _n_sections, total_sec in build_manifest_rows(
                load_all_tracks(data_dir)
            )
        ]
    if not rows:
        raise RuntimeError(
            f"no tracks in {path} -- run training/raveform/raveform_manifest.py first"
        )
    return rows


def is_settled(path: Path, now: float, min_age_sec: float = MIN_AGE_SEC) -> bool:
    # The downloader renames its .part into place and a rename carries the mtime
    # over, so recency -- not the extension -- is the guard that works.
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    return (now - mtime) > min_age_sec


def select_candidates(
    rows: list,
    data_dir: Path,
    now: float | None = None,
    min_age_sec: float = MIN_AGE_SEC,
) -> tuple[list, int, int]:
    now = time.time() if now is None else now
    jobs = []
    missing = 0
    too_recent = 0
    for row in rows:
        path = audio_path(data_dir, row.youtube_id)
        if not path.exists():
            missing += 1
        elif not is_settled(path, now, min_age_sec):
            too_recent += 1
        else:
            jobs.append(
                TrackJob(
                    row.track_id,
                    row.youtube_id,
                    str(path),
                    row.annotation_duration_sec,
                )
            )
    jobs.sort(key=lambda job: job.track_id)
    return jobs, missing, too_recent


def run_checks(jobs: list, workers: int = DEFAULT_WORKERS, progress_every: int = 50) -> list:
    if not jobs:
        return []

    results = []
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            results.append(check_track(job))
            _print_progress(index, len(jobs), progress_every)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(pool.map(check_track, jobs), start=1):
                results.append(result)
                _print_progress(index, len(jobs), progress_every)

    results.sort(key=lambda result: result.track_id)
    return results


def _print_progress(done: int, total: int, every: int) -> None:
    if every and (done % every == 0 or done == total):
        print(f"  checked {done}/{total}", flush=True)


def summarise(results: list) -> collections.Counter:
    return collections.Counter(result.status for result in results)


def _format_duration(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def write_clean_manifest(data_dir: Path, results: list) -> Path:
    path = data_dir / CLEAN_MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CLEAN_MANIFEST_HEADER)
            for result in sorted(results, key=lambda item: item.track_id):
                writer.writerow(
                    (
                        result.track_id,
                        result.youtube_id,
                        result.mp3_path,
                        _format_duration(result.ffprobe_duration_sec),
                        _format_duration(result.decoded_duration_sec),
                        _format_duration(result.annotation_duration_sec),
                        result.status,
                        result.detail,
                    )
                )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def print_report(
    results: list,
    total_tracks: int,
    missing: int,
    too_recent: int,
    elapsed: float,
    max_listed: int = 40,
) -> None:
    counts = summarise(results)
    print()
    print("cleanliness gate")
    print(f"  tracks in manifest        : {total_tracks}")
    print(f"  not yet downloaded        : {missing}")
    print(f"  too recent to touch       : {too_recent}  (mtime younger than the min age)")
    print(f"  checked                   : {len(results)}  in {elapsed:.1f} s")
    for status in STATUS_ORDER:
        print(f"    {status:<24}: {counts.get(status, 0)}")

    for status in (STATUS_CORRUPT, STATUS_MISMATCH):
        rejected = [result for result in results if result.status == status]
        if not rejected:
            continue
        print()
        print(f"{status} ({len(rejected)}):")
        for result in rejected[:max_listed]:
            print(f"  {result.track_id:<20} {result.detail}")
        if len(rejected) > max_listed:
            print(f"  ... and {len(rejected) - max_listed} more (see {CLEAN_MANIFEST_FILE})")


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "training" / "data" / "raveform"


def require_tools(tools: tuple = ("ffmpeg", "ffprobe")) -> None:
    # Without this, a missing binary reads as a corpus of unreadable files.
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} not found on PATH -- the cleanliness gate "
            f"cannot check anything without it"
        )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="corpus root; reads <data-dir>/manifest.csv + audio/, writes "
        "<data-dir>/clean_manifest.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="parallel ffmpeg checks (default: %(default)s)",
    )
    parser.add_argument(
        "--min-age-sec",
        type=float,
        default=MIN_AGE_SEC,
        help="skip audio files written more recently than this -- the corpus "
        "downloader may still be running (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="check at most N tracks (smoke test; 0 = no limit)",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    print("raveform cleanliness gate")
    print(f"data dir: {data_dir}")
    print(f"audio   : {data_dir / AUDIO_DIR}")

    require_tools()
    rows = load_manifest_rows(data_dir)
    jobs, missing, too_recent = select_candidates(rows, data_dir, min_age_sec=args.min_age_sec)
    if args.limit:
        jobs = jobs[: args.limit]
        print(f"NOTE: --limit {args.limit} -- checking a subset only")

    print(f"checking {len(jobs)} track(s) with {args.workers} worker(s)...", flush=True)
    started = time.time()
    results = run_checks(jobs, workers=args.workers)
    elapsed = time.time() - started

    path = write_clean_manifest(data_dir, results)
    print_report(results, len(rows), missing, too_recent, elapsed)
    print()
    print(f"clean manifest: {path}")
    print(f"  columns : {','.join(CLEAN_MANIFEST_HEADER)}")
    print(f"  rows    : {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
