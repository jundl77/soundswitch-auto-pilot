#!/usr/bin/env python
"""Fetch the third-party queue from YouTube, one video at a time.

``raveform_download`` for the quarantine, and deliberately the same shape: one
sequential sweep, a pause between videos, no cookie or credential workaround
ever, a refusal recorded rather than worked around, and a run of consecutive
refusals aborting instead of burning the work list into failure records.  The
failure classifier and ``RETRYABLE_REASONS`` are IMPORTED from it rather than
copied -- a drifted copy of that table is a bug this project has already nearly
shipped once (see CLAUDE.md), and a permanent reason must outrank a transient
one on both sides of the import or a dead video is re-polled forever.

Two things differ from raveform, both forced by the queue's key.  The unit here
is ``track_id``, not the video id: 28 Harmonix videos are shared by two
annotated tracks each, so an id-keyed store would silently drop the second of
every pair.  That also rules out yt-dlp's ``--download-archive``, which is
keyed on the video id and would refuse the twin; ``downloaded.txt`` is written
here instead, after the mp3 is confirmed non-empty on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_TRAINING_DIR.parent), str(_TRAINING_DIR),
              str(_TRAINING_DIR / "raveform"), str(_TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tp_fetch_queue  # noqa: E402
import tp_record  # noqa: E402
from corpus_root import corpus_dir  # noqa: E402
from raveform_download import (  # noqa: E402
    BLOCK_REASONS,
    INTERRUPT_REASON,
    KNOWN_REASONS,
    RETRYABLE_REASONS,
    classify_error,
    format_duration,
    install_interrupt_handler,
    interrupt_requested,
)

ARCHIVE_FILE = "downloaded.txt"
FAILED_FILE = "failed.jsonl"
AUDIO_EXT = "mp3"

DEFAULT_TIMEOUT_SEC = 600
DEFAULT_PAUSE_SEC = 2.0
PAUSE_JITTER = 2.5

RETRY_HINT = "--retry-reasons " + ",".join(sorted(RETRYABLE_REASONS))

_STDERR_TAIL_CHARS = 2000
_PROGRESS_EVERY = 10


def third_party_dir(corpus: Path) -> Path:
    return tp_record.third_party_dir(Path(corpus))


def archive_path(corpus: Path) -> Path:
    return third_party_dir(corpus) / ARCHIVE_FILE


def failed_path(corpus: Path) -> Path:
    return third_party_dir(corpus) / FAILED_FILE


def audio_file(corpus: Path, source: str, track_id: str) -> Path:
    return tp_fetch_queue.audio_path(corpus, source, track_id)


def read_archive_ids(path: Path) -> set:
    path = Path(path)
    if not path.exists():
        return set()
    ids: set = set()
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                ids.add(parts[-1])
    return ids


def append_archive(path: Path, track_id: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(track_id + "\n")
        handle.flush()


def read_failed_reasons(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    reasons: dict = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            value = record.get("track_id")
            if isinstance(value, str) and value:
                reason = record.get("reason")
                reasons[value] = reason if isinstance(reason, str) else "other"
    return reasons


def append_failure(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()


def build_command(corpus: Path, row) -> list:
    target = audio_file(corpus, row.source, row.track_id)
    # The `--` separator is required: a YouTube id may start with `-`.
    return [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", AUDIO_EXT,
        "--audio-quality", "192K",
        "--no-playlist",
        "--retries", "3",
        "--socket-timeout", "30",
        "-o", str(target.with_suffix(".%(ext)s")),
        "--",
        row.youtube_id,
    ]


def _tail(text: str, limit: int = _STDERR_TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def download_one(corpus: Path, row, timeout_sec: int) -> tuple:
    target = audio_file(corpus, row.source, row.track_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            build_command(corpus, row),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        if interrupt_requested():
            return False, INTERRUPT_REASON, "cancelled by SIGINT"
        return False, "timeout", f"yt-dlp exceeded the {timeout_sec}s budget"
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found on PATH") from None

    if completed.returncode != 0:
        if interrupt_requested():
            return False, INTERRUPT_REASON, "cancelled by SIGINT"
        blob = _tail(completed.stderr) or _tail(completed.stdout) or "(no output)"
        return False, classify_error(blob), blob

    if not target.exists():
        return (False, "missing_output",
                _tail(completed.stdout) or "yt-dlp exited 0 but wrote no mp3")
    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        return False, "empty_output", "yt-dlp exited 0 but wrote a zero-byte mp3 (deleted)"
    return True, "", ""


def print_progress(done: int, failed: int, remaining: int, elapsed: float) -> None:
    attempted = done + failed
    per_track = elapsed / attempted if attempted else 0.0
    eta = per_track * remaining if attempted else float("nan")
    print(
        f"  progress: done {done}  failed {failed}  remaining {remaining}  "
        f"({per_track:.1f}s/track, elapsed {format_duration(elapsed)}, "
        f"ETA {format_duration(eta)})",
        flush=True,
    )


def _reason_line(reasons: dict) -> str:
    return ", ".join(f"{reason} x{count}" for reason, count in sorted(reasons.items()))


def _tally(id_to_reason: dict) -> dict:
    counts: dict = {}
    for reason in id_to_reason.values():
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root (default: the resolved corpus dir)")
    parser.add_argument("--queue", type=Path, default=None,
                        help="fetch queue csv "
                             f"(default: <corpus>/third_party/{tp_fetch_queue.QUEUE_FILE})")
    parser.add_argument("--limit", type=int, default=0,
                        help="attempt at most N pending tracks; 0 = all "
                             "(default: %(default)s)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-attempt every track in failed.jsonl, whatever the reason")
    parser.add_argument("--retry-reasons", default="", metavar="R1,R2",
                        help="re-attempt only the failures whose recorded reason is in "
                             "this comma list; the rest stay skipped. Worth retrying: "
                             f"{','.join(sorted(RETRYABLE_REASONS))}. "
                             f"Known: {','.join(sorted(KNOWN_REASONS))}")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE_SEC,
                        help="minimum pause between videos, seconds; the actual wait is "
                             f"jittered up to {PAUSE_JITTER:g}x it (default: %(default)s)")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="stop after this many consecutive sign-in/rate-limit "
                             "refusals; 0 disables the guard (default: %(default)s)")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC,
                        help="per-track wall-clock budget for yt-dlp (default: %(default)s)")
    args = parser.parse_args(argv)

    if args.pause < 0:
        parser.error("--pause must not be negative")

    retry_reasons = {part.strip() for part in args.retry_reasons.split(",") if part.strip()}
    unknown = sorted(retry_reasons - KNOWN_REASONS)
    if unknown:
        parser.error(
            f"unknown --retry-reasons value(s): {', '.join(unknown)}; "
            f"known reasons are {', '.join(sorted(KNOWN_REASONS))}")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    corpus = (args.data_dir or corpus_dir()).resolve()
    queue_file = (args.queue or tp_fetch_queue.queue_path(corpus)).resolve()

    rows = tp_fetch_queue.read_queue(queue_file)
    archived = read_archive_ids(archive_path(corpus))
    previously_failed = read_failed_reasons(failed_path(corpus))
    third_party_dir(corpus).mkdir(parents=True, exist_ok=True)
    install_interrupt_handler()

    retried = {
        track_id
        for track_id, reason in previously_failed.items()
        if args.retry_failed or reason in retry_reasons
    }
    skip = set(archived) | (set(previously_failed) - retried)
    pending = [
        row for row in rows
        if row.track_id not in skip
        and not audio_file(corpus, row.source, row.track_id).exists()
    ]
    if args.limit > 0:
        pending = pending[: args.limit]

    by_source: dict = {}
    for row in pending:
        by_source[row.source] = by_source.get(row.source, 0) + 1

    print("third-party audio download")
    print(f"corpus    : {corpus}")
    print(f"queue     : {queue_file}")
    print(f"audio dir : {third_party_dir(corpus) / tp_record.AUDIO_DIR}")
    print(f"archive   : {archive_path(corpus)}")
    print(f"failures  : {failed_path(corpus)}")
    if args.retry_failed:
        retry_note = "all being retried (--retry-failed)"
    elif retry_reasons:
        retry_note = (
            f"{len(retried)} being retried ({','.join(sorted(retry_reasons))}), "
            f"{len(previously_failed) - len(retried)} still skipped")
    else:
        retry_note = "skipped"
    print(f"queue     : {len(rows)} track(s) | already downloaded {len(archived)} | "
          f"previously failed {len(previously_failed)}  ({retry_note})")
    if previously_failed and not args.retry_failed:
        print(f"            failure reasons on record: "
              f"{_reason_line(_tally(previously_failed))}")
    print(f"this run  : {len(pending)} track(s)"
          + (f" (--limit {args.limit})" if args.limit > 0 else "")
          + f" [{_reason_line(by_source) or 'none'}]"
          + f", pausing {args.pause:g}-{args.pause * PAUSE_JITTER:g}s between videos")
    print()

    done = 0
    failed = 0
    consecutive_blocks = 0
    reasons: dict = {}
    started = time.monotonic()
    interrupted = False
    aborted = False

    for index, row in enumerate(pending, start=1):
        if interrupt_requested():
            interrupted = True
            break

        if index > 1:
            delay = random.Random(row.track_id).uniform(
                args.pause, args.pause * PAUSE_JITTER)
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                interrupted = True
                break

        try:
            ok, reason, error = download_one(corpus, row, args.timeout_sec)
        except KeyboardInterrupt:
            interrupted = True
            break

        if not ok and reason == INTERRUPT_REASON:
            interrupted = True
            print(f"[{index}/{len(pending)}] cancelled  {row.track_id}  (not recorded)",
                  flush=True)
            break

        if ok:
            done += 1
            consecutive_blocks = 0
            append_archive(archive_path(corpus), row.track_id)
            size_mb = audio_file(corpus, row.source, row.track_id).stat().st_size / (1 << 20)
            print(f"[{index}/{len(pending)}] ok    {row.track_id}  {size_mb:.1f} MB",
                  flush=True)
        else:
            failed += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            append_failure(failed_path(corpus), {
                "track_id": row.track_id,
                "source": row.source,
                "youtube_id": row.youtube_id,
                "error": error,
                "reason": reason,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            })
            last_line = error.splitlines()[-1] if error else ""
            print(f"[{index}/{len(pending)}] FAIL  {row.track_id}  [{reason}] "
                  f"{last_line[:120]}", flush=True)
            if reason in BLOCK_REASONS:
                consecutive_blocks += 1
                if 0 < args.max_consecutive_failures <= consecutive_blocks:
                    aborted = True
                    break
            else:
                consecutive_blocks = 0

        if index % _PROGRESS_EVERY == 0:
            print_progress(done, failed, len(pending) - index, time.monotonic() - started)

    elapsed = time.monotonic() - started
    attempted = done + failed

    print()
    print("summary")
    print(f"  attempted   : {attempted}/{len(pending)} planned this run")
    print(f"  downloaded  : {done}")
    print(f"  failed      : {failed}" + (f"  ({_reason_line(reasons)})" if reasons else ""))
    print(f"  elapsed     : {format_duration(elapsed)}")
    if attempted:
        print(f"  mean        : {elapsed / attempted:.1f}s per attempted track")
    on_disk = read_archive_ids(archive_path(corpus))
    remaining_total = sum(1 for row in rows if row.track_id not in on_disk)
    print(f"  queue       : {len(rows) - remaining_total}/{len(rows)} tracks on disk")
    if attempted and remaining_total:
        print(f"  full-run ETA: "
              f"{format_duration(elapsed / attempted * remaining_total)} "
              f"for the remaining {remaining_total}")

    recoverable = sorted(reason for reason in reasons if reason in RETRYABLE_REASONS)
    if recoverable:
        print()
        print(f"  RECOVERABLE : {sum(reasons[reason] for reason in recoverable)} of this "
              f"run's failures may not be permanent ({', '.join(recoverable)}).")
        print(f"                A plain re-run SKIPS them. To re-attempt:  {RETRY_HINT}")

    blocks = reasons.get("bot_check", 0)
    if blocks:
        print()
        print(f"  BOT CHECK   : {blocks} sign-in / rate-limit refusal(s). No cookie or "
              "credential workaround was attempted -- this is an owner decision.")
    forbidden = reasons.get("http_403", 0)
    if forbidden:
        print()
        print(f"  HTTP 403    : {forbidden} refusal(s) on the media URL -- a signature "
              "challenge yt-dlp could not solve. These are NOT dead videos;")
        print(f"                re-run with  {RETRY_HINT}  and let it work through them.")
    if aborted:
        print(f"  ABORTED     : {consecutive_blocks} consecutive refusals hit the "
              "--max-consecutive-failures guard.")
        print("                Those tracks are now in failed.jsonl, so a PLAIN RE-RUN "
              "WILL SKIP THEM. To bring them back:")
        print(f"                {RETRY_HINT}")
        return 2
    if interrupted:
        print("  INTERRUPTED : Ctrl-C. State on disk is resumable; re-run to continue.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
