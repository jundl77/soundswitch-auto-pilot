#!/usr/bin/env python
"""Download the raveform corpus audio from YouTube, one track at a time."""

from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

MANIFEST_FILE = "manifest.csv"
MANIFEST_ID_COLUMN = "youtube_id"
AUDIO_DIR = "audio"
ARCHIVE_FILE = "downloaded.txt"
FAILED_FILE = "failed.jsonl"

AUDIO_EXT = "mp3"

DEFAULT_TIMEOUT_SEC = 600

_STDERR_TAIL_CHARS = 2000

_PROGRESS_EVERY = 10


# Substring matching, first match wins, and the order encodes one rule: a
# PERMANENT condition outranks a TRANSIENT one.  YouTube's permanent errors
# carry transient-looking text ("Sign in to confirm your age", "Use
# --cookies-from-browser"), so a transient bucket placed above them swallows a
# dead video, re-polls it forever and trips --max-consecutive-blocks.
_PERMANENT_PATTERNS = (
    ("unavailable", ("private video", "this video is private")),
    (
        "age_restricted",
        (
            "age-restricted",
            "age restricted",
            "sign in to confirm your age",
            "inappropriate for some users",
        ),
    ),
    ("geo_blocked", ("not available in your country", "geo restricted", "geo-restricted")),
)

_TRANSIENT_PATTERNS = (
    (
        "bot_check",
        (
            "confirm you're not a bot",
            "confirm you\u2019re not a bot",
            "confirm you are not a bot",
            "sign in to confirm",
            "please sign in",
            "use --cookies",
            "--cookies-from-browser",
            "http error 429",
            "too many requests",
        ),
    ),
    ("http_403", ("http error 403", "403: forbidden")),
)

_REASON_PATTERNS = _PERMANENT_PATTERNS + _TRANSIENT_PATTERNS + (
    (
        "unavailable",
        (
            "video unavailable",
            "this video is not available",
            "private video",
            "this video is private",
            "has been removed",
            "account associated with this video has been terminated",
            "video has been removed",
            "unavailable in your",
            "no longer available",
            "does not exist",
        ),
    ),
    ("copyright", ("copyright",)),
)

INTERRUPT_REASON = "interrupted"

BLOCK_REASONS = frozenset(reason for reason, _patterns in _TRANSIENT_PATTERNS)

KNOWN_REASONS = frozenset(
    {reason for reason, _patterns in _REASON_PATTERNS}
    | {"other", "timeout", "missing_output", "empty_output"}
)

RETRYABLE_REASONS = frozenset(
    BLOCK_REASONS | {"timeout", "empty_output", "missing_output", "other"}
)

RETRY_HINT = "--retry-reasons " + ",".join(sorted(RETRYABLE_REASONS))


def classify_error(text: str) -> str:
    lowered = text.lower()
    for reason, patterns in _REASON_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return reason
    return "other"


_interrupt_requested = False


def interrupt_requested() -> bool:
    return _interrupt_requested


def request_interrupt() -> None:
    global _interrupt_requested
    _interrupt_requested = True


def install_interrupt_handler() -> None:
    def _handler(_signum, _frame):
        request_interrupt()

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError, AttributeError):
        pass


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "training" / "data" / "raveform"


def manifest_path(data_dir: Path) -> Path:
    return data_dir / MANIFEST_FILE


def audio_dir(data_dir: Path) -> Path:
    return data_dir / AUDIO_DIR


def archive_path(data_dir: Path) -> Path:
    return data_dir / ARCHIVE_FILE


def failed_path(data_dir: Path) -> Path:
    return data_dir / FAILED_FILE


def audio_file(data_dir: Path, youtube_id: str) -> Path:
    return audio_dir(data_dir) / f"{youtube_id}.{AUDIO_EXT}"


def read_manifest_ids(data_dir: Path) -> list[str]:
    path = manifest_path(data_dir)
    if not path.exists():
        raise RuntimeError(f"no manifest at {path} -- run raveform_manifest.py first")
    ids: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or MANIFEST_ID_COLUMN not in reader.fieldnames:
            raise RuntimeError(
                f"{path} has no {MANIFEST_ID_COLUMN!r} column (found: {reader.fieldnames})"
            )
        for row in reader:
            value = (row.get(MANIFEST_ID_COLUMN) or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ids.append(value)
    return ids


def read_archive_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                ids.add(parts[-1])
    return ids


def forget_download(data_dir: Path, youtube_id: str) -> bool:
    # yt-dlp consults its archive before anything else, so an archived id whose
    # mp3 is gone can never be re-fetched until its line is removed.
    path = archive_path(data_dir)
    if not path.exists():
        return False
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    kept = [line for line in lines if line.split()[-1:] != [youtube_id]]
    if len(kept) == len(lines):
        return False
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(kept)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


def read_failed_reasons(path: Path) -> dict:
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
            value = record.get("youtube_id")
            if isinstance(value, str) and value:
                reason = record.get("reason")
                reasons[value] = reason if isinstance(reason, str) else "other"
    return reasons


def append_failure(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()


def build_command(data_dir: Path, youtube_id: str) -> list[str]:
    # The `--` separator is required: a YouTube id may start with `-` and would
    # otherwise be parsed as an option.
    return [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", AUDIO_EXT,
        "--audio-quality", "192K",
        "--no-playlist",
        "--retries", "3",
        "--socket-timeout", "30",
        "--download-archive", str(archive_path(data_dir)),
        "-o", str(audio_dir(data_dir) / "%(id)s.%(ext)s"),
        "--",
        youtube_id,
    ]


def _tail(text: str, limit: int = _STDERR_TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def download_one(data_dir: Path, youtube_id: str, timeout_sec: int) -> tuple[bool, str, str]:
    command = build_command(data_dir, youtube_id)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        if interrupt_requested():
            return False, INTERRUPT_REASON, "cancelled by SIGINT"
        return False, "timeout", f"yt-dlp exceeded the {timeout_sec}s per-track budget"
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found on PATH -- see the Task 1 tooling report") from None

    if completed.returncode != 0:
        if interrupt_requested():
            return False, INTERRUPT_REASON, "cancelled by SIGINT"
        blob = _tail(completed.stderr) or _tail(completed.stdout) or "(no output)"
        return False, classify_error(blob), blob

    target = audio_file(data_dir, youtube_id)
    if not target.exists():
        forget_download(data_dir, youtube_id)
        return (
            False,
            "missing_output",
            _tail(completed.stdout) or "yt-dlp exited 0 but no mp3 was written",
        )

    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        forget_download(data_dir, youtube_id)
        return False, "empty_output", "yt-dlp exited 0 but wrote a zero-byte mp3 (deleted)"

    return True, "", ""


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "?"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


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


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="corpus root; reads <data-dir>/manifest.csv, writes <data-dir>/audio "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="attempt at most N pending tracks; 0 = all (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep-min",
        type=float,
        default=2.0,
        help="minimum pause between downloads, seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep-max",
        type=float,
        default=5.0,
        help="maximum pause between downloads, seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-attempt every id in failed.jsonl, whatever the reason",
    )
    parser.add_argument(
        "--retry-reasons",
        default="",
        metavar="R1,R2",
        help="re-attempt only the failures whose recorded reason is in this comma list; "
        f"the rest stay skipped. To retry everything that is worth retrying, use "
        f"{','.join(sorted(RETRYABLE_REASONS))} -- the reasons that describe this run "
        f"rather than the video. Known: {','.join(sorted(KNOWN_REASONS))}",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="per-track wall-clock budget for yt-dlp (default: %(default)s)",
    )
    parser.add_argument(
        "--max-consecutive-blocks",
        type=int,
        default=5,
        help="stop after this many consecutive sign-in/rate-limit refusals; "
        "0 disables the guard (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.sleep_min < 0 or args.sleep_max < args.sleep_min:
        parser.error("need 0 <= --sleep-min <= --sleep-max")

    retry_reasons = {part.strip() for part in args.retry_reasons.split(",") if part.strip()}
    unknown = sorted(retry_reasons - KNOWN_REASONS)
    if unknown:
        parser.error(
            f"unknown --retry-reasons value(s): {', '.join(unknown)}; "
            f"known reasons are {', '.join(sorted(KNOWN_REASONS))}"
        )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    data_dir = args.data_dir.resolve()

    all_ids = read_manifest_ids(data_dir)
    archived = read_archive_ids(archive_path(data_dir))
    previously_failed = read_failed_reasons(failed_path(data_dir))
    audio_dir(data_dir).mkdir(parents=True, exist_ok=True)
    install_interrupt_handler()

    retried = {
        track_id
        for track_id, reason in previously_failed.items()
        if args.retry_failed or reason in retry_reasons
    }
    skip = set(archived) | (set(previously_failed) - retried)
    pending = [track_id for track_id in all_ids if track_id not in skip]
    if args.limit > 0:
        pending = pending[: args.limit]

    known = set(all_ids)
    orphans = sorted(
        track_id
        for track_id in archived
        if track_id in known and not audio_file(data_dir, track_id).exists()
    )

    print("raveform audio download")
    print(f"data dir  : {data_dir}")
    print(f"audio dir : {audio_dir(data_dir)}")
    print(f"archive   : {archive_path(data_dir)}")
    print(f"failures  : {failed_path(data_dir)}")
    if args.retry_failed:
        retry_note = "all being retried (--retry-failed)"
    elif retry_reasons:
        retry_note = (
            f"{len(retried)} being retried ({','.join(sorted(retry_reasons))}), "
            f"{len(previously_failed) - len(retried)} still skipped"
        )
    else:
        retry_note = "skipped"
    print(
        f"manifest  : {len(all_ids)} unique ids | already downloaded {len(archived)} | "
        f"previously failed {len(previously_failed)}  ({retry_note})"
    )
    if previously_failed and not args.retry_failed:
        print(f"            failure reasons on record: {_reason_line(_tally(previously_failed))}")
    print(
        f"this run  : {len(pending)} track(s)"
        + (f" (--limit {args.limit})" if args.limit > 0 else "")
        + f", sleeping {args.sleep_min:g}-{args.sleep_max:g}s between videos"
    )
    if orphans:
        print(
            f"WARNING   : {len(orphans)} archived id(s) have no mp3 on disk; yt-dlp will "
            "refuse to re-fetch them until their line is removed from the archive"
        )
        print(f"            {', '.join(orphans[:10])}" + (" ..." if len(orphans) > 10 else ""))
    print()

    done = 0
    failed = 0
    consecutive_blocks = 0
    reasons: dict[str, int] = {}
    started = time.monotonic()
    interrupted = False
    aborted = False

    for index, track_id in enumerate(pending, start=1):
        if interrupt_requested():
            interrupted = True
            break

        if index > 1:
            delay = random.Random(track_id).uniform(args.sleep_min, args.sleep_max)
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                interrupted = True
                break

        try:
            ok, reason, error = download_one(data_dir, track_id, args.timeout_sec)
        except KeyboardInterrupt:
            interrupted = True
            break

        if not ok and reason == INTERRUPT_REASON:
            interrupted = True
            print(f"[{index}/{len(pending)}] cancelled  {track_id}  (not recorded)", flush=True)
            break

        if ok:
            done += 1
            consecutive_blocks = 0
            size_mb = audio_file(data_dir, track_id).stat().st_size / (1 << 20)
            print(f"[{index}/{len(pending)}] ok    {track_id}  {size_mb:.1f} MB", flush=True)
        else:
            failed += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            append_failure(
                failed_path(data_dir),
                {
                    "youtube_id": track_id,
                    "error": error,
                    "reason": reason,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
                },
            )
            last_line = error.splitlines()[-1] if error else ""
            print(
                f"[{index}/{len(pending)}] FAIL  {track_id}  [{reason}] {last_line[:120]}",
                flush=True,
            )
            if reason in BLOCK_REASONS:
                consecutive_blocks += 1
                if 0 < args.max_consecutive_blocks <= consecutive_blocks:
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
    on_disk = read_archive_ids(archive_path(data_dir))
    remaining_total = sum(1 for track_id in all_ids if track_id not in on_disk)
    print(f"  corpus      : {len(all_ids) - remaining_total}/{len(all_ids)} tracks on disk")
    if attempted and remaining_total:
        print(
            f"  full-run ETA: {format_duration(elapsed / attempted * remaining_total)} "
            f"for the remaining {remaining_total}"
        )

    recoverable = sorted(reason for reason in reasons if reason in RETRYABLE_REASONS)
    if recoverable:
        print()
        print(
            f"  RECOVERABLE : {sum(reasons[reason] for reason in recoverable)} of this run's "
            f"failures may not be permanent ({', '.join(recoverable)})."
        )
        print(f"                A plain re-run SKIPS them. To re-attempt:  {RETRY_HINT}")

    blocks = reasons.get("bot_check", 0)
    if blocks:
        print()
        print(
            f"  BOT CHECK   : {blocks} sign-in / rate-limit refusal(s). No cookie or credential "
            "workaround was attempted -- this is an owner decision."
        )
    forbidden = reasons.get("http_403", 0)
    if forbidden:
        print()
        print(
            f"  HTTP 403    : {forbidden} refusal(s) on the media URL -- a signature/nsig "
            "challenge yt-dlp could not solve. These are"
        )
        print(
            "                NOT dead videos, and on the full corpus sweep every one of "
            "them came back on a later attempt:"
        )
        print(f"                re-run with  {RETRY_HINT}  and let it work through them again.")
        print(
            "                If a wave survives repeated patient re-runs, it is a toolchain "
            "problem (yt-dlp version, or a JS"
        )
        print(
            "                runtime not visible to this process) and an owner decision -- "
            "not something to keep retrying."
        )
    if aborted:
        print(
            f"  ABORTED     : {consecutive_blocks} consecutive refusals hit the "
            "--max-consecutive-blocks guard."
        )
        print(
            "                Those tracks are now in failed.jsonl, so a PLAIN RE-RUN WILL SKIP "
            "THEM. To bring them back once"
        )
        print(f"                the block clears, re-run with:  {RETRY_HINT}")
        return 2
    if interrupted:
        print("  INTERRUPTED : Ctrl-C. State on disk is resumable; re-run to continue.")
        return 130
    return 0


def _reason_line(reasons: dict) -> str:
    return ", ".join(f"{reason} x{count}" for reason, count in sorted(reasons.items()))


def _tally(id_to_reason: dict) -> dict:
    counts: dict = {}
    for reason in id_to_reason.values():
        counts[reason] = counts.get(reason, 0) + 1
    return counts


if __name__ == "__main__":
    sys.exit(main())
