#!/usr/bin/env python
"""Download the raveform corpus audio from YouTube, one track at a time.

Consumes ``<data-dir>/manifest.csv`` (column ``youtube_id``, written by
``raveform_manifest.py``) and produces::

    <data-dir>/audio/<youtube_id>.mp3   the audio, 192 kbit/s mp3
    <data-dir>/downloaded.txt           yt-dlp download archive -- resume state
    <data-dir>/failed.jsonl             one JSON object per failed attempt

Downloads are strictly sequential with a pause between videos.  This is a
politeness contract with YouTube, not a performance trade-off: 1,423 tracks
pulled in parallel is indistinguishable from abuse.  Audio only -- the video
stream is never fetched.

**Resume contract.**  The script is safe to kill (Ctrl-C, reboot, `taskkill`)
and re-run at any point:

* an id recorded in ``downloaded.txt`` is skipped -- yt-dlp writes that line
  only after the download *and* the mp3 conversion have both finished, so a
  half-converted track is never mistaken for a finished one;
* an id already present in ``failed.jsonl`` is skipped too, unless
  ``--retry-failed`` (all of them) or ``--retry-reasons`` (only those whose
  recorded reason matches) says otherwise -- otherwise every re-run would
  re-attempt every dead video and the run would never converge;
* a cancelled track is never recorded as failed.  Interruption is decided by
  our own SIGINT flag rather than by yt-dlp's message, which differs depending
  on whether Ctrl-C landed during the fetch or during post-processing;
* ``failed.jsonl`` is append-only and flushed per record, so a hard kill cannot
  lose failures.  An id that failed once and succeeded on a later retry appears
  in *both* files; ``downloaded.txt`` is the authority.  Consumers tallying
  failures must subtract the archive.

**A track counts as downloaded only when a non-empty mp3 is on disk.**  A
zero-byte or absent output is recorded as a failure and its archive line is
removed, so the track is genuinely re-fetchable -- the corpus must never be
silently short a track that merely looked successful.

**Bot checks are never worked around.**  If YouTube answers with a sign-in /
cookie wall or rate-limits the run, the failure is recorded with
``reason: "bot_check"`` and reported; no cookie, credential or IP workaround is
attempted here.  After ``--max-consecutive-blocks`` such answers in a row the
run stops rather than burning the rest of the manifest into failure records --
the state on disk is resumable, so the owner can decide and re-run.

Stdlib only.  Requires ``yt-dlp`` and ``ffmpeg`` on PATH.

Usage::

    uv run python training/raveform_download.py \
        --data-dir C:\\Users\\Julian\\Projects\\soundswitch-auto-pilot\\training\\data\\raveform \
        --limit 10
"""

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

# Bounded per-track wall-clock budget.  ``--socket-timeout`` and ``--retries``
# bound the network, but a wedged ffmpeg post-process is not covered by either,
# and this script is meant to run detached for hours.
DEFAULT_TIMEOUT_SEC = 600

# How much of yt-dlp's stderr to keep in a failure record.  Enough for the full
# error block plus context, short enough that failed.jsonl stays readable.
_STDERR_TAIL_CHARS = 2000

_PROGRESS_EVERY = 10


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #
#
# Categories exist so the owner can act on a run without reading 1,400 stderr
# blobs: `bot_check` means YouTube wants credentials (a policy decision),
# `unavailable` means the video is simply gone (nothing to decide).  Matching is
# substring-on-lowercase; the raw stderr tail is always kept alongside, so a
# mis-classification loses nothing.

# Ordered: the first match wins, and the ordering encodes exactly one rule --
#
#     a PERMANENT condition outranks a TRANSIENT one.
#
# Every mis-classification found in this table so far was a violation of it, and
# they all failed the same way.  YouTube's permanent errors habitually carry
# transient-looking text: a private video ends with "Use --cookies-from-browser",
# an age gate opens with "Sign in to confirm your age".  A transient bucket
# placed above them silently swallows a video that is gone for good, and the
# cost is never just a wrong label -- every retry pass re-polls it forever, and
# a run of them trips --max-consecutive-blocks and aborts a healthy run.
#
# So the permanent buckets come first, then the refusals, then the generic
# permanent text that the specific patterns above have already claimed.
_PERMANENT_PATTERNS = (
    # Access we will never be granted is not a block.
    ("unavailable", ("private video", "this video is private")),
    # "Sign in to confirm your age" contains "sign in to confirm", which is a
    # bot_check pattern.  The age gate is the more specific reading and the
    # permanent one: we do not sign in, so the video is not obtainable.
    (
        "age_restricted",
        (
            "age-restricted",
            "age restricted",
            "sign in to confirm your age",
            "inappropriate for some users",
        ),
    ),
    # Must stay above the generic "this video is not available" below, which
    # would otherwise swallow the country-specific wording.
    ("geo_blocked", ("not available in your country", "geo restricted", "geo-restricted")),
)

_TRANSIENT_PATTERNS = (
    (
        "bot_check",
        (
            "confirm you're not a bot",
            "confirm you\u2019re not a bot",  # curly apostrophe
            "confirm you are not a bot",
            "sign in to confirm",
            "please sign in",
            "use --cookies",
            "--cookies-from-browser",
            "http error 429",
            "too many requests",
        ),
    ),
    # A 403 on the media URL is YouTube refusing *this client* -- a signature/nsig
    # challenge that could not be solved, which is a property of our toolchain and
    # not of the video.  Measured on the first full corpus sweep: 43 of the 62
    # happened while a JS runtime WAS visible to the run, and four of those
    # records carry yt-dlp's own explanation -- its remote challenge-solver
    # components are opt-in and were never enabled.  So a visible runtime is
    # necessary but not sufficient; see the operator note printed in the summary.
    # It gets its own bucket rather than folding into bot_check because the two
    # demand opposite remedies: bot_check is a credential wall and an owner
    # decision, while a 403 wave means fix the extractor and re-run.  Reporting
    # one as the other invites precisely the wrong fix (cookies).  It was worth
    # 56% of the failures on the first full corpus sweep, all of them recovered
    # by a plain re-attempt.
    ("http_403", ("http error 403", "403: forbidden")),
)

_REASON_PATTERNS = _PERMANENT_PATTERNS + _TRANSIENT_PATTERNS + (
    (
        "unavailable",
        (
            "video unavailable",
            # YouTube says this without the word "unavailable" attached to the
            # word "video", so the pattern above misses it and the failure lands
            # in `other` -- which retry passes re-poll.
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

# Categories that mean "YouTube is refusing this client", not "this video is
# gone".  Consecutive ones abort the run: if the refusals are back to back the
# problem is the run, not the manifest, and burning the remaining tracks into
# failure records makes the corpus harder to finish rather than easier.
BLOCK_REASONS = frozenset(reason for reason, _patterns in _TRANSIENT_PATTERNS)

# Every reason this script can record.  `--retry-reasons` is validated against
# it so a typo fails loudly instead of silently selecting nothing.
KNOWN_REASONS = frozenset(
    {reason for reason, _patterns in _REASON_PATTERNS}
    | {"other", "timeout", "missing_output", "empty_output"}
)

# Reasons worth re-attempting: the failure was about this run, not about the
# video.  `other` belongs here precisely because it is the *unclassified*
# bucket -- an error we have no pattern for is far likelier to be something
# transient we have not named than a video that vanished without YouTube
# saying so.  Leaving it out is what makes a "just retry the blocks" hint
# quietly abandon tracks: on the first full corpus sweep 62 of the 65 `other`
# failures were 403s that a plain re-attempt recovered.
#
# This is the list the operator-facing hints print, so the advice this script
# gives can never again be narrower than the advice that actually works.
RETRYABLE_REASONS = frozenset(
    BLOCK_REASONS | {"timeout", "empty_output", "missing_output", "other"}
)

RETRY_HINT = "--retry-reasons " + ",".join(sorted(RETRYABLE_REASONS))


def classify_error(text: str) -> str:
    """Bucket a yt-dlp stderr blob into a coarse, machine-readable reason."""
    lowered = text.lower()
    for reason, patterns in _REASON_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return reason
    return "other"


# --------------------------------------------------------------------------- #
# Interrupt handling
# --------------------------------------------------------------------------- #
#
# Ctrl-C reaches the child too on Windows, so yt-dlp dies non-zero on its own.
# What it *prints* when that happens depends on where the signal landed: during
# the fetch it says "Interrupted by user", but during the ffmpeg post-process it
# says "ERROR: Postprocessing: ..." instead.  Classifying on that text would
# therefore file a cancelled track as a permanent `other` failure, and a plain
# re-run would skip an available track forever.  Worse, the substring test cut
# both ways: a genuine error that happened to contain the word would abort the
# run at the same index every time.
#
# So interruption is decided by a signal flag we own, never by the child's text.

_interrupt_requested = False


def interrupt_requested() -> bool:
    return _interrupt_requested


def request_interrupt() -> None:
    """Set the interrupt flag (also the seam fault-injection tests drive)."""
    global _interrupt_requested
    _interrupt_requested = True


def install_interrupt_handler() -> None:
    """Make SIGINT set the flag instead of raising mid-subprocess.

    A no-op where the platform or thread will not accept the handler; the
    ``KeyboardInterrupt`` guards in the run loop remain as the fallback.
    """
    def _handler(_signum, _frame):
        request_interrupt()

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError, AttributeError):
        pass


# --------------------------------------------------------------------------- #
# Paths and state files
# --------------------------------------------------------------------------- #


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "training" / "data" / "raveform"


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
    """YouTube ids from the manifest, manifest order, duplicates removed.

    A duplicate id would otherwise be attempted twice in one run (the archive is
    only re-read at startup), so dedupe here rather than relying on yt-dlp.
    """
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
    """Ids recorded in a yt-dlp download archive (``<extractor> <id>`` lines)."""
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
    """Drop ``youtube_id`` from the download archive.  Returns True if removed.

    Needed whenever an archived track turns out not to be on disk: yt-dlp
    consults the archive first and would answer "already recorded" forever,
    leaving the corpus permanently short.  Rewritten via a temp file so a kill
    mid-write cannot truncate the archive.
    """
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
    """Map ``youtube_id`` -> the reason of its **most recent** failure record.

    Most-recent wins because failed.jsonl is append-only: a track that was
    rate-limited and later found to be deleted should be treated as deleted.
    Tolerates a torn last line.
    """
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
                continue  # a kill mid-write; the id is retried, which is safe
            value = record.get("youtube_id")
            if isinstance(value, str) and value:
                reason = record.get("reason")
                reasons[value] = reason if isinstance(reason, str) else "other"
    return reasons


def append_failure(path: Path, record: dict) -> None:
    """Append one failure record, flushed -- a hard kill must not lose it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def build_command(data_dir: Path, youtube_id: str) -> list[str]:
    """The yt-dlp argv for one track.

    List form, never a shell string, and a ``--`` separator before the id: a
    YouTube id may legitimately start with ``-`` and would otherwise be parsed
    as an option.
    """
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
    """Run yt-dlp for one id.  Returns ``(ok, reason, error_tail)``.

    ``reason`` and ``error_tail`` are empty on success.  Every non-success path
    -- non-zero exit, timeout, launch failure, or a zero exit that left no mp3
    or an empty one -- yields a classified reason, so the caller can always
    write a complete failure record.

    Success means **a non-empty mp3 on disk**, not merely an exit code: a
    zero-byte or truncated output must never be archived as a finished track.
    """
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
        # Not this track's problem -- the tool is missing.  Fail loudly.
        raise RuntimeError("yt-dlp not found on PATH -- see the Task 1 tooling report") from None

    if completed.returncode != 0:
        # A signal we raised outranks anything the child printed: yt-dlp's own
        # wording differs depending on whether Ctrl-C landed during the fetch or
        # during post-processing, and neither must become a permanent failure.
        if interrupt_requested():
            return False, INTERRUPT_REASON, "cancelled by SIGINT"
        blob = _tail(completed.stderr) or _tail(completed.stdout) or "(no output)"
        return False, classify_error(blob), blob

    target = audio_file(data_dir, youtube_id)
    if not target.exists():
        # yt-dlp exited 0 without producing the file.  The usual cause is an
        # archive line whose mp3 was deleted: yt-dlp then reports "already
        # recorded in the archive" and does nothing.  Record it rather than
        # counting a phantom success.
        forget_download(data_dir, youtube_id)
        return (
            False,
            "missing_output",
            _tail(completed.stdout) or "yt-dlp exited 0 but no mp3 was written",
        )

    if target.stat().st_size == 0:
        # An empty file would otherwise be archived as a finished track and the
        # corpus would be silently short by one.  Remove it, and remove the
        # archive line with it, so a retry actually re-fetches.
        target.unlink(missing_ok=True)
        forget_download(data_dir, youtube_id)
        return False, "empty_output", "yt-dlp exited 0 but wrote a zero-byte mp3 (deleted)"

    return True, "", ""


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # negative or NaN
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


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
        # Silently matching nothing would look like "no failures of that kind"
        # and send the owner hunting for a data problem that does not exist.
        parser.error(
            f"unknown --retry-reasons value(s): {', '.join(unknown)}; "
            f"known reasons are {', '.join(sorted(KNOWN_REASONS))}"
        )

    # This run is meant to be launched detached with stdout redirected to a log.
    # Block buffering would hide hours of progress, so force line buffering.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    data_dir = args.data_dir.resolve()

    # Read the manifest BEFORE creating anything: a mistyped --data-dir must
    # fail with "no manifest at ..." rather than silently mkdir-ing a new empty
    # corpus root somewhere unintended.
    all_ids = read_manifest_ids(data_dir)
    archived = read_archive_ids(archive_path(data_dir))
    previously_failed = read_failed_reasons(failed_path(data_dir))
    audio_dir(data_dir).mkdir(parents=True, exist_ok=True)
    install_interrupt_handler()

    # A failure is retried when --retry-failed is given, or when its recorded
    # reason is in --retry-reasons.  This is what makes recovery from a
    # rate-limit episode cheap: `--retry-reasons bot_check,timeout` re-attempts
    # only the tracks YouTube refused, and leaves the genuinely dead videos
    # alone instead of re-polling ~1,400 of them.
    retried = {
        track_id
        for track_id, reason in previously_failed.items()
        if args.retry_failed or reason in retry_reasons
    }
    skip = set(archived) | (set(previously_failed) - retried)
    pending = [track_id for track_id in all_ids if track_id not in skip]
    if args.limit > 0:
        pending = pending[: args.limit]

    # An archive line whose mp3 has gone missing is a silent trap: yt-dlp would
    # refuse to re-download it.  Surface it instead of letting the corpus be
    # quietly short.
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
            # Seeded per id, so a re-run of the same manifest slice waits the
            # same amount -- reruns behave identically.
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
            # Ctrl-C killed the child mid-fetch or mid-postprocess.  Not a
            # failure -- do not poison failed.jsonl with a track the user
            # merely cancelled, or a plain re-run would skip it forever.
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
            # yt-dlp puts its ERROR line last, so the tail is the useful summary.
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

    # A plain re-run skips every recorded failure, so a run that ends with
    # recoverable ones must say out loud how to get them back -- and must name
    # every such reason, not a sample of them.  Under-naming here is what
    # silently abandons tracks the corpus could have had.
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
        # Deliberately NOT the cookie advice: a 403 on the media URL is an
        # extraction failure on our side, and pointing at credentials would send
        # the owner after a problem they do not have.
        print()
        print(
            f"  HTTP 403    : {forbidden} refusal(s) on the media URL -- a signature/nsig "
            "challenge yt-dlp could not solve. Two causes,"
        )
        print(
            "                in the order they are worth checking:"
        )
        print(
            "                 1. yt-dlp skips its remote challenge-solver components unless "
            "asked. Re-run with:"
        )
        print(
            "                      --remote-components ejs:github        (yt-dlp's own "
            "recommended form; ejs:npm is the alternative)"
        )
        print(
            "                 2. a JS runtime (deno/node) must be visible TO THIS PROCESS. "
            "Installed is not enough -- a shell"
        )
        print(
            "                    started before the install carries a stale PATH. Check from "
            "the runner, not a fresh terminal."
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
