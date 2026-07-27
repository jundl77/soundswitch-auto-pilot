#!/usr/bin/env python
"""Full-corpus validation: every annotated track accounted for, or named.

The cleanliness gate (``build_clean_manifest.py``) answers "which audio may we
learn from?" and is silent about everything it does not find.  This validator
answers the harder question the owner actually asked: **is every one of the
1,423 annotated tracks either present-and-correct, or precisely recorded as
unobtainable?**  Silence is the failure mode it exists to eliminate -- a corpus
that is quietly 40 tracks short looks exactly like a complete one unless
something insists on reconciling the manifest against the disk.

Every manifest row lands in exactly one bucket::

    OK                 decodes fully, and the decoded length agrees with the
                       annotation record's duration
    DURATION_MISMATCH  decodes fully, wrong length -- almost always a different
                       edit or the wrong video; kept on disk, listed for a human
    CORRUPT            on disk but undecodable, or truncated (the decoder
                       produced less audio than the container advertises)
    MISSING            in the manifest, not on disk, and no recorded attempt --
                       the one bucket that means "we lost track of this"
    UNAVAILABLE        never obtained, with the recorded yt-dlp reason and the
                       error tail that justifies it

The first three verdicts come straight from the cleanliness gate, which already
reasons carefully about truncation vs. wrong-video; this module adds the two
that only exist once you reconcile against the manifest and the download state.

**Convergence** is the whole point, and it is arithmetic, not vibes::

    OK + DURATION_MISMATCH + UNAVAILABLE == manifest rows   and
    MISSING == CORRUPT == 0

The sum is checked as well as the two zeroes, because a bucket count can only
prove nothing was dropped if the buckets add up to the manifest.

Convergence is about audio, so two findings cannot appear in it by
construction: orphan files in ``audio/`` that no manifest row claims, and
annotations that do not reconcile with the manifest (a missing or unparsable
beat grid, a disagreeing YouTube id or duration).  A track can be counted OK and
still be untrainable because its beat grid never arrived.  Those get their own
**all-clear** verdict, which is what the exit code reports: converged *and* no
orphans *and* no annotation issues.  Both verdicts are always printed.

**What the disk says outranks what the log says.**  ``failed.jsonl`` is
append-only, so a track that was rate-limited on one cycle and fetched on the
next appears in both it and ``downloaded.txt``.  A file that is on disk is
judged by decoding it; the failure log is consulted only for tracks that are
not there.

**Checksums are the integrity baseline, not a verification.**  YouTube
publishes no canonical hashes, so there is nothing to check the corpus
*against*; ``checksums.sha256`` records what we have, so that a future
re-validation can prove the bytes have not changed since the day the decode
check passed.  Written in ``sha256sum -c`` format with paths relative to the
data dir, so it is portable and checkable without this script.

Read-only over ``audio/`` unless ``--prune-corrupt`` is given, which deletes
CORRUPT files and drops their download-archive lines so a plain re-run of
``raveform_download.py`` genuinely re-fetches them (yt-dlp would otherwise
answer "already recorded" forever).

Stdlib only.  Requires ``ffmpeg`` and ``ffprobe`` on PATH.

Usage::

    uv run python training/raveform_validate.py \\
        --data-dir C:\\Users\\Julian\\Projects\\soundswitch-auto-pilot\\training\\data\\raveform
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_clean_manifest as gate  # noqa: E402  (needs the path insert above)
import raveform_download as downloader  # noqa: E402
from raveform_fetch_annotations import (  # noqa: E402
    beat_csv_path,
    load_tracks,
    parse_beat_csv,
    parse_sections,
    youtube_id as annotation_youtube_id,
)

VALIDATION_JSON = "validation_report.json"
VALIDATION_TXT = "validation_report.txt"
CHECKSUMS_FILE = "checksums.sha256"

STATUS_OK = "OK"
STATUS_DURATION_MISMATCH = "DURATION_MISMATCH"
STATUS_CORRUPT = "CORRUPT"
STATUS_MISSING = "MISSING"
STATUS_UNAVAILABLE = "UNAVAILABLE"

STATUS_ORDER = (
    STATUS_OK,
    STATUS_DURATION_MISMATCH,
    STATUS_CORRUPT,
    STATUS_MISSING,
    STATUS_UNAVAILABLE,
)

# Buckets that mean "this track is accounted for": we have it, or we know
# exactly why we do not.
ACCOUNTED_FOR = (STATUS_OK, STATUS_DURATION_MISMATCH, STATUS_UNAVAILABLE)

# The gate's vocabulary is lower-case and local to the training-table build;
# this module's is the owner-facing one.  One mapping, in one place.
_GATE_TO_STATUS = {
    gate.STATUS_OK: STATUS_OK,
    gate.STATUS_MISMATCH: STATUS_DURATION_MISMATCH,
    gate.STATUS_CORRUPT: STATUS_CORRUPT,
}

# manifest.csv stores the annotation duration rounded to milliseconds, so the
# two can differ by at most half a millisecond.  Anything larger means the
# manifest and the annotations describe different tracks.
MANIFEST_ROUNDING_SEC = 0.001

# Enough of a yt-dlp error to identify the refusal without pasting the whole
# stderr blob into a per-track line.
_DETAIL_CHARS = 300

_HASH_CHUNK = 1 << 20


class Failure(NamedTuple):
    """The most recent recorded download failure for one YouTube id."""

    youtube_id: str
    reason: str
    error: str
    timestamp: str


class TrackVerdict(NamedTuple):
    """One manifest row's final status, and the evidence behind it."""

    track_id: str
    youtube_id: str
    status: str
    detail: str
    mp3_path: str                       # "" when the file is not on disk
    ffprobe_duration_sec: float | None  # container header -- advertised
    decoded_duration_sec: float | None  # what the decoder produced -- the truth
    annotation_duration_sec: float
    failure_reason: str                 # "" unless UNAVAILABLE
    sha256: str                         # "" unless OK and checksums were run


# --------------------------------------------------------------------------- #
# Download state
# --------------------------------------------------------------------------- #


def load_failures(path: Path) -> dict:
    """``youtube_id`` -> its **most recent** failure record.

    Mirrors the downloader's own most-recent-wins rule (a track rate-limited on
    Monday and found deleted on Tuesday is deleted), but keeps the error tail as
    well as the reason: an UNAVAILABLE verdict is only worth anything if it
    carries the evidence for itself.  A torn final line -- the shape of a hard
    kill mid-append -- costs that one record and nothing else.
    """
    if not path.exists():
        return {}
    failures: dict = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            track_id = record.get("youtube_id")
            if not isinstance(track_id, str) or not track_id:
                continue
            failures[track_id] = Failure(
                track_id,
                # A record with no reason becomes "other" -- the same
                # placeholder the downloader uses for an error it has no
                # pattern for.  Worth knowing when reading a report: an
                # `other` in the UNAVAILABLE breakdown means "we did not
                # recognise this failure", never "there was no failure".  The
                # raw error tail is carried alongside for exactly that reason.
                str(record.get("reason") or "other"),
                str(record.get("error") or ""),
                str(record.get("timestamp") or ""),
            )
    return failures


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def _last_line(text: str, limit: int = _DETAIL_CHARS) -> str:
    """yt-dlp prints its ERROR line last, so the tail is the useful summary."""
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1][:limit] if lines else ""


def classify_track(gate_status, gate_detail: str, failure) -> tuple:
    """``(status, detail)`` for one manifest row.

    ``gate_status`` is the cleanliness gate's verdict, or ``None`` when the file
    is not on disk.  It outranks the failure log unconditionally: ``failed.jsonl``
    is append-only, so a track that failed once and was fetched on a later cycle
    is recorded in both places, and only one of them describes the bytes we now
    hold.
    """
    if gate_status is not None:
        return _GATE_TO_STATUS[gate_status], gate_detail
    if failure is not None:
        tail = _last_line(failure.error)
        return STATUS_UNAVAILABLE, f"{failure.reason}: {tail}" if tail else failure.reason
    return (
        STATUS_MISSING,
        "in the manifest, absent from audio/, and no download attempt on record",
    )


def tally(verdicts: list) -> collections.Counter:
    """Status histogram, with an explicit zero for every bucket."""
    counts = collections.Counter({status: 0 for status in STATUS_ORDER})
    counts.update(verdict.status for verdict in verdicts)
    return counts


def convergence(counts, manifest_tracks: int) -> tuple:
    """``(converged, statement)`` -- the corpus verdict, stated in full.

    Two conditions, and both are load-bearing.  The zeroes say no track is in a
    state we refuse to accept; the sum says no track fell out of the accounting
    altogether.  Neither implies the other: a validator that lost 40 rows would
    report zero MISSING and zero CORRUPT quite happily.
    """
    accounted = sum(counts.get(status, 0) for status in ACCOUNTED_FOR)
    missing = counts.get(STATUS_MISSING, 0)
    corrupt = counts.get(STATUS_CORRUPT, 0)
    converged = accounted == manifest_tracks and missing == 0 and corrupt == 0

    head = "CONVERGED" if converged else "NOT CONVERGED"
    statement = (
        f"{head}: OK + DURATION_MISMATCH + UNAVAILABLE = "
        f"{counts.get(STATUS_OK, 0)} + {counts.get(STATUS_DURATION_MISMATCH, 0)} + "
        f"{counts.get(STATUS_UNAVAILABLE, 0)} = {accounted} "
        f"(manifest {manifest_tracks}); MISSING = {missing}; CORRUPT = {corrupt}"
    )
    if not converged:
        blockers = []
        if accounted != manifest_tracks:
            blockers.append(f"{manifest_tracks - accounted} manifest row(s) unaccounted for")
        if missing:
            blockers.append(f"{missing} MISSING")
        if corrupt:
            blockers.append(f"{corrupt} CORRUPT")
        statement += "  [blocked by: " + ", ".join(blockers) + "]"
    return converged, statement


def overall_verdict(converged: bool, orphans: list, annotation_issues: list) -> tuple:
    """``(all_clear, statement)`` -- convergence *plus* the corpus-level checks.

    Convergence is the accounting question: is every manifest row in a bucket we
    accept?  It is deliberately left alone, because that arithmetic is the
    contract this corpus was declared complete against.

    But a track can be counted OK and still be unusable -- its beat grid may
    have failed to download, or ``manifest.csv`` may have gone stale against a
    re-fetched ``segments.json``.  Those findings cannot appear in the five
    buckets by construction, since the buckets only ever describe audio.  So
    they get their own verdict line and their own share of the exit code: a
    supervisor branching on 0/1 must never read "all good" while the annotation
    cross-check has findings.  Two numbers, both stated, neither hidden.
    """
    findings = []
    if not converged:
        findings.append("the corpus does not converge")
    if annotation_issues:
        findings.append(f"{len(annotation_issues)} annotation issue(s)")
    if orphans:
        findings.append(f"{len(orphans)} orphan audio file(s)")
    if not findings:
        return True, "ALL CLEAR: converged, no orphans, no annotation issues"
    return False, "NOT CLEAR: " + ", ".join(findings)


# --------------------------------------------------------------------------- #
# Corpus-level checks
# --------------------------------------------------------------------------- #


def find_orphans(data_dir: Path, known_youtube_ids) -> list:
    """Audio files on disk that no manifest row claims, sorted.

    Only ``*.mp3`` is swept.  ``audio/`` also holds deliberate ``*.npy`` decode
    caches (the eval set writes them beside the source) and can hold transient
    ``*.part`` files; counting either as an orphan would invent a finding on
    every run and bury a real one.
    """
    audio = data_dir / gate.AUDIO_DIR
    if not audio.is_dir():
        return []
    known = set(known_youtube_ids)
    return sorted(
        path.stem for path in audio.glob("*.mp3") if path.stem not in known
    )


def annotation_durations(tracks: list) -> dict:
    """``track_id`` -> annotated duration, at the record's full precision.

    This -- not ``manifest.csv``'s ``total_sec`` -- is the reference length for
    the duration check.  The manifest rounds to milliseconds on the way out, and
    the whole point of the check is to compare against the annotation itself.
    """
    return {str(track["key"]): float(track["duration"]) for track in tracks}


def check_annotations(data_dir: Path, rows: list, tracks: list) -> list:
    """Every issue found reconciling the annotations against the manifest.

    "The corpus is complete" has to mean complete *against the annotations*: a
    track whose beat grid failed to download is as useless for training as one
    whose audio did, and would otherwise be invisible because its mp3 is fine.
    Returns human-readable strings, sorted; empty means clean.
    """
    issues = []
    by_track = {}
    parsed = {}          # track_id -> the record, for records that parsed
    for track in tracks:
        try:
            track_id = str(track["key"])
            youtube = annotation_youtube_id(track)
            duration = float(track["duration"])
            sections = parse_sections(track)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            issues.append(f"{track.get('key', '?')}: annotation record does not parse ({exc})")
            continue
        if not sections:
            issues.append(f"{track_id}: annotation record has no sections")
        if track_id in by_track:
            # Two records under one key: the second silently replaces the first
            # everywhere downstream, and the corpus is a track short without
            # anything looking wrong.
            issues.append(f"{track_id}: duplicate annotation record")
        by_track[track_id] = (youtube, duration)
        parsed[track_id] = track

    manifest = {row.track_id: row for row in rows}
    for track_id, (youtube, duration) in sorted(by_track.items()):
        row = manifest.get(track_id)
        if row is None:
            issues.append(f"{track_id}: annotated but absent from manifest.csv")
            continue
        if row.youtube_id != youtube:
            issues.append(
                f"{track_id}: youtube id disagrees -- manifest {row.youtube_id!r}, "
                f"annotation {youtube!r}"
            )
        if abs(duration - row.annotation_duration_sec) > MANIFEST_ROUNDING_SEC:
            issues.append(
                f"{track_id}: duration disagrees -- manifest "
                f"{row.annotation_duration_sec:.3f} s, annotation {duration:.6f} s"
            )

    for track_id in sorted(manifest):
        if track_id not in by_track:
            issues.append(f"{track_id}: in manifest.csv but has no annotation record")

    # Only records that parsed: a record with no usable ``key`` has already been
    # reported, and asking for its beat-grid path would raise -- crashing the
    # whole validation run before a single artifact is written, on exactly the
    # malformed input the check exists to describe.
    for track_id, track in sorted(parsed.items()):
        path = beat_csv_path(data_dir, track)
        if not path.exists():
            issues.append(f"{track_id}: beat grid missing ({path.name})")
            continue
        try:
            beats = parse_beat_csv(path)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            issues.append(f"{track_id}: beat grid does not parse ({exc})")
            continue
        if not beats:
            issues.append(f"{track_id}: beat grid is empty")

    return sorted(issues)


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks (corpus files run to ~15 MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(data_dir: Path, verdicts: list) -> tuple:
    """Write ``checksums.sha256`` over the OK files.  Returns ``(path, count)``.

    Only OK rows: the baseline records the bytes we have decided are correct,
    so re-validation compares like with like.  ``sha256sum -c`` format with
    data-dir-relative paths, sorted, so the file is stable across runs and
    checkable by tools that know nothing about this corpus.
    """
    path = data_dir / CHECKSUMS_FILE
    entries = sorted(
        (f"{gate.AUDIO_DIR}/{verdict.youtube_id}.mp3", verdict.sha256)
        for verdict in verdicts
        if verdict.status == STATUS_OK and verdict.sha256
    )
    lines = [f"{digest}  {name}" for name, digest in entries]
    _write_atomic(path, "\n".join(lines) + ("\n" if lines else ""))
    return path, len(lines)


def _write_atomic(path: Path, text: str) -> Path:
    """Write via a temp file so a kill can never leave a half-written report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


# --------------------------------------------------------------------------- #
# Retry support
# --------------------------------------------------------------------------- #


def prune_corrupt(data_dir: Path, verdicts: list) -> list:
    """Delete CORRUPT audio and forget it in the download archive.

    The only path in this module that writes inside ``audio/``, and it only runs
    on request.  Both halves are required: yt-dlp consults its archive first, so
    deleting the file alone leaves a track that can never be re-fetched.

    DURATION_MISMATCH is deliberately untouched -- a wrong-length track is a
    judgement call about which recording YouTube served, not a bad byte stream,
    and deleting it would destroy the evidence a human needs.
    """
    pruned = []
    for verdict in verdicts:
        if verdict.status != STATUS_CORRUPT:
            continue
        path = Path(verdict.mp3_path) if verdict.mp3_path else None
        if path is not None:
            path.unlink(missing_ok=True)
        downloader.forget_download(data_dir, verdict.youtube_id)
        pruned.append(verdict.youtube_id)
    return sorted(pruned)


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


def validate(
    data_dir: Path,
    workers: int = gate.DEFAULT_WORKERS,
    checksums: bool = True,
    progress_every: int = 100,
) -> dict:
    """Run one full validation pass and write all three artifacts.

    Returns the same payload that lands in ``validation_report.json``.
    """
    data_dir = Path(data_dir)
    rows = gate.load_manifest_rows(data_dir)
    tracks = load_tracks(data_dir)
    durations = annotation_durations(tracks)
    failures = load_failures(downloader.failed_path(data_dir))

    jobs = []
    absent = []
    for row in rows:
        # The annotation record is the reference length; the manifest value is
        # the ms-rounded copy and is only the fallback if a record is missing
        # (which check_annotations reports separately).
        duration = durations.get(row.track_id, row.annotation_duration_sec)
        path = gate.audio_path(data_dir, row.youtube_id)
        if path.exists():
            jobs.append(gate.TrackJob(row.track_id, row.youtube_id, str(path), duration))
        else:
            absent.append((row, duration))

    print(
        f"decoding {len(jobs)} file(s) with {workers} worker(s); "
        f"{len(absent)} manifest row(s) have no audio",
        flush=True,
    )
    results = gate.run_checks(jobs, workers=workers, progress_every=progress_every)

    verdicts = []
    for result in results:
        status, detail = classify_track(result.status, result.detail, None)
        digest = ""
        if checksums and status == STATUS_OK:
            digest = sha256_file(Path(result.mp3_path))
        verdicts.append(
            TrackVerdict(
                result.track_id,
                result.youtube_id,
                status,
                detail,
                result.mp3_path,
                result.ffprobe_duration_sec,
                result.decoded_duration_sec,
                result.annotation_duration_sec,
                "",
                digest,
            )
        )
    for row, duration in absent:
        failure = failures.get(row.youtube_id)
        status, detail = classify_track(None, "", failure)
        verdicts.append(
            TrackVerdict(
                row.track_id,
                row.youtube_id,
                status,
                detail,
                "",
                None,
                None,
                duration,
                failure.reason if failure else "",
                "",
            )
        )
    verdicts.sort(key=lambda verdict: verdict.track_id)

    checksum_path, checksum_count = (
        write_checksums(data_dir, verdicts) if checksums else (None, 0)
    )

    payload = build_payload(
        data_dir=data_dir,
        rows=rows,
        verdicts=verdicts,
        orphans=find_orphans(data_dir, {row.youtube_id for row in rows}),
        annotation_issues=check_annotations(data_dir, rows, tracks),
        checksums={"file": CHECKSUMS_FILE, "algorithm": "sha256", "files": checksum_count}
        if checksums
        else {"skipped": True},
    )
    _write_atomic(data_dir / VALIDATION_JSON, json.dumps(payload, indent=2, sort_keys=False) + "\n")
    _write_atomic(data_dir / VALIDATION_TXT, render_text_report(payload))
    if checksum_path is not None:
        payload["checksums"]["path"] = str(checksum_path)
    return payload


def build_payload(data_dir, rows, verdicts, orphans, annotation_issues, checksums) -> dict:
    """The machine-readable report: counts, verdict, and every track's row."""
    counts = tally(verdicts)
    converged, statement = convergence(counts, len(rows))
    all_clear, verdict_statement = overall_verdict(converged, orphans, annotation_issues)
    by_reason = collections.Counter(
        verdict.failure_reason for verdict in verdicts if verdict.status == STATUS_UNAVAILABLE
    )
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
        "data_dir": str(data_dir),
        "manifest_tracks": len(rows),
        "counts": {status: counts.get(status, 0) for status in STATUS_ORDER},
        "unavailable_by_reason": dict(sorted(by_reason.items())),
        "converged": converged,
        "convergence_statement": statement,
        "all_clear": all_clear,
        "verdict_statement": verdict_statement,
        "tolerance": {"abs_sec": gate.ABS_TOLERANCE_SEC, "rel": gate.REL_TOLERANCE},
        "checksums": checksums,
        "orphans": orphans,
        "annotation_issues": annotation_issues,
        "tracks": [verdict._asdict() for verdict in verdicts],
    }


# --------------------------------------------------------------------------- #
# Human report
# --------------------------------------------------------------------------- #


def _duration(value) -> str:
    return "?" if value is None else f"{value:.3f}"


def render_text_report(payload: dict) -> str:
    """The human summary: counts, verdict, and every track that is not OK.

    Nothing is elided.  The lists this prints in full -- UNAVAILABLE and
    DURATION_MISMATCH -- are exactly the ones the owner has to read and judge,
    and a "... and 47 more" line in the middle of them would defeat the purpose
    of the whole exercise.
    """
    lines = []
    add = lines.append

    add("raveform corpus validation")
    add("=" * 74)
    add(f"generated : {payload['generated_at_utc']}")
    add(f"data dir  : {payload['data_dir']}")
    add(f"manifest  : {payload['manifest_tracks']} annotated tracks")
    tolerance = payload["tolerance"]
    add(
        f"tolerance : duration agreement within max(+-{tolerance['abs_sec']:g} s, "
        f"+-{tolerance['rel'] * 100:g}%) of the annotation record"
    )
    add("")
    add("counts")
    for status in STATUS_ORDER:
        add(f"  {status:<20}: {payload['counts'].get(status, 0)}")
    add("")
    add(payload["convergence_statement"])
    add("")

    checksums = payload.get("checksums", {})
    if checksums.get("skipped"):
        add("checksums : SKIPPED for this run (--skip-checksums)")
    else:
        add(
            f"checksums : {checksums.get('files', 0)} OK file(s) hashed into "
            f"{checksums.get('file', CHECKSUMS_FILE)} (sha256sum -c format)"
        )
    add(
        f"orphans   : {len(payload['orphans'])} audio file(s) with no manifest row"
        + (f" -- {', '.join(payload['orphans'])}" if payload["orphans"] else "")
    )
    add(f"annotations: {len(payload['annotation_issues'])} issue(s)")
    for issue in payload["annotation_issues"]:
        add(f"  {issue}")
    add("")
    add(payload.get("verdict_statement", ""))

    tracks = payload["tracks"]

    unavailable = [t for t in tracks if t["status"] == STATUS_UNAVAILABLE]
    add("")
    add(f"UNAVAILABLE ({len(unavailable)}) -- never obtained, reason on record")
    by_reason = payload.get("unavailable_by_reason") or {}
    if by_reason:
        add("  by reason: " + ", ".join(f"{k}={v}" for k, v in by_reason.items()))
    add("-" * 74)
    for track in unavailable:
        add(f"  {track['track_id']:<22} {track['detail']}")

    mismatched = [t for t in tracks if t["status"] == STATUS_DURATION_MISMATCH]
    add("")
    add(f"DURATION_MISMATCH ({len(mismatched)}) -- on disk, kept, needs a human's eye")
    add("-" * 74)
    for track in mismatched:
        add(
            f"  {track['track_id']:<22} decoded {_duration(track['decoded_duration_sec'])} s "
            f"vs annotation {_duration(track['annotation_duration_sec'])} s"
        )

    for status in (STATUS_CORRUPT, STATUS_MISSING):
        rows = [t for t in tracks if t["status"] == status]
        add("")
        add(f"{status} ({len(rows)})" + ("" if rows else " -- none"))
        if rows:
            add("-" * 74)
        for track in rows:
            add(f"  {track['track_id']:<22} {track['detail']}")

    return "\n".join(lines) + "\n"


def print_summary(payload: dict) -> None:
    print()
    for status in STATUS_ORDER:
        print(f"  {status:<20}: {payload['counts'].get(status, 0)}")
    print()
    print(payload["convergence_statement"])
    if payload["orphans"]:
        print(f"  orphans           : {len(payload['orphans'])}")
    if payload["annotation_issues"]:
        print(f"  annotation issues : {len(payload['annotation_issues'])}")
    print(payload["verdict_statement"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=gate.default_data_dir(),
        help="corpus root; reads manifest.csv, annotations/, audio/, downloaded.txt, "
        "failed.jsonl and writes the validation artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=gate.DEFAULT_WORKERS,
        help="parallel ffmpeg decodes (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="do not hash the OK files -- for a quick intermediate pass during "
        "the retry loop; the final run must write the baseline",
    )
    parser.add_argument(
        "--prune-corrupt",
        action="store_true",
        help="delete CORRUPT files and drop their download-archive lines so a "
        "plain raveform_download.py re-run re-fetches them",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    data_dir = args.data_dir.resolve()
    print("raveform corpus validation")
    print(f"data dir: {data_dir}")

    gate.require_tools()
    started = time.time()
    payload = validate(data_dir, workers=args.workers, checksums=not args.skip_checksums)
    print_summary(payload)
    print(f"\nelapsed: {time.time() - started:.1f} s")
    print(f"report  : {data_dir / VALIDATION_JSON}")
    print(f"          {data_dir / VALIDATION_TXT}")
    if not args.skip_checksums:
        print(f"checksums: {data_dir / CHECKSUMS_FILE}")

    if args.prune_corrupt:
        verdicts = [TrackVerdict(**track) for track in payload["tracks"]]
        pruned = prune_corrupt(data_dir, verdicts)
        print(f"\npruned {len(pruned)} CORRUPT file(s); they are now re-fetchable")
        for youtube_id in pruned:
            print(f"  {youtube_id}")

    # Exit code is the verdict, so a supervisor or a CI step can branch on it --
    # and it is the FULL verdict, not just convergence.  A corpus that adds up
    # but whose annotations do not reconcile is not a corpus we are done with.
    return 0 if payload["all_clear"] else 1


if __name__ == "__main__":
    sys.exit(main())
