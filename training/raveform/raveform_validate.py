#!/usr/bin/env python
"""Full-corpus validation: every annotated track accounted for, or named."""

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

import build_clean_manifest as gate  # noqa: E402
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

ACCOUNTED_FOR = (STATUS_OK, STATUS_DURATION_MISMATCH, STATUS_UNAVAILABLE)

_GATE_TO_STATUS = {
    gate.STATUS_OK: STATUS_OK,
    gate.STATUS_MISMATCH: STATUS_DURATION_MISMATCH,
    gate.STATUS_CORRUPT: STATUS_CORRUPT,
}

MANIFEST_ROUNDING_SEC = 0.001

_DETAIL_CHARS = 300

_HASH_CHUNK = 1 << 20


class Failure(NamedTuple):
    youtube_id: str
    reason: str
    error: str
    timestamp: str


class TrackVerdict(NamedTuple):
    track_id: str
    youtube_id: str
    status: str
    detail: str
    mp3_path: str
    ffprobe_duration_sec: float | None
    decoded_duration_sec: float | None
    annotation_duration_sec: float
    failure_reason: str
    sha256: str


def load_failures(path: Path) -> dict:
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
                str(record.get("reason") or "other"),
                str(record.get("error") or ""),
                str(record.get("timestamp") or ""),
            )
    return failures


def _last_line(text: str, limit: int = _DETAIL_CHARS) -> str:
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1][:limit] if lines else ""


def classify_track(gate_status, gate_detail: str, failure) -> tuple:
    # The gate's verdict outranks the failure log unconditionally: failed.jsonl
    # is append-only, so a track fetched on a later cycle is in both places and
    # only the bytes on disk describe what we now hold.
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
    counts = collections.Counter({status: 0 for status in STATUS_ORDER})
    counts.update(verdict.status for verdict in verdicts)
    return counts


def convergence(counts, manifest_tracks: int) -> tuple:
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


def retryable_remainder(verdicts: list) -> list:
    return sorted(
        verdict.track_id
        for verdict in verdicts
        if verdict.status == STATUS_UNAVAILABLE
        and verdict.failure_reason in downloader.RETRYABLE_REASONS
    )


def overall_verdict(converged: bool, orphans: list, annotation_issues: list) -> tuple:
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


def find_orphans(data_dir: Path, known_youtube_ids) -> list:
    # Only *.mp3: audio/ also holds .npy decode caches and transient .part
    # files, and counting those would invent a finding on every run.
    audio = data_dir / gate.AUDIO_DIR
    if not audio.is_dir():
        return []
    known = set(known_youtube_ids)
    return sorted(
        path.stem for path in audio.glob("*.mp3") if path.stem not in known
    )


def annotation_durations(tracks: list) -> dict:
    return {str(track["key"]): float(track["duration"]) for track in tracks}


def check_annotations(data_dir: Path, rows: list, tracks: list) -> list:
    issues = []
    by_track = {}
    parsed = {}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(data_dir: Path, verdicts: list) -> tuple:
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


def prune_corrupt(data_dir: Path, verdicts: list) -> list:
    # Both halves are required: yt-dlp consults its archive first, so deleting
    # the file alone leaves a track that can never be re-fetched.
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


def validate(
    data_dir: Path,
    workers: int = gate.DEFAULT_WORKERS,
    checksums: bool = True,
    progress_every: int = 100,
) -> dict:
    data_dir = Path(data_dir)
    rows = gate.load_manifest_rows(data_dir)
    tracks = load_tracks(data_dir)
    durations = annotation_durations(tracks)
    failures = load_failures(downloader.failed_path(data_dir))

    jobs = []
    absent = []
    for row in rows:
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
        "retryable_remainder": retryable_remainder(verdicts),
        "tolerance": {"abs_sec": gate.ABS_TOLERANCE_SEC, "rel": gate.REL_TOLERANCE},
        "checksums": checksums,
        "orphans": orphans,
        "annotation_issues": annotation_issues,
        "tracks": [verdict._asdict() for verdict in verdicts],
    }


def _duration(value) -> str:
    return "?" if value is None else f"{value:.3f}"


def render_text_report(payload: dict) -> str:
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
    remainder = payload.get("retryable_remainder") or []
    if remainder:
        add(
            f"WARNING   : converged with a retryable remainder -- {len(remainder)} "
            "UNAVAILABLE track(s) failed for a reason worth re-attempting."
        )
        add(f"            Re-run the downloader with {downloader.RETRY_HINT}, then re-validate.")
        add("            " + ", ".join(remainder))

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
    if payload["retryable_remainder"]:
        print(
            f"  WARNING           : {len(payload['retryable_remainder'])} UNAVAILABLE "
            "track(s) failed for a retryable reason -- see the report"
        )
    print(payload["verdict_statement"])


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

    return 0 if payload["all_clear"] else 1


if __name__ == "__main__":
    sys.exit(main())
