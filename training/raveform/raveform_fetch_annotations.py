#!/usr/bin/env python
"""Download and validate the Raveform EDM section annotations.

The Hugging Face dataset ``taejunkim/raveform`` publishes everything as a single
479 MB Zip64 archive (``raveform.zip``) at the repo root -- 74,423 members, of
which only the ``raveform/structures/`` subtree is section-annotation material:

    raveform/structures/segments.json   the 1,423 annotated track records
    raveform/structures/beats/*.csv     one beat grid per track (beat -> section)

The remaining ~1.2 GB (``alignments/``, ``beats/mixes/``, ``beats/tracks/``) is
DJ-mix alignment data this project does not use.  Rather than pull the whole
archive for 3% of its content, this script mounts the remote zip over HTTP range
requests and extracts just that subtree -- roughly 15 MB of transfer per run.

Produced layout under ``<data-dir>/annotations/``::

    segments.json
    beats/<index>.<youtube_id>.beat.csv

Stdlib only.  Idempotent: a member already on disk at its recorded uncompressed
size is left untouched, so re-running only re-reads the archive index.

Usage::

    uv run python training/raveform/raveform_fetch_annotations.py \
        --data-dir C:\\Users\\Julian\\Projects\\soundswitch-auto-pilot\\training\\data\\raveform
"""

from __future__ import annotations

import argparse
import collections
import csv
import http.client
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ID = "taejunkim/raveform"
ARCHIVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/raveform.zip"

# Only this subtree of the archive holds section annotations.
MEMBER_PREFIX = "raveform/structures/"

SEGMENTS_FILE = "segments.json"
BEATS_DIR = "beats"

_USER_AGENT = "soundswitch-auto-pilot/raveform-fetch (stdlib urllib)"
_READ_BUFFER = 4 << 20  # 4 MiB -- keeps the number of range requests small
_MAX_ATTEMPTS = 3

# Labels the plan expects; anything outside this set is reported, not rejected.
EXPECTED_LABELS = frozenset(
    {"intro", "buildup", "breakdown", "drop", "cooldown", "outro", "altoutro"}
)


# --------------------------------------------------------------------------- #
# Remote zip access
# --------------------------------------------------------------------------- #


class _RangeUnsupported(RuntimeError):
    """The server would not serve a byte range -- selective extraction is off.

    Deliberately a ``RuntimeError`` and not an ``OSError``: it must escape the
    transient-error retry path rather than be retried three times.
    """


class HttpRangeFile(io.RawIOBase):
    """A read-only seekable file backed by HTTP range requests.

    Requests always target the original Hugging Face ``resolve`` URL: urllib
    follows the redirect to the CDN and carries the ``Range`` header along, so
    there is no signed-URL expiry to manage across a long run.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        self.size = self._probe_size()

    # -- plumbing ----------------------------------------------------------- #

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.pos
        if size <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + size, self.size) - 1
        data = self._get_range(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, buf) -> int:  # BufferedReader talks to us through this
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)

    # -- HTTP --------------------------------------------------------------- #

    def _fetch(self, start: int, end: int) -> tuple:
        """One full request-and-read cycle, retried as a unit.

        The retry must span the body read, not just ``urlopen``: for multi-MB
        ranges the common failure is a mid-body ``IncompleteRead`` or reset
        connection, which would otherwise surface as a raw traceback through
        zipfile. A short read is treated as a failure and retried too.

        Returns ``(payload, content_range_header)``.
        """
        byte_range = f"bytes={start}-{end}"
        expected = end - start + 1
        last_err = None

        for attempt in range(_MAX_ATTEMPTS):
            try:
                req = urllib.request.Request(
                    self.url, headers={"User-Agent": _USER_AGENT, "Range": byte_range}
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    # Validate BEFORE touching the body: a 200 here means the
                    # server ignored Range and is about to hand us the entire
                    # 479 MB entity, which we must never pull into memory.
                    if resp.status != 206:
                        raise _RangeUnsupported(
                            f"server ignored the Range request (HTTP {resp.status}, "
                            f"expected 206) -- cannot stream the archive selectively"
                        )
                    content_range = resp.headers.get("Content-Range")
                    if not content_range:
                        raise _RangeUnsupported(
                            "server returned 206 without a Content-Range header -- "
                            "cannot stream the archive selectively"
                        )
                    payload = resp.read()
                if len(payload) != expected:
                    raise http.client.IncompleteRead(payload, expected - len(payload))
                return payload, content_range
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 404):  # gated / moved / renamed: do not retry
                    raise RuntimeError(
                        f"HTTP {exc.code} for {self.url} -- the dataset may be gated, "
                        f"renamed or removed. Server said: {exc.reason}"
                    ) from exc
                last_err = exc
            except (OSError, http.client.HTTPException) as exc:
                # Covers URLError, ConnectionReset/Aborted, socket timeouts and
                # IncompleteRead -- all transient, all worth another attempt.
                last_err = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))

        raise RuntimeError(
            f"range request {byte_range} failed after {_MAX_ATTEMPTS} attempts: "
            f"{type(last_err).__name__}: {last_err}"
        )

    def _probe_size(self) -> int:
        _payload, content_range = self._fetch(0, 0)
        if "/" not in content_range:
            raise _RangeUnsupported(f"malformed Content-Range: {content_range!r}")
        return int(content_range.rsplit("/", 1)[1])

    def _get_range(self, start: int, end: int) -> bytes:
        payload, _content_range = self._fetch(start, end)
        return payload


def open_remote_zip(url: str = ARCHIVE_URL) -> zipfile.ZipFile:
    """Open the published archive without downloading it in full."""
    raw = HttpRangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=_READ_BUFFER))


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


def annotations_dir(data_dir: Path) -> Path:
    return data_dir / "annotations"


def _destination(ann_dir: Path, member: str) -> Path:
    """Map an archive member to its on-disk path, refusing path traversal."""
    relative = member[len(MEMBER_PREFIX) :]
    dest = (ann_dir / relative).resolve()
    if ann_dir.resolve() not in dest.parents and dest != ann_dir.resolve():
        raise RuntimeError(f"archive member escapes the target directory: {member}")
    return dest


def sweep_partials(ann_dir: Path) -> int:
    """Delete leftover ``*.part`` files from a killed or failed earlier run."""
    removed = 0
    for stale in ann_dir.rglob("*.part"):
        try:
            stale.unlink()
            removed += 1
        except OSError:  # locked/vanished -- not worth failing the fetch over
            pass
    if removed:
        print(f"cleanup : removed {removed} stale .part file(s) from a previous run")
    return removed


def fetch_annotations(data_dir: Path, url: str = ARCHIVE_URL) -> dict:
    """Extract the annotation subtree into ``<data-dir>/annotations/``.

    Returns a small stats dict: extracted / skipped counts and bytes written.
    """
    ann_dir = annotations_dir(data_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"archive : {url}")
    print(f"target  : {ann_dir}")

    sweep_partials(ann_dir)

    archive = open_remote_zip(url)
    with archive:
        members = [
            info
            for info in archive.infolist()
            if info.filename.startswith(MEMBER_PREFIX) and not info.is_dir()
        ]
        if not members:
            raise RuntimeError(
                f"no members under {MEMBER_PREFIX!r} in the archive -- the dataset "
                f"layout changed; re-run layout discovery"
            )
        # Sequential by archive offset keeps the range requests contiguous.
        members.sort(key=lambda info: info.header_offset)

        extracted = skipped = written = 0
        for info in members:
            dest = _destination(ann_dir, info.filename)
            if dest.exists() and dest.stat().st_size == info.file_size:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                with archive.open(info) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out)
                tmp.replace(dest)
            except BaseException:
                # Never leave a half-written .part behind (BaseException so a
                # Ctrl-C mid-copy cleans up too).
                tmp.unlink(missing_ok=True)
                raise
            extracted += 1
            written += info.file_size

    print(
        f"members : {len(members)} under {MEMBER_PREFIX} "
        f"({extracted} extracted, {skipped} already present, "
        f"{written / 1e6:.1f} MB written)"
    )
    return {
        "members": len(members),
        "extracted": extracted,
        "skipped": skipped,
        "bytes_written": written,
    }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def load_tracks(data_dir: Path) -> list:
    """Load the track records from ``annotations/segments.json``."""
    path = annotations_dir(data_dir) / SEGMENTS_FILE
    if not path.exists():
        raise RuntimeError(f"missing {path} -- run the fetch step first")
    with open(path, "r", encoding="utf-8") as handle:
        tracks = json.load(handle)
    if not isinstance(tracks, list):
        raise RuntimeError(f"{SEGMENTS_FILE}: expected a JSON list, got {type(tracks).__name__}")
    return tracks


def parse_sections(track: dict) -> list:
    """Return ``[(start_sec, end_sec, label), ...]`` for one track record."""
    return [
        (float(section["start"]), float(section["end"]), str(section["name"]))
        for section in track["sections"]
    ]


def youtube_id(track: dict) -> str:
    """Return the track's YouTube video ID."""
    return str(track["id"])


def parse_beat_csv(path: Path) -> list:
    """Return ``[(time_sec, beat_in_bar, section_label), ...]`` for one beat grid."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [
            (float(row["time"]), int(row["downbeat"]), row["section"])
            for row in csv.DictReader(handle)
        ]


def beat_csv_path(data_dir: Path, track: dict) -> Path:
    return annotations_dir(data_dir) / BEATS_DIR / f"{track['key']}.beat.csv"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(data_dir: Path) -> int:
    """Print the validation summary. Returns the track count."""
    tracks = load_tracks(data_dir)
    print()
    print(f"tracks found: {len(tracks)}")

    # --- label vocabulary over ALL tracks --------------------------------- #
    labels = collections.Counter()
    for track in tracks:
        for _start, _end, label in parse_sections(track):
            labels[label] += 1
    print(f"section labels ({len(labels)} distinct, raw as published):")
    for label, count in labels.most_common():
        marker = "" if label in EXPECTED_LABELS else "   <-- outside expected set"
        print(f"  {label:<12} {count:>6}{marker}")
    unexpected = sorted(set(labels) - EXPECTED_LABELS)
    if unexpected:
        print(f"NOTE: labels beyond the expected vocabulary: {', '.join(unexpected)}")

    # --- YouTube ID coverage ---------------------------------------------- #
    with_ids = [t for t in tracks if youtube_id(t)]
    unique_ids = {youtube_id(t) for t in with_ids}
    print()
    print(f"youtube ids : {len(with_ids)}/{len(tracks)} tracks ({len(unique_ids)} unique)")
    print(f"  examples  : {', '.join(youtube_id(t) for t in tracks[:3])}")

    # --- beat grid coverage ------------------------------------------------ #
    present = sum(1 for t in tracks if beat_csv_path(data_dir, t).exists())
    print(f"beat grids  : {present}/{len(tracks)} tracks have a beat CSV")

    # --- parse three tracks end to end ------------------------------------- #
    print()
    print("sample parses (3 tracks, sections + beat grid):")
    for track in tracks[:3]:
        sections = parse_sections(track)
        beats = parse_beat_csv(beat_csv_path(data_dir, track))
        print(
            f"  {track['key']}  id={youtube_id(track)}  "
            f"{len(sections)} sections  {len(beats)} beats  "
            f"{float(track['duration']):.1f}s  {track.get('genre', '?')}"
        )

    example = tracks[0]
    print()
    print(f"parsed example -- {example['key']} ({example['title']}):")
    for start, end, label in parse_sections(example):
        print(f"  ({start:9.3f}, {end:9.3f}, {label!r})")

    return len(tracks)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_data_dir() -> Path:
    # parents[2] is the repo root: this file sits in training/raveform/.
    return Path(__file__).resolve().parents[2] / "training" / "data" / "raveform"


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="corpus root; annotations land in <data-dir>/annotations (default: %(default)s)",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip the download and only validate what is already on disk",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    print("raveform annotation fetch")
    print(f"data dir: {data_dir}")

    if not args.no_fetch:
        fetch_annotations(data_dir)

    validate(data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
