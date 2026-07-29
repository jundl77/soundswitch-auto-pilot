#!/usr/bin/env python
"""Download and validate the Raveform EDM section annotations."""

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

MEMBER_PREFIX = "raveform/structures/"

SEGMENTS_FILE = "segments.json"
BEATS_DIR = "beats"

_USER_AGENT = "soundswitch-auto-pilot/raveform-fetch (stdlib urllib)"
_READ_BUFFER = 4 << 20
_MAX_ATTEMPTS = 3

EXPECTED_LABELS = frozenset(
    {"intro", "buildup", "breakdown", "drop", "cooldown", "outro", "altoutro"}
)


# A RuntimeError, not an OSError, so it escapes the transient-retry path below.
class _RangeUnsupported(RuntimeError):
    pass


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        self.size = self._probe_size()

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

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)

    def _fetch(self, start: int, end: int) -> tuple:
        byte_range = f"bytes={start}-{end}"
        expected = end - start + 1
        last_err = None

        for attempt in range(_MAX_ATTEMPTS):
            try:
                req = urllib.request.Request(
                    self.url, headers={"User-Agent": _USER_AGENT, "Range": byte_range}
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    # Check before touching the body: a 200 means the server
                    # ignored Range and is about to hand us the whole 479 MB.
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
                if exc.code in (401, 403, 404):
                    raise RuntimeError(
                        f"HTTP {exc.code} for {self.url} -- the dataset may be gated, "
                        f"renamed or removed. Server said: {exc.reason}"
                    ) from exc
                last_err = exc
            except (OSError, http.client.HTTPException) as exc:
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
    raw = HttpRangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=_READ_BUFFER))


def annotations_dir(data_dir: Path) -> Path:
    return data_dir / "annotations"


def _destination(ann_dir: Path, member: str) -> Path:
    relative = member[len(MEMBER_PREFIX) :]
    dest = (ann_dir / relative).resolve()
    if ann_dir.resolve() not in dest.parents and dest != ann_dir.resolve():
        raise RuntimeError(f"archive member escapes the target directory: {member}")
    return dest


def sweep_partials(ann_dir: Path) -> int:
    removed = 0
    for stale in ann_dir.rglob("*.part"):
        try:
            stale.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"cleanup : removed {removed} stale .part file(s) from a previous run")
    return removed


def fetch_annotations(data_dir: Path, url: str = ARCHIVE_URL) -> dict:
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
        # Sorting by archive offset keeps the range requests contiguous.
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


def load_tracks(data_dir: Path) -> list:
    path = annotations_dir(data_dir) / SEGMENTS_FILE
    if not path.exists():
        raise RuntimeError(f"missing {path} -- run the fetch step first")
    with open(path, "r", encoding="utf-8") as handle:
        tracks = json.load(handle)
    if not isinstance(tracks, list):
        raise RuntimeError(f"{SEGMENTS_FILE}: expected a JSON list, got {type(tracks).__name__}")
    return tracks


def parse_sections(track: dict) -> list:
    return [
        (float(section["start"]), float(section["end"]), str(section["name"]))
        for section in track["sections"]
    ]


def youtube_id(track: dict) -> str:
    return str(track["id"])


def parse_beat_csv(path: Path) -> list:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [
            (float(row["time"]), int(row["downbeat"]), row["section"])
            for row in csv.DictReader(handle)
        ]


def beat_csv_path(data_dir: Path, track: dict) -> Path:
    return annotations_dir(data_dir) / BEATS_DIR / f"{track['key']}.beat.csv"


def validate(data_dir: Path) -> int:
    tracks = load_tracks(data_dir)
    print()
    print(f"tracks found: {len(tracks)}")

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

    with_ids = [t for t in tracks if youtube_id(t)]
    unique_ids = {youtube_id(t) for t in with_ids}
    print()
    print(f"youtube ids : {len(with_ids)}/{len(tracks)} tracks ({len(unique_ids)} unique)")
    print(f"  examples  : {', '.join(youtube_id(t) for t in tracks[:3])}")

    present = sum(1 for t in tracks if beat_csv_path(data_dir, t).exists())
    print(f"beat grids  : {present}/{len(tracks)} tracks have a beat CSV")

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


def default_data_dir() -> Path:
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
