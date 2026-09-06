#!/usr/bin/env python
"""Convert RWC Popular Music into third-party corpus records.

Three sources meet here and none of them is authoritative on its own: AIST's
CHORUS files carry the section labels (100 Hz integer frames, quoted labels, an
optional parenthesised key shift), the 2025 preprocessed CSVs carry the beat
grid that supersedes the archive's own sentinel-encoded one, and ``metadata.csv``
carries the title, artist, genre and the duration measured off the wav.  The
labels are stored VERBATIM in RWC's own 16-name vocabulary -- mapping them into
the show's label space is a later, separate decision, and doing it here would
bake one reading of ``chorus A`` into the dataset with nothing able to see it.

The beat grid is written in the corpus's published three-column format so
admission finds it and keeps it rather than running madmom over music whose
downbeats are already hand-checked; RWC-Pop is not all 4/4, so bar positions are
carried as the source states them and the tracks that exceed four are counted
rather than folded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_TRAINING_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_TRAINING_DIR.parent), str(_TRAINING_DIR),
              str(_TRAINING_DIR / "raveform"),
              str(_TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from corpus_root import corpus_dir  # noqa: E402
from hand_label_admission import beat_rows, write_beat_csv  # noqa: E402
from tp_record import (  # noqa: E402
    ID_PREFIXES,
    RECORD_SCHEMA,
    annotation_path,
    audio_path,
    beat_csv_path,
    third_party_dir,
    validate_record,
    write_record,
)

SOURCE = "rwc"
CONVERTER = "rwc_convert"
PIECE_COUNT = 100

CHORUS_DIR = "AIST_RWC-MDB-P-2001_CHORUS"
BEATS_DIR = Path("01_annotations_preprocessed") / "beats" / "RWC-P"
METADATA_FILE = "metadata.csv"
REPORT_FILE = "rwc_convert_report.json"
PARTIAL_REPORT_FILE = "rwc_convert_report.partial.json"
ALL_PIECES = tuple(range(1, PIECE_COUNT + 1))

FRAME_RATE = 100.0
LABEL_VOCABULARY = "rwc_aist_chorus"
BEATS_SOURCE = "rwc_preprocessed"
LICENSE = ("CC BY-NC 4.0; AIST README: research use only, "
           "do not redistribute")

AUDIO_SUFFIXES = (".wav", ".aiff")
BEAT_CSV_FIELDS = ["t", "beat"]
NOMINAL_BAR = 4

_FRAMES = re.compile(r"\d+")
_KEY_SHIFT = re.compile(r"\(([+-]\d+)\)")


class ConvertError(Exception):
    pass


class Section(NamedTuple):
    name: str
    start: float
    end: float
    key_shift: int | None


class Sources(NamedTuple):
    archive: Path
    preprocessed: Path
    audio: Path | None


class Verdict(NamedTuple):
    track_id: str
    reason: str
    names: tuple
    max_position: int
    labelled_sec: float
    duration_sec: float


def native_id(number: int) -> str:
    return f"RWC_P{number:03d}"


def track_id(number: int) -> str:
    return f"{ID_PREFIXES[SOURCE]}p{number:03d}"


def chorus_path(archive: Path, number: int) -> Path:
    return archive / CHORUS_DIR / f"RM-P{number:03d}.CHORUS.TXT"


def beats_path(preprocessed: Path, number: int) -> Path:
    return preprocessed / BEATS_DIR / f"{native_id(number)}.csv"


def metadata_path(preprocessed: Path) -> Path:
    return preprocessed / METADATA_FILE


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# The CHORUS files
# --------------------------------------------------------------------------- #


def _chorus_rows(text: str) -> list:
    rows = []
    for number, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) not in (3, 4):
            raise ConvertError(
                f"line {number}: {len(fields)} tab-separated fields, want 3 or 4")
        start, end, label = fields[0], fields[1], fields[2]
        if not _FRAMES.fullmatch(start) or not _FRAMES.fullmatch(end):
            raise ConvertError(
                f"line {number}: frame times {start!r} {end!r} are not integers")
        if len(label) < 2 or not label.startswith('"') or not label.endswith('"'):
            raise ConvertError(
                f"line {number}: label {label!r} is not double-quote wrapped")
        shift = None
        if len(fields) == 4:
            found = _KEY_SHIFT.fullmatch(fields[3].strip())
            if found is None:
                raise ConvertError(
                    f"line {number}: key shift {fields[3]!r} is not (+n) or (-n)")
            shift = int(found.group(1))
        rows.append((int(start), int(end), label[1:-1], shift))
    if not rows:
        raise ConvertError("no section lines")
    return rows


def _check_partition(rows: list) -> None:
    previous_end = None
    for index, (start, end, _name, _shift) in enumerate(rows, 1):
        if end <= start:
            raise ConvertError(
                f"line {index}: empty or reversed span {start}..{end} frames")
        if previous_end is not None and start != previous_end:
            gap = "gap" if start > previous_end else "overlap"
            raise ConvertError(
                f"line {index}: {gap} -- starts at {start} frames, previous "
                f"ended at {previous_end}")
        previous_end = end


def parse_chorus(text: str) -> list:
    rows = _chorus_rows(text)
    _check_partition(rows)
    return [Section(name, start / FRAME_RATE, end / FRAME_RATE, shift)
            for start, end, name, shift in rows]


def read_chorus(path: Path) -> list:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ConvertError(f"{path.name}: {error}") from error
    return parse_chorus(text)


def section_dicts(sections: list) -> list:
    out = []
    for section in sections:
        entry = {"name": section.name, "start": section.start,
                 "end": section.end}
        if section.key_shift is not None:
            entry["key_shift"] = section.key_shift
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Metadata, beats and audio
# --------------------------------------------------------------------------- #


def load_metadata(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"no metadata at {path}")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return {row["RWCID"]: row
                for row in csv.DictReader(handle, delimiter=";")}


def _number(text: str, field: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError) as error:
        raise ConvertError(f"metadata {field} is not a number: {text!r}") from error


def read_beats(path: Path) -> list:
    if not path.exists():
        raise ConvertError(f"no preprocessed beat csv at {path}")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames != BEAT_CSV_FIELDS:
            raise ConvertError(
                f"{path.name}: header {reader.fieldnames} is not "
                f"{BEAT_CSV_FIELDS}")
        rows = list(reader)
    beats = []
    for index, row in enumerate(rows, 2):
        try:
            beats.append((float(row["t"]), int(row["beat"])))
        except (TypeError, ValueError) as error:
            raise ConvertError(f"{path.name} line {index}: {error}") from error
    if not beats:
        raise ConvertError(f"{path.name}: no beats")
    return beats


def audio_index(root: Path | None) -> dict:
    if root is None or not root.exists():
        return {}
    found = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in AUDIO_SUFFIXES and path.is_file():
            found.setdefault(path.stem.upper(), path)
    return found


def copy_audio(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return "kept"
    shutil.copy2(source, target)
    return "copied"


# --------------------------------------------------------------------------- #
# One piece
# --------------------------------------------------------------------------- #


def carry_timestamp(path: Path, record: dict) -> dict:
    """Keep an unchanged record's original stamp, so a re-run rewrites nothing.

    A per-run timestamp is the one field that differs when nothing has: without
    this, a second conversion rewrites all 100 files and no diff can say whether
    the corpus actually moved.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except (OSError, ValueError):
        return record
    if not isinstance(existing, dict):
        return record
    stamp = (existing.get("provenance") or {}).get("converted_utc")
    if not isinstance(stamp, str) or not stamp:
        return record
    probe = copy.deepcopy(record)
    probe["provenance"]["converted_utc"] = stamp
    if probe == existing:
        record["provenance"]["converted_utc"] = stamp
    return record


def convert_piece(number: int, sources: Sources, metadata: dict,
                  audio_files: dict, corpus: Path, copy: bool) -> Verdict:
    identifier = track_id(number)
    native = native_id(number)
    chorus = chorus_path(sources.archive, number)
    if not chorus.exists():
        raise ConvertError(f"no chorus file at {chorus}")
    sections = read_chorus(chorus)

    row = metadata.get(native)
    if row is None:
        raise ConvertError(f"no {native} row in {METADATA_FILE}")
    duration = _number(row.get("duration", ""), "duration")
    tempo = row.get("Tempo", "").strip()

    source_audio = audio_files.get(native)
    if source_audio is None and copy:
        raise ConvertError(f"no audio file for {native} under {sources.audio}")
    suffix = source_audio.suffix.lower() if source_audio else AUDIO_SUFFIXES[0]
    audio_name = f"{identifier}{suffix}"

    beats_csv = beats_path(sources.preprocessed, number)
    beats = read_beats(beats_csv)

    record = {
        "schema": RECORD_SCHEMA,
        "source": SOURCE,
        "id": identifier,
        "native_id": native,
        "title": row.get("Title", ""),
        "artist": row.get("Artist", ""),
        "genre": row.get("GenreSub", ""),
        "audio": audio_name,
        "duration": duration,
        "label_vocabulary": LABEL_VOCABULARY,
        "beats": BEATS_SOURCE,
        "sections": section_dicts(sections),
        "masked": [],
        "provenance": {
            "converter": CONVERTER,
            "converted_utc": _now_utc(),
            "chorus_file": chorus.name,
            "chorus_sha256": sha256(chorus),
            "beats_file": beats_csv.name,
            "beats_sha256": sha256(beats_csv),
            "metadata_file": METADATA_FILE,
            "tempo_bpm": float(tempo) if tempo else None,
            "license": LICENSE,
        },
    }
    try:
        validate_record(record)
    except ValueError as error:
        raise ConvertError(str(error)) from error

    if copy:
        copy_audio(source_audio, audio_path(corpus, record))

    grid = beat_csv_path(corpus, record)
    if not grid.exists():
        grid.parent.mkdir(parents=True, exist_ok=True)
        write_beat_csv(grid, beat_rows(beats, record["sections"]))

    target = annotation_path(corpus, identifier, SOURCE)
    write_record(target, carry_timestamp(target, record))

    return Verdict(
        identifier, "",
        tuple(section.name for section in sections),
        max(position for _time, position in beats),
        sum(section.end - section.start for section in sections),
        duration,
    )


# --------------------------------------------------------------------------- #
# The batch
# --------------------------------------------------------------------------- #


def report_file(pieces: list) -> str:
    """A restricted run must not land where a whole-set report belongs.

    A three-piece report sitting at the full name silently stops covering the
    other 97, and nothing downstream can tell the two apart.
    """
    return REPORT_FILE if tuple(pieces) == ALL_PIECES else PARTIAL_REPORT_FILE


def _report(verdicts: list, failures: list, pieces: list) -> dict:
    labels: dict = {}
    odd_meter = []
    for verdict in verdicts:
        for name in verdict.names:
            labels[name] = labels.get(name, 0) + 1
        if verdict.max_position > NOMINAL_BAR:
            odd_meter.append({"id": verdict.track_id,
                              "max_position": verdict.max_position})
    name = report_file(pieces)
    return {
        "converter": CONVERTER,
        "source": SOURCE,
        "report_file": name,
        "partial": name != REPORT_FILE,
        "pieces_requested": len(pieces),
        "converted": len(verdicts),
        "failed": len(failures),
        "failures": [{"id": identifier, "reason": reason}
                     for identifier, reason in failures],
        "labels": dict(sorted(labels.items())),
        "nothing_sections": labels.get("nothing", 0),
        "odd_meter_count": len(odd_meter),
        "odd_meter_pieces": odd_meter,
        "labelled_sec": round(sum(v.labelled_sec for v in verdicts), 3),
        "audio_sec": round(sum(v.duration_sec for v in verdicts), 3),
    }


def write_report(corpus: Path, report: dict) -> Path:
    path = third_party_dir(corpus) / report["report_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def print_report(report: dict) -> None:
    print(f"converted {report['converted']}/{report['pieces_requested']}, "
          f"failed {report['failed']}")
    for failure in report["failures"]:
        print(f"  FAIL {failure['id']}: {failure['reason']}")
    print(f"labels ({len(report['labels'])} distinct): " + ", ".join(
        f"{name}={count}" for name, count in report["labels"].items()))
    print(f"'nothing' sections: {report['nothing_sections']}")
    print(f"beat positions above {NOMINAL_BAR}: {report['odd_meter_count']} "
          f"piece(s)")
    for piece in report["odd_meter_pieces"]:
        print(f"  {piece['id']} max position {piece['max_position']}")
    print(f"labelled {report['labelled_sec']:.3f} s of "
          f"{report['audio_sec']:.3f} s of audio")
    print(f"report written to {report['report_file']}"
          + (" (partial run)" if report["partial"] else ""))


def convert(sources: Sources, corpus: Path, pieces: list,
            copy: bool = True) -> dict:
    metadata = load_metadata(metadata_path(sources.preprocessed))
    audio_files = audio_index(sources.audio)
    verdicts = []
    failures = []
    for number in pieces:
        try:
            verdicts.append(convert_piece(number, sources, metadata,
                                          audio_files, corpus, copy))
        except (ConvertError, OSError) as error:
            # An I/O failure on one piece -- an unreadable csv, a copy that runs
            # out of disk -- is that piece's failure, not the batch's.
            reason = (str(error) if isinstance(error, ConvertError)
                      else f"{type(error).__name__}: {error}")
            failures.append((track_id(number), reason))
    report = _report(verdicts, failures, pieces)
    write_report(corpus, report)
    return report


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, required=True,
                        help="the AIST annotation archive root")
    parser.add_argument("--preprocessed", type=Path, required=True,
                        help="the preprocessed annotation root (metadata.csv, "
                             "01_annotations_preprocessed/)")
    parser.add_argument("--audio", type=Path, default=None,
                        help="root the RWC-P wavs were extracted to")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root (default: the resolved corpus)")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert only the first N pieces")
    parser.add_argument("--pieces", type=int, nargs="+", default=None,
                        help="explicit piece numbers")
    parser.add_argument("--no-audio", action="store_true",
                        help="convert annotations only, skip the audio copy")
    args = parser.parse_args(argv)

    if args.audio is None and not args.no_audio:
        parser.error("--audio is required unless --no-audio is passed")

    pieces = args.pieces or list(range(1, PIECE_COUNT + 1))
    if args.limit is not None:
        pieces = pieces[:args.limit]

    corpus = args.data_dir.resolve() if args.data_dir else corpus_dir()
    report = convert(
        Sources(args.archive.resolve(), args.preprocessed.resolve(),
                args.audio.resolve() if args.audio else None),
        corpus, pieces, copy=not args.no_audio)
    print_report(report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
