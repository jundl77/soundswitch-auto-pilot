#!/usr/bin/env python
"""Convert the SALAMI 2.0 public annotations into third-party corpus records.

The RAW annotator files are the input, never ``annotations/<id>/parsed/*``.  A
raw line is ``TIME<TAB>LABEL`` where LABEL is a comma-separated token list that
interleaves all three of SALAMI's layers on one line -- the large-scale
similarity letter (``A``, ``B'``, ``V'``), the small-scale one (``a``, ``b''``),
the function word (``Verse``, ``Chorus``, ``Silence``) -- plus a parenthesised
instrumentation annotation that opens on one line and closes on a later one.
The parsed ``*_functions.txt`` layer flattens that to one label per line and
writes ``no_function`` wherever a segment carried a structural letter and no
function word, so the letter -- exactly the ``V'``/``W'`` variant the raw file
states -- is destroyed.  This converter reads the raw file and keeps all three.

A boundary is a row carrying a function word or a large-scale letter, which
reproduces SALAMI's own functions grid; a segment with no function word is not a
section at all but a MASKED span, and so is one whose function names no music
(``Silence``).  Masked spans still carry their structural letter, so nothing the
annotator wrote is lost to the masking.  Labels are stored VERBATIM in SALAMI's
own function vocabulary -- the only transform is the CASE normalisation the
shipped ``funct_vocab_dictionary.txt`` states, applied only for the entries
whose two sides differ by case alone.

SALAMI ships no beat grid, so this converter writes no beat CSV and says
``madmom_offline_pending`` in the record: admission generates the grid, and a
consumer must be able to see that it is generated rather than expert.
"""

from __future__ import annotations

import argparse
import collections
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
from tp_record import (  # noqa: E402
    ID_PREFIXES,
    RECORD_SCHEMA,
    annotation_path,
    audio_path,
    third_party_dir,
    validate_record,
    write_record,
)

SOURCE = "salami"
CONVERTER = "salami_convert"

ANNOTATIONS_DIR = "annotations"
AUDIO_DIR = "audio"
METADATA_FILE = Path("metadata") / "metadata.csv"
VOCAB_FILE = "funct_vocab_dictionary.txt"
TEXTFILES = {1: "textfile1.txt", 2: "textfile2.txt"}

REPORT_FILE = "salami_convert_report.json"
PARTIAL_REPORT_FILE = "salami_convert_report.partial.json"

LABEL_VOCABULARY = "salami_function_raw"
BEATS_SOURCE = "madmom_offline_pending"
LICENSE = "annotations CC0 1.0 (SALAMI 2.0); audio is not SALAMI's to license"

LIVE_ARCHIVE_CLASS = "Live_Music_Archive"

# The parsed layer's marker for a segment the annotator gave no function word.
# It never appears in a raw file; it is the name this converter masks under.
NO_FUNCTION = "no_function"
# Function words that name the absence of music rather than a section of it.
NON_MUSICAL = ("silence", "&pause")
END_MARKER = "end"

MIN_SPAN_SEC = 1e-6

_LARGE_SCALE = re.compile(r"^[A-Z]+'*$")
_SMALL_SCALE = re.compile(r"^[a-z]+'*$")


class ConvertError(Exception):
    pass


class Row(NamedTuple):
    time: float
    large: tuple
    small: tuple
    function: tuple


class Segment(NamedTuple):
    start: float
    end: float
    large: tuple
    small: tuple
    function: tuple


class Parse(NamedTuple):
    sections: list
    masked: list
    last_time: float


class Verdict(NamedTuple):
    track_id: str
    annotator: int
    names: tuple
    large_scale: tuple
    masked_count: int
    masked_sec: float
    labelled_sec: float
    duration_sec: float
    duration_source: str


def track_id(salami_id: str) -> str:
    return f"{ID_PREFIXES[SOURCE]}{salami_id}"


def annotation_dir(salami: Path, salami_id: str) -> Path:
    return salami / ANNOTATIONS_DIR / salami_id


def textfile_path(salami: Path, salami_id: str, annotator: int) -> Path:
    return annotation_dir(salami, salami_id) / TEXTFILES[annotator]


def source_audio_path(salami: Path, salami_id: str) -> Path:
    return salami / AUDIO_DIR / f"{salami_id}.mp3"


def metadata_path(salami: Path) -> Path:
    return salami / METADATA_FILE


def vocab_path(salami: Path) -> Path:
    return salami / VOCAB_FILE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# The function vocabulary
# --------------------------------------------------------------------------- #


def load_case_map(path: Path) -> dict:
    """casefolded token -> the dictionary's spelling, for case-only entries.

    The shipped dictionary is mostly semantic (``hook`` -> ``Chorus``,
    ``cadenza`` -> ``outro``), and applying that would map SALAMI's labels into
    a reading nobody chose.  Only the entries whose two sides are the same word
    are taken, so the transform can change nothing but capitalisation.
    """
    if not path.exists():
        raise SystemExit(f"no function vocabulary at {path}")
    case_map = {}
    known = set()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for line in handle:
            raw, _, canonical = line.rstrip("\n").rstrip("\r").partition("\t")
            raw, canonical = raw.strip(), canonical.strip()
            for token in (raw, canonical):
                if token and "," not in token and " " not in token:
                    known.add(token.casefold())
            if not raw or not canonical or raw == canonical:
                continue
            if raw.casefold() != canonical.casefold():
                continue
            if "," in canonical or " " in canonical:
                continue
            case_map[raw.casefold()] = canonical
    if not case_map:
        raise SystemExit(f"{path.name}: no case-only entries")
    return {"case": case_map, "known": known}


def normalise(token: str, vocab: dict) -> str:
    return vocab["case"].get(token.casefold(), token)


# --------------------------------------------------------------------------- #
# The raw annotator file
# --------------------------------------------------------------------------- #


def _classify(token: str, vocab: dict) -> str:
    if "(" in token or ")" in token:
        return "instrument"
    if token.casefold() in vocab["known"]:
        return "function"
    if _LARGE_SCALE.match(token):
        return "large"
    if _SMALL_SCALE.match(token):
        return "small"
    return "unknown"


def parse_rows(text: str, vocab: dict) -> list:
    rows = []
    for number, line in enumerate(
            text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ConvertError(
                f"line {number}: {len(fields)} tab-separated fields, want 2")
        try:
            time = float(fields[0])
        except ValueError:
            raise ConvertError(
                f"line {number}: time {fields[0]!r} is not a number") from None
        if time < 0.0:
            raise ConvertError(f"line {number}: time {time!r} is negative")
        large, small, function = [], [], []
        for token in (part.strip() for part in fields[1].split(",")):
            if not token:
                continue
            kind = _classify(token, vocab)
            if kind == "large":
                large.append(token)
            elif kind == "small":
                small.append(token)
            elif kind == "function":
                function.append(normalise(token, vocab))
        rows.append(Row(time, tuple(large), tuple(small), tuple(function)))
    if not rows:
        raise ConvertError("no annotation lines")
    for index, (before, after) in enumerate(zip(rows, rows[1:]), 2):
        if after.time < before.time:
            raise ConvertError(
                f"line {index}: time {after.time!r} is before line "
                f"{index - 1}'s {before.time!r} -- unsorted")
    return rows


def _is_end(row: Row) -> bool:
    return any(name.casefold() == END_MARKER for name in row.function)


def _musical(function: tuple) -> tuple:
    return tuple(name for name in function
                 if name.casefold() not in NON_MUSICAL
                 and name.casefold() != END_MARKER)


def segments(rows: list) -> list:
    """The function layer's own grid: every row stating a letter or a word."""
    boundaries = [row for row in rows if row.large or row.function]
    if not boundaries:
        raise ConvertError("no large-scale or function tokens anywhere")
    ends = [row.time for row in boundaries[1:]] + [rows[-1].time]
    return [Segment(row.time, end, row.large, row.small, row.function)
            for row, end in zip(boundaries, ends)
            if end - row.time > MIN_SPAN_SEC]


def _layers(segment: Segment) -> dict:
    entry = {}
    if segment.large:
        entry["large_scale"] = ", ".join(segment.large)
    if segment.small:
        entry["small_scale"] = ", ".join(segment.small)
    return entry


def parse_annotation(text: str, vocab: dict) -> Parse:
    rows = parse_rows(text, vocab)
    sections, masked = [], []
    for segment in segments(rows):
        span = {"start": segment.start, "end": segment.end}
        musical = _musical(segment.function)
        if musical:
            sections.append({"name": ", ".join(musical), **span,
                             **_layers(segment)})
            continue
        # The parsed layer would write no_function here and lose the letter.
        reason = NO_FUNCTION if not segment.function else "non_musical"
        entry = {**span, "reason": reason, **_layers(segment)}
        if segment.function:
            entry["function"] = ", ".join(segment.function)
        masked.append(entry)
    if not sections:
        raise ConvertError(
            "no section survives the mask -- the annotator named no musical "
            "function anywhere in the file")
    return Parse(sections, masked, rows[-1].time)


def read_annotation(path: Path, vocab: dict) -> Parse:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConvertError(f"{path.name}: {error}") from error
    return parse_annotation(text, vocab)


# --------------------------------------------------------------------------- #
# Metadata and audio
# --------------------------------------------------------------------------- #


def load_metadata(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"no metadata at {path}")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "SONG_ID" not in (rows[0] or {}):
        raise SystemExit(f"{path.name}: no SONG_ID column")
    return {row["SONG_ID"].strip(): row for row in rows}


def _sort_key(salami_id: str) -> tuple:
    return (0, int(salami_id), "") if salami_id.isdigit() else (1, 0, salami_id)


def track_duration(row: dict, parse: Parse) -> tuple:
    """SONG_DURATION when it covers the annotation, else the annotation's end.

    SONG_DURATION is whole seconds, so it is under the true length by up to a
    second and half the corpus annotates past it -- a record whose sections end
    after its own duration is one no validator can accept, and the annotator's
    final timestamp is the more precise of the two readings anyway.
    """
    stated = (row.get("SONG_DURATION") or "").strip()
    if not stated:
        if parse.last_time <= 0.0:
            raise ConvertError(
                "no SONG_DURATION and the annotation ends at zero")
        return parse.last_time, "annotation_last_time"
    try:
        duration = float(stated)
    except ValueError:
        raise ConvertError(
            f"SONG_DURATION {stated!r} is not a number") from None
    if duration <= 0.0 and parse.last_time <= 0.0:
        raise ConvertError(f"SONG_DURATION {stated!r} is not positive")
    if parse.last_time > duration:
        return parse.last_time, "annotation_last_time_over_metadata"
    return duration, "metadata"


def copy_audio(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return "kept"
    shutil.copy2(source, target)
    return "copied"


# --------------------------------------------------------------------------- #
# One track
# --------------------------------------------------------------------------- #


def choose_annotator(salami: Path, salami_id: str, vocab: dict) -> tuple:
    """textfile1 wins when both parse; textfile2 is the fallback, not a vote.

    Nothing in the data ranks one annotator above the other, so a stable rule
    beats a clever one: a second reading of the same track would otherwise
    depend on which file a later SALAMI release happened to touch.
    """
    reasons = []
    for annotator in sorted(TEXTFILES):
        path = textfile_path(salami, salami_id, annotator)
        if not path.exists():
            continue
        try:
            return annotator, path, read_annotation(path, vocab)
        except ConvertError as error:
            reasons.append(f"{path.name}: {error}")
    if not reasons:
        raise ConvertError("no textfile1.txt or textfile2.txt")
    raise ConvertError("; ".join(reasons))


def convert_track(salami_id: str, salami: Path, row: dict, vocab: dict,
                  corpus: Path, copy: bool) -> Verdict:
    identifier = track_id(salami_id)
    annotator, path, parse = choose_annotator(salami, salami_id, vocab)

    duration, duration_source = track_duration(row, parse)

    source_audio = source_audio_path(salami, salami_id)
    if not source_audio.exists():
        raise ConvertError(f"no audio at {source_audio}")

    record = {
        "schema": RECORD_SCHEMA,
        "source": SOURCE,
        "id": identifier,
        "native_id": salami_id,
        "title": (row.get("SONG_TITLE") or "").strip(),
        "artist": (row.get("ARTIST") or "").strip(),
        "genre": (row.get("GENRE") or "").strip(),
        "salami_class": (row.get("CLASS") or "").strip(),
        "audio": f"{identifier}{source_audio.suffix.lower()}",
        "duration": duration,
        "label_vocabulary": LABEL_VOCABULARY,
        "beats": BEATS_SOURCE,
        "sections": parse.sections,
        "masked": parse.masked,
        "provenance": {
            "converter": CONVERTER,
            "converted_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "annotator": annotator,
            "annotation_file": path.name,
            "annotation_sha256": sha256(path),
            "annotators_available": sorted(
                number for number in TEXTFILES
                if textfile_path(salami, salami_id, number).exists()),
            "duration_source": duration_source,
            "salami_source": (row.get("SOURCE") or "").strip(),
            "vocabulary_transform": "case_only",
            "license": LICENSE,
        },
    }
    try:
        validate_record(record)
    except ValueError as error:
        raise ConvertError(str(error)) from error

    if copy:
        copy_audio(source_audio, audio_path(corpus, record))

    # No beat CSV: SALAMI ships no grid, so admission generates one.
    write_record(annotation_path(corpus, identifier, SOURCE), record)

    return Verdict(
        identifier, annotator,
        tuple(section["name"] for section in parse.sections),
        tuple(name for section in parse.sections
              for name in section.get("large_scale", "").split(", ") if name),
        len(parse.masked),
        sum(span["end"] - span["start"] for span in parse.masked),
        sum(section["end"] - section["start"] for section in parse.sections),
        duration, duration_source,
    )


# --------------------------------------------------------------------------- #
# The batch
# --------------------------------------------------------------------------- #


def eligible(salami: Path, metadata: dict, ids: list | None,
             include_live: bool) -> tuple:
    wanted = ids if ids is not None else sorted(metadata, key=_sort_key)
    convertible, excluded, no_annotation, no_audio, unknown = [], [], [], [], []
    for salami_id in wanted:
        row = metadata.get(salami_id)
        if row is None:
            unknown.append(salami_id)
            continue
        if not include_live and (row.get("CLASS") or "").strip() == \
                LIVE_ARCHIVE_CLASS:
            excluded.append(salami_id)
            continue
        if not annotation_dir(salami, salami_id).is_dir():
            no_annotation.append(salami_id)
            continue
        if not source_audio_path(salami, salami_id).exists():
            no_audio.append(salami_id)
            continue
        convertible.append(salami_id)
    return convertible, excluded, no_annotation, no_audio, unknown


def _report(verdicts: list, failures: list, buckets: tuple,
            requested: int, partial: bool) -> dict:
    labels: dict = {}
    letters: dict = {}
    two_annotators = []
    fallbacks = []
    for verdict in verdicts:
        for name in verdict.names:
            labels[name] = labels.get(name, 0) + 1
        for name in verdict.large_scale:
            letters[name] = letters.get(name, 0) + 1
        if verdict.annotator != 1:
            fallbacks.append(verdict.track_id)
    excluded, no_annotation, no_audio, unknown, both = buckets
    return {
        "converter": CONVERTER,
        "source": SOURCE,
        "partial": partial,
        "tracks_requested": requested,
        "converted": len(verdicts),
        "failed": len(failures),
        "failures": [{"id": identifier, "reason": reason}
                     for identifier, reason in failures],
        "excluded_live_music_archive": len(excluded),
        "excluded_live_music_archive_ids": excluded,
        "no_annotation": len(no_annotation),
        "no_annotation_ids": no_annotation,
        "no_audio": len(no_audio),
        "no_audio_ids": no_audio,
        "not_in_metadata": unknown,
        "labels": dict(sorted(labels.items())),
        "large_scale_labels": dict(sorted(letters.items())),
        "masked_spans": sum(verdict.masked_count for verdict in verdicts),
        "masked_sec": round(sum(v.masked_sec for v in verdicts), 3),
        "tracks_with_two_annotators": len(both),
        "tracks_with_two_annotators_ids": both,
        "annotator2_used": fallbacks,
        "duration_sources": dict(sorted(collections.Counter(
            verdict.duration_source for verdict in verdicts).items())),
        "duration_from_annotation": [
            verdict.track_id for verdict in verdicts
            if verdict.duration_source == "annotation_last_time"],
        "labelled_sec": round(sum(v.labelled_sec for v in verdicts), 3),
        "audio_sec": round(sum(v.duration_sec for v in verdicts), 3),
    }


def write_report(corpus: Path, report: dict) -> Path:
    name = PARTIAL_REPORT_FILE if report["partial"] else REPORT_FILE
    path = third_party_dir(corpus) / name
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
    print(f"converted {report['converted']}/{report['tracks_requested']}, "
          f"failed {report['failed']}"
          + (" (partial run)" if report["partial"] else ""))
    for failure in report["failures"]:
        print(f"  FAIL {failure['id']}: {failure['reason']}")
    print(f"excluded {LIVE_ARCHIVE_CLASS}: "
          f"{report['excluded_live_music_archive']}; no annotation: "
          f"{report['no_annotation']}; no audio: {report['no_audio']}")
    print(f"labels ({len(report['labels'])} distinct): " + ", ".join(
        f"{name}={count}" for name, count in report["labels"].items()))
    print(f"large-scale ({len(report['large_scale_labels'])} distinct): "
          + ", ".join(f"{name}={count}"
                      for name, count in report["large_scale_labels"].items()))
    print(f"masked {report['masked_spans']} span(s), "
          f"{report['masked_sec']:.3f} s")
    print(f"two annotators available: {report['tracks_with_two_annotators']}; "
          f"annotator 2 used: {len(report['annotator2_used'])}")
    print("duration sources: " + ", ".join(
        f"{name}={count}"
        for name, count in report["duration_sources"].items()))
    print(f"labelled {report['labelled_sec']:.3f} s of "
          f"{report['audio_sec']:.3f} s of audio")


def convert(salami: Path, corpus: Path, ids: list | None = None,
            limit: int | None = None, copy: bool = True,
            include_live: bool = False) -> dict:
    metadata = load_metadata(metadata_path(salami))
    vocab = load_case_map(vocab_path(salami))
    convertible, excluded, no_annotation, no_audio, unknown = eligible(
        salami, metadata, ids, include_live)
    partial = ids is not None or limit is not None
    if limit is not None:
        convertible = convertible[:limit]
    both = [salami_id for salami_id in convertible
            if all(textfile_path(salami, salami_id, number).exists()
                   for number in TEXTFILES)]

    verdicts, failures = [], []
    for salami_id in convertible:
        try:
            verdicts.append(convert_track(salami_id, salami,
                                          metadata[salami_id], vocab, corpus,
                                          copy))
        except ConvertError as error:
            failures.append((track_id(salami_id), str(error)))
    report = _report(verdicts, failures,
                     (excluded, no_annotation, no_audio, unknown, both),
                     len(convertible), partial)
    write_report(corpus, report)
    return report


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salami-dir", type=Path, required=True,
                        help="the salami-data-public root")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root (default: the resolved corpus)")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="explicit SALAMI song ids")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert only the first N eligible tracks")
    parser.add_argument("--no-audio", action="store_true",
                        help="convert annotations only, skip the audio copy")
    parser.add_argument("--include-live-archive", action="store_true",
                        help=f"convert {LIVE_ARCHIVE_CLASS} tracks, which are "
                             f"excluded from admission by default")
    args = parser.parse_args(argv)

    corpus = args.data_dir.resolve() if args.data_dir else corpus_dir()
    report = convert(args.salami_dir.resolve(), corpus, ids=args.ids,
                     limit=args.limit, copy=not args.no_audio,
                     include_live=args.include_live_archive)
    print_report(report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
