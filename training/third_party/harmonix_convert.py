#!/usr/bin/env python
"""Convert the Harmonix Set into third-party corpus records.

The segments files are SINGLE-SPACE separated -- the repository's own README
says tab, and a tab split yields one field per line rather than an error, so a
reader that believes the documentation gets an empty timeline and no complaint.
They are start-only: a row states the time a segment begins and the label it
begins with, and its end is the next row's time.  ``end`` is a tail sentinel in
exactly the sense Raveform's is -- it marks where the annotation stops, not
where the audio does (it sits a median 4.4 s inside the stated duration) -- so
it terminates the segment before it and is never itself a section.  Three
shapes occur and all three are handled here: the normal trailing sentinel, a
duplicated one (the first terminates, the second is redundant), and two tracks
that carry none at all, whose final segment is closed on the metadata duration
and says so in ``final_segment_end_source``.

Labels are stored VERBATIM in Harmonix's own 126-name function vocabulary.
Adjacent identical labels are NOT merged: merging answers "how long is a
musical section", which is an evaluation-time view, and storage must keep what
the annotator wrote.  ``silence`` is a real label in this vocabulary and is
stored as a section -- unlike SALAMI's ``no_function``, which genuinely means
the annotator named nothing and is masked there.

The beat grid is expert and is written straight through into the corpus's
published three-column format so admission finds it and keeps it: madmom never
runs over downbeats a human placed.  Column 2 is the 1-based position in the
bar and ``1`` is the downbeat marker; positions above four are carried as the
file states them rather than folded, because folding would fabricate downbeats
nobody marked.  The tracks that exceed four, and the ones that open mid-bar,
are counted and named for the owner instead.

Audio is not copied: the fetcher writes ``hx-<id>.mp3`` into the corpus
directly, and annotations convert whether or not it has landed yet.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
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
    beat_csv_path,
    third_party_dir,
    validate_record,
    write_record,
)

SOURCE = "harmonix"
CONVERTER = "harmonix_convert"

DATASET_DIR = "dataset"
SEGMENTS_DIR = "segments"
BEATS_DIR = "beats_and_downbeats"
METADATA_FILE = "metadata.csv"
URLS_FILE = "youtube_urls.csv"
SCORES_FILE = "youtube_alignment_scores.csv"

REPORT_FILE = "harmonix_convert_report.json"
PARTIAL_REPORT_FILE = "harmonix_convert_report.partial.json"

LABEL_VOCABULARY = "harmonix_function"
BEATS_SOURCE = "harmonix_expert"
LICENSE = "annotations MIT (Nieto et al. 2019); audio not distributed"

AUDIO_SUFFIX = ".mp3"
AUDIO_DIR = "audio"

END_LABEL = "end"
END_FROM_SENTINEL = "end_sentinel"
END_FROM_DURATION = "metadata_duration"
MIN_SPAN_SEC = 1e-3
NOMINAL_BAR = 4
DOWNBEAT_POSITION = 1

_YOUTUBE_ID = re.compile(r"[?&]v=([A-Za-z0-9_-]+)")


class ConvertError(Exception):
    pass


class Section(NamedTuple):
    name: str
    start: float
    end: float


class Parse(NamedTuple):
    sections: list
    end_source: str
    duplicate_end: bool


class Sources(NamedTuple):
    dataset: Path
    metadata: dict
    urls: dict
    scores: dict


class Verdict(NamedTuple):
    track_id: str
    names: tuple
    adjacent_identical: int
    first_position: int
    max_position: int
    end_source: str
    duplicate_end: bool
    dtw_score: float | None
    labelled_sec: float
    duration_sec: float


def track_id(native: str) -> str:
    return f"{ID_PREFIXES[SOURCE]}{native}"


def dataset_dir(harmonixset: Path) -> Path:
    return harmonixset / DATASET_DIR


def segments_path(dataset: Path, native: str) -> Path:
    return dataset / SEGMENTS_DIR / f"{native}.txt"


def beats_path(dataset: Path, native: str) -> Path:
    return dataset / BEATS_DIR / f"{native}.txt"


def corpus_audio_path(corpus: Path, native: str) -> Path:
    return (third_party_dir(corpus) / AUDIO_DIR / SOURCE
            / f"{track_id(native)}{AUDIO_SUFFIX}")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lines(text: str) -> list:
    return [line for line in text.replace("\r\n", "\n").replace("\r", "\n")
            .split("\n") if line.strip()]


# --------------------------------------------------------------------------- #
# The segments file
# --------------------------------------------------------------------------- #


def segment_rows(text: str) -> list:
    rows = []
    for number, line in enumerate(_lines(text), 1):
        fields = line.strip().split(" ")
        if len(fields) != 2:
            raise ConvertError(
                f"segments line {number}: {len(fields)} space-separated "
                f"fields, want 2")
        try:
            time = float(fields[0])
        except ValueError:
            raise ConvertError(
                f"segments line {number}: time {fields[0]!r} is not a "
                f"number") from None
        if time < 0.0:
            raise ConvertError(f"segments line {number}: time {time!r} is negative")
        if not fields[1]:
            raise ConvertError(f"segments line {number}: empty label")
        rows.append((time, fields[1]))
    if not rows:
        raise ConvertError("no segment lines")
    for index, (before, after) in enumerate(zip(rows, rows[1:]), 2):
        if after[0] <= before[0]:
            raise ConvertError(
                f"segments line {index}: time {after[0]!r} does not advance on "
                f"line {index - 1}'s {before[0]!r}")
    return rows


def parse_segments(text: str, duration: float) -> Parse:
    rows = segment_rows(text)
    sections = []
    pending = None
    duplicate_end = False
    for time, label in rows:
        if pending is not None:
            sections.append(Section(pending[1], pending[0], time))
        elif label == END_LABEL and sections:
            duplicate_end = True
        pending = None if label == END_LABEL else (time, label)
    end_source = END_FROM_SENTINEL
    if pending is not None:
        end_source = END_FROM_DURATION
        sections.append(Section(pending[1], pending[0],
                                max(duration, pending[0] + MIN_SPAN_SEC)))
    if not sections:
        raise ConvertError("no section survives the end sentinel")
    return Parse(sections, end_source, duplicate_end)


def read_segments(path: Path, duration: float) -> Parse:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConvertError(f"{path.name}: {error}") from error
    return parse_segments(text, duration)


def section_dicts(sections: list) -> list:
    return [{"name": section.name, "start": section.start, "end": section.end}
            for section in sections]


def adjacent_identical(sections: list) -> int:
    return sum(1 for before, after in zip(sections, sections[1:])
               if before.name == after.name)


# --------------------------------------------------------------------------- #
# The beats file
# --------------------------------------------------------------------------- #


def parse_beats(text: str) -> list:
    beats = []
    for number, line in enumerate(_lines(text), 1):
        fields = line.rstrip().split("\t")
        if len(fields) != 3:
            raise ConvertError(
                f"beats line {number}: {len(fields)} tab-separated fields, "
                f"want 3")
        try:
            time, position = float(fields[0]), int(fields[1])
        except ValueError:
            raise ConvertError(
                f"beats line {number}: {fields[0]!r} {fields[1]!r} are not a "
                f"time and a bar position") from None
        if position < 1:
            raise ConvertError(
                f"beats line {number}: bar position {position} is not 1-based")
        beats.append((time, position))
    if not beats:
        raise ConvertError("no beat lines")
    for index, (before, after) in enumerate(zip(beats, beats[1:]), 2):
        if after[0] <= before[0]:
            raise ConvertError(
                f"beats line {index}: time {after[0]!r} does not advance on "
                f"line {index - 1}'s {before[0]!r}")
    return beats


def read_beats(path: Path) -> list:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConvertError(f"{path.name}: {error}") from error
    return parse_beats(text)


# --------------------------------------------------------------------------- #
# Metadata, urls and alignment scores
# --------------------------------------------------------------------------- #


def _keyed(path: Path, column: str) -> dict:
    if not path.exists():
        raise SystemExit(f"no {path.name} at {path}")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in (rows[0] or {}):
        raise SystemExit(f"{path.name}: no {column} column")
    return {(row[column] or "").strip(): row for row in rows}


def load_metadata(dataset: Path) -> dict:
    return _keyed(dataset / METADATA_FILE, "File")


def load_urls(dataset: Path) -> dict:
    return _keyed(dataset / URLS_FILE, "File")


def load_scores(dataset: Path) -> dict:
    return _keyed(dataset / SCORES_FILE, "File")


def youtube_id(url: str) -> str:
    found = _YOUTUBE_ID.search(url or "")
    return found.group(1) if found else ""


def _number(text: str, field: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError) as error:
        raise ConvertError(
            f"metadata {field} is not a number: {text!r}") from error


def _optional(text: str) -> float | None:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# One track
# --------------------------------------------------------------------------- #


def carry_timestamp(path: Path, record: dict) -> dict:
    """Keep an unchanged record's original stamp, so a re-run rewrites nothing.

    A per-run timestamp is the one field that differs when nothing has: without
    this, a second conversion rewrites all 912 files and no diff can say whether
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


def convert_track(native: str, sources: Sources, corpus: Path) -> Verdict:
    identifier = track_id(native)
    row = sources.metadata.get(native)
    if row is None:
        raise ConvertError(f"no {native} row in {METADATA_FILE}")
    duration = _number(row.get("Duration", ""), "Duration")

    segments = segments_path(sources.dataset, native)
    if not segments.exists():
        raise ConvertError(f"no segments file at {segments}")
    parse = read_segments(segments, duration)

    beats_file = beats_path(sources.dataset, native)
    if not beats_file.exists():
        raise ConvertError(f"no beats file at {beats_file}")
    beats = read_beats(beats_file)

    score = _optional((sources.scores.get(native) or {}).get("score", ""))
    record = {
        "schema": RECORD_SCHEMA,
        "source": SOURCE,
        "id": identifier,
        "native_id": native,
        "title": (row.get("Title") or "").strip(),
        "artist": (row.get("Artist") or "").strip(),
        "release": (row.get("Release") or "").strip(),
        "genre": (row.get("Genre") or "").strip(),
        "audio": f"{identifier}{AUDIO_SUFFIX}",
        "duration": duration,
        "label_vocabulary": LABEL_VOCABULARY,
        "beats": BEATS_SOURCE,
        "sections": section_dicts(parse.sections),
        "masked": [],
        "provenance": {
            "converter": CONVERTER,
            "converted_utc": _now_utc(),
            "segments_file": segments.name,
            "segments_sha256": sha256(segments),
            "beats_file": beats_file.name,
            "beats_sha256": sha256(beats_file),
            "metadata_file": METADATA_FILE,
            "final_segment_end_source": parse.end_source,
            "bpm": _optional(row.get("BPM", "")),
            "time_signature": (row.get("Time Signature") or "").strip(),
            "ratio_bars_in_4": _optional(row.get("Ratio Bars in 4", "")),
            "dtw_score": score,
            "youtube_id": youtube_id(
                (sources.urls.get(native) or {}).get("URL", "")),
            "license": LICENSE,
        },
    }
    try:
        validate_record(record)
    except ValueError as error:
        raise ConvertError(str(error)) from error

    grid = beat_csv_path(corpus, record)
    if not grid.exists():
        grid.parent.mkdir(parents=True, exist_ok=True)
        write_beat_csv(grid, beat_rows(beats, record["sections"]))

    target = annotation_path(corpus, identifier, SOURCE)
    write_record(target, carry_timestamp(target, record))

    return Verdict(
        identifier,
        tuple(section.name for section in parse.sections),
        adjacent_identical(parse.sections),
        beats[0][1],
        max(position for _time, position in beats),
        parse.end_source, parse.duplicate_end, score,
        sum(section.end - section.start for section in parse.sections),
        duration,
    )


# --------------------------------------------------------------------------- #
# The batch
# --------------------------------------------------------------------------- #


def eligible(sources: Sources, corpus: Path, ids: list | None,
             require_audio: bool) -> tuple:
    wanted = ids if ids is not None else list(sources.metadata)
    convertible, no_annotation, no_audio, unknown = [], [], [], []
    for native in wanted:
        if native not in sources.metadata:
            unknown.append(native)
        elif not (segments_path(sources.dataset, native).exists()
                  and beats_path(sources.dataset, native).exists()):
            no_annotation.append(native)
        elif require_audio and not corpus_audio_path(corpus, native).exists():
            no_audio.append(native)
        else:
            convertible.append(native)
    return convertible, no_annotation, no_audio, unknown


def report_file(ids: list | None, limit: int | None,
                require_audio: bool) -> str:
    """A restricted run must not land where a whole-set report belongs.

    A twenty-five track report sitting at the full name silently stops covering
    the other 887, and nothing downstream can tell the two apart.  A track the
    dataset is simply missing does not restrict the run: it is inside the set
    and named in a bucket, so the report still accounts for it.
    """
    return (PARTIAL_REPORT_FILE
            if ids is not None or limit is not None or require_audio
            else REPORT_FILE)


def _at(ordered: list, fraction: float):
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _distribution(values: list) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "p25": round(_at(ordered, 0.25), 6),
        "median": round(_at(ordered, 0.5), 6),
        "p75": round(_at(ordered, 0.75), 6),
        "max": round(ordered[-1], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def _report(verdicts: list, failures: list, buckets: tuple, requested: int,
            report_name: str) -> dict:
    labels: dict = {}
    odd_meter = []
    mid_bar = []
    for verdict in verdicts:
        for name in verdict.names:
            labels[name] = labels.get(name, 0) + 1
        if verdict.max_position > NOMINAL_BAR:
            odd_meter.append({"id": verdict.track_id,
                              "max_position": verdict.max_position})
        if verdict.first_position != DOWNBEAT_POSITION:
            mid_bar.append({"id": verdict.track_id,
                            "first_position": verdict.first_position})
    no_annotation, no_audio, unknown = buckets
    scores = [verdict.dtw_score for verdict in verdicts
              if verdict.dtw_score is not None]
    return {
        "converter": CONVERTER,
        "source": SOURCE,
        "report_file": report_name,
        "partial": report_name != REPORT_FILE,
        "tracks_requested": requested,
        "converted": len(verdicts),
        "failed": len(failures),
        "failures": [{"id": identifier, "reason": reason}
                     for identifier, reason in failures],
        "no_annotation": len(no_annotation),
        "no_annotation_ids": no_annotation,
        "no_audio": len(no_audio),
        "no_audio_ids": no_audio,
        "not_in_metadata": unknown,
        "labels": dict(sorted(labels.items())),
        "label_count": len(labels),
        "adjacent_identical_pairs": sum(verdict.adjacent_identical
                                        for verdict in verdicts),
        "tracks_with_adjacent_identical": sum(
            1 for verdict in verdicts if verdict.adjacent_identical),
        "odd_meter_count": len(odd_meter),
        "odd_meter_tracks": odd_meter,
        "mid_bar_start_count": len(mid_bar),
        "mid_bar_start_tracks": mid_bar,
        "duplicate_end_ids": [verdict.track_id for verdict in verdicts
                              if verdict.duplicate_end],
        "no_end_sentinel_ids": [verdict.track_id for verdict in verdicts
                                if verdict.end_source == END_FROM_DURATION],
        "dtw_score": _distribution(scores),
        "dtw_score_missing": len(verdicts) - len(scores),
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
    print(f"converted {report['converted']}/{report['tracks_requested']}, "
          f"failed {report['failed']}")
    for failure in report["failures"]:
        print(f"  FAIL {failure['id']}: {failure['reason']}")
    print(f"no annotation: {report['no_annotation']}; no audio yet: "
          f"{report['no_audio']}; not in metadata: "
          f"{len(report['not_in_metadata'])}")
    print(f"labels ({report['label_count']} distinct): " + ", ".join(
        f"{name}={count}" for name, count in report["labels"].items()))
    print(f"adjacent identical pairs: {report['adjacent_identical_pairs']} "
          f"over {report['tracks_with_adjacent_identical']} track(s) -- kept, "
          f"not merged")
    print(f"beat positions above {NOMINAL_BAR}: {report['odd_meter_count']} "
          f"track(s)")
    for track in report["odd_meter_tracks"]:
        print(f"  {track['id']} max position {track['max_position']}")
    print(f"starting mid-bar: {report['mid_bar_start_count']} track(s)")
    for track in report["mid_bar_start_tracks"]:
        print(f"  {track['id']} first position {track['first_position']}")
    print(f"duplicated end sentinel: {len(report['duplicate_end_ids'])} "
          f"track(s) {report['duplicate_end_ids']}")
    print(f"no end sentinel (final end from metadata duration): "
          f"{len(report['no_end_sentinel_ids'])} track(s) "
          f"{report['no_end_sentinel_ids']}")
    print(f"dtw score: {report['dtw_score']} (missing "
          f"{report['dtw_score_missing']})")
    print(f"labelled {report['labelled_sec']:.3f} s of "
          f"{report['audio_sec']:.3f} s of audio")
    print(f"report written to {report['report_file']}"
          + (" (partial run)" if report["partial"] else ""))


def convert(harmonixset: Path, corpus: Path, ids: list | None = None,
            limit: int | None = None, require_audio: bool = False) -> dict:
    dataset = dataset_dir(harmonixset)
    sources = Sources(dataset, load_metadata(dataset), load_urls(dataset),
                      load_scores(dataset))
    convertible, no_annotation, no_audio, unknown = eligible(
        sources, corpus, ids, require_audio)
    if limit is not None:
        convertible = convertible[:limit]

    verdicts, failures = [], []
    for native in convertible:
        try:
            verdicts.append(convert_track(native, sources, corpus))
        except (ConvertError, OSError) as error:
            reason = (str(error) if isinstance(error, ConvertError)
                      else f"{type(error).__name__}: {error}")
            failures.append((track_id(native), reason))
    report = _report(verdicts, failures,
                     (no_annotation, no_audio, unknown), len(convertible),
                     report_file(ids, limit, require_audio))
    write_report(corpus, report)
    return report


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harmonixset", type=Path, required=True,
                        help="the harmonixset repository root (holds dataset/)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root (default: the resolved corpus)")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="explicit Harmonix track ids (e.g. 0001_12step)")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert only the first N eligible tracks")
    parser.add_argument("--require-audio", action="store_true",
                        help="convert only tracks whose fetched mp3 has landed")
    args = parser.parse_args(argv)

    corpus = args.data_dir.resolve() if args.data_dir else corpus_dir()
    report = convert(args.harmonixset.resolve(), corpus, ids=args.ids,
                     limit=args.limit, require_audio=args.require_audio)
    print_report(report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
