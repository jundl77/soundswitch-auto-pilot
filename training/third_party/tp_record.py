#!/usr/bin/env python
"""The quarantined third-party annotation record: its shape, its IO, its rules.

RWC, SALAMI and Harmonix sections land in ``annotations/<id>.<source>.json``,
which nothing in the corpus flow globs -- the loaders take ``*.hand.json`` and
``segments.json`` and nothing else -- so a converted track is invisible to the
manifest, the clean manifest, the splits, the training table and the priors
until an integration ruling says otherwise.  The quarantine is that filename;
this module is what makes the file trustworthy once it is opened.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_TRAINING_DIR.parent), str(_TRAINING_DIR),
              str(_TRAINING_DIR / "raveform"), str(_TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from raveform_fetch_annotations import BEATS_DIR, annotations_dir  # noqa: E402

RECORD_SCHEMA = 1
THIRD_PARTY_SOURCES = ("rwc", "salami", "harmonix")
ID_PREFIXES = {"rwc": "rwc-", "salami": "salami-", "harmonix": "hx-"}
REQUIRED_FIELDS = ("schema", "source", "id", "native_id", "title", "audio",
                   "duration", "label_vocabulary", "sections", "provenance")

RECORD_EXTENSION = ".json"
AUDIO_DIR = "audio"
THIRD_PARTY_DIR = "third_party"

# Annotators round the last section end; the slack is not a licence to overrun.
END_SLACK_SEC = 0.5
OVERLAP_TOLERANCE_SEC = 1e-6

PROVENANCE_FIELDS = ("converter", "converted_utc")


def record_suffix(source: str) -> str:
    return f".{source}{RECORD_EXTENSION}"


def annotation_path(corpus: Path, track_id: str, source: str) -> Path:
    return annotations_dir(Path(corpus)) / f"{track_id}{record_suffix(source)}"


def third_party_dir(corpus: Path) -> Path:
    return Path(corpus) / THIRD_PARTY_DIR


def audio_path(corpus: Path, record: dict) -> Path:
    return (third_party_dir(corpus) / AUDIO_DIR / str(record["source"])
            / str(record["audio"]))


def beat_csv_path(corpus: Path, record: dict) -> Path:
    return annotations_dir(Path(corpus)) / BEATS_DIR / f"{record['id']}.beat.csv"


def labelled_span(record: dict) -> tuple:
    sections = record.get("sections") or []
    if not sections:
        return 0.0, 0.0
    return float(sections[0]["start"]), float(sections[-1]["end"])


def _number(value, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: not a number ({value!r})") from None
    if not math.isfinite(number):
        raise ValueError(f"{what}: not finite ({value!r})")
    return number


def _span(entry, what: str, duration: float) -> tuple:
    if not isinstance(entry, dict):
        raise ValueError(f"{what}: expected an object, got {type(entry).__name__}")
    start = _number(entry.get("start"), f"{what} start")
    end = _number(entry.get("end"), f"{what} end")
    if start < 0.0:
        raise ValueError(f"{what}: starts at {start!r}, before zero")
    if end <= start:
        raise ValueError(f"{what}: start {start!r} is not before end {end!r}")
    if end > duration + END_SLACK_SEC:
        raise ValueError(
            f"{what}: ends at {end!r}, past the record duration {duration!r}")
    return start, end


def _check_order(spans: list, what: str) -> None:
    for index, (before, after) in enumerate(zip(spans, spans[1:])):
        if after[0] < before[0]:
            raise ValueError(
                f"{what} {index + 1} starts at {after[0]!r}, before {what} "
                f"{index} at {before[0]!r} -- unsorted")
        if before[1] > after[0] + OVERLAP_TOLERANCE_SEC:
            raise ValueError(
                f"{what} {index} ends at {before[1]!r}, past the start of "
                f"{what} {index + 1} at {after[0]!r} -- overlapping")


def _check_id(track_id, source: str) -> None:
    if not isinstance(track_id, str):
        raise ValueError(f"id: expected a str, got {type(track_id).__name__}")
    prefix = ID_PREFIXES[source]
    if not track_id.startswith(prefix):
        raise ValueError(f"id {track_id!r}: a {source} id must start with {prefix!r}")
    if "/" in track_id or "\\" in track_id:
        raise ValueError(f"id {track_id!r}: contains a path separator")
    if any(character.isspace() for character in track_id):
        raise ValueError(f"id {track_id!r}: contains whitespace")
    if track_id != track_id.lower():
        raise ValueError(f"id {track_id!r}: must be lowercase")


def validate_record(record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"expected a JSON object, got {type(record).__name__}")
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"record lacks {', '.join(missing)}")
    if record["schema"] != RECORD_SCHEMA:
        raise ValueError(
            f"schema: expected {RECORD_SCHEMA}, got {record['schema']!r}")
    source = record["source"]
    if source not in THIRD_PARTY_SOURCES:
        raise ValueError(f"source: {source!r} is not a third-party source")
    _check_id(record["id"], source)

    duration = _number(record["duration"], "duration")
    if duration <= 0.0:
        raise ValueError(f"duration: {duration!r} is not positive")

    sections = record["sections"]
    if not isinstance(sections, list):
        raise ValueError(f"sections: expected a list, got {type(sections).__name__}")
    if not sections:
        raise ValueError("sections: empty -- the record labels nothing")
    spans = []
    for index, section in enumerate(sections):
        what = f"section {index}"
        span = _span(section, what, duration)
        name = section.get("name")
        # No vocabulary check anywhere: a source's labels are its own ground
        # truth, and mapping them into ours is a later training decision.
        if not isinstance(name, str):
            raise ValueError(f"{what}: name is not a str ({name!r})")
        if not name:
            raise ValueError(f"{what}: name is empty")
        if name != name.strip():
            raise ValueError(f"{what}: name {name!r} carries stray whitespace")
        spans.append(span)
    # Gaps between sections are legal: unlabelled audio is dropped downstream,
    # never absorbed into a neighbour, so closing one here would invent labels.
    _check_order(spans, "section")

    masked = record.get("masked")
    if masked is not None:
        if not isinstance(masked, list):
            raise ValueError(f"masked: expected a list, got {type(masked).__name__}")
        _check_order([_span(entry, f"masked {index}", duration)
                      for index, entry in enumerate(masked)], "masked")

    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError(
            f"provenance: expected an object, got {type(provenance).__name__}")
    absent = [field for field in PROVENANCE_FIELDS if not provenance.get(field)]
    if absent:
        raise ValueError(f"provenance lacks {', '.join(absent)}")


def write_record(path: Path, record: dict) -> Path:
    validate_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_record(path: Path) -> dict:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except ValueError as exc:
        raise RuntimeError(f"{path.name}: does not parse ({exc})") from None
    try:
        validate_record(record)
    except ValueError as exc:
        raise RuntimeError(f"{path.name}: {exc}") from None
    stem = f"{record['id']}.{record['source']}"
    if path.name[: -len(RECORD_EXTENSION)] != stem:
        raise RuntimeError(
            f"{path.name}: the record says {stem}{RECORD_EXTENSION} -- a renamed "
            f"record would label the wrong track")
    return record


def load_records(corpus: Path, source: str) -> list:
    directory = annotations_dir(Path(corpus))
    records = [read_record(path)
               for path in sorted(directory.glob(f"*{record_suffix(source)}"))]
    records.sort(key=lambda record: str(record["id"]))
    return records


def load_all_third_party(corpus: Path) -> list:
    records = [record for source in THIRD_PARTY_SOURCES
               for record in load_records(corpus, source)]
    records.sort(key=lambda record: str(record["id"]))
    return records
