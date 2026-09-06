#!/usr/bin/env python
"""Admit a converted third-party track into the quarantine, and no further.

This is ``hand_label_admission`` for annotations we did not author: the same
beat grid in the published format, the same cleanliness gate measuring the same
durations, the same idempotence -- but every output lands under
``<corpus>/third_party/``.  The corpus's own ``manifest.csv``,
``clean_manifest.csv``, ``segments.json``, ``splits.json`` and
``checksums.sha256`` are never read for writing and never written, because an
RWC track is not a dataset member until the integration ruling makes it one.
No split is assigned for the same reason: a quarantined track is not a
candidate, and reporting a split would say it was.

One thing this path does that the hand one does not is name a masked span in the
grid: hand masks are edge-only, so ``section_of``'s ``end`` is honest there,
while a SALAMI record's masks are routinely interior and an ``end`` sitting mid
track reads to any tail-truncating consumer as the last beat of the song.

The one shared write is the beat grid at ``annotations/beats/<id>.beat.csv``,
which is inert -- that directory is only ever read by exact path from a tracks
list the quarantined record is absent from.  A source that ships an expert grid
(Harmonix, RWC) keeps it: madmom never runs over a grid better than madmom's.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_TRAINING_DIR.parent), str(_TRAINING_DIR),
              str(_TRAINING_DIR / "raveform"), str(_TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_clean_manifest as gate  # noqa: E402
import tp_record  # noqa: E402
from corpus_root import corpus_dir  # noqa: E402
from hand_label_admission import (  # noqa: E402
    BEAT_CSV_HEADER,
    END_SENTINEL,
    beat_rows,
    detect_beats,
    write_beat_csv,
)
from raveform_fetch_annotations import parse_beat_csv  # noqa: E402

TP_DIR = "third_party"
TP_MANIFEST_FILE = "manifest.csv"
TP_CLEAN_MANIFEST_FILE = "clean_manifest.csv"
TP_MANIFEST_HEADER = ("track_id", "source", "native_id", "title", "artist",
                      "genre", "n_sections", "n_masked", "labelled_sec",
                      "total_sec")
TP_CLEAN_HEADER = ("track_id", "source", "audio_path", "probe_duration_sec",
                   "decoded_duration_sec", "annotation_duration_sec",
                   "status", "detail")

MASKED_SENTINEL = "masked"


def tp_dir(corpus: Path) -> Path:
    return tp_record.third_party_dir(corpus)


def _masked_spans(record: dict) -> list:
    return sorted(((float(span["start"]), float(span["end"]))
                   for span in record.get("masked", [])),
                  key=lambda span: span[0])


def _is_masked(spans: list, t: float) -> bool:
    return any(start <= t < end for start, end in spans)


def third_party_beat_rows(beats, record: dict) -> list:
    # ``section_of`` calls everything outside a section span ``end``, which is
    # edge-only for a hand label and wrong here: SALAMI masks are frequently
    # interior, so a consumer truncating at the first ``end`` would drop the
    # back half of the track.  A mask is a hole, not a tail.
    spans = _masked_spans(record)
    return [(time, position,
             MASKED_SENTINEL if _is_masked(spans, time) else section)
            for time, position, section in beat_rows(beats, record["sections"])]


def labelled_seconds(record: dict) -> float:
    # The sum of the sections, not the span they cover: a gap is unlabelled
    # audio and must not be counted as labelled.
    return sum(float(section["end"]) - float(section["start"])
               for section in record["sections"])


def manifest_rows(records: list) -> list:
    rows = [
        (
            str(record["id"]),
            str(record["source"]),
            str(record["native_id"]),
            str(record["title"]),
            str(record.get("artist", "")),
            str(record.get("genre", "")),
            len(record["sections"]),
            len(record.get("masked", [])),
            f"{labelled_seconds(record):.3f}",
            f"{float(record['duration']):.3f}",
        )
        for record in records
    ]
    rows.sort(key=lambda row: row[0])
    return rows


def write_manifest(corpus: Path, rows: list) -> Path:
    return _write_csv(tp_dir(corpus) / TP_MANIFEST_FILE, TP_MANIFEST_HEADER, rows)


def _write_csv(path: Path, header: tuple, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_beat_csv(path: Path) -> list:
    # A hand-placed expert grid with the wrong columns must fail by name here,
    # not as a KeyError from inside the published parser.
    with open(path, "r", encoding="utf-8", newline="") as handle:
        header = tuple(csv.DictReader(handle).fieldnames or ())
    if header != BEAT_CSV_HEADER:
        raise RuntimeError(
            f"{path.name}: beat grid columns are {header}, expected "
            f"{BEAT_CSV_HEADER}")
    return parse_beat_csv(path)


def read_raw_beat_rows(path: Path) -> list:
    # Repair rewrites one column of a grid madmom took hours to place, so the
    # other two travel as the text they were written as rather than through a
    # parse and a re-format that could round them.
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != BEAT_CSV_HEADER:
            raise RuntimeError(
                f"{path.name}: beat grid columns are {header}, expected "
                f"{BEAT_CSV_HEADER}")
        return [(row["time"], row["downbeat"], row["section"]) for row in reader]


def repair_grid(corpus: Path, record: dict) -> tuple:
    grid = tp_record.beat_csv_path(corpus, record)
    if not grid.exists():
        raise RuntimeError(
            f"{record['id']}: no beat grid at {grid} -- repair never creates one")
    raw = read_raw_beat_rows(grid)
    wanted = [section for _time, _position, section in third_party_beat_rows(
        [(float(time), int(position)) for time, position, _section in raw],
        record)]
    before = [section for _time, _position, section in raw]
    if wanted == before:
        return False, 0
    promoted = sum(1 for was, now in zip(before, wanted)
                   if was == END_SENTINEL and now == MASKED_SENTINEL)
    _write_csv(grid, BEAT_CSV_HEADER,
               [(time, position, section)
                for (time, position, _section), section in zip(raw, wanted)])
    return True, promoted


def repair_sections(track_ids: list, source: str, corpus: Path) -> int:
    examined = changed = promoted = skipped = 0
    for track_id in track_ids:
        path = tp_record.annotation_path(corpus, track_id, source)
        if not path.exists():
            print(f"skipped {track_id}: no {source} record at {path}")
            skipped += 1
            continue
        record = tp_record.read_record(path)
        if not tp_record.beat_csv_path(corpus, record).exists():
            print(f"skipped {track_id}: no beat grid to repair")
            skipped += 1
            continue
        examined += 1
        moved, beats = repair_grid(corpus, record)
        if moved:
            changed += 1
            promoted += beats
            print(f"repaired {track_id}: {beats} beats end -> "
                  f"{MASKED_SENTINEL}")
    print(f"examined {examined} grids, changed {changed}, "
          f"{promoted} beats moved from {END_SENTINEL} to {MASKED_SENTINEL}, "
          f"skipped {skipped}")
    return 0


def _duration(text: str) -> float | None:
    return None if text == "" else float(text)


def load_clean_results(corpus: Path) -> list:
    path = tp_dir(corpus) / TP_CLEAN_MANIFEST_FILE
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [
            gate.CheckResult(
                row["track_id"], row["source"], row["audio_path"],
                _duration(row["probe_duration_sec"]),
                _duration(row["decoded_duration_sec"]),
                _duration(row["annotation_duration_sec"]),
                row["status"], row["detail"],
            )
            for row in csv.DictReader(handle)
        ]


def write_clean_manifest(corpus: Path, results: list) -> Path:
    rows = [
        (
            result.track_id,
            result.youtube_id,
            result.mp3_path,
            gate._format_duration(result.ffprobe_duration_sec),
            gate._format_duration(result.decoded_duration_sec),
            gate._format_duration(result.annotation_duration_sec),
            result.status,
            result.detail,
        )
        for result in sorted(results, key=lambda item: item.track_id)
    ]
    return _write_csv(tp_dir(corpus) / TP_CLEAN_MANIFEST_FILE, TP_CLEAN_HEADER, rows)


def upsert_clean_row(corpus: Path, result) -> None:
    results = [existing for existing in load_clean_results(corpus)
               if existing.track_id != result.track_id]
    results.append(result)
    write_clean_manifest(corpus, results)


def admit(track_id: str, source: str, corpus: Path | None = None) -> str:
    corpus = Path(corpus) if corpus is not None else corpus_dir()
    path = tp_record.annotation_path(corpus, track_id, source)
    if not path.exists():
        raise RuntimeError(f"{track_id}: no {source} record at {path}")
    record = tp_record.read_record(path)

    audio = tp_record.audio_path(corpus, record)
    if not audio.exists():
        raise RuntimeError(f"missing audio: {audio}")

    grid = tp_record.beat_csv_path(corpus, record)
    rows = read_beat_csv(grid) if grid.exists() else []
    kept = bool(rows)
    if not kept:
        rows = third_party_beat_rows(detect_beats(audio), record)
        if not rows:
            raise RuntimeError(
                f"{record['id']}: the offline tracker heard no beats in {audio.name}")
        grid.parent.mkdir(parents=True, exist_ok=True)
        write_beat_csv(grid, rows)

    # The gate's job is duration measurement, so its record travels with the
    # source in the slot a corpus track uses for its youtube id -- that is the
    # column this manifest carries, and the one the upsert reads back.
    result = gate.check_track(gate.TrackJob(
        str(record["id"]), str(record["source"]), str(audio),
        float(record["duration"])))
    upsert_clean_row(corpus, result)
    if result.status != gate.STATUS_OK:
        raise RuntimeError(
            f"{record['id']}: the cleanliness gate says {result.status} "
            f"({result.detail}) -- recorded, not admitted")

    write_manifest(corpus, manifest_rows(tp_record.load_all_third_party(corpus)))
    return (
        f"admitted {record['id']} ({record['source']}): beat grid {len(rows)} "
        f"beats ({'kept' if kept else 'generated'}), third-party manifest row, "
        f"clean row ok (decoded {result.decoded_duration_sec:.3f} s) -- "
        f"quarantined from the training table pending the integration ruling"
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("track_ids", nargs="*",
                        help="third-party track ids (e.g. rwc-p001)")
    parser.add_argument("--source", required=True,
                        choices=list(tp_record.THIRD_PARTY_SOURCES),
                        help="which converted annotation source to admit from")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root (default: the resolved corpus)")
    parser.add_argument("--all", action="store_true",
                        help="admit every converted record of that source")
    parser.add_argument("--repair-sections", action="store_true",
                        help="recompute only the section column of existing "
                             "beat grids; never runs the tracker")
    args = parser.parse_args(argv)

    corpus = args.data_dir.resolve() if args.data_dir else corpus_dir()
    wanted = list(args.track_ids)
    if args.all:
        wanted += [str(record["id"])
                   for record in tp_record.load_records(corpus, args.source)
                   if str(record["id"]) not in wanted]
    if not wanted:
        parser.error("name at least one track id, or pass --all")

    if args.repair_sections:
        return repair_sections(wanted, args.source, corpus)

    admitted = 0
    failed = 0
    for track_id in wanted:
        try:
            print(admit(track_id, args.source, corpus=corpus))
            admitted += 1
        except Exception as exc:  # one bad track must not end a batch of a hundred
            print(f"FAILED {track_id}: {exc}")
            failed += 1
    print(f"admitted {admitted}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
