#!/usr/bin/env python
"""Make a committed hand label dataset-complete: beat grid, manifest row, clean row.

The labelling tool's Commit places two files -- ``annotations/<track_id>.hand.json``
and, for new audio, ``audio/<track_id>.mp3`` -- and then calls ``admit`` here.
Everything else a dataset member needs is this module's job: a beat grid in the
published format (madmom's offline downbeat tracker -- never the online one the
show runs, whose warm-up and causality are runtime constraints this side does
not have), the manifest row, and a clean-manifest row whose durations are
measured by the same gate every downloaded track passes. Fields with no source
stay absent. ``segments.json``, ``checksums.sha256``, the splits file and every
eval artifact are never touched; the split is reported, and materialises at the
next dataset build through the same hash every corpus track goes through.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent
for _path in (str(_TRAINING_DIR.parent), str(_TRAINING_DIR),
              str(_TRAINING_DIR / "raveform")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_clean_manifest as gate  # noqa: E402
from corpus_root import corpus_dir  # noqa: E402
from lib.label_space import SECTION_LABELS  # noqa: E402
from raveform_fetch_annotations import (  # noqa: E402
    HAND_LABEL_SOURCE,
    annotations_dir,
    beat_csv_path,
    load_all_tracks,
    load_hand_tracks,
    parse_beat_csv,
    youtube_id,
)
from raveform_manifest import build_manifest_rows, write_manifest  # noqa: E402

BEAT_CSV_HEADER = ("time", "downbeat", "section")
HAND_GRID_HEADER = ("time", "downbeat")
HAND_GRID_SUFFIX = ".hand.beat.csv"

BEATS_PER_BAR = 4
DBN_FPS = 100

START_SENTINEL = "start"
END_SENTINEL = "end"


def track_record(corpus: Path, track_id: str) -> dict:
    wanted = str(track_id)
    for record in load_all_tracks(corpus):
        if wanted in (youtube_id(record), str(record.get("key", ""))):
            if str(record.get("source", "")) != HAND_LABEL_SOURCE:
                raise RuntimeError(
                    f"{wanted} has no hand label in {annotations_dir(corpus)} -- "
                    f"admission is only for hand-labelled tracks"
                )
            return record
    raise RuntimeError(f"{wanted}: no annotation record in {corpus}")


def check_vocabulary(record: dict) -> None:
    names = {str(section["name"]) for section in record["sections"]}
    unknown = sorted(names - set(SECTION_LABELS))
    if unknown:
        raise RuntimeError(
            f"{record['key']}: label(s) outside the vocabulary: {', '.join(unknown)}"
        )


def section_of(spans: list, t: float) -> str:
    if spans and t < spans[0][0]:
        return START_SENTINEL
    for start, end, name in spans:
        if start <= t < end:
            return name
    return END_SENTINEL


def beat_rows(beats, sections: list) -> list:
    spans = sorted(
        ((float(section["start"]), float(section["end"]), str(section["name"]))
         for section in sections),
        key=lambda span: span[0],
    )
    return [(float(time), int(position), section_of(spans, float(time)))
            for time, position in beats]


def _time_field(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _write_csv(path: Path, header: tuple, rows: list) -> None:
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


def write_beat_csv(path: Path, rows: list) -> None:
    _write_csv(path, BEAT_CSV_HEADER,
               [(_time_field(time), position, section)
                for time, position, section in rows])


def write_hand_grid(path: Path, rows: list) -> None:
    _write_csv(path, HAND_GRID_HEADER,
               [(_time_field(time), 1 if position == 1 else 0)
                for time, position, _section in rows])


def detect_beats(audio: Path):
    from madmom.features.downbeats import (DBNDownBeatTrackingProcessor,
                                           RNNDownBeatProcessor)
    activations = RNNDownBeatProcessor()(str(audio))
    return DBNDownBeatTrackingProcessor(
        beats_per_bar=[BEATS_PER_BAR], fps=DBN_FPS)(activations)


def _duration(text: str) -> float | None:
    return None if text == "" else float(text)


def load_clean_results(data_dir: Path) -> list:
    path = data_dir / gate.CLEAN_MANIFEST_FILE
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [
            gate.CheckResult(
                row["track_id"], row["youtube_id"], row["mp3_path"],
                _duration(row["ffprobe_duration_sec"]),
                _duration(row["decoded_duration_sec"]),
                _duration(row["annotation_duration_sec"]),
                row["status"], row["detail"],
            )
            for row in csv.DictReader(handle)
        ]


def upsert_clean_row(data_dir: Path, result) -> None:
    results = [existing for existing in load_clean_results(data_dir)
               if existing.track_id != result.track_id]
    results.append(result)
    gate.write_clean_manifest(data_dir, results)


def admit(track_id: str, audio: Path | None = None, labels: Path | None = None,
          corpus: Path | None = None) -> str:
    corpus = Path(corpus) if corpus is not None else corpus_dir()
    record = track_record(corpus, track_id)
    check_vocabulary(record)

    mp3 = corpus / gate.AUDIO_DIR / str(
        record.get("audio") or f"{youtube_id(record)}.mp3")
    if audio is not None and Path(audio).resolve() != mp3.resolve():
        raise RuntimeError(f"audio mismatch: told {audio}, the record names {mp3}")
    if not mp3.exists():
        raise RuntimeError(f"missing audio: {mp3}")

    grid = beat_csv_path(corpus, record)
    generated = False
    if grid.exists():
        rows = parse_beat_csv(grid)
    else:
        rows = beat_rows(detect_beats(mp3), record["sections"])
        if not rows:
            raise RuntimeError(
                f"{record['key']}: the offline tracker heard no beats in {mp3.name}"
            )
        grid.parent.mkdir(parents=True, exist_ok=True)
        write_beat_csv(grid, rows)
        generated = True
    hand_grid = grid.parent / f"{youtube_id(record)}{HAND_GRID_SUFFIX}"
    if not hand_grid.exists():
        write_hand_grid(hand_grid, rows)

    write_manifest(corpus, build_manifest_rows(load_all_tracks(corpus)))

    result = gate.check_track(gate.TrackJob(
        str(record["key"]), youtube_id(record), str(mp3),
        float(record["duration"])))
    upsert_clean_row(corpus, result)
    if result.status != gate.STATUS_OK:
        raise RuntimeError(
            f"{record['key']}: the cleanliness gate says {result.status} "
            f"({result.detail}) -- recorded, not admitted"
        )

    from nn.dataset import assign_split
    split = assign_split(youtube_id(record))
    return (
        f"admitted {record['key']}: beat grid {len(rows)} beats "
        f"({'generated' if generated else 'kept'}), manifest row, clean row ok "
        f"(decoded {result.decoded_duration_sec:.3f} s), split '{split}' at the "
        f"next dataset build"
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("track_ids", nargs="*",
                        help="hand-label track ids (hand-<sha> or a native id)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root (default: the resolved corpus)")
    parser.add_argument("--all", action="store_true",
                        help="admit every committed hand label")
    args = parser.parse_args(argv)

    corpus = args.data_dir.resolve() if args.data_dir else corpus_dir()
    wanted = list(args.track_ids)
    if args.all:
        wanted += [youtube_id(record) for record in load_hand_tracks(corpus)
                   if youtube_id(record) not in wanted]
    if not wanted:
        parser.error("name at least one track id, or pass --all")

    for track_id in wanted:
        print(admit(track_id, corpus=corpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
