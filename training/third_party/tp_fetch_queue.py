#!/usr/bin/env python
"""Build the merged third-party fetch queue: SALAMI's remainder, then Harmonix.

There is ONE queue because there is one downloader.  The corpus's politeness
contract is strictly sequential downloads with a pause between videos, so two
work lists swept by two processes would double the request rate against
YouTube.  SALAMI rows come first: there are fewer of them and that conversion
is closer to done.

The Harmonix cut is by DTW alignment score, which is the mean slope of the
warping path between the dataset's own audio and the YouTube upload (higher is
better, 1.0 is a straight diagonal).  Its floor across all 912 tracks is 0.803,
so every threshold at or below 0.80 is a no-op; 0.95 keeps 784 tracks inside
the dense high-confidence mass.  Path straightness is provably blind to a
UNIFORM speed offset -- a track played 3% fast warps to a straight line of the
wrong gradient -- so the nine tracks the dataset annotates as needing a speed
or direction correction score well (8 of 9 above 0.95) and are excluded by name
regardless.  The repo's own ``align_thres=0.9`` is a never-referenced default
argument, not published guidance.

None of this is the verification.  The score is a first filter; what catches a
wrong video is the cleanliness gate at admission, which decodes the audio and
holds its duration against the annotation's.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlsplit

_TRAINING_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_TRAINING_DIR.parent), str(_TRAINING_DIR),
              str(_TRAINING_DIR / "raveform"), str(_TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tp_record  # noqa: E402
from corpus_root import corpus_dir  # noqa: E402

QUEUE_FILE = "fetch_queue.csv"
QUEUE_HEADER = ("track_id", "source", "youtube_id", "title", "artist",
                "annotation_duration_sec", "note")

SALAMI_SOURCE = "salami"
HARMONIX_SOURCE = "harmonix"

SALAMI_MIN_COVERAGE = 0.9
SALAMI_EXCLUDED_CLASS = "Live_Music_Archive"

HARMONIX_DEFAULT_THRESHOLD = 0.95

# The URL field of these nine carries free text after the link ("backwards",
# "needs to be 2.45% sped-up"): the upload is not time-aligned to the
# annotation without a speed correction we do not apply.  The DTW score cannot
# see it -- a uniform offset is still a straight path -- so they go by name.
HARMONIX_UNALIGNED = frozenset({
    "0046_castlesmadeofsand",
    "0218_policyoftruth",
    "0541_youdroppedabombonme",
    "0653_dynamite",
    "0772_kissmebackcrytonight",
    "0777_lebump",
    "0913_somuchlove",
    "0980_whenyouleave",
    "0998_youregonnamissthis",
})

SALAMI_PAIRINGS_FILE = "salami_youtube_pairings.csv"
HARMONIX_URLS_FILE = "youtube_urls.csv"
HARMONIX_SCORES_FILE = "youtube_alignment_scores.csv"
HARMONIX_METADATA_FILE = "metadata.csv"

DEFAULT_HARMONIXSET = Path(r"D:\rwc_stage\harmonixset")
DEFAULT_MATCHING_SALAMI = Path(r"C:\Users\Julian\Projects\matching-salami")


class QueueRow(NamedTuple):
    track_id: str
    source: str
    youtube_id: str
    title: str
    artist: str
    annotation_duration_sec: str
    note: str


class Tally(NamedTuple):
    rows: list
    counts: dict


def queue_path(corpus: Path) -> Path:
    return tp_record.third_party_dir(Path(corpus)) / QUEUE_FILE


def audio_path(corpus: Path, source: str, track_id: str) -> Path:
    return (tp_record.third_party_dir(Path(corpus)) / tp_record.AUDIO_DIR
            / source / f"{track_id}.mp3")


def default_salami_dir() -> Path:
    return corpus_dir().parent / "salami-data-public"


def parse_video_id(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    values = parse_qs(urlsplit(text.split()[0]).query).get("v") or []
    return values[0].strip() if values else ""


def _bump(counts: dict, key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _read_csv(path: Path) -> list:
    if not path.is_file():
        raise RuntimeError(f"missing {path}")
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_salami_metadata(salami_dir: Path) -> dict:
    rows = _read_csv(Path(salami_dir) / "metadata" / "metadata.csv")
    return {(row.get("SONG_ID") or "").strip(): row for row in rows}


def salami_rows(corpus: Path, salami_dir: Path, pairings: Path) -> Tally:
    salami_dir = Path(salami_dir)
    metadata = read_salami_metadata(salami_dir)
    counts: dict = {}
    rows: list = []
    for row in _read_csv(Path(pairings)):
        native_id = (row.get("salami_id") or "").strip()
        track_id = f"salami-{native_id}"
        video_id = (row.get("youtube_id") or "").strip()
        try:
            coverage = float(row.get("coverage_percent") or "")
        except ValueError:
            coverage = -1.0
        if coverage < SALAMI_MIN_COVERAGE:
            _bump(counts, "salami_low_coverage")
            continue
        if not (salami_dir / "annotations" / native_id).is_dir():
            _bump(counts, "salami_no_annotation")
            continue
        record = metadata.get(native_id, {})
        if (record.get("CLASS") or "").strip() == SALAMI_EXCLUDED_CLASS:
            _bump(counts, "salami_live_music_archive")
            continue
        if (salami_dir / "audio" / f"{native_id}.mp3").exists():
            _bump(counts, "salami_already_on_disk")
            continue
        if audio_path(corpus, SALAMI_SOURCE, track_id).exists():
            _bump(counts, "salami_already_fetched")
            continue
        if not video_id:
            _bump(counts, "salami_no_video_id")
            continue
        rows.append(QueueRow(
            track_id=track_id,
            source=SALAMI_SOURCE,
            youtube_id=video_id,
            title=(record.get("SONG_TITLE") or "").strip(),
            artist=(record.get("ARTIST") or "").strip(),
            annotation_duration_sec=(row.get("salami_length") or "").strip(),
            note=f"coverage={coverage:.4f}",
        ))
        _bump(counts, "salami_kept")
    return Tally(rows, counts)


def harmonix_rows(corpus: Path, harmonixset: Path, threshold: float) -> Tally:
    dataset = Path(harmonixset) / "dataset"
    scores = {
        (row.get("File") or "").strip(): row.get("score") or ""
        for row in _read_csv(dataset / HARMONIX_SCORES_FILE)
    }
    metadata = {
        (row.get("File") or "").strip(): row
        for row in _read_csv(dataset / HARMONIX_METADATA_FILE)
    }
    counts: dict = {}
    rows: list = []
    for row in _read_csv(dataset / HARMONIX_URLS_FILE):
        name = (row.get("File") or "").strip()
        track_id = f"hx-{name}"
        if name in HARMONIX_UNALIGNED:
            _bump(counts, "harmonix_unaligned_by_name")
            continue
        try:
            score = float(scores.get(name, ""))
        except ValueError:
            _bump(counts, "harmonix_no_score")
            continue
        if score < threshold:
            _bump(counts, "harmonix_below_threshold")
            continue
        video_id = parse_video_id(row.get("URL") or "")
        if not video_id:
            _bump(counts, "harmonix_no_video_id")
            continue
        if audio_path(corpus, HARMONIX_SOURCE, track_id).exists():
            _bump(counts, "harmonix_already_fetched")
            continue
        record = metadata.get(name, {})
        rows.append(QueueRow(
            track_id=track_id,
            source=HARMONIX_SOURCE,
            youtube_id=video_id,
            title=(record.get("Title") or "").strip(),
            artist=(record.get("Artist") or "").strip(),
            annotation_duration_sec=(record.get("Duration") or "").strip(),
            note=f"align={score:.4f}",
        ))
        _bump(counts, "harmonix_kept")
    return Tally(rows, counts)


def write_queue(path: Path, rows: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(QUEUE_HEADER)
            writer.writerows(rows)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_queue(path: Path) -> list:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"no fetch queue at {path} -- run tp_fetch_queue.py")
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(QUEUE_HEADER) - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"{path} is missing column(s): {', '.join(sorted(missing))}")
        return [QueueRow(*(row.get(name) or "" for name in QUEUE_HEADER))
                for row in reader]


def build_queue(corpus: Path, salami_dir: Path, pairings: Path,
                harmonixset: Path, threshold: float) -> Tally:
    salami = salami_rows(corpus, salami_dir, pairings)
    harmonix = harmonix_rows(corpus, harmonixset, threshold)
    counts = dict(salami.counts)
    counts.update(harmonix.counts)
    return Tally(salami.rows + harmonix.rows, counts)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root (default: the resolved corpus dir)")
    parser.add_argument("--harmonix-threshold", type=float,
                        default=HARMONIX_DEFAULT_THRESHOLD,
                        help="minimum DTW alignment score (default: %(default)s)")
    parser.add_argument("--salami-dir", type=Path, default=None,
                        help="salami-data-public checkout "
                             "(default: <corpus>/../salami-data-public)")
    parser.add_argument("--harmonixset", type=Path, default=DEFAULT_HARMONIXSET,
                        help="harmonixset checkout (default: %(default)s)")
    parser.add_argument("--matching-salami", type=Path,
                        default=DEFAULT_MATCHING_SALAMI,
                        help=f"directory holding {SALAMI_PAIRINGS_FILE} "
                             "(default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the breakdown and write nothing")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    corpus = (args.data_dir or corpus_dir()).resolve()
    salami_dir = (args.salami_dir or default_salami_dir()).resolve()
    pairings = Path(args.matching_salami).resolve() / SALAMI_PAIRINGS_FILE
    target = queue_path(corpus)

    tally = build_queue(corpus, salami_dir, pairings, args.harmonixset,
                        args.harmonix_threshold)

    print("third-party fetch queue")
    print(f"corpus          : {corpus}")
    print(f"salami-data     : {salami_dir}")
    print(f"pairings        : {pairings}")
    print(f"harmonixset     : {Path(args.harmonixset).resolve()}")
    print(f"hx threshold    : {args.harmonix_threshold:g}")
    print(f"queue           : {target}")
    print()
    for source in (SALAMI_SOURCE, HARMONIX_SOURCE):
        kept = tally.counts.get(f"{source}_kept", 0)
        print(f"{source:9s} kept  : {kept}")
        for key in sorted(tally.counts):
            if key.startswith(f"{source}_") and not key.endswith("_kept"):
                print(f"            rejected {key[len(source) + 1:]:22s} "
                      f"{tally.counts[key]}")
    print()
    print(f"queue rows      : {len(tally.rows)}")
    if args.dry_run:
        print("dry run         : nothing written")
        return 0
    write_queue(target, tally.rows)
    print(f"wrote           : {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
