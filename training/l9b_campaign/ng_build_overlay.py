"""PREP 3: the drop-demotion overlay -- mellow drops rewritten to breakdown.

Ruling D2 (decision #341): demote published drop sections whose measured
mean_db sits under -16.5 (the measured -15.5 zero-false-demotion point on mean
power plus the chartered 1.0 dB margin).  Time spans are re-derived by aligning
each track's CSV rows, in order, to its published drop sections in order --
count or per-row duration disagreement (>0.05 s) is a drifted input and fatal.
Kept sections must be train-split, hand-free (hand wins), and outside
test/val/eval.  Writes CAMP/drop_demotion_overlay.json plus the report-only
CAMP/val_demotion_diagnostic.json (ruling 1: how the demotion WOULD read on
val -- never applied).  Reads segments.json read-only; writes only under CAMP.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import sys
from pathlib import Path

from ng_common import (CAMP, CORPUS, DEMOTION_BASE_THRESHOLD_DB,
                       DEMOTION_FEATURE, DEMOTION_MARGIN_DB,
                       DEMOTION_THRESHOLD_DB, inject_repo_paths)

inject_repo_paths()

from raveform_fetch_annotations import load_tracks  # noqa: E402

DURATION_TOLERANCE = 0.05
DEFAULT_CSV = (CORPUS / "models" / "spec_mapped_campaign" / "mellow_drop_measurement"
               / "drop_sections.csv")


def csv_rows_by_id(path: Path) -> dict:
    grouped: dict = collections.OrderedDict()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["youtube_id"], []).append(row)
    return grouped


def aligned_sections(track: dict, rows: list) -> list:
    track_id = str(track["id"])
    drops = [s for s in track["sections"] if str(s["name"]) == "drop"]
    if len(drops) != len(rows):
        raise SystemExit(
            f"{track_id}: {len(rows)} CSV drop rows vs {len(drops)} published "
            f"drop sections -- drifted inputs, refusing")
    out = []
    for index, (section, row) in enumerate(zip(drops, rows)):
        start, end = float(section["start"]), float(section["end"])
        if abs(float(row["duration"]) - (end - start)) > DURATION_TOLERANCE:
            raise SystemExit(
                f"{track_id} drop #{index}: CSV duration {row['duration']} vs "
                f"published {end - start:.2f} -- drifted inputs, refusing")
        out.append({"id": track_id, "start": start, "end": end,
                    "duration": float(row["duration"]),
                    "mean_db": float(row["mean_db"]),
                    "genre": row["genre"]})
    return out


def sizing(sections: list) -> dict:
    per_genre: dict = collections.Counter()
    per_genre_sec: dict = collections.Counter()
    for section in sections:
        genre = section["genre"] or "(none)"
        per_genre[genre] += 1
        per_genre_sec[genre] += section["end"] - section["start"]
    return {
        "sections": len(sections),
        "tracks": len({s["id"] for s in sections}),
        "total_seconds": round(sum(s["end"] - s["start"] for s in sections), 2),
        "per_genre": {g: {"sections": per_genre[g],
                          "seconds": round(per_genre_sec[g], 2)}
                      for g in sorted(per_genre)},
    }


def print_sizing(label: str, report: dict) -> None:
    hours = report["total_seconds"] / 3600.0
    print(f"{label}: {report['sections']} sections on {report['tracks']} tracks, "
          f"{report['total_seconds']:.0f} s ({hours:.2f} h)")
    for genre, entry in sorted(report["per_genre"].items(),
                               key=lambda kv: -kv[1]["seconds"]):
        print(f"    {genre:<22} {entry['sections']:>4} sections  "
              f"{entry['seconds']:>9.0f} s")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=CORPUS)
    parser.add_argument("--camp", type=Path, default=CAMP)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--splits-file", type=Path, default=None)
    args = parser.parse_args(argv)

    corpus = args.data_dir.resolve()
    camp = args.camp.resolve()
    splits = json.loads((args.splits_file or corpus / "splits.json")
                        .read_text(encoding="utf-8"))
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    eval_ids = set(splits["eval_set"])
    hand_ids = {p.name[: -len(".hand.json")]
                for p in (corpus / "annotations").glob("*.hand.json")}

    tracks = {str(t["id"]): t for t in load_tracks(corpus)}
    grouped = csv_rows_by_id(args.csv)
    aligned: list = []
    for track_id, rows in grouped.items():
        track = tracks.get(track_id)
        if track is None:
            raise SystemExit(f"{track_id}: in the CSV but not in segments.json")
        aligned.extend(aligned_sections(track, rows))

    def eligible(split_members: set, threshold: float) -> tuple:
        kept, overridden = [], set()
        for section in aligned:
            track_id = section["id"]
            if section["mean_db"] >= threshold or track_id not in split_members:
                continue
            if track_id in test or track_id in val or track_id in eval_ids:
                if split_members is train:
                    raise SystemExit(f"{track_id}: train member also in "
                                     f"test/val/eval -- splits are inconsistent")
            if track_id in hand_ids:
                overridden.add(track_id)
                continue
            kept.append(section)
        return kept, sorted(overridden)

    kept, overridden = eligible(train, DEMOTION_THRESHOLD_DB)
    unmargined, _ = eligible(train, DEMOTION_BASE_THRESHOLD_DB)
    val_kept, val_overridden = eligible(val, DEMOTION_THRESHOLD_DB)

    camp.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    overlay = {
        "kind": "nextgen_drop_demotion",
        "threshold_db": DEMOTION_THRESHOLD_DB,
        "feature": DEMOTION_FEATURE,
        "base_threshold_db": DEMOTION_BASE_THRESHOLD_DB,
        "margin_db": DEMOTION_MARGIN_DB,
        "source": str(args.csv.resolve()),
        "generated_utc": now,
        "excluded_hand_overridden": overridden,
        "sections": kept,
    }
    (camp / "drop_demotion_overlay.json").write_text(
        json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
    diagnostic = {
        "kind": "nextgen_drop_demotion_val_diagnostic",
        "note": "report-only per #341 ruling 1 -- never applied to any fit",
        "threshold_db": DEMOTION_THRESHOLD_DB,
        "feature": DEMOTION_FEATURE,
        "source": str(args.csv.resolve()),
        "generated_utc": now,
        "excluded_hand_overridden": val_overridden,
        "sizing": sizing(val_kept),
        "sections": val_kept,
    }
    (camp / "val_demotion_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")

    print(f"aligned {len(aligned)} drop sections across {len(grouped)} CSV tracks")
    whole = [s for s in aligned if s["mean_db"] < DEMOTION_BASE_THRESHOLD_DB]
    print_sizing(f"whole measurement unmargined (mean_db < "
                 f"{DEMOTION_BASE_THRESHOLD_DB}, #340's ballpark is "
                 f"~380 sections / ~5.9 h)", sizing(whole))
    print_sizing(f"TRAIN overlay (mean_db < {DEMOTION_THRESHOLD_DB})", sizing(kept))
    print_sizing(f"TRAIN unmargined (mean_db < {DEMOTION_BASE_THRESHOLD_DB}, "
                 f"for the record)", sizing(unmargined))
    print_sizing(f"VAL diagnostic (mean_db < {DEMOTION_THRESHOLD_DB}, report-only)",
                 sizing(val_kept))
    if overridden:
        print(f"hand-overridden train tracks excluded: {overridden}")
    print(f"\nwrote {camp / 'drop_demotion_overlay.json'}")
    print(f"wrote {camp / 'val_demotion_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
