"""L9C conditional arm: the anti-supervision mask overlay, hand-labels excluded.

l9b's ARM H-MK builder (#343) with one selection change: every mined span whose
track now carries a hand label on disk (annotations/<id>.hand.json) is DROPPED
from the overlay rather than fatally refused.  Since the l9b cut the owner
hand-relabeled ~28 corpus tracks, and the trainer's mask loader
(training/nn/ceiling/mask.py, phase-b worktree) hard-refuses any masked span on
a hand-labeled track -- the owner's relabel wins and the overlay may not touch
it.  The exclusion is recorded in the output (span count, track ids) so the cut
is auditable, and the surviving span count is stated beside it.

Spans are RE-MINED from the l9b campaign's registered mining artifact
(models/l9b_campaign/label_mining/corpus_mining.json, read-only) rather than
filtered out of the banked l9b overlay: the re-mine re-runs every build-time
check against TODAY's disk state -- published-section alignment (drift is
fatal), drop-abut shape, train membership, val/test/eval refusal -- where
filtering the banked JSON would inherit l9b's checks without re-running them.
The hand exclusion set is read live from the annotations dir at build time, so
a label committed after this run is picked up by rebuilding.  Writes
CAMP/supervision_mask_overlay.json only, same kind the trainer's loader checks.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

MAIN = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot")
CORPUS = MAIN / "training" / "data" / "raveform"
CAMP = CORPUS / "models" / "l9c_campaign"
MINING = CORPUS / "models" / "l9b_campaign" / "label_mining" / "corpus_mining.json"

W = str(MAIN)
sys.path[:0] = [W, W + r"\training", W + r"\training\raveform"]

from raveform_fetch_annotations import load_tracks  # noqa: E402

OVERLAY_KIND = "nextgen_supervision_mask"
DELTA_Q_MIN = 0.6
RHO_MIN = 0.5
SPAN_TOLERANCE = 0.05
DROP_ABUT_TOLERANCE = 0.5
MASKABLE = {"breakdown", "bridge"}
EXPECTED_SPANS = 279  # the registered pre-exclusion read of the mining artifact


def aligned_section(track_id: str, span: tuple, rows: list) -> dict:
    start, end = span
    inside = [row for row in rows
              if float(row[1]) > start + SPAN_TOLERANCE
              and float(row[0]) < end - SPAN_TOLERANCE]
    if len(inside) != 1:
        raise SystemExit(f"{track_id} [{start}, {end}]: {len(inside)} published "
                         f"sections overlap the mined span -- drifted inputs")
    s, e, label = inside[0]
    if label not in MASKABLE:
        raise SystemExit(f"{track_id} [{start}, {end}]: overlapping section is "
                         f"{label!r}, not breakdown/bridge")
    if abs(float(s) - start) > SPAN_TOLERANCE or abs(float(e) - end) > SPAN_TOLERANCE:
        raise SystemExit(f"{track_id}: mined span [{start}, {end}] vs published "
                         f"section [{s}, {e}] -- drifted inputs")
    follows_drop = any(r[2] == "drop" and abs(float(r[0]) - end) <= DROP_ABUT_TOLERANCE
                       for r in rows)
    if not follows_drop:
        raise SystemExit(f"{track_id} [{start}, {end}]: no drop section starts "
                         f"at the span end -- not the mined shape")
    return {"id": track_id, "start": float(s), "end": float(e), "label": label}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=CORPUS)
    parser.add_argument("--camp", type=Path, default=CAMP)
    parser.add_argument("--mining", type=Path, default=MINING)
    args = parser.parse_args(argv)

    corpus, camp = args.data_dir.resolve(), args.camp.resolve()
    mining_path = args.mining.resolve()
    mining = json.loads(mining_path.read_text(encoding="utf-8"))
    selected = [c for c in mining["candidates_breakdown_abutting_drop"]
                if c["split"] == "train"
                and c["ramp"]["delta_q"] >= DELTA_Q_MIN
                and c["ramp"]["rho"] >= RHO_MIN]
    if len(selected) != EXPECTED_SPANS:
        raise SystemExit(f"selection produced {len(selected)} spans, the "
                         f"registered read is {EXPECTED_SPANS} -- mining "
                         f"artifact drifted")

    splits = json.loads((corpus / "splits.json").read_text(encoding="utf-8"))
    train = set(splits["train"])
    off_limits = set(splits["val"]) | set(splits["test"]) | set(splits["eval_set"])
    hand_ids = {p.name[: -len(".hand.json")]
                for p in (corpus / "annotations").glob("*.hand.json")}

    sections_by_id = {}
    for track in load_tracks(corpus):
        sections_by_id[str(track["id"])] = [
            (float(s["start"]), float(s["end"]), str(s["name"]))
            for s in track["sections"]]

    kept = []
    excluded = []
    for candidate in selected:
        track_id = str(candidate["youtube_id"])
        if track_id not in train:
            raise SystemExit(f"{track_id}: mined as train but not in the "
                             f"corpus train list")
        if track_id in off_limits:
            raise SystemExit(f"{track_id}: in val/test/eval -- refusing")
        if track_id in hand_ids or candidate.get("hand_labeled"):
            excluded.append(track_id)
            continue
        entry = aligned_section(track_id, tuple(candidate["span"]),
                                sections_by_id[track_id])
        entry["genre"] = candidate.get("genre")
        entry["delta_q"] = candidate["ramp"]["delta_q"]
        entry["rho"] = candidate["ramp"]["rho"]
        kept.append(entry)

    per_genre = collections.Counter(e["genre"] or "(none)" for e in kept)
    total_sec = round(sum(e["end"] - e["start"] for e in kept), 2)
    overlay = {
        "kind": OVERLAY_KIND,
        "selection": {"source": str(mining_path),
                      "pool": "candidates_breakdown_abutting_drop",
                      "split": "train", "delta_q_min": DELTA_Q_MIN,
                      "rho_min": RHO_MIN},
        "hand_excluded": {
            "reason": "track carries annotations/<id>.hand.json -- the "
                      "owner's relabel wins and the overlay may not touch it "
                      "(the trainer's mask loader hard-refuses)",
            "spans": len(excluded),
            "track_ids": sorted(set(excluded)),
        },
        "surviving_spans": len(kept),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sections": kept,
    }
    camp.mkdir(parents=True, exist_ok=True)
    out = camp / "supervision_mask_overlay.json"
    out.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
    print(f"mask overlay: {len(kept)} sections on "
          f"{len({e['id'] for e in kept})} tracks, {total_sec:.0f} s "
          f"({total_sec / 3600.0:.2f} h)")
    print(f"hand-label exclusion: {len(excluded)} spans dropped on "
          f"{len(set(excluded))} tracks")
    for genre, count in per_genre.most_common():
        print(f"    {genre:<22} {count:>4} sections")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
