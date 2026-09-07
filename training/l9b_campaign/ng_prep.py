"""PREP 1: verify the six new hand tracks are admitted; emit the extraction inputs.

Enumerates the new hand labels (hand-* prefixed, mtime past the cutoff),
refuses on any surprise, verifies full admission per id (hand.json, audio,
beat grid, ok clean-manifest row), runs training/hand_label_admission.py for
any id missing a grid or clean row, then writes CAMP/extract_ids.txt and
CAMP/audio_map.json in the exact format stream_extract's --ids-file and
--audio-map consume.  --post-extract re-checks that every F3 sidecar exists
and is a loadable npz with the F3/hop1 geometry.  Never touches the GPU.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

from ng_common import CAMP, CORPUS, F3_DIR, NEW_HAND_IDS, REPO, VENV_PY

MTIME_CUTOFF = datetime.datetime(2026, 9, 5)


def new_hand_ids(corpus: Path) -> list:
    found = []
    for path in sorted((corpus / "annotations").glob("*.hand.json")):
        if not path.name.startswith("hand-"):
            continue
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        if mtime > MTIME_CUTOFF:
            found.append(path.name[: -len(".hand.json")])
    if set(found) != set(NEW_HAND_IDS):
        raise SystemExit(
            f"new hand labels on disk {sorted(found)} != the chartered six "
            f"{sorted(NEW_HAND_IDS)} -- refusing to proceed on a surprise"
        )
    return list(NEW_HAND_IDS)


def clean_rows(corpus: Path) -> dict:
    path = corpus / "clean_manifest.csv"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return {row["track_id"]: row for row in csv.DictReader(handle)}


def check_id(corpus: Path, track_id: str, rows: dict) -> dict:
    state = {"id": track_id}
    hand = corpus / "annotations" / f"{track_id}.hand.json"
    try:
        record = json.loads(hand.read_text(encoding="utf-8"))
        state["hand_json"] = "ok" if record.get("id") == track_id else "id-mismatch"
    except (OSError, ValueError) as error:
        state["hand_json"] = f"UNREADABLE ({error})"
    state["audio"] = "ok" if (corpus / "audio" / f"{track_id}.mp3").exists() else "MISSING"
    state["beat_grid"] = ("ok" if (corpus / "annotations" / "beats"
                                   / f"{track_id}.beat.csv").exists() else "MISSING")
    row = rows.get(track_id)
    if row is None:
        state["clean_row"] = "MISSING"
    else:
        state["clean_row"] = row["status"] if row["status"] == "ok" else f"NOT-OK ({row['status']})"
    state["f3_sidecar"] = "ok" if (F3_DIR / f"{track_id}.npz").exists() else "missing"
    return state


def print_table(states: list) -> None:
    columns = ("id", "hand_json", "audio", "beat_grid", "clean_row", "f3_sidecar")
    widths = {c: max(len(c), max(len(str(s[c])) for s in states)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for state in states:
        print("  ".join(str(state[c]).ljust(widths[c]) for c in columns))


def admit(corpus: Path, track_id: str) -> None:
    flags = 0x00004000 if sys.platform == "win32" else 0
    print(f"\nadmitting {track_id} via training/hand_label_admission.py ...", flush=True)
    result = subprocess.run(
        [str(VENV_PY), str(REPO / "training" / "hand_label_admission.py"),
         track_id, "--data-dir", str(corpus)],
        capture_output=True, text=True, cwd=str(REPO), creationflags=flags,
        timeout=1800)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"admission failed for {track_id} (exit {result.returncode})")


def write_extraction_inputs(corpus: Path, camp: Path, ids: list) -> list:
    camp.mkdir(parents=True, exist_ok=True)
    missing = [i for i in ids if not (F3_DIR / f"{i}.npz").exists()]
    lines = ["# nextgen campaign (decision #341): new hand tracks missing an F3 sidecar",
             "# consumed by training.nn.ceiling.stream_extract --ids-file"]
    (camp / "extract_ids.txt").write_text("\n".join(lines + missing) + "\n",
                                          encoding="utf-8")
    audio_map = {i: str((corpus / "audio" / f"{i}.mp3").resolve()) for i in missing}
    (camp / "audio_map.json").write_text(
        json.dumps(audio_map, indent=2) + "\n", encoding="utf-8")
    return missing


def post_extract(corpus: Path, ids: list) -> None:
    import numpy as np
    rows = clean_rows(corpus)
    failures = []
    for track_id in ids:
        path = F3_DIR / f"{track_id}.npz"
        try:
            with np.load(path) as archive:
                emb = archive["emb"]
                geometry = (int(archive["stream_causal"]),
                            float(archive["stream_margin_sec"]),
                            float(archive["stream_hop_sec"]))
                cell_sec = float(archive["label_frame_sec"])
            if emb.ndim != 3 or emb.shape[1] != 2 or emb.shape[2] != 1024 or emb.shape[0] < 1:
                raise RuntimeError(f"emb shape {emb.shape}")
            if geometry != (1, 3.0, 1.0):
                raise RuntimeError(f"stream geometry {geometry} != (causal, F3, hop1)")
            duration = float(rows[track_id]["decoded_duration_sec"])
            covered = emb.shape[0] * cell_sec
            if not 0.9 * duration <= covered <= 1.1 * duration:
                raise RuntimeError(f"{covered:.1f} s of cells vs {duration:.1f} s of audio")
            print(f"{track_id}: emb {emb.shape} covers {covered:.1f} s -- ok")
        except Exception as error:  # noqa: BLE001 - collected, then fatal
            failures.append(f"{track_id}: {error}")
    if failures:
        raise SystemExit("post-extract FAILED:\n  " + "\n  ".join(failures))
    print(f"\nall {len(ids)} F3 sidecars present and sane")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=CORPUS)
    parser.add_argument("--camp", type=Path, default=CAMP)
    parser.add_argument("--post-extract", action="store_true")
    args = parser.parse_args(argv)

    corpus = args.data_dir.resolve()
    ids = new_hand_ids(corpus)
    if args.post_extract:
        post_extract(corpus, ids)
        return 0

    states = [check_id(corpus, i, clean_rows(corpus)) for i in ids]
    print_table(states)

    needs = [s["id"] for s in states
             if s["beat_grid"] == "MISSING" or s["clean_row"] != "ok"]
    for track_id in needs:
        admit(corpus, track_id)
    if needs:
        states = [check_id(corpus, i, clean_rows(corpus)) for i in ids]
        print()
        print_table(states)

    broken = [s["id"] for s in states
              if not (s["hand_json"] == "ok" and s["audio"] == "ok"
                      and s["beat_grid"] == "ok" and s["clean_row"] == "ok")]
    if broken:
        raise SystemExit(f"still not fully admitted after admission: {broken}")

    missing = write_extraction_inputs(corpus, args.camp.resolve(), ids)
    print(f"\nwrote {args.camp / 'extract_ids.txt'} ({len(missing)} ids to extract)")
    print(f"wrote {args.camp / 'audio_map.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
