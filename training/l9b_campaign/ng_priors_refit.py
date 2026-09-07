"""PREP 4: per-arm priors refit on the label9 instrument (branch label9_migration).

Arm H is the merged annotation view, train split only -- the same computation
as running label9's training/nn/priors.py directly, and --verify-anchor proves
it: it runs that CLI as a subprocess and this module's H arm side by side and
diffs the bytes (Priors.save is timestamp-free and byte-stable by content).
Arm HD is identical except the overlay's drop sections are rewritten to
breakdown in the annotation records BEFORE fitting; the overlay is train-only
by construction and an overlay span that matches no published drop section
within 0.05 s is a drifted input and fatal.  The seam: corpus_bar_runs is
mirrored with a per-track section transform, then priors.py's own fit_runs and
Priors.save do everything else.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

from ng_common import CAMP, CORPUS, LABEL9_WORKTREE, VENV_PY

W = str(LABEL9_WORKTREE)
sys.path[:0] = [W, W + r"\training"]

try:
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402

from nn import priors as P  # noqa: E402

SPAN_TOLERANCE = 0.05


def load_overlay(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != "nextgen_drop_demotion":
        raise SystemExit(f"{path} is not a nextgen_drop_demotion overlay "
                         f"(kind={document.get('kind')!r})")
    spans: dict = {}
    for section in document["sections"]:
        spans.setdefault(str(section["id"]), []).append(
            (float(section["start"]), float(section["end"])))
    return {"document": document, "spans": spans}


def demote(track_id: str, sections: list, spans: list) -> list:
    used = [False] * len(spans)
    out = []
    for start, end, label in sections:
        hit = None
        if label == "drop":
            for index, (a, b) in enumerate(spans):
                if (not used[index] and abs(start - a) <= SPAN_TOLERANCE
                        and abs(end - b) <= SPAN_TOLERANCE):
                    hit = index
                    break
        if hit is None:
            out.append((start, end, label))
        else:
            used[hit] = True
            out.append((start, end, "breakdown"))
    if not all(used):
        unmatched = [spans[i] for i, u in enumerate(used) if not u]
        raise SystemExit(f"{track_id}: overlay span(s) {unmatched} match no "
                         f"published drop section -- drifted inputs, refusing")
    return out


def bar_run_sequences(data_dir: Path, youtube_ids: list, spans_by_id: dict) -> tuple:
    """corpus_bar_runs, mirrored, with the demotion applied before bar_runs."""
    from raveform_fetch_annotations import (
        beat_csv_path, load_all_tracks, parse_beat_csv, parse_sections)
    wanted = set(str(i) for i in youtube_ids)
    by_id = {str(track.get("id")): track for track in load_all_tracks(data_dir)}

    sequences: list = []
    skipped: list = []
    demoted = 0
    for youtube_id in sorted(wanted):
        track = by_id.get(youtube_id)
        if track is None:
            skipped.append(youtube_id)
            continue
        path = beat_csv_path(data_dir, track)
        if not path.exists():
            skipped.append(youtube_id)
            continue
        downbeats = np.array(
            [time for time, position, _section in parse_beat_csv(path)
             if position == 1], dtype=np.float64)
        if downbeats.size < 2:
            skipped.append(youtube_id)
            continue
        sections = parse_sections(track)
        spans = spans_by_id.get(youtube_id)
        if spans:
            sections = demote(youtube_id, sections, spans)
            demoted += len(spans)
        runs = [(label, bars) for label, bars in
                P.bar_runs(sections, downbeats) if bars > 0]
        if runs:
            sequences.append(runs)
        else:
            skipped.append(youtube_id)
    return sequences, skipped, demoted


def fit_arm(data_dir: Path, split: str, overlay: dict | None):
    ids = P.split_ids(data_dir, split)
    spans_by_id = overlay["spans"] if overlay else {}
    outside = sorted(set(spans_by_id) - set(str(i) for i in ids))
    if outside:
        raise SystemExit(f"overlay names {len(outside)} track(s) outside the "
                         f"'{split}' split ({outside[:5]}...) -- the overlay is "
                         f"train-only by construction, refusing")
    sequences, skipped, demoted = bar_run_sequences(data_dir, ids, spans_by_id)
    if not sequences:
        raise SystemExit(f"no usable tracks in split '{split}' of {data_dir}")
    provenance = {
        "split": split,
        "split_size": len(ids),
        "skipped_no_beat_grid": skipped,
    }
    if overlay:
        document = overlay["document"]
        expected = len(document["sections"])
        if demoted != expected:
            raise SystemExit(f"demoted {demoted} sections, overlay carries "
                             f"{expected} -- refusing")
        provenance["demotion_overlay"] = {
            "kind": document["kind"],
            "feature": document["feature"],
            "threshold_db": document["threshold_db"],
            "source": document["source"],
            "sections_demoted": demoted,
            "tracks_demoted": len(spans_by_id),
        }
    return P.fit_runs(sequences, strict=True, provenance=provenance)


def floor_rows(priors, labels=("buildup", "drop")) -> str:
    lines = []
    for label in labels:
        index = priors.index(label)
        stats = priors.corpus["duration_bars"][label]
        lines.append(f"  {label:<9} n={stats['n']:>4} median={stats['median']:>5.1f} "
                     f"floor={int(priors.floor_bars[index]):>2} "
                     f"hazard={priors.hazard[index]:.4f} "
                     f"E[bars]={stats['expected_bars']:.1f}")
    return "\n".join(lines)


def verify_anchor(data_dir: Path, split: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cli_out = out_dir / "priors_label9_cli.json"
    mine_out = out_dir / "priors_ng_arm_h.json"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([W, W + r"\training"])
    print(f"running label9 priors.py CLI -> {cli_out}", flush=True)
    result = subprocess.run(
        [str(VENV_PY), "-m", "nn.priors", "--data-dir", str(data_dir),
         "--split", split, "--out", str(cli_out)],
        capture_output=True, text=True, env=env, cwd=W, timeout=3600)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"label9 priors.py CLI failed (exit {result.returncode})")

    print(f"running this module's arm H -> {mine_out}", flush=True)
    fit_arm(data_dir, split, None).save(mine_out)

    a, b = cli_out.read_bytes(), mine_out.read_bytes()
    if a == b:
        print(f"ANCHOR OK: byte-identical ({len(a)} bytes)")
        return 0
    print(f"ANCHOR FAILED: {len(a)} vs {len(b)} bytes differ", file=sys.stderr)
    return 1


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("H", "HD"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overlay", type=Path,
                        default=CAMP / "drop_demotion_overlay.json")
    parser.add_argument("--data-dir", type=Path, default=CORPUS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--verify-anchor", action="store_true")
    parser.add_argument("--anchor-dir", type=Path, default=CAMP / "anchor")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    if args.verify_anchor:
        return verify_anchor(data_dir, args.split, args.anchor_dir.resolve())
    if not args.arm or not args.out:
        parser.error("--arm and --out are required (or pass --verify-anchor)")

    overlay = load_overlay(args.overlay.resolve()) if args.arm == "HD" else None
    priors = fit_arm(data_dir, args.split, overlay)
    priors.save(args.out)
    corpus = priors.corpus
    print(f"arm {args.arm}: fitted {corpus['tracks']} tracks / {corpus['runs']} "
          f"runs / {corpus['bars']} bars (split={args.split})")
    print(floor_rows(priors))

    if args.arm == "HD":
        reference = fit_arm(data_dir, args.split, None)
        moved = int(np.sum(~np.isclose(reference.transition, priors.transition,
                                       rtol=0, atol=1e-12)))
        print(f"\nvs arm H: {moved} transition cells moved, "
              f"floor deltas {dict(zip(priors.classes, (priors.floor_bars - reference.floor_bars).tolist()))}")
        print("arm H reference rows:")
        print(floor_rows(reference))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
