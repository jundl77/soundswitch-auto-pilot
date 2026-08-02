"""Where do the slips come from, and can anything cheap see them coming?

No decoding: this is a property of the live beat stream, the annotated grid and
the boundary head.  It reproduces 1c's slip inventory as an anchor, measures
where the first DETECTED beat sits in the bar, dissects every slip into an
interval signature, and sweeps the candidate trackers so the decoding gate only
pays for configs worth paying for.

Candidates are ranked on per-beat PHASE ACCURACY against the annotated grid, not
on bar-line placement.  Placement is reported and is not a selector: at the
decoder's +/-0.5 s window and a 0.4725 s median beat, a grid rotated by one beat
still places almost every line "near a downbeat", so the proxy is blind to
exactly the error that costs the most crispness.

Val only.  Zero GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE_B = Path(r"C:\Users\Julian\Projects\soundswitch-phase-b-worktree")
sys.path.insert(0, str(PHASE_B))
sys.path.insert(0, str(HERE))

import training.nn.decoder  # noqa: E402
from training.nn.evaluate_v1 import split_ids, write_json  # noqa: E402
from training.nn.priors import MODELS_DIR  # noqa: E402
from raveform_fetch_annotations import BEATS_DIR, annotations_dir  # noqa: E402
from build_training_table import TABLE_FILE  # noqa: E402
from evaluate_against_labels import load_tracks  # noqa: E402

import phase_tracking as pt  # noqa: E402

if not Path(training.nn.decoder.__file__).is_relative_to(PHASE_B):
    raise SystemExit("the decoder generation must come from the phase_b worktree")

PLACEMENT_TOL = 0.5
WINDOWS = (4, 8, 16)
ANCHORS = (0, 1, 2, 3)


class BoundaryStream:
    """One track's raw boundary curve plus the geometry to sample it.

    Lifted from ``task1c_phase_vote`` so the two artifacts read the head the
    same way.  The score is raw by contract and is never a probability.
    """

    def __init__(self, sidecar: Path, min_coverage: int):
        with np.load(sidecar) as archive:
            self.boundary = np.asarray(archive["boundary"], dtype=np.float64)
            coverage = np.asarray(archive["coverage"], dtype=np.int64)
            self.frame_sec = float(archive["frame_sec"])
            self.t0 = float(archive["t0"])
        self.ok = coverage[:len(self.boundary)] >= int(min_coverage)
        self.times = self.t0 + np.arange(len(self.boundary)) * self.frame_sec

    def sample(self, lines: np.ndarray, tol: float, how: str = "max") -> np.ndarray:
        out = np.full(len(lines), np.nan, dtype=np.float64)
        if tol <= 0.0:
            idx = np.rint((np.asarray(lines) - self.t0) / self.frame_sec)
            idx = np.clip(idx.astype(np.int64), 0, len(self.boundary) - 1)
            good = self.ok[idx]
            out[good] = self.boundary[idx[good]]
            return out
        lo = np.searchsorted(self.times, np.asarray(lines) - tol, "left")
        hi = np.searchsorted(self.times, np.asarray(lines) + tol, "right")
        for i in range(len(lines)):
            window = self.ok[lo[i]:hi[i]]
            if window.any():
                values = self.boundary[lo[i]:hi[i]][window]
                out[i] = values.max() if how == "max" else values.mean()
        return out


def placement(lines: np.ndarray, downbeats: np.ndarray, tol: float = PLACEMENT_TOL):
    """One-to-one precision / recall / F1 of bar lines against real downbeats.

    Each line may claim at most one downbeat, closest first, so emitting more
    lines can no longer buy recall.  Reported, never selected on -- see the
    module docstring for why the tolerance makes it a poor ranker.
    """
    if lines.size == 0 or downbeats.size == 0:
        return 0.0, 0.0, 0.0
    idx = np.clip(np.searchsorted(downbeats, lines), 1, len(downbeats) - 1)
    nearest = np.where(np.abs(lines - downbeats[idx - 1])
                       <= np.abs(lines - downbeats[idx]), idx - 1, idx)
    distance = np.abs(lines - downbeats[nearest])
    used = np.zeros(downbeats.size, dtype=bool)
    matched = 0
    for i in np.argsort(distance):
        if distance[i] <= tol and not used[nearest[i]]:
            used[nearest[i]] = True
            matched += 1
    precision = matched / lines.size
    recall = matched / downbeats.size
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def span_repairs(js: np.ndarray, required: np.ndarray, phases: np.ndarray):
    """Truth and detected phase movement across each inter-match span."""
    truth = (required[1:] - required[:-1]) % 4
    detected = (phases[js[1:]] - phases[js[:-1]]) % 4
    return truth, detected


def slip_pr(truth: np.ndarray, detected: np.ndarray) -> dict:
    real = truth != 0
    tp = int(((truth == detected) & real).sum())
    fn = int(real.sum() - tp)
    fp = int(((detected != 0) & ~real).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def stat(values) -> dict | None:
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size == 0:
        return None
    return {"median": round(float(np.median(values)), 4),
            "mean": round(float(values.mean()), 4),
            "p10": round(float(np.percentile(values, 10)), 4),
            "p90": round(float(np.percentile(values, 90)), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "n": int(values.size)}


def signature(track, k: int) -> dict:
    """What the live stream looks like across one slip's span."""
    js, live = track["js"], track["live"]
    lo, hi = int(js[k - 1]), int(js[k])
    local = track["ratios"][8][lo + 1:hi + 1]
    finite = local[np.isfinite(local)]
    return {"live_gap": hi - lo,
            "annotated_gap": int(track["isx"][k] - track["isx"][k - 1]),
            "ratios": [round(float(v), 3) for v in local[:6]],
            "max_abs_deviation": (round(float(np.abs(finite - 1.0).max()), 3)
                                  if finite.size else None)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(
        r"C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    data_dir = args.data_dir
    if args.out_dir is None:
        args.out_dir = data_dir / MODELS_DIR / "phase_b" / "phase_tracking"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids = split_ids(data_dir, "val")
    by_yt = {t.track_id.split(".", 1)[-1]: t
             for t in load_tracks(data_dir / TABLE_FILE)}
    beats_dir = annotations_dir(data_dir) / BEATS_DIR
    posteriors_dir = data_dir / "posteriors_phase_b" / "student_kd_t2_w05_s1234"

    inventory = json.loads((data_dir / MODELS_DIR / "phase_b" / "integration_gate"
                            / "task1c_beat_slip.json").read_text(encoding="utf-8"))
    known_slips = {r["youtube_id"]: r["phase_slips"]
                   for r in inventory["per_track"] if r.get("usable")}

    tracks = []
    for yt in ids:
        times, positions = pt.annotation_beats(beats_dir / f"{by_yt[yt].track_id}.beat.csv")
        live = np.asarray(np.load(data_dir / "madmom_beats" / f"{yt}.npy"),
                          dtype=np.float64).reshape(-1)
        live.sort()
        js, isx = pt.match(live, times)
        phase = pt.expert_phase(times, positions)
        if js.size < 8 or phase is None:
            continue
        required = pt.required_phase(js, isx, phase)
        truth, known = pt.truth_phase_track(len(live), js, required)
        tracks.append({
            "youtube_id": yt, "annotated": times, "phase": phase, "live": live,
            "downbeats": np.asarray(sorted(t for t, p in zip(times, positions)
                                           if p == 1)),
            "js": js, "isx": isx, "required": required, "truth": truth,
            "known": known, "slips": int((np.diff(required) != 0).sum()),
            "ratios": {w: pt.interval_ratios(live, window=w) for w in WINDOWS},
        })
    print(f"loaded {len(tracks)} val tracks", flush=True)

    mismatches = [{"youtube_id": t["youtube_id"], "ours": t["slips"],
                   "task1c": known_slips.get(t["youtube_id"])}
                  for t in tracks if known_slips.get(t["youtube_id"]) != t["slips"]]
    if mismatches:
        raise SystemExit(f"SLIP ANCHOR FAILED on {len(mismatches)}: {mismatches[:5]}")
    print(f"SLIP ANCHOR reproduced on {len(tracks)} tracks, "
          f"{sum(t['slips'] for t in tracks)} slips total", flush=True)

    first = {}
    for track in tracks:
        first[int(track["required"][0])] = first.get(int(track["required"][0]), 0) + 1
    offset = {}
    for track in tracks:
        key = int(track["isx"][0] - track["js"][0])
        offset[key] = offset.get(key, 0) + 1
    print("required phase at the first matched live beat:",
          json.dumps(dict(sorted(first.items()))), flush=True)
    print("annotated index of the first matched live beat:",
          json.dumps(dict(sorted(offset.items()))), flush=True)

    shapes: dict = {}
    invisible = 0
    deviations = []
    for track in tracks:
        required = track["required"]
        for k in np.nonzero(np.diff(required) != 0)[0] + 1:
            sig = signature(track, int(k))
            delta = int((required[k] - required[k - 1]) % 4)
            key = f"delta{delta}_livegap{min(sig['live_gap'], 4)}"
            shapes[key] = shapes.get(key, 0) + 1
            if sig["max_abs_deviation"] is not None:
                deviations.append(sig["max_abs_deviation"])
                invisible += sig["max_abs_deviation"] <= 0.2
    print("slip shapes:",
          json.dumps(dict(sorted(shapes.items(), key=lambda kv: -kv[1]))), flush=True)
    print(f"slips with no interval evidence (max |r-1| <= 0.2): "
          f"{invisible} of {len(deviations)} "
          f"({invisible / max(len(deviations), 1):.3f})", flush=True)

    def evaluate(name: str, positions_fn, anchors=(0,), config=None) -> list:
        rows = {a: {"agg": {"tp": 0, "fp": 0, "fn": 0}, "per_track": []}
                for a in anchors}
        for track in tracks:
            positions, advances, interpolate = positions_fn(track)
            live = track["live"]
            index = np.arange(live.size)
            good = track["known"]
            for anchor in anchors:
                shifted = (positions + anchor) % 4
                lines = pt.edges_from_positions(live, shifted, advances, interpolate)
                precision, recall, f1 = placement(lines, track["downbeats"])
                phases = (index - shifted) % 4
                accuracy = (float((phases[good] == track["truth"][good]).mean())
                            if good.any() else 0.0)
                truth_delta, detected = span_repairs(track["js"], track["required"],
                                                     phases)
                counts = slip_pr(truth_delta, detected)
                for key in rows[anchor]["agg"]:
                    rows[anchor]["agg"][key] += counts[key]
                rows[anchor]["per_track"].append(
                    {"youtube_id": track["youtube_id"], "precision": precision,
                     "recall": recall, "f1": f1, "phase_accuracy": accuracy,
                     "n_lines": int(lines.size), **counts})
        out = []
        for anchor in anchors:
            block = rows[anchor]
            agg = block["agg"]
            p = agg["tp"] / max(agg["tp"] + agg["fp"], 1)
            r = agg["tp"] / max(agg["tp"] + agg["fn"], 1)
            per_track = block["per_track"]
            out.append({
                "name": name if len(anchors) == 1 and anchor == 0
                        else f"{name}|anchor{anchor}",
                "anchor": anchor,
                "phase_accuracy": round(float(np.mean(
                    [x["phase_accuracy"] for x in per_track])), 6),
                "phase_accuracy_median": round(float(np.median(
                    [x["phase_accuracy"] for x in per_track])), 6),
                "placement_f1": round(float(np.mean(
                    [x["f1"] for x in per_track])), 6),
                "placement_precision": round(float(np.mean(
                    [x["precision"] for x in per_track])), 6),
                "placement_recall": round(float(np.mean(
                    [x["recall"] for x in per_track])), 6),
                "slip_precision": round(p, 6), "slip_recall": round(r, 6),
                "slip_counts": agg,
                "config": dict(config, anchor=anchor) if config else {"anchor": anchor},
                "per_track": per_track,
            })
        return out

    def dump(stage: str):
        """Write the artifact after every sweep stage.

        The box is shared, so a run can be many times slower than its own CPU
        time; a stage that finished should not be re-run because a later one
        was still going.
        """
        ordered = sorted(results, key=lambda r: -r["phase_accuracy"])
        write_json(args.out_dir / "phase_probe.json", {
            "task": "phase tracking -- the no-decode probe: slip anatomy and the sweep",
            "split": "val",
            "stage": stage,
            "tracks": len(tracks),
            "placement_tolerance_sec": PLACEMENT_TOL,
            "ranked_on": "phase_accuracy",
            "why_not_placement": (
                "at +/-0.5 s and a 0.4725 s median beat a one-beat rotation still "
                "places nearly every bar line 'near a downbeat', so placement cannot "
                "rank the error that costs the most crispness"),
            "slip_anchor": {"source": "task1c_beat_slip.json",
                            "tracks_checked": len(tracks),
                            "mismatches": mismatches,
                            "total_slips": int(sum(t["slips"] for t in tracks))},
            "first_beat": {
                "required_phase_histogram": dict(sorted(first.items())),
                "annotated_index_of_first_matched_live_beat": dict(sorted(offset.items())),
                "live_start_prior": list(pt.LIVE_START_PRIOR),
            },
            "slip_shapes": dict(sorted(shapes.items(), key=lambda kv: -kv[1])),
            "slip_interval_evidence": {
                "n": len(deviations),
                "share_invisible_at_0.2": round(invisible / max(len(deviations), 1), 6),
                "max_abs_deviation": stat(deviations),
            },
            "sweep": [{k: v for k, v in row.items() if k != "per_track"}
                      for row in results],
            "per_track": {row["name"]: row["per_track"] for row in ordered[:5]},
        })
        print(f"[{stage}] wrote {args.out_dir / 'phase_probe.json'} "
              f"({len(results)} rows)", flush=True)

    def plain(track):
        n = track["live"].size
        advances = np.ones(n, dtype=np.int64)
        advances[0] = 0
        return np.arange(n, dtype=np.int64) % 4, advances, False

    results = evaluate("fallback", plain, anchors=ANCHORS)
    results[0]["name"] = "fallback_phase0"
    for lag in (0, 4, 8, 16):
        results.extend(evaluate(f"oracle_lag{lag}", lambda t, lag=lag: (
            pt.oracle_positions(t["live"].size, t["truth"], t["known"], lag),
            np.ones(t["live"].size, dtype=np.int64), False)))

    a_grid = [{"del_lo": d, "ins_hi": i, "window": w, "max_advance": m,
               "interpolate": p}
              for d in (1.4, 1.5, 1.6, 1.75) for i in (0.6, 0.65)
              for w in WINDOWS for m in (2, 4) for p in (True, False)]
    for config in a_grid:
        def positions_fn(track, config=config):
            positions, advances = pt.repair_from_ratios(
                track["ratios"][config["window"]], del_lo=config["del_lo"],
                ins_hi=config["ins_hi"], max_advance=config["max_advance"])
            return positions, advances, config["interpolate"]
        name = ("A|del%.2f|ins%.2f|w%d|max%d|%s"
                % (config["del_lo"], config["ins_hi"], config["window"],
                   config["max_advance"],
                   "interp" if config["interpolate"] else "plain"))
        results.extend(evaluate(name, positions_fn, anchors=ANCHORS, config=config))
    print(f"swept {len(a_grid)} A configs x {len(ANCHORS)} anchors", flush=True)
    dump("A")

    for track in tracks:
        stream = BoundaryStream(posteriors_dir / f"{track['youtube_id']}.npz", 1)
        period = float(np.median(np.diff(track["live"])))
        track["boundary"] = {
            "frame": stream.sample(track["live"], 0.0),
            "quarter": stream.sample(track["live"], 0.25 * period),
            "half": stream.sample(track["live"], 0.5 * period),
        }
    print("sampled the boundary head at every live beat", flush=True)

    best_a = max((r for r in results if r["name"].startswith("A|")),
                 key=lambda r: r["phase_accuracy"])
    print(f"best A: {best_a['name']} phase_acc={best_a['phase_accuracy']:.4f} "
          f"placement_f1={best_a['placement_f1']:.4f} "
          f"slip P/R {best_a['slip_precision']:.3f}/{best_a['slip_recall']:.3f}",
          flush=True)

    def make_b(config):
        def positions_fn(track):
            base = None
            if config["base"] == "A":
                base = pt.repair_from_ratios(
                    track["ratios"][best_a["config"]["window"]],
                    del_lo=best_a["config"]["del_lo"],
                    ins_hi=best_a["config"]["ins_hi"],
                    max_advance=best_a["config"]["max_advance"])
            samples = track["boundary"][config["read"]] if config["beta"] else None
            committed, advances, _, _ = pt.phase_filter(
                track["live"], samples, lag_beats=config["lag"],
                sigma=config["sigma"], eta=config["eta"], beta=config["beta"],
                slip_rate=config.get("slip_rate"), base=base,
                anchor=best_a["anchor"] if config["base"] == "A" else 0,
                start_prior=(pt.LIVE_START_PRIOR if config["prior"] == "live"
                             else (0.25, 0.25, 0.25, 0.25)),
                ratios=track["ratios"][8], max_advance=config["max_advance"])
            return committed, advances, config["interpolate"]
        return positions_fn

    b_grid = [{"lag": g, "sigma": s, "eta": 0.02, "beta": b, "read": "quarter",
               "base": "none", "prior": p, "max_advance": 4, "interpolate": False}
              for g in (4, 8, 16) for s in (0.15, 0.22)
              for b in (0.0, 0.5, 1.0, 2.0) for p in ("live", "uniform")]
    b_grid += [{"lag": 8, "sigma": 0.15, "eta": 0.02, "beta": b, "read": r,
                "base": "none", "prior": "live", "max_advance": 4,
                "interpolate": False}
               for b in (0.5, 1.0, 2.0) for r in ("frame", "half")]
    b_grid += [{"lag": 8, "sigma": 0.15, "eta": 0.02, "beta": b, "read": "quarter",
                "base": "none", "prior": "live", "max_advance": 2,
                "interpolate": False, "slip_rate": s}
               for s in (0.002, 0.01, 0.05) for b in (0.0, 2.0)]
    for config in b_grid:
        name = ("B|lag%d|sig%.2f|eta%.3f|beta%.1f|%s|%s%s"
                % (config["lag"], config["sigma"], config["eta"], config["beta"],
                   config["read"], config["prior"],
                   "|flat%.3f" % config["slip_rate"] if "slip_rate" in config else ""))
        results.extend(evaluate(name, make_b(config), config=config))
    print(f"swept {len(b_grid)} B configs", flush=True)
    dump("AB")

    c_grid = [{"lag": g, "sigma": 0.15, "eta": 0.02, "beta": b, "read": "quarter",
               "base": "A", "prior": "live", "max_advance": 4, "interpolate": p}
              for g in (4, 8, 16) for b in (0.0, 0.5, 1.0, 2.0) for p in (True, False)]
    for config in c_grid:
        name = ("C|lag%d|beta%.1f|%s" % (config["lag"], config["beta"],
                                         "interp" if config["interpolate"] else "plain"))
        results.extend(evaluate(name, make_b(config), config=config))
    print(f"swept {len(c_grid)} C configs", flush=True)

    ranked = sorted(results, key=lambda r: -r["phase_accuracy"])
    print("\ntop 20 by per-beat phase accuracy:", flush=True)
    for row in ranked[:20]:
        print(f"  {row['name']:<48} acc={row['phase_accuracy']:.4f} "
              f"med={row['phase_accuracy_median']:.4f} "
              f"place={row['placement_f1']:.4f} "
              f"slipP/R={row['slip_precision']:.3f}/{row['slip_recall']:.3f}",
              flush=True)
    print("\nreference rows:", flush=True)
    for row in results[:8]:
        print(f"  {row['name']:<48} acc={row['phase_accuracy']:.4f} "
              f"place={row['placement_f1']:.4f}", flush=True)

    dump("ABC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
