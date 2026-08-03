"""What does the shipping decoder score on a TRACKED bar grid?

1d priced the fallback -- bars every 4 madmom online beats from the first
detected beat -- at 0.4966 crispness@0.5, against 0.6362 for the same counting
rule on annotated beats and 0.6739 on expert downbeats.  The whole -0.1396 is
booked by phase slips, so the question this answers is whether a live tracker
can win any of it back.

The candidates come from ``phase_tracking_probe``'s no-decode sweep, selected on
bar-line placement F1 -- a different metric from the one they are then scored on,
so the pick is made without reading the gate's own answer.  Every grid is built
causally with corrections applied FORWARD ONLY: a correction that arrives N beats
late re-times future bar lines and never rewrites one already emitted.

Val only (#112), lag_bars = 2 (#154).  Zero GPU.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE_B = Path(r"C:\Users\Julian\Projects\soundswitch-phase-b-worktree")
sys.path.insert(0, str(PHASE_B))
sys.path.insert(0, str(HERE))

from training.nn.decoder import DecodeParams, bar_observations  # noqa: E402
from training.nn.ceiling.decoder_frontier import score_stream  # noqa: E402
from training.nn.evaluate_v1 import (  # noqa: E402
    DEFAULT_SPACE, aggregate, build_decoder, decode_beats, generation_model_sha,
    load_inputs, restricted_macro_f1, score_predicted, split_ids, write_json)
from training.nn.paired import CONTESTED  # noqa: E402
from training.nn.priors import MODELS_DIR, PRIORS_FILE, Priors  # noqa: E402
from raveform_fetch_annotations import BEATS_DIR, annotations_dir  # noqa: E402

import phase_tracking as pt  # noqa: E402
from phase_tracking_probe import (  # noqa: E402
    WINDOWS, BoundaryStream, placement, slip_pr, span_repairs)

LAG_BARS = 2
ANCHOR = {"boundary_f1_0.5": 0.673856, "macro_f1_contested": 0.708425,
          "flicker_2.0": 0.312882}
ANCHOR_TOL = 1e-6
ANNOTATED_ANCHOR = 0.636185
ANNOTATED_TOL = 1e-5
# 1d's published live row, to the digits it was published at.  The exact values
# are read back out of its artifact below and checked against these, so a
# rounding-width agreement can never stand in for reproduction.
LIVE_PUBLISHED = {"boundary_f1_0.5": 0.4966, "macro_f1_contested": 0.7045,
                  "flicker_2.0": 0.3213}
LIVE_TOL = 1e-9

DRAWS = 2000
SEED = 20260803

# Pre-registered, and written here before any candidate was decoded.  A win must
# clear all three: the crispness gain survives its own error bars, the decoder's
# class decisions do not get worse beyond noise, and the show does not start
# twitching to pay for the placement.
PASS_RULE = {
    "crispness_ci_excludes_zero": True,
    "macro_contested_ci_may_not_lie_entirely_below_zero": True,
    "flicker_relative_rise_max": 0.05,
}


def config_sha(params: DecodeParams) -> str:
    payload = json.dumps(dataclasses.asdict(params), sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_sha(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def pick_params(frontier_path: Path):
    document = json.loads(Path(frontier_path).read_text(encoding="utf-8"))
    name = document["pick"]
    for row in document["frontier"]:
        if row["name"] == name:
            return name, DecodeParams(**row["params"])
    raise RuntimeError(f"{frontier_path} names pick {name!r} but has no such row")


def regrid(item, edges: np.ndarray, sidecar: Path, params: DecodeParams):
    posteriors, boundary = bar_observations(
        sidecar, edges, min_coverage=params.min_coverage,
        boundary_tolerance_sec=params.boundary_tolerance_sec,
        temperature=params.temperature)
    return dataclasses.replace(item, edges=edges, posteriors=posteriors,
                               boundary=boundary)


def metrics(scores) -> dict:
    total = aggregate(scores)
    return {
        "crispness_0.5": float(total.boundary_prf("class", 0.5)[2]),
        "macro_contested": float(restricted_macro_f1(total, CONTESTED)),
        "flicker_2.0": float(total.flicker_per_minute["class"][2.0]),
    }


def accumulators(score) -> list:
    """The raw additive cells the three bootstrap metrics are derived from.

    Every accumulator in ``Score`` is additive, so a resampled corpus metric is
    a function of column sums -- which makes the bootstrap a matrix product
    instead of two thousand Python-level aggregations.  Equivalence to
    ``metrics`` is a gate below, not an assumption.
    """
    cell = score.boundary["class"][0.5]["overall"]
    row = [cell["matched"], cell["n_pred"], cell["n_truth"]]
    for label in CONTESTED:
        row.extend(score.counts[label])
    row.append(score.flicker["class"][2.0])
    row.append(score.weight_total_sec)
    return row


def _f1(tp, fp, fn):
    precision = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1), 0.0)
    recall = np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1), 0.0)
    denominator = precision + recall
    return np.where(denominator > 0, 2 * precision * recall,
                    0.0) / np.where(denominator > 0, denominator, 1)


def derive(sums: np.ndarray) -> dict:
    sums = np.atleast_2d(sums)
    matched, n_pred, n_truth = sums[:, 0], sums[:, 1], sums[:, 2]
    crisp = _f1(matched, n_pred - matched, n_truth - matched)
    macro = np.zeros(len(sums))
    for k in range(len(CONTESTED)):
        tp, fp, fn = sums[:, 3 + 3 * k], sums[:, 4 + 3 * k], sums[:, 5 + 3 * k]
        macro = macro + _f1(tp, fp, fn)
    macro = macro / len(CONTESTED)
    minutes = sums[:, -1] / 60.0
    flicker = np.where(minutes > 0, sums[:, -2] / np.where(minutes > 0, minutes, 1), 0.0)
    return {"crispness_0.5": crisp, "macro_contested": macro, "flicker_2.0": flicker}


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


def fallback_positions(n: int):
    advances = np.ones(n, dtype=np.int64)
    advances[0] = 0
    return np.arange(n, dtype=np.int64) % 4, advances


def candidate_positions(track, spec):
    """``(bar position per live beat, beats each interval crosses)``.

    ``anchor`` is the bar position handed to the FIRST DETECTED beat.  The
    shipping fallback hardcodes 0; the probe measures that the live stream
    starts on position 1 on 147 of 215 tracks, because madmom's online warmup
    normally costs the first annotated beat.  B and C carry the anchor inside
    the filter's start prior instead, so it is not applied twice.
    """
    live = track["live"]
    n = len(live)
    family = spec["family"]
    anchor = int(spec.get("anchor", 0))
    if family == "oracle":
        return (pt.oracle_positions(n, track["truth"], track["known"], spec["lag"]),
                np.ones(n, dtype=np.int64))
    if family == "fallback":
        positions, advances = fallback_positions(n)
        return (positions + anchor) % 4, advances
    if family == "A":
        positions, advances = pt.repair_from_ratios(
            track["ratios"][spec["window"]], del_lo=spec["del_lo"],
            ins_hi=spec["ins_hi"], max_advance=spec["max_advance"])
        return (positions + anchor) % 4, advances
    if family in ("B", "C"):
        base = None
        if family == "C":
            base = pt.repair_from_ratios(
                track["ratios"][spec["base_window"]], del_lo=spec["base_del_lo"],
                ins_hi=spec["base_ins_hi"], max_advance=spec["base_max_advance"])
        samples = track["boundary"][spec["read"]] if spec["beta"] else None
        committed, advances, corrections, _ = pt.phase_filter(
            live, samples, lag_beats=spec["lag"], sigma=spec["sigma"],
            eta=spec["eta"], beta=spec["beta"], slip_rate=spec.get("slip_rate"),
            base=base, anchor=spec.get("base_anchor", 0) if base else 0,
            start_prior=(pt.LIVE_START_PRIOR if spec["prior"] == "live"
                         else (0.25, 0.25, 0.25, 0.25)),
            ratios=track["ratios"][8], max_advance=spec["max_advance"])
        track.setdefault("corrections", {})[spec["name"]] = len(corrections)
        return committed, advances
    raise RuntimeError(f"unknown family {family!r}")


def candidate_grid(track, spec) -> np.ndarray:
    """The candidate's bar grid, or the shipping fallback where it has none.

    A tracker that rotates its count forward can skip bar position 0 every time
    it corrects, and a tracker that corrects often enough starves the grid of
    bar lines entirely.  That is a real property of rotate-forward designs, not
    a decoding accident, so it is counted per arm and reported rather than
    hidden -- and the track is scored on the grid the live system would actually
    be running, which is the fallback.  Dropping the track instead would score
    the arms on different track sets.
    """
    positions, advances = candidate_positions(track, spec)
    lines = pt.edges_from_positions(track["live"], positions, advances,
                                    spec.get("interpolate", False))
    if lines.size < 2:
        track.setdefault("degenerate", {})[spec["name"]] = int(lines.size)
        positions, advances = fallback_positions(len(track["live"]))
        lines = pt.edges_from_positions(track["live"], positions, advances, False)
    track.setdefault("positions", {})[spec["name"]] = positions
    track.setdefault("lines", {})[spec["name"]] = lines
    return pt.close_grid(lines)


def select(sweep: list, runners_up: int = 2) -> list:
    """One config per family, chosen on the probe's PHASE ACCURACY, plus runners-up.

    The proxy is not the gate's metric, so this is a pre-registration rather
    than a peek: nothing here has been decoded.  It is phase accuracy and not
    bar-line placement because placement is measured at +/-0.5 s against a
    0.4725 s beat, which cannot see a one-beat rotation -- the error that costs
    the most crispness.  The two ablations are the best config in their own
    restricted family, so a lost win is attributable to the term that was
    removed and not to a worse search.

    Selection and scoring both happen on val (test is spent), so the runners-up
    are decoded too and reported beside the pick.  They cannot win the gate --
    only ``primary`` rows can -- but a pick that beats them by more than they
    differ from each other is a pick chosen by noise, and this is what makes
    that visible instead of arguable.
    """
    def ranked(predicate):
        return sorted((r for r in sweep if predicate(r)),
                      key=lambda r: -r["phase_accuracy"])

    def is_b(row):
        return row["name"].startswith("B|") and "flat" not in row["name"]

    families = [
        ("D_anchor", "fallback", lambda r: r["name"].startswith("fallback"), False),
        ("A_interval_repair", "A", lambda r: r["name"].startswith("A|"), True),
        ("B_phase_filter", "B", lambda r: is_b(r) and r["config"]["beta"] != 0.0, True),
        ("B_minus_boundary", "B", lambda r: is_b(r) and r["config"]["beta"] == 0.0, False),
        ("B_minus_intervals", "B",
         lambda r: r["name"].startswith("B|") and "flat" in r["name"], False),
        ("C_union", "C", lambda r: r["name"].startswith("C|"), True),
    ]
    a_row = ranked(lambda r: r["name"].startswith("A|"))[0]
    chosen = []
    missing = []
    for name, family, predicate, sweepable in families:
        rows = ranked(predicate)
        if not rows:
            missing.append(name)
            continue
        for rank, row in enumerate(rows[:1 + (runners_up if sweepable else 0)]):
            spec = dict(row["config"], family=family,
                        name=name if rank == 0 else f"{name}_alt{rank}",
                        primary=rank == 0, probe=row["name"],
                        probe_phase_accuracy=row["phase_accuracy"],
                        probe_placement_f1=row["placement_f1"])
            if family == "C":
                spec.update({f"base_{k}": v for k, v in a_row["config"].items()
                             if k != "interpolate"})
            chosen.append(spec)
    return chosen, missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(
        r"C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    posteriors_dir = data_dir / "posteriors_phase_b" / "student_kd_t2_w05_s1234"
    frontier_path = (data_dir / MODELS_DIR / "phase_b"
                     / "student_kd_t2_w05_s1234" / "frontier.json")
    madmom_dir = data_dir / "madmom_beats"
    gate_dir = data_dir / MODELS_DIR / "phase_b" / "integration_gate"
    if args.out_dir is None:
        args.out_dir = data_dir / MODELS_DIR / "phase_b" / "phase_tracking"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    name, base = pick_params(frontier_path)
    params = dataclasses.replace(base, lag_bars=LAG_BARS)
    print(f"pick {name}  lag_bars={params.lag_bars}  sha={config_sha(params)[:12]}",
          flush=True)

    model_dir = data_dir / MODELS_DIR / "phase_b_student_kd_t2_w05_s1234"
    model_sha = generation_model_sha(model_dir)
    priors = Priors.load(model_dir / PRIORS_FILE)
    ids = split_ids(data_dir, "val")
    if args.limit:
        ids = ids[:args.limit]

    inputs, skipped = load_inputs(
        data_dir, ids, min_coverage=params.min_coverage,
        boundary_tolerance_sec=params.boundary_tolerance_sec,
        temperature=params.temperature, posteriors_dir=posteriors_dir,
        model_sha=model_sha)
    if skipped:
        raise SystemExit(f"{len(skipped)} tracks unusable: {skipped[:5]}")
    print(f"loaded {len(inputs)} val tracks", flush=True)

    decoder = build_decoder(priors, params)

    def scored(items):
        streams = [decode_beats(item, decoder) for item in items]
        per_track = [score_predicted(item.as_track_beats(), DEFAULT_SPACE, stream)
                     for item, stream in zip(items, streams)]
        return score_stream(items, streams), per_track

    expert_row, expert_scores = scored(inputs)
    observed = {"boundary_f1_0.5": expert_row["boundary_f1"]["0.5"],
                "macro_f1_contested": expert_row["macro_f1_contested"],
                "flicker_2.0": expert_row["flicker_per_audience_minute"]["2.0"]}
    deltas = {k: round(observed[k] - ANCHOR[k], 9) for k in ANCHOR}
    if not all(abs(v) <= ANCHOR_TOL for v in deltas.values()):
        raise SystemExit(f"EXPERT ANCHOR FAILED {deltas} -- no conclusion may be drawn")
    print("EXPERT ANCHOR", json.dumps(deltas), flush=True)

    beats_dir = annotations_dir(data_dir) / BEATS_DIR
    tracks = []
    for item in inputs:
        times, positions = pt.annotation_beats(beats_dir / f"{item.track_id}.beat.csv")
        live = np.asarray(np.load(madmom_dir / f"{item.youtube_id}.npy"),
                          dtype=np.float64).reshape(-1)
        live.sort()
        js, isx = pt.match(live, times)
        phase = pt.expert_phase(times, positions)
        if js.size < 8 or phase is None:
            raise SystemExit(f"{item.youtube_id} has no usable phase truth")
        required = pt.required_phase(js, isx, phase)
        truth, known = pt.truth_phase_track(len(live), js, required)
        stream = BoundaryStream(posteriors_dir / f"{item.youtube_id}.npz", 1)
        period = float(np.median(np.diff(live)))
        tracks.append({
            "youtube_id": item.youtube_id, "item": item, "annotated": times,
            "downbeats": np.asarray(sorted(t for t, p in zip(times, positions)
                                           if p == 1)),
            "phase": phase, "live": live, "js": js, "isx": isx,
            "required": required, "truth": truth, "known": known,
            "slips": int((np.diff(required) != 0).sum()),
            "ratios": {w: pt.interval_ratios(live, window=w) for w in WINDOWS},
            "boundary": {"frame": stream.sample(live, 0.0),
                         "quarter": stream.sample(live, 0.25 * period),
                         "half": stream.sample(live, 0.5 * period)},
        })

    inventory = json.loads((gate_dir / "task1c_beat_slip.json").read_text(encoding="utf-8"))
    known_slips = {r["youtube_id"]: r["phase_slips"]
                   for r in inventory["per_track"] if r.get("usable")}
    mismatched = [t["youtube_id"] for t in tracks
                  if known_slips.get(t["youtube_id"]) != t["slips"]]
    if mismatched:
        raise SystemExit(f"SLIP ANCHOR FAILED on {len(mismatched)}: {mismatched[:5]}")
    print(f"SLIP ANCHOR reproduced, {sum(t['slips'] for t in tracks)} slips", flush=True)

    def build(edges_fn):
        return [regrid(track["item"], edges_fn(track),
                       posteriors_dir / f"{track['youtube_id']}.npz", params)
                for track in tracks]

    annotated_items = build(
        lambda t: pt.close_grid(t["annotated"][0::4]))
    annotated_row, annotated_scores = scored(annotated_items)
    delta = round(annotated_row["boundary_f1"]["0.5"] - ANNOTATED_ANCHOR, 9)
    if abs(delta) > ANNOTATED_TOL and not args.limit:
        raise SystemExit(f"ANNOTATED ANCHOR FAILED delta {delta}")
    print(f"ANNOTATED ANCHOR b@0.5 {annotated_row['boundary_f1']['0.5']:.6f} "
          f"delta {delta:+.9f}", flush=True)

    specs = [dict(family="fallback", name="live_fallback_phase0", interpolate=False,
                  primary=False)]
    probe_path = args.out_dir / "phase_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    selected, missing_families = select(probe["sweep"])
    specs.extend(selected)
    if missing_families:
        print(f"WARNING probe stage {probe.get('stage')!r} carries no rows for "
              f"{missing_families} -- those candidates are ABSENT from this gate",
              flush=True)
    specs.append(dict(family="oracle", name="oracle_lag8", lag=8, interpolate=False,
                      primary=False))
    specs.append(dict(family="oracle", name="oracle_lag0", lag=0, interpolate=False,
                      primary=False))

    arms = {"expert": expert_scores, "annotated_phase0": annotated_scores}
    rows = {"expert": expert_row, "annotated_phase0": annotated_row}
    geometry = {}
    for spec in specs:
        items = build(lambda t, s=spec: candidate_grid(t, s))
        row, per_track = scored(items)
        arms[spec["name"]] = per_track
        rows[spec["name"]] = row
        degenerate = [t["youtube_id"] for t in tracks
                      if spec["name"] in t.get("degenerate", {})]
        geometry[spec["name"]] = {
            "bar_lines": int(sum(t["lines"][spec["name"]].size for t in tracks)),
            "corrections": int(sum(t.get("corrections", {}).get(spec["name"], 0)
                                   for t in tracks)),
            "degenerate_tracks": degenerate,
            "scored_on_fallback_grid": len(degenerate),
        }
        print(f"{spec['name']:<22} b@0.5 {row['boundary_f1']['0.5']:.4f}  "
              f"contested {row['macro_f1_contested']:.4f}  "
              f"flicker@2 {row['flicker_per_audience_minute']['2.0']:.4f}  "
              f"sw/min {row['switches_per_audience_minute']:.4f}  "
              f"undec {row['undecoded_share']:.5f}  "
              f"lines {geometry[spec['name']]['bar_lines']}  "
              f"degen {geometry[spec['name']]['scored_on_fallback_grid']}", flush=True)

    live_row = rows["live_fallback_phase0"]
    live_observed = {"boundary_f1_0.5": live_row["boundary_f1"]["0.5"],
                     "macro_f1_contested": live_row["macro_f1_contested"],
                     "flicker_2.0": live_row["flicker_per_audience_minute"]["2.0"]}
    task1d = json.loads((gate_dir / "task1d_live_grid.json").read_text(encoding="utf-8"))
    published = task1d["grids"]["live_phase0"]
    live_anchor = {"boundary_f1_0.5": published["boundary_f1"]["0.5"],
                   "macro_f1_contested": published["macro_f1_contested"],
                   "flicker_2.0": published["flicker_per_audience_minute"]["2.0"]}
    if any(abs(live_anchor[k] - LIVE_PUBLISHED[k]) > 5e-5 for k in LIVE_PUBLISHED):
        raise SystemExit(f"task1d_live_grid.json does not carry the row 1d "
                         f"published: {live_anchor} vs {LIVE_PUBLISHED}")
    live_delta = {k: round(live_observed[k] - live_anchor[k], 12) for k in live_anchor}
    if not args.limit and not all(abs(v) <= LIVE_TOL for v in live_delta.values()):
        raise SystemExit(f"LIVE ANCHOR FAILED {live_delta} -- the shipping "
                         f"fallback does not reproduce 1d; no comparison may be drawn")
    print("LIVE ANCHOR", json.dumps(live_delta), flush=True)

    table = {k: np.asarray([accumulators(s) for s in v], dtype=np.float64)
             for k, v in arms.items()}
    for key, matrix in table.items():
        exact = metrics(arms[key])
        fast = {m: float(v[0]) for m, v in derive(matrix.sum(axis=0)).items()}
        drift = {m: abs(exact[m] - fast[m]) for m in exact}
        if max(drift.values()) > 1e-9:
            raise SystemExit(f"vectorised metrics disagree on {key}: {drift}")
    print("vectorised bootstrap metrics reproduce aggregate() on every arm", flush=True)

    n = len(tracks)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((DRAWS, n), dtype=np.float64)
    for d in range(DRAWS):
        weights[d] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    draws = {k: derive(weights @ m) for k, m in table.items()}

    for d in range(3):
        idx = np.nonzero(weights[d])[0]
        slow = metrics([arms["live_fallback_phase0"][i] for i in idx
                        for _ in range(int(weights[d, i]))])
        fast = {m: float(v[d]) for m, v in draws["live_fallback_phase0"].items()}
        if max(abs(slow[m] - fast[m]) for m in slow) > 1e-9:
            raise SystemExit(f"bootstrap draw {d} disagrees: {slow} vs {fast}")
    print("bootstrap resampling checked against aggregate() on 3 draws", flush=True)

    point = {k: {m: float(v[0]) for m, v in derive(mtx.sum(axis=0)).items()}
             for k, mtx in table.items()}
    baseline = "live_fallback_phase0"
    winner = max((s for s in specs if s.get("primary")),
                 key=lambda s: point[s["name"]]["crispness_0.5"])
    paired = {}
    for key in table:
        if key == baseline:
            continue
        block = {}
        for metric in ("crispness_0.5", "macro_contested", "flicker_2.0"):
            series = draws[key][metric] - draws[baseline][metric]
            block[metric] = {
                "point": round(point[key][metric] - point[baseline][metric], 6),
                "bootstrap_mean": round(float(series.mean()), 6),
                "ci95": [round(float(np.percentile(series, 2.5)), 6),
                         round(float(np.percentile(series, 97.5)), 6)],
                "p_delta_gt_0": round(float((series > 0).mean()), 4),
            }
        crisp = block["crispness_0.5"]
        macro = block["macro_contested"]
        relative = ((point[key]["flicker_2.0"] - point[baseline]["flicker_2.0"])
                    / point[baseline]["flicker_2.0"])
        checks = {
            "crispness_positive_and_ci_excludes_zero":
                bool(crisp["point"] > 0 and crisp["ci95"][0] > 0),
            "macro_not_regressed_beyond_noise": bool(macro["ci95"][1] >= 0),
            "flicker_relative_rise": round(float(relative), 6),
            "flicker_within_budget":
                bool(relative <= PASS_RULE["flicker_relative_rise_max"]),
        }
        checks["passes"] = bool(checks["crispness_positive_and_ci_excludes_zero"]
                                and checks["macro_not_regressed_beyond_noise"]
                                and checks["flicker_within_budget"])
        block["pass_rule"] = checks
        paired[f"{key}_minus_{baseline}"] = block
        print(f"{key:<22} crisp {crisp['point']:+.4f} CI {crisp['ci95']} "
              f"P(>0)={crisp['p_delta_gt_0']:.3f} | macro {macro['point']:+.4f} "
              f"CI {macro['ci95']} | flicker {relative:+.2%} | "
              f"PASS={checks['passes']}", flush=True)

    per_track = []
    slip_totals = {}
    for spec in specs:
        slip_totals[spec["name"]] = {"tp": 0, "fp": 0, "fn": 0}
    for i, track in enumerate(tracks):
        entry = {"youtube_id": track["youtube_id"], "phase_slips": track["slips"],
                 "n_live_beats": int(track["live"].size),
                 "beat_recall": round(float(track["js"].size / track["annotated"].size), 6),
                 "annotated_phase": track["phase"],
                 "first_matched_live_index": int(track["js"][0]),
                 "first_matched_annotated_index": int(track["isx"][0]),
                 "live_start_minus_annotated_start_sec": round(float(
                     track["live"][0] - track["annotated"][0]), 4),
                 "required_phase_at_start": int(track["required"][0])}
        for key in table:
            entry[f"crisp_{key}"] = round(float(
                aggregate([arms[key][i]]).boundary_prf("class", 0.5)[2]), 6)
        for spec in specs:
            positions = track["positions"][spec["name"]]
            phases = (np.arange(track["live"].size) - positions) % 4
            good = track["known"]
            entry[f"phase_acc_{spec['name']}"] = round(float(
                (phases[good] == track["truth"][good]).mean()) if good.any() else 0.0, 6)
            truth_delta, detected = span_repairs(track["js"], track["required"], phases)
            counts = slip_pr(truth_delta, detected)
            for k in slip_totals[spec["name"]]:
                slip_totals[spec["name"]][k] += counts[k]
            precision, recall, f1 = placement(track["lines"][spec["name"]],
                                              track["downbeats"])
            entry[f"placement_f1_{spec['name']}"] = round(f1, 6)
        entry["delta_vs_fallback"] = round(
            entry[f"crisp_{winner['name']}"] - entry[f"crisp_{baseline}"], 6)
        per_track.append(entry)

    slip_pr_rows = {}
    for spec in specs:
        counts = slip_totals[spec["name"]]
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        slip_pr_rows[spec["name"]] = {
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
            **counts}
        print(f"slip P/R {spec['name']:<22} "
              f"{precision:.3f}/{recall:.3f} tp={counts['tp']} fp={counts['fp']} "
              f"fn={counts['fn']}", flush=True)

    worst = sorted(per_track,
                   key=lambda r: r[f"crisp_{winner['name']}"] - r[f"crisp_{baseline}"])[:5]
    best = sorted(per_track,
                  key=lambda r: -(r[f"crisp_{winner['name']}"] - r[f"crisp_{baseline}"]))[:5]

    ceiling = {
        "annotated_beats_phase0": round(point["annotated_phase0"]["crispness_0.5"], 6),
        "expert_downbeats": round(point["expert"]["crispness_0.5"], 6),
        "gap_remaining_to_annotated": round(
            point["annotated_phase0"]["crispness_0.5"]
            - point[winner["name"]]["crispness_0.5"], 6),
        "share_of_1d_gap_recovered": round(
            (point[winner["name"]]["crispness_0.5"] - point[baseline]["crispness_0.5"])
            / (point["annotated_phase0"]["crispness_0.5"]
               - point[baseline]["crispness_0.5"]), 6),
    }

    cache_digest = hashlib.sha256()
    for track in sorted(tracks, key=lambda t: t["youtube_id"]):
        cache_digest.update((madmom_dir / f"{track['youtube_id']}.npy").read_bytes())

    write_json(args.out_dir / "phase_tracking_gate.json", {
        "task": "phase tracking -- the decoding gate: candidate live bar grids priced",
        "question": ("can a live bar-PHASE tracker win back any of the -0.1396 "
                     "crispness the counted fallback loses to phase slips?"),
        "provenance": {
            "posteriors_dir": str(posteriors_dir),
            "generation_model_sha": model_sha,
            "priors_sha256": hashlib.sha256(
                (model_dir / PRIORS_FILE).read_bytes()).hexdigest(),
            "shipped_config_name": name,
            "shipped_config": dataclasses.asdict(params),
            "shipped_config_sha256": config_sha(params),
            "lag_bars": LAG_BARS,
            "decoder_generation_git_sha": git_sha(PHASE_B),
            "decoder_generation_worktree": str(PHASE_B),
            "madmom_beats_sha256": cache_digest.hexdigest(),
            "probe": str(probe_path),
            "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
            "probe_stage": probe.get("stage"),
            "candidate_families_absent_from_this_gate": missing_families,
            "split": "val", "tracks": len(tracks),
            "seed": SEED, "bootstrap_draws": DRAWS,
        },
        "caveat": (
            "Candidates were both TUNED and SCORED on val -- test is spent (#112) "
            "-- so every margin here is an in-sample reading. The selection was "
            "made on bar-line placement F1 in the no-decode probe, a different "
            "metric from the gate's, which limits but does not remove the "
            "optimism. The posteriors and the madmom cache come from the same "
            "ffmpeg-decoded audio, so this isolates the GRID."),
        "pass_rule": PASS_RULE,
        "anchors": {
            "expert_lag2": {"expected": ANCHOR, "observed": observed, "delta": deltas},
            "annotated_phase0_lag2": {"expected": ANNOTATED_ANCHOR,
                                      "observed": annotated_row["boundary_f1"]["0.5"],
                                      "delta": delta},
            "live_fallback_lag2": {"expected": live_anchor,
                                   "published": LIVE_PUBLISHED,
                                   "observed": live_observed, "delta": live_delta},
            "slip_inventory": {"source": str(gate_dir / "task1c_beat_slip.json"),
                               "total_slips": int(sum(t["slips"] for t in tracks)),
                               "mismatches": mismatched},
        },
        "candidates": specs,
        "grids": rows,
        "grid_geometry": geometry,
        "point_estimates": {k: {m: round(v, 6) for m, v in val.items()}
                            for k, val in point.items()},
        "bootstrap": {"draws": DRAWS, "seed": SEED, "tracks": n,
                      "baseline": baseline,
                      "resample": "tracks with replacement, one multiset for every arm",
                      "paired_deltas": paired},
        "slip_detection": slip_pr_rows,
        "first_beat": {
            "note": ("the shipping fallback calls the first DETECTED beat bar "
                     "position 0; these are the measurements that say what it "
                     "actually is, and whether the live stream simply starts "
                     "late (warmup) rather than the cache having lost a beat"),
            "required_phase_at_start": {
                str(v): sum(1 for r in per_track if r["required_phase_at_start"] == v)
                for v in range(4)},
            "first_matched_annotated_index": {
                str(v): sum(1 for r in per_track
                            if r["first_matched_annotated_index"] == v)
                for v in sorted({r["first_matched_annotated_index"]
                                 for r in per_track})},
            "first_matched_live_index": {
                str(v): sum(1 for r in per_track
                            if r["first_matched_live_index"] == v)
                for v in sorted({r["first_matched_live_index"] for r in per_track})},
            "live_start_minus_annotated_start_sec": stat(
                [r["live_start_minus_annotated_start_sec"] for r in per_track]),
        },
        "ceiling": ceiling,
        "winner": winner["name"],
        "phase_accuracy": {s["name"]: stat([r[f"phase_acc_{s['name']}"]
                                            for r in per_track]) for s in specs},
        "placement_f1": {s["name"]: stat([r[f"placement_f1_{s['name']}"]
                                          for r in per_track]) for s in specs},
        "worst_5": worst, "best_5": best,
        "per_track": per_track,
    })
    print(f"\nwrote {args.out_dir / 'phase_tracking_gate.json'}", flush=True)
    print(json.dumps(ceiling, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
