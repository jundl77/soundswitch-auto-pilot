"""Does the rebirth principle repair the wrong state a beat gap teleports the show into?

The case study measured the failure on ``cXBIZOiSaxA``: a 5.01 s hole in the
beat stream re-anchors the bar grid, the re-anchor restarts the committer, and a
restarted committer does not resume where the music is -- it restarts where a
*track* starts.  The decoder committed ``intro`` for 53.4 s against a model that
was 0.978 confident of ``breakdown``, because mid-track ``intro`` is unreachable
by any fitted transition and therefore reachable ONLY by reset.

This gate prices the three coupled changes (see ``rebirth``) on all 215 val
tracks decoded on their LIVE grids -- the cached madmom beat streams, anchored
the way the runtime anchors them, re-anchored the way the runtime re-anchors
them.  The offline phase-tracking gate did not model the re-anchor at all, so
its ``D_anchor`` row is reproduced here as the harness anchor and then the
re-anchor is switched on; the difference between those two rows is the entire
population this package can move.

Pre-registered: the named regressions must be fixed, crispness@0.5 and contested
macro must each not be WORSE beyond noise, flicker must not rise more than 5 %
relative, and mid-track ``intro`` commits must be eliminated or individually
justified.  This is wrong-state repair -- an aggregate win is not the bar.

Val only (#112), lag_bars = 2 (#154), zero GPU.
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
ROOT = HERE.parents[1]
PHASE_B = Path(r"C:\Users\Julian\Projects\soundswitch-phase-b-worktree")
for entry in (str(HERE), str(ROOT / "training" / "raveform"), str(PHASE_B)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from training.nn.decoder import DecodeParams, bar_observations  # noqa: E402
from training.nn.ceiling.decoder_frontier import (  # noqa: E402
    false_drop_entries, score_stream)
from training.nn.evaluate_v1 import (  # noqa: E402
    DEFAULT_SPACE, UNDECODED, aggregate, beat_classes, build_decoder, decode_beats,
    generation_model_sha, load_inputs, restricted_macro_f1, score_predicted,
    split_ids, write_json)
from training.nn.paired import CONTESTED  # noqa: E402
from training.nn.priors import MODELS_DIR, PRIORS_FILE, Priors  # noqa: E402

import rebirth as RB  # noqa: E402

LAG_BARS = 2
DRAWS = 2000
SEED = 20260804

D_ANCHOR = {"crispness_0.5": 0.547596, "macro_contested": 0.705057,
            "flicker_2.0": 0.311473}
D_ANCHOR_BAR_LINES = 46134
ANCHOR_TOL = 1e-6

PASS_RULE = {
    "crispness_0.5_may_not_regress_beyond_noise": True,
    "macro_contested_may_not_regress_beyond_noise": True,
    "flicker_relative_rise_max": 0.05,
    "mid_track_intro_commits": "eliminated, or each survivor individually justified",
    "named_regressions": ["cXBIZOiSaxA re-anchor must not commit intro",
                          "_cHwb8tOEHM beat-in captured, cliff and bar-0 residual closed"],
}

REGRESSION_TRACK = "cXBIZOiSaxA"

METRICS = ("crispness_0.5", "crispness_2.0", "macro_contested", "macro_all",
           "flicker_2.0", "switches_per_min", "false_drop_per_min")


def git_sha(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def config_sha(params: DecodeParams) -> str:
    payload = json.dumps(dataclasses.asdict(params), sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pick_params(frontier_path: Path):
    document = json.loads(Path(frontier_path).read_text(encoding="utf-8"))
    name = document["pick"]
    for row in document["frontier"]:
        if row["name"] == name:
            return name, DecodeParams(**row["params"])
    raise RuntimeError(f"{frontier_path} names pick {name!r} but has no such row")


def accumulators(score, labels, exposure_sec: float, false_drops: int) -> list:
    """The additive cells every bootstrap metric is a function of column sums of."""
    row: list = []
    for tolerance in (0.5, 2.0):
        cell = score.boundary["class"][tolerance]["overall"]
        row.extend([cell["matched"], cell["n_pred"], cell["n_truth"]])
    for label in labels:
        row.extend(score.counts[label])
    row.append(score.flicker["class"][2.0])
    row.append(score.weight_total_sec)
    row.append(exposure_sec)
    row.append(false_drops)
    return row


def _f1(tp, fp, fn):
    precision = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1), 0.0)
    recall = np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1), 0.0)
    denominator = precision + recall
    return np.where(denominator > 0, 2 * precision * recall,
                    0.0) / np.where(denominator > 0, denominator, 1)


def derive(sums: np.ndarray, labels) -> dict:
    sums = np.atleast_2d(sums)
    out = {}
    for index, tolerance in enumerate((0.5, 2.0)):
        matched, n_pred, n_truth = (sums[:, 3 * index], sums[:, 3 * index + 1],
                                    sums[:, 3 * index + 2])
        out[f"crispness_{tolerance}"] = _f1(matched, n_pred - matched, n_truth - matched)
    per_label = {}
    for k, label in enumerate(labels):
        base = 6 + 3 * k
        per_label[label] = _f1(sums[:, base], sums[:, base + 1], sums[:, base + 2])
    out["macro_contested"] = sum(per_label[c] for c in CONTESTED) / len(CONTESTED)
    out["macro_all"] = sum(per_label.values()) / len(labels)
    audience = sums[:, -3] / 60.0
    exposure = sums[:, -2] / 60.0
    out["flicker_2.0"] = np.where(audience > 0, sums[:, -4]
                                  / np.where(audience > 0, audience, 1), 0.0)
    out["switches_per_min"] = np.where(exposure > 0, sums[:, 4]
                                       / np.where(exposure > 0, exposure, 1), 0.0)
    out["false_drop_per_min"] = np.where(exposure > 0, sums[:, -1]
                                         / np.where(exposure > 0, exposure, 1), 0.0)
    return out


def segments_sec(labels, edges) -> list:
    return [{"bar": start, "bars": end - start, "label": label,
             "start_sec": round(float(edges[start]), 3),
             "end_sec": round(float(edges[end]), 3),
             "duration_sec": round(float(edges[end] - edges[start]), 3)}
            for start, end, label in RB.runs(labels)]


def truth_at(times, truth_labels, lo: float, hi: float) -> dict:
    times = np.asarray(times, dtype=np.float64)
    inside = (times >= lo) & (times < hi)
    if not inside.any():
        return {}
    values, counts = np.unique(np.asarray(truth_labels, dtype=object)[inside],
                               return_counts=True)
    return {str(v): int(c) for v, c in sorted(zip(values, counts), key=lambda p: -p[1])}


def birth_audit(labels, births, grid, item) -> list:
    """What each rebirth committed, how long it held, and what was actually playing."""
    spans = [s for s in RB.runs(labels) if s[2]]
    truth = item.labels[DEFAULT_SPACE]
    audit = []
    for birth in births:
        bar = birth["bar"]
        run = next(((start, end, label) for start, end, label in spans if end > bar),
                   None)
        if run is None:
            continue
        start, end, label = run
        lo, hi = float(grid.edges[max(start, bar)]), float(grid.edges[end])
        agreement = truth_at(item.times, truth, lo, hi)
        held = sum(agreement.values())
        audit.append({
            "bar": bar, "at_sec": round(lo, 3), "carried": birth["carried"],
            "committed": label, "undecoded_bars": max(0, start - bar),
            "bars_held": end - max(start, bar), "duration_sec": round(hi - lo, 3),
            "truth_beats": agreement,
            "share_correct": round(agreement.get(label, 0) / held, 4) if held else None,
        })
    return audit


def mid_track_intro(labels, edges) -> list:
    """``intro`` runs committed after a non-intro one -- reachable only by a reset."""
    seen_other = False
    found = []
    for start, end, label in RB.runs(labels):
        if label == UNDECODED:
            continue
        if label == "intro" and seen_other:
            found.append({"bar": start, "bars": end - start,
                          "start_sec": round(float(edges[start]), 3),
                          "duration_sec": round(float(edges[end] - edges[start]), 3)})
        elif label != "intro":
            seen_other = True
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(
        r"C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    posteriors_dir = data_dir / "posteriors_phase_b" / "student_kd_t2_w05_s1234"
    frontier_path = (data_dir / MODELS_DIR / "phase_b" / "student_kd_t2_w05_s1234"
                     / "frontier.json")
    madmom_dir = data_dir / "madmom_beats"
    if args.out_dir is None:
        args.out_dir = data_dir / MODELS_DIR / "phase_b" / "decoder_rebirth"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if UNDECODED != RB.UNDECODED:
        raise SystemExit(f"sentinel drift: evaluator {UNDECODED!r} vs "
                         f"rebirth {RB.UNDECODED!r}")

    name, base = pick_params(frontier_path)
    params = dataclasses.replace(base, lag_bars=LAG_BARS)
    print(f"pick {name}  lag_bars={params.lag_bars}  sha={config_sha(params)[:12]}",
          flush=True)

    model_dir = data_dir / MODELS_DIR / "phase_b_student_kd_t2_w05_s1234"
    model_sha = generation_model_sha(model_dir)
    priors = Priors.load(model_dir / PRIORS_FILE)
    ids = split_ids(data_dir, "val")
    inputs, skipped = load_inputs(
        data_dir, ids, min_coverage=params.min_coverage,
        boundary_tolerance_sec=params.boundary_tolerance_sec,
        temperature=params.temperature, posteriors_dir=posteriors_dir,
        model_sha=model_sha)
    if skipped:
        raise SystemExit(f"{len(skipped)} tracks unusable: {skipped[:5]}")
    print(f"loaded {len(inputs)} val tracks", flush=True)

    decoder = build_decoder(priors, params)
    labels_space = list(aggregate([score_predicted(
        inputs[0].as_track_beats(), DEFAULT_SPACE,
        tuple(UNDECODED for _ in inputs[0].times))]).labels)

    beats = {}
    for item in inputs:
        stream = np.asarray(np.load(madmom_dir / f"{item.youtube_id}.npy"),
                            dtype=np.float64).reshape(-1)
        stream.sort()
        beats[item.youtube_id] = stream

    def regrid(item, grid):
        posteriors, boundary = bar_observations(
            posteriors_dir / f"{item.youtube_id}.npz", grid.edges,
            min_coverage=params.min_coverage,
            boundary_tolerance_sec=params.boundary_tolerance_sec,
            temperature=params.temperature)
        return dataclasses.replace(item, edges=grid.edges, posteriors=posteriors,
                                   boundary=boundary), posteriors, boundary

    frozen = []
    for item in inputs:
        grid = RB.live_grid(beats[item.youtube_id], gap_sec=float("inf"))
        regridded, posteriors, boundary = regrid(item, grid)
        decoded, _ = RB.decode_live(grid, posteriors, boundary, decoder, RB.SHIPPED)
        stream = beat_classes(regridded.times, regridded.edges, decoded)
        if stream != decode_beats(regridded, decoder):
            raise SystemExit(f"{item.youtube_id}: the rebirth committer's SHIPPED "
                             f"policy is not the shipped decode path")
        frozen.append((regridded, stream, grid))
    anchor_row = score_stream([f[0] for f in frozen], [f[1] for f in frozen])
    observed = {"crispness_0.5": anchor_row["boundary_f1"]["0.5"],
                "macro_contested": anchor_row["macro_f1_contested"],
                "flicker_2.0": anchor_row["flicker_per_audience_minute"]["2.0"]}
    deltas = {k: round(observed[k] - D_ANCHOR[k], 9) for k in D_ANCHOR}
    lines = sum(int(f[2].edges.size - 1) for f in frozen)
    if not all(abs(v) <= ANCHOR_TOL for v in deltas.values()):
        raise SystemExit(f"D_ANCHOR FAILED {deltas} -- no conclusion may be drawn")
    if lines != D_ANCHOR_BAR_LINES:
        raise SystemExit(f"D_ANCHOR grid geometry FAILED: {lines} bars against "
                         f"{D_ANCHOR_BAR_LINES}")
    print(f"D_ANCHOR reproduced {json.dumps(deltas)}  bars {lines}", flush=True)

    tracks = []
    for item in inputs:
        stream = beats[item.youtube_id]
        grid = RB.live_grid(stream)
        regridded, posteriors, boundary = regrid(item, grid)
        gaps = np.diff(stream)
        tracks.append({
            "youtube_id": item.youtube_id, "item": regridded, "grid": grid,
            "posteriors": posteriors, "boundary": boundary,
            "re_anchor_sec": [round(float(grid.edges[bar]), 3)
                              for bar in grid.re_anchors],
            "max_gap_sec": round(float(gaps.max()) if gaps.size else 0.0, 3),
        })
    gapped = [t for t in tracks if t["grid"].re_anchors]
    print(f"live grids: {len(gapped)} of {len(tracks)} tracks re-anchor, "
          f"{sum(len(t['grid'].re_anchors) for t in gapped)} re-anchors total",
          flush=True)

    arms = {}
    for arm, policy in RB.ARMS.items():
        per_track = []
        for track in tracks:
            decoded, births = RB.decode_live(
                track["grid"], track["posteriors"], track["boundary"], decoder, policy)
            item = track["item"]
            stream = beat_classes(item.times, item.edges, decoded)
            score = score_predicted(item.as_track_beats(), DEFAULT_SPACE, stream)
            drops = false_drop_entries(stream, item.times,
                                       item.as_track_beats().labels[DEFAULT_SPACE])
            exposure = float(np.diff(np.asarray(item.times, dtype=np.float64)).sum())
            per_track.append({
                "track": track, "decoded": decoded, "stream": stream, "score": score,
                "births": births, "exposure": exposure,
                "false_drops": drops["false_drop_entries"],
            })
        arms[arm] = per_track
        row = score_stream([e["track"]["item"] for e in per_track],
                           [e["stream"] for e in per_track])
        print(f"{arm:<16} b@0.5 {row['boundary_f1']['0.5']:.6f}  "
              f"b@2.0 {row['boundary_f1']['2.0']:.6f}  "
              f"contested {row['macro_f1_contested']:.6f}  "
              f"all {row['macro_f1_all']:.6f}  "
              f"flick {row['flicker_per_audience_minute']['2.0']:.6f}  "
              f"sw/min {row['switches_per_audience_minute']:.4f}  "
              f"fdrop/min {row['false_drop_entries_per_audience_minute']:.4f}  "
              f"undec {row['undecoded_share']:.5f}", flush=True)
        arms[arm] = {"entries": per_track, "row": row}

    table = {arm: np.asarray([accumulators(e["score"], labels_space, e["exposure"],
                                           e["false_drops"])
                              for e in payload["entries"]], dtype=np.float64)
             for arm, payload in arms.items()}
    for arm, matrix in table.items():
        total = aggregate([e["score"] for e in arms[arm]["entries"]])
        exposure = sum(e["exposure"] for e in arms[arm]["entries"]) / 60.0
        exact = {
            "crispness_0.5": float(total.boundary_prf("class", 0.5)[2]),
            "crispness_2.0": float(total.boundary_prf("class", 2.0)[2]),
            "macro_contested": float(restricted_macro_f1(total, CONTESTED)),
            "macro_all": float(total.macro_f1),
            "flicker_2.0": float(total.flicker_per_minute["class"][2.0]),
            "switches_per_min": float(
                total.boundary["class"][2.0]["overall"]["n_pred"]) / exposure,
            "false_drop_per_min": sum(e["false_drops"] for e in
                                      arms[arm]["entries"]) / exposure,
        }
        fast = {m: float(v[0]) for m, v in derive(matrix.sum(axis=0), labels_space).items()}
        drift = {m: abs(exact[m] - fast[m]) for m in exact}
        if max(drift.values()) > 1e-9:
            raise SystemExit(f"vectorised metrics disagree on {arm}: {drift}")
    print("vectorised metrics reproduce aggregate() on every arm", flush=True)

    n = len(tracks)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((DRAWS, n), dtype=np.float64)
    for draw in range(DRAWS):
        weights[draw] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    draws = {arm: derive(weights @ matrix, labels_space)
             for arm, matrix in table.items()}
    point = {arm: {m: float(v[0]) for m, v in
                   derive(matrix.sum(axis=0), labels_space).items()}
             for arm, matrix in table.items()}

    baseline = "shipped"
    paired = {}
    for arm in table:
        if arm == baseline:
            continue
        block = {}
        for metric in METRICS:
            series = draws[arm][metric] - draws[baseline][metric]
            block[metric] = {
                "point": round(point[arm][metric] - point[baseline][metric], 6),
                "ci95": [round(float(np.percentile(series, 2.5)), 6),
                         round(float(np.percentile(series, 97.5)), 6)],
                "p_delta_gt_0": round(float((series > 0).mean()), 4),
            }
        relative = ((point[arm]["flicker_2.0"] - point[baseline]["flicker_2.0"])
                    / point[baseline]["flicker_2.0"])
        checks = {
            "crispness_not_worse_beyond_noise":
                bool(block["crispness_0.5"]["ci95"][1] >= 0),
            "macro_contested_not_worse_beyond_noise":
                bool(block["macro_contested"]["ci95"][1] >= 0),
            "flicker_relative_rise": round(float(relative), 6),
            "flicker_within_budget":
                bool(relative <= PASS_RULE["flicker_relative_rise_max"]),
        }
        checks["passes"] = bool(checks["crispness_not_worse_beyond_noise"]
                                and checks["macro_contested_not_worse_beyond_noise"]
                                and checks["flicker_within_budget"])
        block["pass_rule"] = checks
        paired[f"{arm}_minus_{baseline}"] = block
        crisp, macro = block["crispness_0.5"], block["macro_contested"]
        print(f"{arm:<16} crisp {crisp['point']:+.5f} CI {crisp['ci95']} | "
              f"contested {macro['point']:+.5f} CI {macro['ci95']} | "
              f"flicker {relative:+.2%} | PASS={checks['passes']}", flush=True)

    audit = {}
    for arm, payload in arms.items():
        intro_runs, intro_sec, per_track_intro = 0, 0.0, []
        births_after_gap = []
        for entry in payload["entries"]:
            track = entry["track"]
            found = mid_track_intro(entry["decoded"], track["grid"].edges)
            if found:
                intro_runs += len(found)
                intro_sec += sum(f["duration_sec"] for f in found)
                per_track_intro.append({"youtube_id": track["youtube_id"],
                                        "runs": found})
            if track["grid"].re_anchors:
                for record in birth_audit(entry["decoded"], entry["births"][1:],
                                          track["grid"], track["item"]):
                    births_after_gap.append({"youtube_id": track["youtube_id"],
                                             **record})
        held = [b["duration_sec"] for b in births_after_gap]
        correct = [b["share_correct"] for b in births_after_gap
                   if b["share_correct"] is not None]
        audit[arm] = {
            "mid_track_intro_runs": intro_runs,
            "mid_track_intro_sec": round(intro_sec, 2),
            "mid_track_intro_tracks": len(per_track_intro),
            "per_track": per_track_intro,
            "post_gap_births": len(births_after_gap),
            "post_gap_first_run_sec": {
                "median": round(float(np.median(held)), 3) if held else None,
                "mean": round(float(np.mean(held)), 3) if held else None,
                "max": round(float(np.max(held)), 3) if held else None,
                "total": round(float(np.sum(held)), 2) if held else 0.0},
            "post_gap_first_run_share_correct": {
                "median": round(float(np.median(correct)), 4) if correct else None,
                "mean": round(float(np.mean(correct)), 4) if correct else None,
                "n": len(correct)},
            "post_gap_births_detail": births_after_gap,
        }
        print(f"{arm:<16} mid-track intro: {intro_runs} runs / "
              f"{intro_sec:.1f} s over {len(per_track_intro)} tracks | "
              f"post-gap first run median {audit[arm]['post_gap_first_run_sec']['median']} s "
              f"correct {audit[arm]['post_gap_first_run_share_correct']['median']}",
              flush=True)

    stress = {}
    for arm in ("R1_carry", "R1_R2", "full_R1_R2_R3"):
        policy = RB.ARMS[arm]
        held = []
        for entry in arms[arm]["entries"]:
            track = entry["track"]
            if not track["grid"].re_anchors:
                continue
            for wrong in range(len(decoder.classes)):
                decoded, births = RB.decode_live(
                    track["grid"], track["posteriors"], track["boundary"],
                    decoder, policy, carry_override=wrong)
                for record in birth_audit(decoded, births[1:], track["grid"],
                                          track["item"]):
                    if record["carried"] != decoder.classes[wrong]:
                        continue
                    held.append({"youtube_id": track["youtube_id"],
                                 "bar": record["bar"],
                                 "forced": decoder.classes[wrong],
                                 "committed": record["committed"],
                                 "bars_held": record["bars_held"]
                                 if record["committed"] == decoder.classes[wrong] else 0,
                                 "duration_sec": record["duration_sec"]
                                 if record["committed"] == decoder.classes[wrong] else 0.0})
        survived = [h["bars_held"] for h in held]
        stress[arm] = {
            "forced_births": len(held),
            "accepted_the_forced_class": sum(1 for h in held if h["bars_held"] > 0),
            "bars_held": {"median": float(np.median(survived)) if survived else None,
                          "mean": round(float(np.mean(survived)), 3) if survived else None,
                          "max": int(np.max(survived)) if survived else None},
            "detail": held,
        }
        print(f"{arm:<16} forced-wrong-carry: {stress[arm]['accepted_the_forced_class']}"
              f"/{len(held)} accepted, bars held median "
              f"{stress[arm]['bars_held']['median']} max "
              f"{stress[arm]['bars_held']['max']}", flush=True)

    regression = {}
    for arm, payload in arms.items():
        entry = next(e for e in payload["entries"]
                     if e["track"]["youtube_id"] == REGRESSION_TRACK)
        track = entry["track"]
        regression[arm] = {
            "re_anchor_sec": track["re_anchor_sec"],
            "births": birth_audit(entry["decoded"], entry["births"][1:],
                                  track["grid"], track["item"]),
            "timeline": [s for s in segments_sec(entry["decoded"], track["grid"].edges)
                         if s["label"]],
            "crispness_0.5": round(float(aggregate([entry["score"]])
                                         .boundary_prf("class", 0.5)[2]), 6),
        }

    per_track_rows = []
    for index, track in enumerate(tracks):
        row = {"youtube_id": track["youtube_id"],
               "re_anchors": len(track["grid"].re_anchors),
               "re_anchor_sec": track["re_anchor_sec"],
               "max_beat_gap_sec": track["max_gap_sec"]}
        for arm in arms:
            score = arms[arm]["entries"][index]["score"]
            row[f"crisp_{arm}"] = round(float(
                aggregate([score]).boundary_prf("class", 0.5)[2]), 6)
        per_track_rows.append(row)

    digest = hashlib.sha256()
    for track in sorted(tracks, key=lambda t: t["youtube_id"]):
        digest.update((madmom_dir / f"{track['youtube_id']}.npy").read_bytes())

    write_json(args.out_dir / "rebirth_gate.json", {
        "task": "the rebirth principle -- a committer born mid-track priced on val",
        "question": ("does carrying the belief, pre-aging the duration state and "
                     "pre-rolling one bar repair the wrong state a beat gap "
                     "teleports the show into, without costing the show elsewhere?"),
        "provenance": {
            "posteriors_dir": str(posteriors_dir),
            "generation_model_sha": model_sha,
            "priors_sha256": hashlib.sha256(
                (model_dir / PRIORS_FILE).read_bytes()).hexdigest(),
            "shipped_config_name": name,
            "shipped_config": dataclasses.asdict(params),
            "shipped_config_sha256": config_sha(params),
            "lag_bars": LAG_BARS,
            "grid": {"anchor": RB.FIRST_BEAT_BAR_POSITION,
                     "re_anchor_position": RB.RE_ANCHOR_BAR_POSITION,
                     "beat_gap_sec": RB.BEAT_GAP_SEC},
            "madmom_beats_sha256": digest.hexdigest(),
            "rebirth_git_sha": git_sha(ROOT),
            "decoder_generation_git_sha": git_sha(PHASE_B),
            "split": "val", "tracks": len(tracks),
            "seed": SEED, "bootstrap_draws": DRAWS,
        },
        "caveat": (
            "Everything is tuned and scored on val -- test is spent (#112). The "
            "baseline is the live grid WITH the runtime's re-anchor, which the "
            "phase-tracking gate did not model; the D_anchor row reproduced below "
            "is the same grid with the re-anchor switched off, and the gap "
            "between them is the whole population this package can move. "
            "macro_all is bootstrapped over the space's full label set held "
            "fixed, not over Score.macro_classes, so a resample cannot silently "
            "change the denominator."),
        "pass_rule": PASS_RULE,
        "anchors": {
            "D_anchor_no_re_anchor": {"expected": D_ANCHOR, "observed": observed,
                                      "delta": deltas, "bar_lines": lines,
                                      "expected_bar_lines": D_ANCHOR_BAR_LINES},
            "shipped_policy_is_the_shipped_decode_path": True,
        },
        "arms": {arm: RB.ARMS[arm]._asdict() for arm in arms},
        "grids": {arm: payload["row"] for arm, payload in arms.items()},
        "gap_population": {
            "tracks_with_re_anchor": len(gapped),
            "re_anchors": sum(len(t["grid"].re_anchors) for t in gapped),
            "tracks": [{"youtube_id": t["youtube_id"],
                        "re_anchor_sec": t["re_anchor_sec"],
                        "max_beat_gap_sec": t["max_gap_sec"]} for t in gapped],
        },
        "point_estimates": {arm: {m: round(v, 6) for m, v in value.items()}
                            for arm, value in point.items()},
        "bootstrap": {"draws": DRAWS, "seed": SEED, "tracks": n,
                      "baseline": baseline,
                      "resample": "tracks with replacement, one multiset for every arm",
                      "paired_deltas": paired},
        "verdict": {
            "arms_passing_the_pre_registered_rule": sorted(
                arm for arm in paired if paired[arm]["pass_rule"]["passes"]),
            "arms_failing": sorted(
                arm for arm in paired if not paired[arm]["pass_rule"]["passes"]),
            "mid_track_intro_sec": {arm: audit[arm]["mid_track_intro_sec"]
                                    for arm in audit},
            "named_regressions": "see rebirth_cases.json beside this file",
            "recommendation": (
                "ship all three or none. R1 alone fixes the wrong state and "
                "leaves a wrong carry unbounded; R3 alone is a regression -- the "
                "virtual bar inherits the start-of-track prior and intro has no "
                "fitted way out below its floor, so a rebirth locks into it. Only "
                "the full arm eliminates mid-track intro, rejects a wrong carry "
                "before it commits, and lowers flicker."),
        },
        "edge_audit": audit,
        "carry_stress": {
            "question": ("if the carried belief is WRONG, how long does it survive "
                         "contrary evidence?  Every re-anchor is forced to carry "
                         "each class in turn; a birth is counted only where the "
                         "forced class is the one the policy would have carried."),
            "arms": stress,
        },
        "named_regression_cXBIZOiSaxA": regression,
        "per_track": per_track_rows,
    })
    print(f"\nwrote {args.out_dir / 'rebirth_gate.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
