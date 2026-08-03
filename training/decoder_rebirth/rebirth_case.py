"""The two named regressions, replayed on the beat streams that produced them.

The val sweep decodes on the CACHED madmom arrays, which is what pins it to the
phase-tracking gate's anchor.  Both named failures were measured on the fast
sim's own live beat stream, and the two streams do not agree bar for bar -- so
re-scoring the failures on the cache would answer a different question than the
one the case study asked.  This replays each track's runtime beats and the
runtime posterior trace (the sim's cached MERT cells pushed through the shipped
ONNX graph), builds the grid the runtime built, and decodes it under every arm.

``cXBIZOiSaxA`` is the wrong-state case: a 5 s hole re-anchors the grid at
351.5 s and the reborn committer commits ``intro`` for 53.4 s over the back half
of a breakdown.  ``_cHwb8tOEHM`` is the floor case: its beat-in at 69.3 s was
captured only because madmom's false beats accidentally pre-aged the grid, and
replacing that grid with a synthetic one that opens later exposes a 27 s cliff
plus a residual at the grid's own first bar.  Both tables are re-run here.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASE_B = Path(r"C:\Users\Julian\Projects\soundswitch-phase-b-worktree")
for entry in (str(HERE), str(ROOT / "training" / "raveform"), str(PHASE_B)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from training.nn.decoder import DecodeParams, FixedLagViterbi, temper  # noqa: E402
from training.nn.evaluate_v1 import write_json  # noqa: E402
from training.nn.priors import MODELS_DIR, PRIORS_FILE, Priors  # noqa: E402

import rebirth as RB  # noqa: E402

LAG_BARS = 2

CASES = {
    "cXBIZOiSaxA": {"report": "case_cXBIZOiSaxA.json",
                    "posteriors": "runtime_posteriors_cXBIZ.npz",
                    "bar_sec": None},
    "_cHwb8tOEHM": {"report": "case_cHwb8tOEHM.json",
                    "posteriors": "runtime_posteriors.npz",
                    "bar_sec": 1.7291},
}

CLIFF_GRID_STARTS = (40.0, 55.0, 60.0, 64.0, 66.5, 68.93)
BEAT_IN_TOLERANCE_SEC = 0.5


def pick_params(frontier_path: Path) -> tuple:
    document = json.loads(Path(frontier_path).read_text(encoding="utf-8"))
    name = document["pick"]
    for row in document["frontier"]:
        if row["name"] == name:
            return name, dataclasses.replace(DecodeParams(**row["params"]),
                                             lag_bars=LAG_BARS)
    raise RuntimeError(f"{frontier_path} names pick {name!r} but has no such row")


def observe(trace: dict, edges: np.ndarray, params: DecodeParams) -> tuple:
    """The runtime's bar aggregation: mean posterior over the bar, peak boundary at its line."""
    times, posterior, boundary = trace["t"], trace["posterior"], trace["boundary"]
    n_bars = len(edges) - 1
    rows = np.full((n_bars, posterior.shape[1]), np.nan, dtype=np.float64)
    scores = np.full(n_bars, np.nan, dtype=np.float64)
    tolerance = params.boundary_tolerance_sec
    for bar in range(n_bars):
        lo, hi = float(edges[bar]), float(edges[bar + 1])
        inside = (times >= lo) & (times < hi)
        if inside.any():
            rows[bar] = temper(posterior[inside], params.temperature).mean(axis=0)
        near = (times >= lo - tolerance) & (times <= lo + tolerance)
        if near.any():
            scores[bar] = boundary[near].max()
    return rows, scores


def timeline(labels, edges) -> list:
    return [{"bar": start, "label": label,
             "start_sec": round(float(edges[start]), 3),
             "end_sec": round(float(edges[end]), 3),
             "duration_sec": round(float(edges[end] - edges[start]), 3)}
            for start, end, label in RB.runs(labels) if label]


def first_change(labels, edges) -> dict | None:
    """The case study's metric: when does the show stop saying ``intro``.

    Stated absolutely rather than as "the first change", because an arm that
    never says ``intro`` at all has no change to report and would otherwise be
    scored as the failure it is the fix for.
    """
    spans = [s for s in RB.runs(labels) if s[2]]
    if not spans:
        return None
    for start, end, label in spans:
        if label != "intro":
            return {"opening": spans[0][2], "to": label,
                    "at_sec": round(float(edges[start]), 3),
                    "opened_non_intro": spans[0][2] != "intro"}
    return None


def build(priors: Priors, params: DecodeParams) -> FixedLagViterbi:
    return FixedLagViterbi(
        priors, params.lag_bars, class_prior_division=params.class_prior_division,
        drop_miss_cost=params.drop_miss_cost, prior_strength=params.prior_strength,
        boundary_weight=params.boundary_weight, boundary_ref=params.boundary_ref,
        floor_scale=params.floor_scale, floor_bars=params.floor_bars,
        outro_escape=params.outro_escape)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(
        r"C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    if args.out_dir is None:
        args.out_dir = data_dir / MODELS_DIR / "phase_b" / "decoder_rebirth"
    case_dir = args.out_dir / "case_inputs"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    name, params = pick_params(data_dir / MODELS_DIR / "phase_b"
                               / "student_kd_t2_w05_s1234" / "frontier.json")
    priors = Priors.load(data_dir / MODELS_DIR
                         / "phase_b_student_kd_t2_w05_s1234" / PRIORS_FILE)
    decoder = build(priors, params)
    print(f"pick {name}  lag_bars={params.lag_bars}  classes {priors.classes}",
          flush=True)

    segments = {row["id"]: row for row in
                json.loads((data_dir / "annotations" / "segments.json")
                           .read_text(encoding="utf-8"))}

    document = {"task": "the rebirth principle -- the two named regressions",
                "provenance": {"shipped_config_name": name,
                               "shipped_config": dataclasses.asdict(params),
                               "case_inputs": str(case_dir),
                               "beat_source": "the fast sim's own live stream, "
                                              "read out of the case-study report"},
                "cases": {}}

    for youtube_id, spec in CASES.items():
        report = json.loads((case_dir / spec["report"]).read_text(encoding="utf-8"))
        beats = np.asarray([b["t"] for b in report["beats"]], dtype=np.float64)
        with np.load(case_dir / spec["posteriors"]) as archive:
            trace = {"t": np.asarray(archive["t"], dtype=np.float64),
                     "posterior": np.asarray(archive["posterior"], dtype=np.float64),
                     "boundary": np.asarray(archive["boundary"], dtype=np.float64)}
        sections = [{"name": s["name"], "start": s["start"], "end": s["end"]}
                    for s in segments[youtube_id]["sections"]]

        grid = RB.live_grid(beats)
        rows, scores = observe(trace, grid.edges, params)
        print(f"\n### {youtube_id}  {len(beats)} beats, {grid.edges.size - 1} bars, "
              f"re-anchors at {[round(float(grid.edges[b]), 2) for b in grid.re_anchors]}",
              flush=True)

        live = {}
        for arm, policy in RB.ARMS.items():
            labels, births = RB.decode_live(grid, rows, scores, decoder, policy)
            spans = timeline(labels, grid.edges)
            after = []
            for birth in births[1:]:
                run = next((s for s in spans if s["bar"] >= birth["bar"]), None)
                after.append({"re_anchor_sec": round(float(grid.edges[birth["bar"]]), 3),
                              "carried": birth["carried"],
                              "committed": run["label"] if run else None,
                              "committed_at_sec": run["start_sec"] if run else None,
                              "duration_sec": run["duration_sec"] if run else None})
            live[arm] = {"re_anchor_commits": after, "timeline": spans,
                         "first_change": first_change(labels, grid.edges)}
            summary = "; ".join(f"{a['re_anchor_sec']}s -> {a['committed']} "
                                f"{a['duration_sec']}s" for a in after) or "no re-anchor"
            print(f"  {arm:<16} first change {live[arm]['first_change']} | {summary}",
                  flush=True)

        case = {"beats": int(beats.size), "bars": int(grid.edges.size - 1),
                "re_anchor_sec": [round(float(grid.edges[b]), 3)
                                  for b in grid.re_anchors],
                "sections": sections, "live_grid": live}

        if spec["bar_sec"]:
            truth = float(sections[0]["end"])
            bar_sec = float(spec["bar_sec"])
            cliff = {}
            for start in CLIFF_GRID_STARTS:
                edges = np.arange(start, float(trace["t"][-1]), bar_sec)
                synthetic = RB.LiveGrid(edges, (), ())
                rows_s, scores_s = observe(trace, edges, params)
                row = {}
                for arm, policy in RB.ARMS.items():
                    labels, _ = RB.decode_live(synthetic, rows_s, scores_s,
                                               decoder, policy)
                    change = first_change(labels, edges)
                    row[arm] = change
                cliff[f"{start:.2f}"] = row
                shipped = row["shipped"]
                full = row["full_R1_R2_R3"]
                print(f"  grid opens {start:6.2f}s  shipped "
                      f"{shipped['at_sec'] if shipped else 'never':>8}  full "
                      f"{full['at_sec'] if full else 'never':>8}  "
                      f"(truth {truth:.2f})", flush=True)
            case["beat_in_truth_sec"] = round(float(truth), 3)
            case["beat_in_tolerance_sec"] = BEAT_IN_TOLERANCE_SEC
            case["beat_in_captured"] = {
                arm: (None if live[arm]["first_change"] is None else
                      round(abs(live[arm]["first_change"]["at_sec"] - truth), 3))
                for arm in RB.ARMS}
            case["synthetic_cliff"] = cliff
        document["cases"][youtube_id] = case

    write_json(args.out_dir / "rebirth_cases.json", document)
    print(f"\nwrote {args.out_dir / 'rebirth_cases.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
