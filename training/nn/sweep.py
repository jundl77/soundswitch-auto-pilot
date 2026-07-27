#!/usr/bin/env python
"""Decoder parameter sweep on the VAL split, against cached posteriors.

The acoustic model is frozen by the time this runs.  What is still free is the
decoder's policy -- how much prior mass to hand back, what a missed drop is
worth, how far to trust a ranking-score boundary head, how long a run must be,
and how late to commit -- and those five knobs interact, so this searches them
rather than arguing about them.

**Why it is cheap.**  ``bar_observations`` depends only on ``min_coverage`` and
``boundary_tolerance_sec``; everything else the sweep varies lives inside the
trellis.  So the sidecars are read once into ``TrackInputs`` and each config is
a pure-numpy Viterbi over ~48 states per bar plus the scoring pass.  Configs are
grouped by the observation-affecting pair rather than assumed to share it, so
the cache cannot go stale if a future sweep does move them.

**Why it is joint where it matters.**  ``prior_strength`` and
``drop_miss_cost`` both push probability mass toward the rare, expensive classes
-- a line search over one at the other's default finds a compromise that neither
axis would pick and calls it optimal.  Same for ``boundary_weight`` and
``boundary_ref``: the gain is meaningless without the neutral point it is
measured from, since the bonus is ``weight * (score - ref)``.  Those pairs are
full grids.  The remaining axes are swept in stages around the running winner
and then a joint refinement grid re-opens all of them together, which is what
catches an interaction a staged search would have walked past.

**Why the sensitivity numbers come last and separately.**  A best-per-value
curve read off a staged search conflates the axis with whatever the rest of the
config happened to be in the stage that produced it, so it cannot answer "what
did this knob cost".  The final pass ablates each axis one at a time around the
chosen config, reusing already-measured configs rather than re-running them, and
those curves are what the report quotes.  The ablation is also a search: it can
find a config the staged pass missed, and the selection re-runs over everything
including it (``anchor_survived_ablation`` records whether it did).

**Why there is no RNG.**  Every axis is an explicit tuple and the enumeration is
a fixed-order cartesian product, so the sweep is a pure function of the cached
posteriors.  Re-running it produces the same winner, and a config's number can
be reproduced on its own without replaying the search.

**Selection.**  Best val macro-F1 SUBJECT TO class-stream flicker at or below
the rule classifier's own val flicker -- a model that wins on average while
twitching more than what ships is not an improvement, it is a different failure.
Configs outside the latency budget are evaluated (the lag curve is a deliverable)
but cannot be chosen.

Usage::

    uv run python -m training.nn.sweep --data-dir <data> [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import itertools
import time
from pathlib import Path

from .decoder import DEFAULT_LAG_BARS, DecodeParams
from .evaluate_v1 import (
    DECODER_CONFIG_FILE,
    DEFAULT_SPACE,
    artifact_provenance,
    build_report,
    default_data_dir,
    evaluate_config,
    identity_claims,
    load_inputs,
    render,
    rule_baseline,
    split_ids,
    write_json,
)
from .priors import MODEL_VERSION, MODELS_DIR, PRIORS_FILE, Priors

from evaluate_against_labels import (  # noqa: E402  (nn/__init__ sets the path)
    PRIMARY_TOLERANCE_SEC,
    prf,
)

# The look-ahead the show is built around is 8 s and the spec spends 5-6 s of it
# on the decoder; 3 bars is ~5.6 s at the corpus median bar of 1.875 s.  Longer
# lags are still MEASURED -- the accuracy-vs-lag curve is a deliverable, and it
# is the evidence for whether the budget should ever move -- but they cannot be
# selected, because a config that needs 15 s of future audio is not a config
# this system can run.
LATENCY_BUDGET_BARS = DEFAULT_LAG_BARS

# --------------------------------------------------------------------------- #
# The axes
# --------------------------------------------------------------------------- #

# Signed on purpose.  The label head was trained with inverse-frequency class
# weights, so it is already rebalanced toward a uniform prior; a positive
# strength corrects that twice, and a negative one hands corpus occupancy back.
# The useful half of this axis is expected to be the negative one.
PRIOR_STRENGTHS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25)

# Log-spaced across [1, 10]: the knob is odds, so equal ratios are equal steps.
DROP_MISS_COSTS = tuple(round(10.0 ** (step / 6.0), 4) for step in range(7))

# 0.0 disables the boundary head entirely, which is the control the rest of the
# axis is measured against.
BOUNDARY_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)
BOUNDARY_REFS = (0.1, 0.2, 0.3, 0.5, 0.7)

# drop's fitted floor is exactly 16 bars -- the DJ-cut length -- so this axis is
# really "is the annotator's drop the same unit as the dance floor's".
FLOOR_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

LAG_BARS = (0, 1, 2, 3, 4, 6, 8)

QUICK_AXES = {
    "prior_strength": (-0.5, 0.0),
    "drop_miss_cost": (1.0, 2.1544),
}

# Every axis, for the one-at-a-time ablation around the chosen config.  Same
# tuples the search used, so an ablation point is a config the search could have
# picked rather than a value invented for the report.
ABLATION_AXES = {
    "prior_strength": PRIOR_STRENGTHS,
    "drop_miss_cost": DROP_MISS_COSTS,
    "boundary_weight": BOUNDARY_WEIGHTS,
    "boundary_ref": BOUNDARY_REFS,
    "floor_scale": FLOOR_SCALES,
    "lag_bars": LAG_BARS,
}


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #


def enumerate_configs(base: DecodeParams, axes: dict) -> list:
    """The cartesian product of ``axes``, in declared order, over ``base``.

    Deterministic by construction: dicts preserve insertion order, the product
    is lexicographic in that order, and nothing samples.  Knobs not named keep
    the base's value, so a stage inherits the running winner rather than
    silently resetting to defaults.
    """
    fields = {field.name for field in dataclasses.fields(DecodeParams)}
    unknown = [name for name in axes if name not in fields]
    if unknown:
        raise TypeError(f"DecodeParams has no knob(s) {unknown}; "
                        f"known: {sorted(fields)}")
    names = list(axes)
    return [dataclasses.replace(base, **dict(zip(names, values)))
            for values in itertools.product(*(tuple(axes[name]) for name in names))]


def observation_key(params: DecodeParams) -> tuple:
    """The part of a config that invalidates the cached bar observations."""
    return (int(params.min_coverage), float(params.boundary_tolerance_sec))


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


class InputCache:
    """Sidecars read once per distinct observation setting, then reused.

    The sweep's whole cost model rests on this: a config that shares the
    observation key with the previous one costs a decode plus a score, not 123
    ``npz`` loads.
    """

    def __init__(self, data_dir, ids, table_path=None) -> None:
        self.data_dir = Path(data_dir)
        self.ids = list(ids)
        self.table_path = table_path
        self._cache: dict = {}
        self.skipped: list = []

    def for_params(self, params: DecodeParams) -> list:
        key = observation_key(params)
        if key not in self._cache:
            inputs, skipped = load_inputs(
                self.data_dir, self.ids, min_coverage=key[0],
                boundary_tolerance_sec=key[1], table_path=self.table_path)
            self._cache[key] = inputs
            if not self.skipped:
                self.skipped = skipped
        return self._cache[key]


def summarise(result: dict, space: str) -> dict:
    """A config's row in the sweep table -- scalars only, no ``Score`` objects."""
    score = result["score"]
    drop_precision, drop_recall, drop_f1 = prf(*score.counts["drop"])
    return {
        "params": dataclasses.asdict(result["params"]),
        "macro_f1": round(float(result["macro_f1"]), 6),
        "flicker_per_min": round(float(result["flicker_per_min"]), 6),
        "accuracy": round(float(score.accuracy), 6),
        "drop_recall": round(float(drop_recall), 6),
        "drop_precision": round(float(drop_precision), 6),
        "drop_f1": round(float(drop_f1), 6),
        "boundary_f1_2s": round(
            float(score.boundary_prf("class", PRIMARY_TOLERANCE_SEC)[2]), 6),
        "to_drop_boundary_f1_2s": round(
            float(score.boundary_prf("class", PRIMARY_TOLERANCE_SEC, "type", "drop")[2]),
            6),
        "per_class_f1": {label: round(float(score.f1(label)), 6)
                         for label in score.labels},
        "changes": score.boundary["class"][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"],
        "undecoded_share": round(
            float(score.no_intent_sec / score.exposure_sec)
            if score.exposure_sec else 0.0, 6),
    }


def run_configs(cache: InputCache, priors: Priors, configs, *,
                space: str = DEFAULT_SPACE, stage: str = "",
                seen: dict | None = None, log=None) -> list:
    """Evaluate each config once; ``seen`` de-duplicates across stages.

    Stages overlap by design -- each starts from the running winner, which the
    previous stage already measured -- so re-evaluating would be pure waste and,
    worse, would let the same config appear twice in the selection pool.
    """
    seen = {} if seen is None else seen
    rows: list = []
    for params in configs:
        if params in seen:
            continue
        started = time.perf_counter()
        result = evaluate_config(cache.for_params(params), priors, params,
                                 space=space, claims=identity_claims(space))
        row = summarise(result, space)
        row["stage"] = stage
        row["seconds"] = round(time.perf_counter() - started, 3)
        seen[params] = row
        rows.append(row)
        if log is not None:
            log(row)
    return rows


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def select_config(rows, flicker_ceiling: float, *,
                  budget_bars: int = LATENCY_BUDGET_BARS) -> dict:
    """Highest macro-F1 whose flicker clears the ceiling and whose lag fits.

    Ties break on lower flicker and then on enumeration order, so the winner is
    a function of the rows and not of dict iteration or sort stability luck.
    Nothing eligible is an error rather than a quiet fallback to the best
    ineligible config: "the sweep found no config that is both better and no
    twitchier" is a finding, and silently shipping a twitchier one would bury it.
    """
    rows = list(rows)
    eligible = []
    for index, row in enumerate(rows):
        params = row["params"]
        lag = int(params["lag_bars"] if isinstance(params, dict) else params.lag_bars)
        if lag > budget_bars:
            continue
        if float(row["flicker_per_min"]) > float(flicker_ceiling) + 1e-9:
            continue
        eligible.append((-float(row["macro_f1"]), float(row["flicker_per_min"]),
                         index, row))
    if not eligible:
        raise RuntimeError(
            f"no config reaches flicker <= {flicker_ceiling:.4f}/min within a "
            f"{budget_bars}-bar lag -- the decoder cannot beat the baseline on "
            f"continuity, which is a result, not a reason to relax the rule")
    eligible.sort(key=lambda item: item[:3])
    return eligible[0][3]


def ablate(cache: InputCache, priors: Priors, chosen: DecodeParams, seen: dict, *,
           space: str = DEFAULT_SPACE, log=None) -> tuple:
    """One axis at a time from the chosen config -- the controlled sensitivity.

    The pooled best-per-value curve over a staged search answers "what is the
    best we ever saw at this value", which conflates the axis with whatever the
    rest of the config happened to be in the stage that produced it.  This is
    the question the report actually needs: holding the shipped config fixed,
    what does moving ONE knob cost?  That is the number that says whether a knob
    mattered, and it is the only one from which "getting this wrong would have
    cost us X" can be read honestly.

    Configs the search already measured are reused from ``seen`` rather than
    re-run -- same config, same cached posteriors, same number -- so the anchor
    point appears in every curve instead of being deduplicated out of it.
    """
    curves: dict = {}
    fresh: list = []
    for axis, values in ABLATION_AXES.items():
        if log is not None:
            log({"stage_start": f"ablation:{axis}", "configs": len(values)})
        curve = []
        for value in values:
            params = dataclasses.replace(chosen, **{axis: value})
            if params not in seen:
                fresh.extend(run_configs(cache, priors, [params], space=space,
                                         stage=f"ablation:{axis}", seen=seen,
                                         log=log))
            curve.append(seen[params])
        curves[axis] = curve
    return curves, fresh


def sensitivity(rows, axis: str) -> dict:
    """Macro-F1 spread attributable to one axis, holding the rest as measured.

    Not a partial derivative: it is the range of the best-per-value curve, which
    is the number that answers "would getting this knob wrong have cost us
    anything". A knob whose best value scores within noise of its worst is a
    knob the report should say does not matter.
    """
    best: dict = {}
    for row in rows:
        params = row["params"]
        value = params[axis] if isinstance(params, dict) else getattr(params, axis)
        current = best.get(value)
        if current is None or row["macro_f1"] > current["macro_f1"]:
            best[value] = row
    if not best:
        return {"axis": axis, "values": {}, "spread": 0.0}
    ordered = dict(sorted(best.items(), key=lambda item: item[0]))
    scores = [row["macro_f1"] for row in ordered.values()]
    return {
        "axis": axis,
        "values": {str(value): {"macro_f1": row["macro_f1"],
                                "flicker_per_min": row["flicker_per_min"],
                                "drop_recall": row["drop_recall"],
                                "accuracy": row["accuracy"]}
                   for value, row in ordered.items()},
        "best_value": max(ordered.items(), key=lambda item: item[1]["macro_f1"])[0],
        "spread": round(max(scores) - min(scores), 6),
    }


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def refinement_axes(best: DecodeParams) -> dict:
    """A joint grid around the staged winner, re-opening every axis at once.

    Staged searches can only find a coordinate-wise optimum.  This is the check
    that the winner is not one: each axis is re-offered its winning value plus
    its neighbours, and all of them move together.
    """
    def around(values, chosen):
        values = list(values)
        if chosen not in values:
            return tuple(sorted({chosen, *values[:1], *values[-1:]}))
        index = values.index(chosen)
        window = values[max(0, index - 1):index + 2]
        return tuple(sorted(set(window)))

    return {
        "prior_strength": around(PRIOR_STRENGTHS, best.prior_strength),
        "drop_miss_cost": around(DROP_MISS_COSTS, best.drop_miss_cost),
        "boundary_weight": around(BOUNDARY_WEIGHTS, best.boundary_weight),
        "boundary_ref": around(BOUNDARY_REFS, best.boundary_ref),
        "floor_scale": around(FLOOR_SCALES, best.floor_scale),
    }


def run_sweep(cache: InputCache, priors: Priors, *, space: str = DEFAULT_SPACE,
              flicker_ceiling: float, quick: bool = False, log=None) -> dict:
    """The staged plan, then the joint refinement, then the selection."""
    seen: dict = {}
    rows: list = []
    stages: list = []

    def stage(name, axes, start):
        """Run one block, then re-select over EVERY config measured so far.

        Selecting from the whole pool rather than from this block alone is what
        keeps a later stage from inheriting a winner that a constraint already
        disqualified, and it means the running ``best`` is always the best
        *eligible* config known -- which is the point the next block searches
        around.
        """
        configs = enumerate_configs(start, axes)
        if log is not None:
            log({"stage_start": name, "configs": len(configs)})
        produced = run_configs(cache, priors, configs, space=space, stage=name,
                               seen=seen, log=log)
        rows.extend(produced)
        stages.append({"name": name, "requested": len(configs),
                       "evaluated": len(produced)})
        return DecodeParams(**select_config(seen.values(), flicker_ceiling)["params"])

    if quick:
        stage("quick", QUICK_AXES, DecodeParams())
    else:
        best = stage("prior_x_dropcost",
                     {"prior_strength": PRIOR_STRENGTHS,
                      "drop_miss_cost": DROP_MISS_COSTS}, DecodeParams())
        best = stage("boundary_weight_x_ref",
                     {"boundary_weight": BOUNDARY_WEIGHTS,
                      "boundary_ref": BOUNDARY_REFS}, best)
        best = stage("floor_scale", {"floor_scale": FLOOR_SCALES}, best)
        best = stage("lag_bars", {"lag_bars": LAG_BARS}, best)
        stage("joint_refine", refinement_axes(best), best)

    # Ablate around the search's winner, then re-select over everything: the
    # ablation is a measurement, but it is still a set of evaluated configs, and
    # one of them beating the anchor would mean the search had missed it.
    anchor = DecodeParams(**select_config(rows, flicker_ceiling)["params"])
    curves, fresh = ablate(cache, priors, anchor, seen, space=space, log=log)
    rows.extend(fresh)
    stages.append({"name": "ablation", "requested": sum(len(v) for v in curves.values()),
                   "evaluated": len(fresh)})

    chosen = select_config(rows, flicker_ceiling)
    return {
        "rows": rows,
        "chosen": chosen,
        "anchor": dataclasses.asdict(anchor),
        "anchor_is_chosen": chosen["params"] == dataclasses.asdict(anchor),
        "flicker_ceiling": flicker_ceiling,
        "stages": stages,
        "lag_curve": sorted(curves.get("lag_bars", []),
                            key=lambda row: row["params"]["lag_bars"]),
        # Controlled: one axis moved, everything else at the anchor.
        "sensitivity": [sensitivity(curve, axis) for axis, curve in curves.items()],
        # Uncontrolled, for contrast: the best ever seen at each value.
        "sensitivity_pooled": [sensitivity(rows, axis) for axis in curves],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split", default="val",
                        help="val by default; test is Task 6's to read")
    parser.add_argument("--quick", action="store_true",
                        help="a two-axis smoke sweep, for wiring checks")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--eval-out", type=Path, default=None,
                        help="also write the full val report for the chosen config")
    args = parser.parse_args(argv)

    if args.split == "test":
        parser.error("the sweep never touches the test split")

    model_dir = args.data_dir / MODELS_DIR / MODEL_VERSION
    priors = Priors.load(model_dir / PRIORS_FILE)
    ids = split_ids(args.data_dir, args.split)
    cache = InputCache(args.data_dir, ids)
    inputs = cache.for_params(DecodeParams())
    if not inputs:
        parser.error(f"no usable tracks in split {args.split!r}")

    baseline = rule_baseline(inputs)
    ceiling = baseline["flicker_per_min"]
    print(f"{len(inputs)} {args.split} tracks; rule baseline macro-F1 "
          f"{baseline['macro_f1']:.4f}, class-stream flicker {ceiling:.3f}/min "
          f"-- that flicker is the ceiling", flush=True)

    def log(row):
        if "stage_start" in row:
            print(f"\n-- {row['stage_start']}: {row['configs']} configs", flush=True)
            return
        print(f"   macro-F1 {row['macro_f1']:.4f}  flicker {row['flicker_per_min']:6.3f}"
              f"  dropR {row['drop_recall']:.3f}  {row['seconds']:5.2f}s  "
              f"{row['params']}", flush=True)

    started = time.perf_counter()
    result = run_sweep(cache, priors, flicker_ceiling=ceiling,
                       quick=args.quick, log=log)
    elapsed = time.perf_counter() - started

    chosen = DecodeParams(**result["chosen"]["params"])
    provenance = artifact_provenance(args.data_dir)
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provenance": provenance,
        "split": args.split,
        "tracks": len(inputs),
        "space": DEFAULT_SPACE,
        "selection_rule": "max val macro-F1 subject to class-stream flicker <= "
                          "rule baseline val flicker and lag_bars <= "
                          f"{LATENCY_BUDGET_BARS}",
        "flicker_ceiling_per_min": round(float(ceiling), 6),
        "rule_baseline": {"macro_f1": round(float(baseline["macro_f1"]), 6),
                          "flicker_per_min": round(float(ceiling), 6)},
        "chosen": dataclasses.asdict(chosen),
        "chosen_metrics": result["chosen"],
        "search_anchor": result["anchor"],
        "anchor_survived_ablation": result["anchor_is_chosen"],
        "defaults": dataclasses.asdict(DecodeParams()),
        "configs_evaluated": len(result["rows"]),
        "stages": result["stages"],
        "lag_curve": result["lag_curve"],
        "sensitivity": result["sensitivity"],
        "sensitivity_pooled": result["sensitivity_pooled"],
        "results": result["rows"],
        "elapsed_sec": round(elapsed, 2),
        "skipped": cache.skipped,
    }
    out = args.out or model_dir / DECODER_CONFIG_FILE
    write_json(out, payload)
    print(f"\n{len(result['rows'])} configs in {elapsed:.1f}s; chose {chosen}")
    print(f"wrote {out}")

    if args.eval_out is not None or not args.quick:
        report = build_report(
            cache.for_params(chosen), priors, chosen, split=args.split,
            skipped=cache.skipped,
            provenance={"config_source": str(out), "requested_tracks": len(ids),
                        "configs_evaluated": len(result["rows"]),
                        "artifacts": provenance})
        eval_out = args.eval_out or model_dir / "eval_val.json"
        write_json(eval_out, report)
        print()
        print(render(report))
        print(f"\nwrote {eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
