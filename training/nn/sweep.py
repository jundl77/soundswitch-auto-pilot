#!/usr/bin/env python
"""Decoder parameter sweep on the VAL split, against cached posteriors."""
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
    EVAL_FILE,
    MODEL_FILE,
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

from evaluate_against_labels import (  # noqa: E402
    PRIMARY_TOLERANCE_SEC,
    file_sha256,
    prf,
)

LATENCY_BUDGET_BARS = DEFAULT_LAG_BARS

PRIOR_STRENGTHS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25)

_DROP_MISS_STEPS_PER_DECADE = 6
DROP_MISS_COSTS = tuple(round(10.0 ** (step / _DROP_MISS_STEPS_PER_DECADE), 4)
                        for step in range(_DROP_MISS_STEPS_PER_DECADE + 1))

BOUNDARY_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)
BOUNDARY_REFS = (0.1, 0.2, 0.3, 0.5, 0.7)

FLOOR_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

LAG_BARS = (0, 1, 2, 3, 4, 6, 8)

QUICK_AXES = {
    "prior_strength": (-0.5, 0.0),
    "drop_miss_cost": (1.0, 2.1544),
}

ABLATION_AXES = {
    "prior_strength": PRIOR_STRENGTHS,
    "drop_miss_cost": DROP_MISS_COSTS,
    "boundary_weight": BOUNDARY_WEIGHTS,
    "boundary_ref": BOUNDARY_REFS,
    "floor_scale": FLOOR_SCALES,
    "lag_bars": LAG_BARS,
}


def enumerate_configs(base: DecodeParams, axes: dict) -> list:
    fields = {field.name for field in dataclasses.fields(DecodeParams)}
    unknown = [name for name in axes if name not in fields]
    if unknown:
        raise TypeError(f"DecodeParams has no knob(s) {unknown}; "
                        f"known: {sorted(fields)}")
    names = list(axes)
    return [dataclasses.replace(base, **dict(zip(names, values)))
            for values in itertools.product(*(tuple(axes[name]) for name in names))]


def observation_key(params: DecodeParams) -> tuple:
    return (int(params.min_coverage), float(params.boundary_tolerance_sec),
            float(params.temperature))


class InputCache:
    def __init__(self, data_dir, ids, table_path=None, posteriors_dir=None,
                 model_sha=None) -> None:
        self.data_dir = Path(data_dir)
        self.ids = list(ids)
        self.table_path = table_path
        self.posteriors_dir = posteriors_dir
        self.model_sha = model_sha
        self._cache: dict = {}
        self.skipped: list = []

    def for_params(self, params: DecodeParams) -> list:
        key = observation_key(params)
        if key not in self._cache:
            inputs, skipped = load_inputs(
                self.data_dir, self.ids, min_coverage=key[0],
                boundary_tolerance_sec=key[1], temperature=key[2],
                table_path=self.table_path,
                posteriors_dir=self.posteriors_dir, model_sha=self.model_sha)
            self._cache[key] = inputs
            if not self.skipped:
                self.skipped = skipped
        return self._cache[key]


def summarise(result: dict, space: str) -> dict:
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


CRISPNESS_TOLERANCE_SEC = 0.5


def summarise_multi(results: list, space: str) -> dict:
    """One row over several posterior sets of the same generation (seeds).

    Selection reads ``macro_f1`` (the seed mean) and ``flicker_per_min`` (the
    seed MAX -- a ceiling any exported seed must clear, so the constraint holds
    whichever seed ships).  Everything per-seed survives under ``seeds``.
    """
    rows = [summarise(result, space) for result in results]
    for row, result in zip(rows, results):
        row["crispness_05"] = round(float(
            result["score"].boundary_prf("class", CRISPNESS_TOLERANCE_SEC)[2]), 6)

    def mean(key):
        return round(sum(float(row[key]) for row in rows) / len(rows), 6)

    return {
        "params": rows[0]["params"],
        "macro_f1": mean("macro_f1"),
        "flicker_per_min": round(max(float(row["flicker_per_min"])
                                     for row in rows), 6),
        "flicker_per_min_mean": mean("flicker_per_min"),
        "accuracy": mean("accuracy"),
        "drop_recall": mean("drop_recall"),
        "drop_precision": mean("drop_precision"),
        "drop_f1": mean("drop_f1"),
        "boundary_f1_2s": mean("boundary_f1_2s"),
        "crispness_05": mean("crispness_05"),
        "to_drop_boundary_f1_2s": mean("to_drop_boundary_f1_2s"),
        "changes": sum(row["changes"] for row in rows),
        "undecoded_share": mean("undecoded_share"),
        "seeds": rows,
    }


def run_configs(cache, priors: Priors, configs, *,
                space: str = DEFAULT_SPACE, stage: str = "",
                seen: dict | None = None, log=None) -> list:
    seen = {} if seen is None else seen
    caches = list(cache) if isinstance(cache, (list, tuple)) else None
    rows: list = []
    for params in configs:
        if params in seen:
            continue
        started = time.perf_counter()
        if caches is None:
            result = evaluate_config(cache.for_params(params), priors, params,
                                     space=space, claims=identity_claims(space))
            row = summarise(result, space)
        else:
            results = [evaluate_config(one.for_params(params), priors, params,
                                       space=space, claims=identity_claims(space))
                       for one in caches]
            row = summarise_multi(results, space)
        row["stage"] = stage
        row["seconds"] = round(time.perf_counter() - started, 3)
        seen[params] = row
        rows.append(row)
        if log is not None:
            log(row)
    return rows


def _raise_if_none_eligible(eligible_configs, flicker_ceiling: float,
                            budget_bars: int) -> None:
    if not eligible_configs:
        raise RuntimeError(
            f"no config reaches flicker <= {flicker_ceiling:.4f}/min within a "
            f"{budget_bars}-bar lag -- the decoder cannot beat the baseline on "
            f"continuity, which is a result, not a reason to relax the rule")


def select_config(rows, flicker_ceiling: float, *,
                  budget_bars: int = LATENCY_BUDGET_BARS) -> dict:
    rows = list(rows)
    eligible_configs = []
    for index, row in enumerate(rows):
        params = row["params"]
        lag = int(params["lag_bars"] if isinstance(params, dict) else params.lag_bars)
        if lag > budget_bars:
            continue
        if float(row["flicker_per_min"]) > float(flicker_ceiling) + 1e-9:
            continue
        eligible_configs.append((-float(row["macro_f1"]),
                                 float(row["flicker_per_min"]), index, row))
    _raise_if_none_eligible(eligible_configs, flicker_ceiling, budget_bars)
    eligible_configs.sort(key=lambda item: item[:3])
    return eligible_configs[0][3]


def ablate(cache, priors: Priors, chosen: DecodeParams, seen: dict, *,
           space: str = DEFAULT_SPACE, log=None, axes: dict | None = None) -> tuple:
    curves: dict = {}
    fresh: list = []
    for axis, values in (ABLATION_AXES if axes is None else axes).items():
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


def refinement_axes(best: DecodeParams, extra_axes: dict | None = None) -> dict:
    def around(values, chosen):
        values = list(values)
        if chosen not in values:
            return tuple(sorted({chosen, *values[:1], *values[-1:]}))
        index = values.index(chosen)
        window = values[max(0, index - 1):index + 2]
        return tuple(sorted(set(window)))

    axes = {
        "prior_strength": around(PRIOR_STRENGTHS, best.prior_strength),
        "drop_miss_cost": around(DROP_MISS_COSTS, best.drop_miss_cost),
        "boundary_weight": around(BOUNDARY_WEIGHTS, best.boundary_weight),
        "boundary_ref": around(BOUNDARY_REFS, best.boundary_ref),
        "floor_scale": around(FLOOR_SCALES, best.floor_scale),
    }
    for name, values in (extra_axes or {}).items():
        axes[name] = around(values, getattr(best, name))
    return axes


def run_sweep(cache, priors: Priors, *, space: str = DEFAULT_SPACE,
              flicker_ceiling: float, quick: bool = False, log=None,
              base: DecodeParams | None = None,
              budget_bars: int = LATENCY_BUDGET_BARS,
              extra_stage_axes: dict | None = None) -> dict:
    base = DecodeParams() if base is None else base
    seen: dict = {}
    rows: list = []
    stages: list = []

    def stage(name, axes, start):
        configs = enumerate_configs(start, axes)
        if log is not None:
            log({"stage_start": name, "configs": len(configs)})
        produced = run_configs(cache, priors, configs, space=space, stage=name,
                               seen=seen, log=log)
        rows.extend(produced)
        stages.append({"name": name, "requested": len(configs),
                       "evaluated": len(produced)})
        return DecodeParams(**select_config(seen.values(), flicker_ceiling,
                                            budget_bars=budget_bars)["params"])

    if quick:
        stage("quick", QUICK_AXES, base)
    else:
        best = stage("prior_x_dropcost",
                     {"prior_strength": PRIOR_STRENGTHS,
                      "drop_miss_cost": DROP_MISS_COSTS}, base)
        best = stage("boundary_weight_x_ref",
                     {"boundary_weight": BOUNDARY_WEIGHTS,
                      "boundary_ref": BOUNDARY_REFS}, best)
        best = stage("floor_scale", {"floor_scale": FLOOR_SCALES}, best)
        for name, values in (extra_stage_axes or {}).items():
            best = stage(name, {name: values}, best)
        best = stage("lag_bars", {"lag_bars": LAG_BARS}, best)
        stage("joint_refine", refinement_axes(best, extra_stage_axes), best)

    anchor = DecodeParams(**select_config(rows, flicker_ceiling,
                                          budget_bars=budget_bars)["params"])
    ablation_axes = dict(ABLATION_AXES)
    ablation_axes.update(extra_stage_axes or {})
    curves, fresh = ablate(cache, priors, anchor, seen, space=space, log=log,
                           axes=ablation_axes)
    rows.extend(fresh)
    stages.append({"name": "ablation", "requested": sum(len(v) for v in curves.values()),
                   "evaluated": len(fresh)})

    chosen = select_config(rows, flicker_ceiling, budget_bars=budget_bars)
    return {
        "rows": rows,
        "chosen": chosen,
        "anchor": dataclasses.asdict(anchor),
        "anchor_is_chosen": chosen["params"] == dataclasses.asdict(anchor),
        "flicker_ceiling": flicker_ceiling,
        "budget_bars": budget_bars,
        "stages": stages,
        "lag_curve": sorted(curves.get("lag_bars", []),
                            key=lambda row: row["params"]["lag_bars"]),
        "sensitivity": [sensitivity(curve, axis) for axis, curve in curves.items()],
        "sensitivity_pooled": [sensitivity(rows, axis) for axis in curves],
    }


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
    parser.add_argument("--model-version", default=MODEL_VERSION,
                        help="artifact generation to sweep: reads priors from "
                             f"<data-dir>/{MODELS_DIR}/<model-version>/ and writes the "
                             "chosen config beside them (default: %(default)s)")
    parser.add_argument("--posteriors-dir", type=Path, default=None,
                        help="sidecar directory (default: <data-dir>/posteriors); a "
                             "retrain writes its own so the sidecars backing a "
                             "published verdict are never overwritten")
    args = parser.parse_args(argv)

    if args.split == "test":
        parser.error("the sweep never touches the test split")

    model_dir = args.data_dir / MODELS_DIR / args.model_version
    graph = model_dir / MODEL_FILE
    if not graph.exists():
        parser.error(f"no exported graph at {graph} -- without it the sidecars "
                     f"cannot be checked against the generation being swept")
    priors = Priors.load(model_dir / PRIORS_FILE)
    ids = split_ids(args.data_dir, args.split)
    cache = InputCache(args.data_dir, ids, posteriors_dir=args.posteriors_dir,
                       model_sha=file_sha256(graph))
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
    provenance = artifact_provenance(
        args.data_dir, args.model_version, args.posteriors_dir)
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
        eval_out = args.eval_out or model_dir / EVAL_FILE.format(split=args.split)
        write_json(eval_out, report)
        print()
        print(render(report))
        print(f"\nwrote {eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
