"""The per-arm decoder sweep for the nextgen retrain campaign (#341).

Same instrument as the l9 sweep (nn.sweep.run_sweep out of the label9 tree),
with the #341 floor_bars axis: explicit per-class floor vectors derived from
the arm's own fitted priors floors, buildup opened down to 1 bar.  floor_bars
is set on the base config and on every axis value, so no row ever carries
None -- the sweep's sensitivity()/around() sorts would raise on a None/tuple
mix.  With floor_bars set the floor_scale stage is inert (floor_bars overrides
it entirely); accepted and disclosed rather than patched around in sweep.py.

--smoke proves the floor_bars machinery end-to-end against the EXISTING l9
posteriors + priors on a tiny grid, writing only under CAMP/smoke_sweep/.
"""
import argparse
import ctypes
import dataclasses
import datetime
import json
import sys
import time
from pathlib import Path

import psutil

W = r"C:\Users\Julian\Projects\soundswitch-label9-worktree"
sys.path[:0] = [W, W + r"\training"]

import nn.sweep as nn_sweep  # noqa: E402
from nn.decoder import DecodeParams  # noqa: E402
from nn.evaluate_v1 import rule_baseline, split_ids, write_json  # noqa: E402
from nn.priors import Priors  # noqa: E402
from nn.sweep import InputCache, run_sweep  # noqa: E402

from evaluate_against_labels import file_sha256  # noqa: E402
from lib.label_space import SECTION_LABELS, check_class_space  # noqa: E402

MAIN = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot")
DATA = MAIN / "training" / "data" / "raveform"
CAMP = DATA / "models" / "nextgen_campaign"
L9_CAMPAIGN = DATA / "models" / "l9_campaign"
REGISTERED_CEILING = 0.274220
BUDGET_BARS = 2
OUTRO_ESCAPES = (0.0, 0.01, 0.02, 0.04)
MIN_FREE_GB = 4.0

SMOKE_AXES = {
    "PRIOR_STRENGTHS": (0.0, 0.25),
    "DROP_MISS_COSTS": (1.0, 6.8129),
    "BOUNDARY_WEIGHTS": (2.0, 4.0),
    "BOUNDARY_REFS": (0.2,),
    "FLOOR_SCALES": (0.5,),
    "LAG_BARS": (1, 2),
}
SMOKE_OUTRO_ESCAPES = (0.0, 0.02)


def below_normal() -> None:
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:  # noqa: BLE001
        pass


def floor_axis(priors: Priors, smoke: bool) -> tuple:
    classes = tuple(priors.classes)
    check_class_space(classes, "the swept priors")
    if classes != SECTION_LABELS:
        raise RuntimeError(f"priors speak {classes}, not the full vocabulary")
    fitted = tuple(int(v) for v in priors.floor_bars)
    buildup = classes.index("buildup")

    def scaled(scale):
        return tuple(max(1, int(round(v * scale))) for v in fitted)

    def with_buildup(vector, bars):
        return vector[:buildup] + (int(bars),) + vector[buildup + 1:]

    half, three_quarter = scaled(0.5), scaled(0.75)
    if smoke:
        axis = [fitted, half, with_buildup(half, 2)]
    else:
        axis = [fitted, three_quarter, half]
        axis += [with_buildup(half, bars) for bars in (4, 3, 2, 1)]
        axis.append(with_buildup(three_quarter, 2))
    return fitted, tuple(dict.fromkeys(axis))


def ceiling_or_die() -> float:
    reference = json.loads((L9_CAMPAIGN / "l9_decoder_reference.json")
                           .read_text(encoding="utf-8"))
    ceiling = float(reference["flicker_ceiling_per_min"])
    if abs(ceiling - REGISTERED_CEILING) > 5e-7:
        raise RuntimeError(
            f"reference ceiling {ceiling} is not the registered "
            f"{REGISTERED_CEILING} -- same split, same instrument, so a "
            f"different number means the instrument moved: STOPPING")
    return ceiling


def apply_smoke_grid() -> None:
    for name, values in SMOKE_AXES.items():
        setattr(nn_sweep, name, values)
    nn_sweep.ABLATION_AXES = {
        "prior_strength": SMOKE_AXES["PRIOR_STRENGTHS"],
        "drop_miss_cost": SMOKE_AXES["DROP_MISS_COSTS"],
        "boundary_weight": SMOKE_AXES["BOUNDARY_WEIGHTS"],
        "boundary_ref": SMOKE_AXES["BOUNDARY_REFS"],
        "floor_scale": SMOKE_AXES["FLOOR_SCALES"],
        "lag_bars": SMOKE_AXES["LAG_BARS"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("H", "HD"))
    parser.add_argument("--runs", default=None,
                        help="comma-separated run names (default: "
                             "ng_<ARM>_w128_s1234,ng_<ARM>_w128_s1235)")
    parser.add_argument("--priors", type=Path, default=None)
    parser.add_argument("--out-config", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and args.arm is None:
        parser.error("--arm is required unless --smoke")

    below_normal()
    free_gb = psutil.virtual_memory().available / 2**30
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(f"only {free_gb:.1f} GB RAM free; the gate is "
                           f"{MIN_FREE_GB} GB -- yielding rather than starting")

    if args.smoke:
        apply_smoke_grid()
        label = "smoke"
        runs = ("l9_w128_s1234", "l9_w128_s1235")
        priors_path = DATA / "models" / "l9" / "priors.json"
        posteriors = {run: DATA / "posteriors_l9_campaign" / run for run in runs}
        reports = {run: L9_CAMPAIGN / run / "training_report.json" for run in runs}
        out_dir = CAMP / "smoke_sweep"
        sweep_out = out_dir / "sweep_smoke.json"
        config_out = out_dir / "decoder_config_smoke.json"
        log_path = out_dir / "sweep_smoke.log"
        escapes = SMOKE_OUTRO_ESCAPES
    else:
        label = args.arm
        runs = tuple((args.runs or f"ng_{label}_w128_s1234,"
                                   f"ng_{label}_w128_s1235").split(","))
        priors_path = args.priors or CAMP / f"priors_{label}.json"
        posteriors = {run: CAMP / f"posteriors_{run}" for run in runs}
        reports = {run: CAMP / run / "training_report.json" for run in runs}
        out_dir = CAMP
        sweep_out = out_dir / f"sweep_{label}.json"
        config_out = args.out_config or out_dir / f"decoder_config_{label}.json"
        log_path = out_dir / f"sweep_{label}.log"
        escapes = OUTRO_ESCAPES

    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")

    def say(message):
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    ceiling = ceiling_or_die()
    priors = Priors.load(priors_path)
    fitted, floors = floor_axis(priors, args.smoke)
    say(f"fitted floors {dict(zip(priors.classes, fitted))}")
    say(f"floor_bars axis ({len(floors)} vectors): "
        + "; ".join(",".join(map(str, v)) for v in floors))

    ids = split_ids(DATA, "val")
    caches = []
    hashes = {}
    for run in runs:
        report = json.loads(reports[run].read_text(encoding="utf-8"))
        hashes[run] = report["weight_hash"]
        caches.append(InputCache(DATA, ids, posteriors_dir=posteriors[run],
                                 model_sha=hashes[run]))

    base = DecodeParams(min_coverage=1, lag_bars=BUDGET_BARS, floor_bars=fitted)
    inputs = caches[0].for_params(base)
    for cache in caches[1:]:
        cache.for_params(base)
    if len(inputs) != 215 or any(cache.skipped for cache in caches):
        raise RuntimeError(
            f"expected 215 clean val tracks, got {len(inputs)} "
            f"(skipped {[cache.skipped for cache in caches]}) -- a skip here "
            f"means a sidecar's model_sha disagrees with its "
            f"training_report.json weight_hash, or a sidecar is missing")

    table_stream = rule_baseline(inputs)
    say(f"{len(inputs)} val tracks; ceiling {ceiling:.6f}/min (registered "
        f"{REGISTERED_CEILING}); table-stream context macro "
        f"{table_stream['macro_f1']:.4f} flicker "
        f"{table_stream['flicker_per_min']:.4f}/min")

    def log(row):
        if "stage_start" in row:
            say(f"\n-- {row['stage_start']}: {row['configs']} configs")
            return
        say(f"   macro {row['macro_f1']:.4f}  flick(max) "
            f"{row['flicker_per_min']:6.4f}  crisp {row['crispness_05']:.4f}  "
            f"{row['seconds']:6.2f}s  {row['params']}")

    started = time.perf_counter()
    result = run_sweep(caches, priors, flicker_ceiling=ceiling, log=log,
                       base=base, budget_bars=BUDGET_BARS,
                       extra_stage_axes={"floor_bars": floors,
                                         "outro_escape": escapes})
    elapsed = time.perf_counter() - started

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": f"nextgen campaign decoder sweep, arm {label}"
                 + (" (SMOKE: l9 posteriors, tiny grid, machinery proof only)"
                    if args.smoke else ""),
        "registered_by": "#341",
        "split": "val",
        "tracks": len(inputs),
        "space": "raw9",
        "selection_rule": "max two-seed-mean val macro-F1 (raw9, class stream) "
                          "subject to max-seed flicker@2s <= the 5-class "
                          f"decoded reference ({ceiling:.6f}/min) and lag_bars "
                          f"<= {BUDGET_BARS}",
        "flicker_ceiling_per_min": ceiling,
        "reference": str(L9_CAMPAIGN / "l9_decoder_reference.json"),
        "table_stream_context": {
            "macro_f1": round(float(table_stream["macro_f1"]), 6),
            "flicker_per_min": round(float(table_stream["flicker_per_min"]), 6)},
        "provenance": {
            "priors": {"path": str(priors_path),
                       "sha256": file_sha256(priors_path)},
            "table": {"path": str(DATA / "training_table.csv.gz"),
                      "sha256": file_sha256(DATA / "training_table.csv.gz")},
            "posteriors": {run: {"dir": str(posteriors[run]),
                                 "weight_hash": hashes[run]} for run in runs},
            "modules_from": W,
        },
        "base": dataclasses.asdict(base),
        "budget_bars": BUDGET_BARS,
        "fitted_floor_bars": list(fitted),
        "floor_bars_axis": [list(v) for v in floors],
        "outro_escape_axis": list(escapes),
        "floor_scale_note": "floor_scale is inert in this sweep: floor_bars is "
                            "set on every row and overrides it (disclosed, "
                            "sweep.py unmodified)",
        "chosen": result["chosen"]["params"],
        "chosen_metrics": result["chosen"],
        "search_anchor": result["anchor"],
        "anchor_survived_ablation": result["anchor_is_chosen"],
        "configs_evaluated": len(result["rows"]),
        "stages": result["stages"],
        "lag_curve": result["lag_curve"],
        "sensitivity": result["sensitivity"],
        "sensitivity_pooled": result["sensitivity_pooled"],
        "results": result["rows"],
        "elapsed_sec": round(elapsed, 2),
    }
    write_json(sweep_out, payload)

    config = {
        "chosen": result["chosen"]["params"],
        "name": f"ng_{label}_sweep_pick",
        "generated_at": payload["generated_at"],
        "class_space": list(SECTION_LABELS),
        "provenance": {
            "source": str(sweep_out),
            "registered_by": "#341",
            "selection_rule": payload["selection_rule"],
            "priors": payload["provenance"]["priors"],
            "posteriors": payload["provenance"]["posteriors"],
            "split": "val",
            "tracks": len(inputs),
        },
        "notes": [
            "floor_bars is an explicit per-class vector in vocabulary order "
            "(#341, buildup opened down to 1 bar); it overrides floor_scale "
            "entirely, so a priors refit does NOT move these floors",
        ] + (["SMOKE artifact -- l9 posteriors, tiny grid; never ship this"]
             if args.smoke else []),
    }
    check_class_space(config["class_space"], str(config_out))
    write_json(config_out, config)

    say(f"\n{len(result['rows'])} configs in {elapsed/60:.1f} min")
    say(f"chosen: {result['chosen']['params']}")
    say(f"chosen row: macro {result['chosen']['macro_f1']:.6f}  flicker(max) "
        f"{result['chosen']['flicker_per_min']:.6f}  crisp "
        f"{result['chosen']['crispness_05']:.6f}")
    say(f"wrote {sweep_out}")
    say(f"wrote {config_out}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
