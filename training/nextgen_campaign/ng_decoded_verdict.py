"""The nextgen campaign's DECODED raveform-val verdict (#341): one instrument.

spec_decoded_verdict.py's structure, with one disclosed difference: each arm is
judged under its OWN swept decoder config + refit priors (that is the campaign
design), where the precedent froze a single config for every row.  Each row
records the config and priors it was decoded under, by path and sha256.

The shipped l9_w128_s1234 row is decoded under the COMMITTED
training/nn/decoder_config.json + models/l9/priors.json and asserted equal to
the l9 campaign's banked digits before anything else is reported, so this is
provably the campaign's decode and not a second instrument.

Ground truth on val is published labels only: every hand relabel is
train-split, and that is asserted (annotations/*.hand.json vs the val ids)
rather than assumed.
"""
import ctypes
import dataclasses
import datetime
import hashlib
import json
import sys
from pathlib import Path

W = r"C:\Users\Julian\Projects\soundswitch-label9-worktree"
sys.path[:0] = [W, W + r"\training"]

try:
    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.SetPriorityClass(_kernel32.GetCurrentProcess(), 0x00004000)
except Exception:  # noqa: BLE001
    pass

import psutil  # noqa: E402

from nn.decoder import decoder_config_classes, load_decoder_config  # noqa: E402
from nn.evaluate_v1 import (  # noqa: E402
    UNDECODED, beat_classes, build_decoder, decode_bars, identity_claims,
    load_inputs, restricted_macro_f1, score_predicted, split_ids, write_json)
from nn.priors import Priors  # noqa: E402

from evaluate_against_labels import aggregate  # noqa: E402
from lib.label_space import SECTION_LABELS, check_class_space  # noqa: E402

DATA = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform")
MAIN = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot")
CAMP = DATA / "models" / "nextgen_campaign"
L9_CAMPAIGN = DATA / "models" / "l9_campaign"
L9_PRIORS = DATA / "models" / "l9" / "priors.json"
COMMITTED_CONFIG = MAIN / "training" / "nn" / "decoder_config.json"
SHIPPED_CONFIG = DATA / "models" / "l9" / "decoder_config.json"
SEGMENTS = DATA / "annotations" / "segments.json"
CORE6 = ("intro", "buildup", "breakdown", "drop", "cooldown", "outro")
# Owner ruling: drop, breakdown, buildup, bridge are what matters (bridge may
# be confused with breakdown).  bridge-merged is a RE-SCORE of the decoded
# timeline under the fold on both truth and prediction -- the fold changes
# TP/FP/FN structure, so it is never arithmetic on the 9-class F1s.
CORE4 = ("drop", "breakdown", "buildup", "bridge")
CORE4_FOLDED = ("drop", "breakdown", "buildup")
OWNER_RULING = ("I dont care a ton about intro/outro/altoutro etc. - the most "
                "important are drop, breakdown, buildup, bridge (might also "
                "be confused with breakdown).")
# D5: available-memory floor under the supervisor's 900 MB park threshold.
MIN_AVAILABLE_MB = 700

sys.path.insert(0, str(L9_CAMPAIGN))
from l9_decoder_reference import decoded_drop_deficit, drop_spans_by_id  # noqa: E402

RUNS = {
    "l9_w128_s1234": {
        "posteriors": DATA / "posteriors_l9_campaign" / "l9_w128_s1234",
        "report": L9_CAMPAIGN / "l9_w128_s1234" / "training_report.json",
        "config": COMMITTED_CONFIG,
        "priors": L9_PRIORS,
        "role": "shipped reference",
    },
}
for _arm in ("H", "HD"):
    for _seed in (1234, 1235):
        _run = f"ng_{_arm}_w128_s{_seed}"
        RUNS[_run] = {
            "posteriors": CAMP / f"posteriors_{_run}",
            "report": CAMP / _run / "training_report.json",
            "config": CAMP / f"decoder_config_{_arm}.json",
            "priors": CAMP / f"priors_{_arm}.json",
            "role": f"nextgen arm {_arm} "
                    f"({'hand-corrected corpus' if _arm == 'H' else 'H + drop demotion overlay'}),"
                    f" seed {_seed}",
        }

# #342: arm H-OS is arm H's recipe with the owner's six hand tracks
# oversampled 16x (duplicated ids in splits_hos.json; the dataset multiplies
# slots, no dedup on the path).  Labels are unchanged, so the row is decoded
# under arm H's swept config + refit priors -- the handoff's first-read
# choice; a re-sweep would be its own recorded decision.
RUNS["ng_HOS_w128_s1234"] = {
    "posteriors": CAMP / "posteriors_ng_HOS_w128_s1234",
    "report": CAMP / "ng_HOS_w128_s1234" / "training_report.json",
    "config": CAMP / "decoder_config_H.json",
    "priors": CAMP / "priors_H.json",
    "role": "nextgen arm H-OS (#342: H + owner's 6 hand tracks oversampled "
            "16x via splits_hos.json; decoded under arm H's config/priors), "
            "seed 1234",
}

# #343 rung 1: arm H-OS8 is H-OS at factor 8 (7 extra copies per id in
# splits_hos8.json) -- the dose-response point between 1x (arm H) and 16x
# (H-OS).  Same labels, so the same arm-H config + priors, for the same
# reason: the three dose points must decode under one fixed decoder.
RUNS["ng_HOS8_w128_s1234"] = {
    "posteriors": CAMP / "posteriors_ng_HOS8_w128_s1234",
    "report": CAMP / "ng_HOS8_w128_s1234" / "training_report.json",
    "config": CAMP / "decoder_config_H.json",
    "priors": CAMP / "priors_H.json",
    "role": "nextgen arm H-OS8 (#343: H + owner's 6 hand tracks oversampled "
            "8x via splits_hos8.json; decoded under arm H's config/priors), "
            "seed 1234",
}

# #343 rung 2: arm H-FT8 initialises from arm H's best checkpoint and
# fine-tunes 3 epochs at 1/10th LR on the 8x oversampled mix -- a mechanism
# change, not a dose point: the init is what protects the board the
# from-scratch doses taxed.  Same labels, same arm-H config + priors.
RUNS["ng_HFT8_w128_s1234"] = {
    "posteriors": CAMP / "posteriors_ng_HFT8_w128_s1234",
    "report": CAMP / "ng_HFT8_w128_s1234" / "training_report.json",
    "config": CAMP / "decoder_config_H.json",
    "priors": CAMP / "priors_H.json",
    "role": "nextgen arm H-FT8 (#343 rung 2: fine-tune from ng_H_w128_s1234, "
            "3 epochs at lr 3e-5 on splits_hos8.json; decoded under arm H's "
            "config/priors), seed 1234",
}

# #343 rung 2 continued: arm H-SP initialises from arm H's best checkpoint
# and trains 3 epochs at the FULL campaign LR on the 8x mix, with an L2-SP
# quadratic anchor to H's weights -- the path-geometry counter to H-FT8's
# measured null (the unanchored LR dial only interpolates between null and
# too-far).  HSP2 is the capped second calibration point (lambda one decade
# over), pre-registered before running.  Same labels, same arm-H config +
# priors, for the same comparability reason as every rung-2 row.
for _arm in ("HSP", "HSP2"):
    _run = f"ng_{_arm}_w128_s1234"
    RUNS[_run] = {
        "posteriors": CAMP / f"posteriors_{_run}",
        "report": CAMP / _run / "training_report.json",
        "config": CAMP / "decoder_config_H.json",
        "priors": CAMP / "priors_H.json",
        "role": f"nextgen arm H-SP{'2' if _arm == 'HSP2' else ''} (#343: "
                f"L2-SP-anchored fine-tune from ng_H_w128_s1234 at campaign "
                f"LR on splits_hos8.json; decoded under arm H's "
                f"config/priors), seed 1234",
    }

# l9_decoder_verdict.json -> seeds.l9_w128_s1234, the banked decoded row.
BANKED = {"macro_f1_9": 0.523542, "core6_macro": 0.640287,
          "accuracy": 0.718635, "crispness_05": 0.708681,
          "boundary_f1_2s": 0.741427, "changes": 1576}
BANKED_DEFICIT = 107

DEFICIT_CAVEAT = ("#338: part of the decoded drop deficit is the model "
                  "correctly disbelieving inflated drop labels -- read it "
                  "beside the per-genre drop rows, not as a pure failure count")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_bridge(label: str) -> str:
    return "breakdown" if label == "bridge" else label


def fold_bridge_track(track):
    labels = dict(track.labels)
    labels["raw9"] = tuple(_fold_bridge(one) for one in labels["raw9"])
    return dataclasses.replace(track, labels=labels)


def fold_bridge_predicted(predicted) -> tuple:
    return tuple(one if one == UNDECODED else _fold_bridge(one)
                 for one in predicted)


def row_of(total) -> dict:
    return {
        "macro_f1_9": round(float(total.macro_f1), 6),
        "core6_macro": round(float(restricted_macro_f1(total, CORE6)), 6),
        "accuracy": round(float(total.accuracy), 6),
        "per_class_f1": {label: round(float(total.f1(label)), 6)
                         for label in total.labels},
        "crispness_05": round(float(total.boundary_prf("class", 0.5)[2]), 6),
        "boundary_f1_2s": round(float(total.boundary_prf("class", 2.0)[2]), 6),
        "to_drop_boundary_f1_2s": round(
            float(total.boundary_prf("class", 2.0, "type", "drop")[2]), 6),
        "changes": total.boundary["class"][2.0]["overall"]["n_pred"],
        "flicker_per_audience_minute": {
            f"{tol}": round(float(rate), 6)
            for tol, rate in total.flicker_per_minute["class"].items()},
        "exposure_min": round(float(total.exposure_sec) / 60.0, 3),
        "undecoded_share": round(float(total.no_intent_sec / total.exposure_sec)
                                 if total.exposure_sec else 0.0, 6),
    }


def assert_val_is_published(ids) -> dict:
    hand_ids = {path.name[:-len(".hand.json")]
                for path in (DATA / "annotations").glob("*.hand.json")}
    overridden = sorted(hand_ids & set(ids))
    if overridden:
        raise RuntimeError(f"val ids with hand overrides: {overridden} -- the "
                           f"instrument's published-label ground truth no "
                           f"longer holds on val")
    return {"hand_labels_seen": len(hand_ids),
            "val_ids_with_hand_override": 0,
            "read": "val ground truth is published labels; every hand relabel "
                    "is outside the val split (asserted, would have raised)"}


def markdown(rows: dict, genre_rows: dict, skipped: dict, ceiling: float) -> str:
    cols = ("macro_f1_9", "core4_macro", "core4_bridge_merged", "core6_macro",
            "accuracy", "crispness_05", "boundary_f1_2s",
            "to_drop_boundary_f1_2s", "changes")
    lines = ["# NG decoded verdict (val-215, raw9, per-arm swept configs)", ""]
    header = "| run | " + " | ".join(cols) + " | flicker@2s (vs ceiling) | deficit | undecoded |"
    lines += [header, "|" + "---|" * (len(cols) + 4)]
    for run, row in rows.items():
        cells = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                 for c in cols]
        flick = row["flicker_per_audience_minute"]["2.0"]
        deficit = row["decoded_drop_deficit"]
        lines.append(
            f"| {run} | " + " | ".join(cells)
            + f" | {flick:.4f} ({row['flicker_vs_ceiling']:+.4f})"
            + f" | {deficit['DEFICIT_sections']}/{deficit['drop_sections_seen']}"
            + f" | {row['undecoded_share']:.4f} |")
    lines += ["", f"Flicker ceiling: {ceiling:.6f}/min (5-class decoded "
                  f"reference).  Deficit caveat -- {DEFICIT_CAVEAT}.", ""]
    for run, per_genre in genre_rows.items():
        lines += [f"## per-genre: {run}", "",
                  "| genre | n | macro_f1_9 | drop_f1 | crispness_05 | "
                  "flicker@2s | deficit |", "|---|---|---|---|---|---|---|"]
        for genre, row in per_genre.items():
            lines.append(f"| {genre} | {row['n_tracks']} | "
                         f"{row['macro_f1_9']:.4f} | {row['drop_f1']:.4f} | "
                         f"{row['crispness_05']:.4f} | {row['flicker_2s']:.4f} "
                         f"| {row['deficit']} |")
        lines.append("")
    if skipped:
        lines += ["## skipped rows", ""]
        lines += [f"- {run}: missing {', '.join(missing)}"
                  for run, missing in skipped.items()]
        lines.append("")
    return "\n".join(lines)


def ram_gate_mb() -> float:
    for n, argument in enumerate(sys.argv[1:], 1):
        if argument.startswith("--ram-gate-mb="):
            return float(argument.split("=", 1)[1])
        if argument == "--ram-gate-mb" and n < len(sys.argv) - 1:
            return float(sys.argv[n + 1])
    return MIN_AVAILABLE_MB


def main() -> int:
    allow_missing = "--allow-missing-arms" in sys.argv[1:]
    gate_mb = ram_gate_mb()
    available_mb = psutil.virtual_memory().available / 2**20
    if available_mb < gate_mb:
        raise RuntimeError(f"only {available_mb:.0f} MB available; the gate is "
                           f"{gate_mb:.0f} MB -- yielding rather than "
                           f"starting")

    committed_cfg = json.loads(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    shipped_cfg = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    if committed_cfg["chosen"] != shipped_cfg["chosen"]:
        raise RuntimeError("committed decoder_config.json disagrees with the "
                           "shipped models/l9/decoder_config.json -- not the "
                           "frozen config")

    sweep = json.loads((L9_CAMPAIGN / "l9_decoder_sweep.json")
                       .read_text(encoding="utf-8"))
    ceiling = float(sweep["flicker_ceiling_per_min"])

    ids = list(split_ids(DATA, "val"))
    val_truth = assert_val_is_published(ids)
    spans_by_id = drop_spans_by_id(set(ids))
    with open(SEGMENTS, encoding="utf-8") as handle:
        genre_by_id = {str(r["id"]): r.get("genre") for r in json.load(handle)}

    rows, genre_rows, skipped_rows = {}, {}, {}
    for run, spec in RUNS.items():
        missing = [str(p) for p in (spec["posteriors"], spec["report"],
                                    spec["config"], spec["priors"])
                   if not p.exists()]
        if missing:
            if not allow_missing:
                raise RuntimeError(f"{run}: missing {missing} (pass "
                                   f"--allow-missing-arms to skip)")
            skipped_rows[run] = missing
            print(f"{run}: SKIPPED (missing inputs)", flush=True)
            continue

        params = load_decoder_config(spec["config"])
        if tuple(decoder_config_classes(spec["config"])) != SECTION_LABELS:
            raise RuntimeError(f"{run}: config class_space is not the "
                               f"vocabulary")
        priors = Priors.load(spec["priors"])
        check_class_space(priors.classes, str(spec["priors"]))
        report = json.loads(spec["report"].read_text(encoding="utf-8"))

        inputs, skipped = load_inputs(
            DATA, ids, min_coverage=params.min_coverage,
            boundary_tolerance_sec=params.boundary_tolerance_sec,
            temperature=params.temperature,
            posteriors_dir=spec["posteriors"],
            model_sha=report["weight_hash"])
        if skipped:
            raise RuntimeError(f"{run}: {len(skipped)} tracks missing inputs: "
                               f"{skipped[:3]}")

        decoder = build_decoder(priors, params)
        scores, folded_scores, segments = {}, {}, {}
        for n, item in enumerate(inputs, 1):
            bar_labels = decode_bars(item, decoder)
            segments[item.youtube_id] = (item.edges, bar_labels)
            predicted = beat_classes(item.times, item.edges, bar_labels)
            track = item.as_track_beats()
            scores[item.youtube_id] = score_predicted(
                track, "raw9", predicted, claims=identity_claims("raw9"))
            folded_scores[item.youtube_id] = score_predicted(
                fold_bridge_track(track), "raw9",
                fold_bridge_predicted(predicted),
                claims=identity_claims("raw9"))
            if n % 50 == 0:
                print(f"  {run}: decoded {n}/{len(inputs)}", flush=True)
        order = [item.youtube_id for item in inputs]
        total = aggregate([scores[i] for i in order])
        row = row_of(total)
        row["core4_macro"] = round(float(restricted_macro_f1(total, CORE4)), 6)
        folded_total = aggregate([folded_scores[i] for i in order])
        row["core4_bridge_merged"] = round(
            float(restricted_macro_f1(folded_total, CORE4_FOLDED)), 6)
        row["decoded_drop_deficit"] = decoded_drop_deficit(segments, spans_by_id)
        row["decoded_drop_deficit_caveat"] = DEFICIT_CAVEAT
        row["weight_hash"] = report["weight_hash"]
        row["posteriors_dir"] = str(spec["posteriors"])
        row["decoder_config"] = {"path": str(spec["config"]),
                                 "sha256": sha256_file(spec["config"])}
        row["priors"] = {"path": str(spec["priors"]),
                         "sha256": sha256_file(spec["priors"])}
        row["role"] = spec["role"]
        row["tracks"] = len(inputs)
        row["flicker_vs_ceiling"] = round(
            row["flicker_per_audience_minute"]["2.0"] - ceiling, 6)
        rows[run] = row

        if run == "l9_w128_s1234":
            for key, banked in BANKED.items():
                if row[key] != banked:
                    raise RuntimeError(
                        f"shipped {key} {row[key]!r} != banked {banked!r} -- "
                        f"this is not the campaign's decode")
            if row["decoded_drop_deficit"]["DEFICIT_sections"] != BANKED_DEFICIT:
                raise RuntimeError("shipped deficit != banked 107")
            print("  REPRODUCED the banked l9 decoded row to all digits",
                  flush=True)

        genre_rows[run] = {}
        for genre in sorted({genre_by_id[i] for i in order}):
            track_ids = [i for i in order if genre_by_id[i] == genre]
            total = aggregate([scores[i] for i in track_ids])
            deficit = decoded_drop_deficit(
                {i: segments[i] for i in track_ids}, spans_by_id)
            genre_rows[run][genre] = {
                "n_tracks": len(track_ids),
                "macro_f1_9": round(float(total.macro_f1), 6),
                "core6_macro": round(float(restricted_macro_f1(total, CORE6)), 6),
                "drop_f1": round(float(total.f1("drop")), 6),
                "crispness_05": round(float(total.boundary_prf("class", 0.5)[2]), 6),
                "flicker_2s": round(float(total.flicker_per_minute["class"][2.0]), 6),
                "deficit": f"{deficit['DEFICIT_sections']}"
                           f"/{deficit['drop_sections_seen']}",
            }
        print(f"{run}: macro_9 {row['macro_f1_9']:.6f}  core4 "
              f"{row['core4_macro']:.6f}  core4_bm "
              f"{row['core4_bridge_merged']:.6f}  core6 "
              f"{row['core6_macro']:.6f}  drop {row['per_class_f1']['drop']:.6f}"
              f"  crisp {row['crispness_05']:.6f}  flicker@2 "
              f"{row['flicker_per_audience_minute']['2.0']:.6f}  deficit "
              f"{row['decoded_drop_deficit']['DEFICIT_sections']}/"
              f"{row['decoded_drop_deficit']['drop_sections_seen']}", flush=True)

    verdict = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": "the nextgen campaign decoded raveform-val verdict (#341): "
                 "shipped l9_w128_s1234 under the frozen committed config vs "
                 "the H and HD arms, EACH under its own swept config + refit "
                 "priors (disclosed difference from the spec precedent's one "
                 "frozen config; that per-arm judging is the campaign design)",
        "split": "val",
        "tracks": 215,
        "harness": {
            "instrument": "l9_decoder_verdict.score_run verbatim (load_inputs/"
                          "build_decoder/decode_bars/beat_classes/"
                          "score_predicted, raw9 identity claims, "
                          "evaluate_against_labels.aggregate)",
            "modules_from": W,
            "driver": str(Path(__file__).resolve()),
            "banked_reproduction": "shipped row equals l9_decoder_verdict.json "
                                   "seeds.l9_w128_s1234 to all published "
                                   "digits (asserted, would have raised)"
                                   if "l9_w128_s1234" in rows else
                                   "NOT RUN -- shipped row skipped",
        },
        "val_ground_truth": val_truth,
        "core4": {
            "ruling_verbatim": OWNER_RULING,
            "core4_macro_classes": list(CORE4),
            "core4_bridge_merged_classes": list(CORE4_FOLDED),
            "method": "core4_macro is the mean per-class F1 over the four on "
                      "the 9-class score; core4_bridge_merged folds bridge "
                      "into breakdown on BOTH truth and prediction and "
                      "re-scores the same decoded timeline (the legacy_v1 "
                      "fold-as-a-view pattern), then takes the macro over the "
                      "three -- never arithmetic on 9-class F1s",
        },
        "committed_config": {"path": str(COMMITTED_CONFIG),
                             "sha256": sha256_file(COMMITTED_CONFIG)},
        "flicker_ceiling_per_min": ceiling,
        "decoded_drop_deficit_caveat": DEFICIT_CAVEAT,
        "runs": rows,
        "per_genre_decoded": genre_rows,
        "skipped_rows": skipped_rows,
    }
    CAMP.mkdir(parents=True, exist_ok=True)
    out = CAMP / "NG_DECODED.json"
    write_json(out, verdict)
    (CAMP / "NG_DECODED.md").write_text(
        markdown(rows, genre_rows, skipped_rows, ceiling), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {CAMP / 'NG_DECODED.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
