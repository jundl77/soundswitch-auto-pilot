"""The l9c campaign's DECODED raveform-val verdict (#346): one instrument.

l9b's verdict with two disclosed changes:

1. Modules import from the MAIN repo rather than the label9 worktree, because
   the #345 buildup_drop_bonus knob lives there.  The instrument is proven
   unmoved by anchors, not asserted: the shipped l9 row must reproduce the l9
   campaign's banked digits, and BOTH ng_H seed rows must reproduce the l9b
   NG_DECODED board's digits, before any new row is trusted.

2. The #344 transition instrument: every row also reports the per-track
   buildup-preceded-drop rate (was BUILDUP the decoded class on the bar
   immediately before the matched drop entry; entries matched to labeled drop
   onsets within the same 2.0 s tolerance the to-drop boundary F1 uses) and
   the buildup entry-lag distribution (first decoded buildup bar inside
   [start - 2 s, end) of each labeled buildup section; median/p90 + a
   never-entered count).  Reported beside the class scores, NOT gated.

Ground truth on val is published labels only (asserted: no val id carries a
hand override).
"""
import ctypes
import dataclasses
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot")
W = str(MAIN)
sys.path[:0] = [W, W + r"\training"]

try:
    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.SetPriorityClass(_kernel32.GetCurrentProcess(), 0x00004000)
except Exception:  # noqa: BLE001
    pass

import psutil  # noqa: E402

from nn.decoder import (decoder_config_classes, load_decoder_config)  # noqa: E402
from nn.evaluate_v1 import (  # noqa: E402
    UNDECODED, beat_classes, build_decoder, decode_bars, identity_claims,
    load_inputs, restricted_macro_f1, score_predicted, split_ids, write_json)
from nn.priors import Priors  # noqa: E402

from build_training_table import label_coverage  # noqa: E402
from evaluate_against_labels import aggregate  # noqa: E402
from lib.label_space import SECTION_LABELS, check_class_space  # noqa: E402
from raveform_fetch_annotations import load_all_tracks, parse_sections  # noqa: E402

DATA = MAIN / "training" / "data" / "raveform"
CAMP = DATA / "models" / "l9c_campaign"
L9B_CAMP = DATA / "models" / "l9b_campaign"
L9_CAMPAIGN = DATA / "models" / "l9_campaign"
L9_PRIORS = DATA / "models" / "l9" / "priors.json"
# The l9 anchor row decodes under the l9 GENERATION's own config -- the one
# its banked digits were cut under.  The committed config is the l9b ship's
# ng_H pick now, so it is checked against models/l9b instead.
L9_GEN_CONFIG = DATA / "models" / "l9" / "decoder_config.json"
COMMITTED_CONFIG = MAIN / "training" / "nn" / "decoder_config.json"
SHIPPED_CONFIG = DATA / "models" / "l9b" / "decoder_config.json"
SEGMENTS = DATA / "annotations" / "segments.json"
CORE6 = ("intro", "buildup", "breakdown", "drop", "cooldown", "outro")
CORE4 = ("drop", "breakdown", "buildup", "bridge")
CORE4_FOLDED = ("drop", "breakdown", "buildup")
OWNER_RULING = ("I dont care a ton about intro/outro/altoutro etc. - the most "
                "important are drop, breakdown, buildup, bridge (might also "
                "be confused with breakdown).")
MIN_AVAILABLE_MB = 700
ENTRY_TOLERANCE_SEC = 2.0

sys.path.insert(0, str(L9_CAMPAIGN))
from l9_decoder_reference import decoded_drop_deficit, drop_spans_by_id  # noqa: E402

RUNS = {
    "l9_w128_s1234": {
        "posteriors": DATA / "posteriors_l9_campaign" / "l9_w128_s1234",
        "report": L9_CAMPAIGN / "l9_w128_s1234" / "training_report.json",
        "config": L9_GEN_CONFIG,
        "priors": L9_PRIORS,
        "role": "l9 generation reference (decoded under its own generation "
                "config, the one the banked digits were cut under)",
    },
}
for _seed in (1234, 1235):
    _run = f"ng_H_w128_s{_seed}"
    RUNS[_run] = {
        "posteriors": L9B_CAMP / f"posteriors_{_run}",
        "report": L9B_CAMP / _run / "training_report.json",
        "config": L9B_CAMP / "decoder_config_H.json",
        "priors": L9B_CAMP / "priors_H.json",
        "role": f"l9b arm H (the binding baseline), seed {_seed} -- anchor "
                f"row, must reproduce the l9b NG_DECODED digits",
    }
for _seed in (1234, 1235):
    _run = f"ng_N_w128_s{_seed}"
    RUNS[_run] = {
        "posteriors": CAMP / f"posteriors_{_run}",
        "report": CAMP / _run / "training_report.json",
        "config": CAMP / "decoder_config_N.json",
        "priors": CAMP / "priors_N.json",
        "role": f"l9c arm N (#346: natural dose -- the owner's completed "
                f"slow-climb label session, no oversampling; decoded under "
                f"the arm's own swept config incl. the #345 transition axis "
                f"+ refit priors), seed {_seed}",
    }

# Registered post-N arms are appended here as they are born (each
# pre-registered in the decisions log before its train launches).
# N-MK (#346 addendum ~04:20): owner labels + the supervision mask over the
# 268 remaining suspect climb spans.  A mask is not a relabel, so the row
# decodes under arm N's swept config + refit priors (the l9b HMK pattern).
EXTRA_ARM_SPECS: dict = {
    "ng_NMK_w128_s1234": {
        "posteriors": CAMP / "posteriors_ng_NMK_w128_s1234",
        "report": CAMP / "ng_NMK_w128_s1234" / "training_report.json",
        "config": CAMP / "decoder_config_N.json",
        "priors": CAMP / "priors_N.json",
        "role": "l9c arm N-MK (#346: arm N's labels + supervision mask over "
                "the 268 surviving climb-shaped breakdown/bridge->drop train "
                "spans; mask proven zero-loss; decoded under arm N's "
                "config/priors), seed 1234",
    },
    "ng_NIW_w128_s1234": {
        "posteriors": CAMP / "posteriors_ng_NIW_w128_s1234",
        "report": CAMP / "ng_NIW_w128_s1234" / "training_report.json",
        "config": CAMP / "decoder_config_N.json",
        "priors": CAMP / "priors_N.json",
        "role": "l9c arm N-IW (#346: arm N + mild clean-exemplar weighting via "
                "splits_niw.json, +6 slots -- inyathi x4, opus/yai/skylark x2; "
                "no mask; decoded under arm N's config/priors), seed 1234",
    },
}
RUNS.update(EXTRA_ARM_SPECS)

# l9 campaign banked decoded row (same values l9b asserted).
BANKED = {"macro_f1_9": 0.523542, "core6_macro": 0.640287,
          "accuracy": 0.718635, "crispness_05": 0.708681,
          "boundary_f1_2s": 0.741427, "changes": 1576}
BANKED_DEFICIT = 107

# l9b NG_DECODED board digits for the H rows (the anchor discipline: the new
# instrument must reproduce them before any new row is read).
H_ANCHOR_KEYS = ("macro_f1_9", "core4_macro", "core4_bridge_merged",
                 "core6_macro", "accuracy", "crispness_05", "boundary_f1_2s",
                 "to_drop_boundary_f1_2s", "changes", "undecoded_share")

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


def class_spans_by_id(ids, wanted=("drop", "buildup")) -> dict:
    """Labeled spans per val id, from the same annotation view the deficit
    reads (merged; equals published on val, which is asserted)."""
    spans: dict = {}
    wanted = set(wanted)
    id_set = set(ids)
    for track in load_all_tracks(DATA):
        youtube_id = str(track.get("id"))
        if youtube_id not in id_set:
            continue
        covered = sorted(label_coverage(parse_sections(track)),
                         key=lambda span: span[0])
        spans[youtube_id] = {
            label: [(float(s), float(e)) for s, e, one in covered
                    if one == label]
            for label in wanted}
    return spans


def _entries(edges, bar_labels, label) -> list:
    """(entry_time, previous_class) per decoded run of ``label``."""
    out = []
    for index, one in enumerate(bar_labels):
        if one == label and (index == 0 or bar_labels[index - 1] != label):
            out.append((float(edges[index]),
                        None if index == 0 else bar_labels[index - 1]))
    return out


def transition_read(edges, bar_labels, spans) -> dict:
    """#344: buildup-preceded-drop + buildup entry lag, one decoded track."""
    drop_entries = _entries(edges, bar_labels, "drop")
    drops = []
    for start, _end in spans.get("drop", ()):
        best = None
        for time, previous in drop_entries:
            delta = time - start
            if abs(delta) <= ENTRY_TOLERANCE_SEC and (
                    best is None or abs(delta) < abs(best[0])):
                best = (delta, previous)
        drops.append({
            "onset": round(start, 3),
            "landed": best is not None,
            "entry_delta": round(best[0], 3) if best else None,
            "preceded_by": best[1] if best else None,
            "preceded_by_buildup": bool(best and best[1] == "buildup"),
        })

    buildup_entries = _entries(edges, bar_labels, "buildup")
    buildups = []
    for start, end in spans.get("buildup", ()):
        entry = next((time for time, _prev in buildup_entries
                      if start - ENTRY_TOLERANCE_SEC <= time < end), None)
        buildups.append({
            "onset": round(start, 3),
            "end": round(end, 3),
            "entered": entry is not None,
            "entry_lag": round(entry - start, 3) if entry is not None else None,
        })
    return {"drops": drops, "buildups": buildups}


def aggregate_transition(per_track: dict) -> dict:
    drops = [d for reads in per_track.values() for d in reads["drops"]]
    buildups = [b for reads in per_track.values() for b in reads["buildups"]]
    landed = [d for d in drops if d["landed"]]
    preceded = [d for d in landed if d["preceded_by_buildup"]]
    lags = sorted(b["entry_lag"] for b in buildups if b["entered"])

    def pct(values, q):
        return round(float(np.percentile(values, q)), 3) if values else None

    return {
        "method": {
            "drop": "decoded drop entries matched to labeled drop onsets "
                    f"within +-{ENTRY_TOLERANCE_SEC} s (the to-drop boundary "
                    "tolerance); preceded = the decoded class of the bar "
                    "immediately before the entry bar is buildup",
            "buildup": "first decoded buildup bar edge inside "
                       f"[onset - {ENTRY_TOLERANCE_SEC} s, section end); lag "
                       "is vs the labeled onset (#344: soft, graded, never "
                       "per-second-charged)",
            "gating": "#344/#346: reported beside the class scores, NOT gated",
        },
        "drop_onsets": len(drops),
        "drop_entries_landed": len(landed),
        "preceded_by_buildup": len(preceded),
        "preceded_rate_of_landed": round(len(preceded) / len(landed), 6)
                                   if landed else None,
        "preceded_rate_of_all": round(len(preceded) / len(drops), 6)
                                if drops else None,
        "buildup_sections": len(buildups),
        "buildup_never_entered": sum(1 for b in buildups if not b["entered"]),
        "entry_lag_median_sec": pct(lags, 50),
        "entry_lag_p90_sec": pct(lags, 90),
    }


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
    lines = ["# L9C decoded verdict (val-215, raw9, per-arm swept configs, "
             "#344 transition instrument)", ""]
    header = ("| run | " + " | ".join(cols)
              + " | flicker@2s (vs ceiling) | deficit | undecoded "
              "| bu>drop (landed) | bu lag med/p90 | bu never |")
    lines += [header, "|" + "---|" * (len(cols) + 7)]
    for run, row in rows.items():
        cells = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                 for c in cols]
        flick = row["flicker_per_audience_minute"]["2.0"]
        deficit = row["decoded_drop_deficit"]
        t = row["transition_read"]["aggregate"]
        rate = t["preceded_rate_of_landed"]
        rate_cell = "n/a" if rate is None else f"{rate:.3f}"
        med, p90 = t["entry_lag_median_sec"], t["entry_lag_p90_sec"]
        lag_cell = "-" if med is None else f"{med}/{p90}s"
        lines.append(
            f"| {run} | " + " | ".join(cells)
            + f" | {flick:.4f} ({row['flicker_vs_ceiling']:+.4f})"
            + f" | {deficit['DEFICIT_sections']}/{deficit['drop_sections_seen']}"
            + f" | {row['undecoded_share']:.4f}"
            + f" | {rate_cell} ({t['preceded_by_buildup']}"
              f"/{t['drop_entries_landed']} of {t['drop_onsets']})"
            + f" | {lag_cell}"
            + f" | {t['buildup_never_entered']}/{t['buildup_sections']} |")
    lines += ["", f"Flicker ceiling: {ceiling:.6f}/min (5-class decoded "
                  f"reference).  Deficit caveat -- {DEFICIT_CAVEAT}.",
              "", "bu>drop / bu lag / bu never are the #344 transition "
                  "instrument: reported, NOT gated.", ""]
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


def assert_h_anchor(run: str, row: dict, banked_rows: dict) -> None:
    banked = banked_rows[run]
    for key in H_ANCHOR_KEYS:
        if row[key] != banked[key]:
            raise RuntimeError(
                f"{run}: {key} {row[key]!r} != l9b banked {banked[key]!r} -- "
                f"the instrument moved; STOPPING before any new row is read")
    if (row["flicker_per_audience_minute"]["2.0"]
            != banked["flicker_per_audience_minute"]["2.0"]):
        raise RuntimeError(f"{run}: flicker@2s differs from the l9b board")
    if (row["decoded_drop_deficit"]["DEFICIT_sections"]
            != banked["decoded_drop_deficit"]["DEFICIT_sections"]):
        raise RuntimeError(f"{run}: deficit differs from the l9b board")
    print(f"  REPRODUCED the l9b banked {run} row to all digits", flush=True)


def main() -> int:
    allow_missing = "--allow-missing-arms" in sys.argv[1:]
    gate_mb = ram_gate_mb()
    available_mb = psutil.virtual_memory().available / 2**20
    if available_mb < gate_mb:
        raise RuntimeError(f"only {available_mb:.0f} MB available; the gate is "
                           f"{gate_mb:.0f} MB -- yielding rather than "
                           f"starting")

    # The shipped generation is l9b; its config predates the #345 knob, so
    # semantic equality (both load to the same DecodeParams) is the check.
    if load_decoder_config(COMMITTED_CONFIG) != load_decoder_config(SHIPPED_CONFIG):
        raise RuntimeError("committed decoder_config.json disagrees with the "
                           "shipped models/l9b/decoder_config.json -- not the "
                           "frozen config")

    sweep = json.loads((L9_CAMPAIGN / "l9_decoder_sweep.json")
                       .read_text(encoding="utf-8"))
    ceiling = float(sweep["flicker_ceiling_per_min"])

    banked_h = json.loads((L9B_CAMP / "NG_DECODED.json")
                          .read_text(encoding="utf-8"))["runs"]

    ids = list(split_ids(DATA, "val"))
    val_truth = assert_val_is_published(ids)
    spans_by_id = drop_spans_by_id(set(ids))
    transition_spans = class_spans_by_id(ids)
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
        per_track_transition = {}
        for n, item in enumerate(inputs, 1):
            bar_labels = decode_bars(item, decoder)
            segments[item.youtube_id] = (item.edges, bar_labels)
            per_track_transition[item.youtube_id] = transition_read(
                item.edges, bar_labels,
                transition_spans.get(item.youtube_id, {}))
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
        row["transition_read"] = {
            "aggregate": aggregate_transition(per_track_transition),
            "per_track": per_track_transition,
        }
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
        if run in ("ng_H_w128_s1234", "ng_H_w128_s1235"):
            assert_h_anchor(run, row, banked_h)

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
        t = row["transition_read"]["aggregate"]
        print(f"{run}: macro_9 {row['macro_f1_9']:.6f}  core4 "
              f"{row['core4_macro']:.6f}  core4_bm "
              f"{row['core4_bridge_merged']:.6f}  crisp "
              f"{row['crispness_05']:.6f}  flicker@2 "
              f"{row['flicker_per_audience_minute']['2.0']:.6f}  deficit "
              f"{row['decoded_drop_deficit']['DEFICIT_sections']}/"
              f"{row['decoded_drop_deficit']['drop_sections_seen']}  "
              f"bu>drop {t['preceded_rate_of_landed']}  "
              f"bu_lag {t['entry_lag_median_sec']}/{t['entry_lag_p90_sec']}s  "
              f"bu_never {t['buildup_never_entered']}/{t['buildup_sections']}",
              flush=True)

    verdict = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": "the l9c campaign decoded raveform-val verdict (#346): "
                 "shipped l9 + l9b arm H anchors + the l9c arms, each under "
                 "its own swept config + refit priors, with the #344 "
                 "transition instrument on every row",
        "split": "val",
        "tracks": 215,
        "harness": {
            "instrument": "l9_decoder_verdict.score_run verbatim (load_inputs/"
                          "build_decoder/decode_bars/beat_classes/"
                          "score_predicted, raw9 identity claims, "
                          "evaluate_against_labels.aggregate)",
            "modules_from": W,
            "driver": str(Path(__file__).resolve()),
            "anchors": "shipped l9 row == l9 campaign banked digits; both "
                       "ng_H rows == l9b NG_DECODED digits (all asserted, "
                       "would have raised)",
        },
        "val_ground_truth": val_truth,
        "core4": {
            "ruling_verbatim": OWNER_RULING,
            "core4_macro_classes": list(CORE4),
            "core4_bridge_merged_classes": list(CORE4_FOLDED),
            "method": "core4_macro is the mean per-class F1 over the four on "
                      "the 9-class score; core4_bridge_merged folds bridge "
                      "into breakdown on BOTH truth and prediction and "
                      "re-scores the same decoded timeline, then takes the "
                      "macro over the three",
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
