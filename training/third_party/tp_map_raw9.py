#!/usr/bin/env python
"""Evidence-informed raw-9 drafts for the admitted third-party tracks (#314).

Each admitted track's verbatim source sections are translated into the show's
raw nine, with contested labels decided per section from measured energy
evidence: a chorus becomes ``drop`` only when its measured entry contrast sits
in this corpus's own top tier, and everything with no honest raw-9 target
(verse, solo, inst, ...) stays an explicit ``masked`` span rather than a lie.
Thresholds are derived from the measured distribution over these tracks and
registered in ``third_party/mapped_thresholds.json`` BEFORE any mapping is
applied.

Drafts land as ``annotations/<id>.mapped.json`` -- a suffix nothing in the
dataset flow globs (the loaders take ``*.hand.json`` and ``segments.json``
only), so a draft is structurally incapable of hand-precedence and inert to
the training table until the integration ruling.  ``--stage-spot-check``
additionally seeds a stratified sample into ``<corpus>/tmp_labels/`` in the
labelling tool's own working format, never overwriting existing work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_TRAINING_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_TRAINING_DIR.parent), str(_TRAINING_DIR),
              str(_TRAINING_DIR / "raveform"), str(_TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tp_record  # noqa: E402
from corpus_root import corpus_dir  # noqa: E402
from raveform_fetch_annotations import annotations_dir  # noqa: E402

MAPPED_SUFFIX = ".mapped.json"
MAPPED_SCHEMA = 1
MAPPED_VOCABULARY = "raveform_raw9_plus_masked"
MASKED = "masked"

DECODE_RATE = 22050
ENTRY_WINDOW_SEC = 5.0
MIN_ENTRY_EVIDENCE_SEC = 1.0
SILENCE_FLOOR_DB = -100.0

# Verbatim label -> raw-9, after normalisation. Only labels whose meaning
# survives the translation; the charter's contested ones route elsewhere.
DIRECT = {
    "intro": "intro", "fadein": "intro", "opening": "intro", "intropt": "intro",
    "outro": "outro", "ending": "outro",
    "bridge": "bridge",
    "breakdown": "breakdown",
    "prechorus": "buildup", "build": "buildup", "buildup": "buildup",
    "transition": "buildup",
}
# Chorus-family labels all face the same evidence split; a quiet or
# instrumental chorus simply falls below the drop threshold on its own merits.
CHORUS_FAMILY = {"chorus", "altchorus", "quietchorus", "instchorus",
                 "chorusinst", "chorushalf", "choruspart", "refrain"}
# Track-tail functions: outro only where they actually end the track.
TAIL_FUNCTIONS = {"fadeout", "coda", "out"}
END_SENTINEL = "end"

RULE_DIRECT = "direct"
RULE_CHORUS_DROP = "chorus_entry_top_tier_drop"
RULE_CHORUS_BREAKDOWN = "chorus_entry_below_tier_breakdown"
RULE_CHORUS_DEFAULT = "chorus_no_entry_evidence_default_breakdown"
RULE_TAIL_OUTRO = "tail_function_final_outro"
RULE_TAIL_MASKED = "tail_function_interior_masked"
RULE_END = "end_sentinel_masked"
RULE_NO_TARGET = "no_honest_target_masked"
RULE_CARRYOVER = "source_masked_carryover"


def normalise_label(name: str) -> str:
    """Verbatim -> canonical: 'chorus A' / 'Chorus2' / 'chorus, fade-out' -> 'chorus'."""
    first = str(name).lower().split(",")[0].strip()
    tokens = first.split()
    if len(tokens) > 1 and len(tokens[-1]) == 1 and tokens[-1].isalpha():
        tokens = tokens[:-1]
    collapsed = re.sub(r"[^a-z0-9]", "", "".join(tokens))
    return re.sub(r"[0-9]+$", "", collapsed)


def rule_for(canonical: str, final: bool) -> tuple:
    """-> (rule, target); target 'chorus' means the evidence split decides."""
    if canonical == END_SENTINEL:
        return RULE_END, MASKED
    if canonical in CHORUS_FAMILY:
        return "chorus_energy_split", "chorus"
    if canonical in DIRECT:
        return RULE_DIRECT, DIRECT[canonical]
    if canonical in TAIL_FUNCTIONS:
        if final:
            return RULE_TAIL_OUTRO, "outro"
        return RULE_TAIL_MASKED, MASKED
    return RULE_NO_TARGET, MASKED


def decide_chorus(entry_contrast_db, threshold_db: float) -> tuple:
    if entry_contrast_db is None:
        return RULE_CHORUS_DEFAULT, "breakdown"
    if entry_contrast_db >= threshold_db:
        return RULE_CHORUS_DROP, "drop"
    return RULE_CHORUS_BREAKDOWN, "breakdown"


# --------------------------------------------------------------------------- #
# Energy evidence
# --------------------------------------------------------------------------- #


def decode_audio(path: Path) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
         "-ar", str(DECODE_RATE), "-f", "f32le", "-"],
        capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise RuntimeError(f"ffmpeg failed on {path.name}: {tail}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def rms_db(samples: np.ndarray, start: float, end: float):
    lo = max(0, int(start * DECODE_RATE))
    hi = min(len(samples), int(end * DECODE_RATE))
    if hi <= lo:
        return None
    power = float(np.mean(np.square(samples[lo:hi], dtype=np.float64)))
    if power <= 10.0 ** (SILENCE_FLOOR_DB / 10.0):
        return SILENCE_FLOOR_DB
    return 10.0 * math.log10(power)


def measure_sections(samples: np.ndarray, sections: list) -> list:
    measured = []
    for span in sections:
        start, end = float(span["start"]), float(span["end"])
        head_end = min(start + ENTRY_WINDOW_SEC, end)
        pre_db = (rms_db(samples, max(0.0, start - ENTRY_WINDOW_SEC), start)
                  if start >= MIN_ENTRY_EVIDENCE_SEC else None)
        measured.append({
            "start": start, "end": end,
            "rms_db": rms_db(samples, start, end),
            "head_db": rms_db(samples, start, head_end),
            "pre_db": pre_db,
        })
    return measured


def _spans_of(sections: list) -> list:
    return [[round(float(s["start"]), 3), round(float(s["end"]), 3)]
            for s in sections]


def measure_track(record: dict, audio: Path) -> dict:
    samples = decode_audio(audio)
    sections = sorted(record["sections"], key=lambda s: float(s["start"]))
    return {
        "audio_size": audio.stat().st_size,
        "audio_mtime_ns": audio.stat().st_mtime_ns,
        "decoded_sec": round(len(samples) / DECODE_RATE, 3),
        "track_rms_db": rms_db(samples, 0.0, len(samples) / DECODE_RATE),
        "spans": _spans_of(sections),
        "sections": measure_sections(samples, sections),
    }


# --------------------------------------------------------------------------- #
# Thresholds -- measured, registered, then applied
# --------------------------------------------------------------------------- #


def quantiles(values: list) -> dict:
    arr = np.asarray(sorted(values), dtype=np.float64)
    if not len(arr):
        return {"n": 0}
    result = {name: round(float(np.quantile(arr, p)), 3)
              for name, p in (("min", 0.0), ("q1", 0.25), ("median", 0.5),
                              ("q3", 0.75), ("p90", 0.9), ("max", 1.0))}
    result.update(n=int(len(arr)), mean=round(float(arr.mean()), 3),
                  sd=round(float(arr.std()), 3))
    return result


def derive_thresholds(chorus_entries: dict) -> dict:
    pooled = [v for values in chorus_entries.values() for v in values]
    if not pooled:
        raise RuntimeError("no chorus entry contrasts measured; nothing to derive")
    dist = {"pooled": quantiles(pooled)}
    for source, values in sorted(chorus_entries.items()):
        dist[source] = quantiles(values)
    return {
        "schema": 1,
        "derived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measure": ("entry_contrast_db = mean RMS dB of the chorus's first "
                    f"{ENTRY_WINDOW_SEC:g} s minus mean RMS dB of the "
                    f"{ENTRY_WINDOW_SEC:g} s of audio before its start"),
        "chorus_entry_contrast_db": dist,
        "chorus_drop_threshold_db": dist["pooled"]["q3"],
        "basis": ("top tier = pooled upper quartile (Q3) of the measured chorus "
                  "entry-contrast distribution over the admitted third-party "
                  "corpus; deltas cancel per-source mastering level, so pooled"),
    }


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #


def _r(value):
    return None if value is None else round(value, 3)


def map_track(record: dict, features: dict, threshold_db: float) -> dict:
    sections = sorted(record["sections"], key=lambda s: float(s["start"]))
    measured = features["sections"]
    mapped = []
    for index, (span, energy) in enumerate(zip(sections, measured)):
        canonical = normalise_label(span["name"])
        rule, target = rule_for(canonical, final=index == len(sections) - 1)
        entry_step = None
        if index and measured[index - 1]["rms_db"] is not None \
                and energy["rms_db"] is not None:
            entry_step = round(energy["rms_db"] - measured[index - 1]["rms_db"], 3)
        entry_contrast = None
        if energy["pre_db"] is not None and energy["head_db"] is not None:
            entry_contrast = round(energy["head_db"] - energy["pre_db"], 3)
        threshold = None
        if target == "chorus":
            rule, target = decide_chorus(entry_contrast, threshold_db)
            threshold = threshold_db
        evidence = {
            "source_label": span["name"],
            "canonical": canonical,
            "rms_db": _r(energy["rms_db"]),
            "entry_step_db": entry_step,
            "entry_contrast_db": entry_contrast,
            "rule": rule,
            "threshold_db": threshold,
        }
        if threshold is not None and entry_contrast is not None:
            evidence["margin_db"] = round(entry_contrast - threshold, 3)
        mapped.append({"start": float(span["start"]), "end": float(span["end"]),
                       "label": target, "evidence": evidence})
    for span in record.get("masked", []):
        origin = span.get("function") or span.get("reason") or "unknown"
        mapped.append({
            "start": float(span["start"]), "end": float(span["end"]),
            "label": MASKED,
            "evidence": {
                "source_label": f"__masked__({origin})",
                "canonical": MASKED, "rms_db": None, "entry_step_db": None,
                "entry_contrast_db": None, "rule": RULE_CARRYOVER,
                "threshold_db": None,
            }})
    mapped.sort(key=lambda s: float(s["start"]))
    return {
        "schema": MAPPED_SCHEMA,
        "kind": "raw9_mapped_draft",
        "source": record["source"],
        "id": record["id"],
        "native_id": record.get("native_id"),
        "title": record.get("title"),
        "artist": record.get("artist"),
        "genre": record.get("genre"),
        "audio": record.get("audio"),
        "duration": record.get("duration"),
        "label_vocabulary": MAPPED_VOCABULARY,
        "mapped_from": f"{record['id']}.{record['source']}.json",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": {"chorus_drop_entry_contrast_db": threshold_db,
                       "registered_in": "third_party/mapped_thresholds.json"},
        "sections": mapped,
    }


def masked_fraction(mapped: dict) -> float:
    total = sum(s["end"] - s["start"] for s in mapped["sections"])
    if total <= 0:
        return 0.0
    dark = sum(s["end"] - s["start"] for s in mapped["sections"]
               if s["label"] == MASKED)
    return dark / total


# --------------------------------------------------------------------------- #
# Spot-check staging (the labelling tool's working format)
# --------------------------------------------------------------------------- #


def seed_lines(mapped: dict) -> list:
    """Boundaries at each labelled section start, in label_tool's CSV shape.

    The tool's vocabulary cannot say "no truth here", so a masked span gets no
    boundary and visually inherits its predecessor; a track that opens masked
    gets a 0.0 ``intro`` placeholder so the file starts where the tool expects.
    """
    lines = [(float(s["start"]), s["label"]) for s in mapped["sections"]
             if s["label"] != MASKED]
    if not lines:
        return []
    lines.sort()
    if lines[0][0] <= 0.5:
        lines[0] = (0.0, lines[0][1])
    else:
        lines.insert(0, (0.0, "intro"))
    return [f"{start:.3f},{label}" for start, label in lines]


def stage_spot_check(picks: list, corpus: Path) -> tuple:
    tmp = corpus / "tmp_labels"
    tmp.mkdir(parents=True, exist_ok=True)
    seeded, skipped = [], []
    for pick in picks:
        audio = Path(pick["audio_path"])
        target = tmp / f"{audio.name}.labels.csv"
        if target.exists():
            skipped.append((pick["id"], str(target),
                            "already exists -- not overwritten"))
            continue
        lines = seed_lines(pick["mapped"])
        if not lines:
            skipped.append((pick["id"], str(target),
                            "fully masked -- nothing to seed"))
            continue
        target.write_text("\n".join(lines) + "\n", encoding="utf-8",
                          newline="\n")
        seeded.append(pick)
    return seeded, skipped


def choose_sample(stats: list, per_bucket: int = 2) -> list:
    """Per source: drop-heaviest, breakdown-heaviest, masked-heaviest tracks.

    A fully-masked track has no draft to audit, so only seedable tracks
    qualify -- the heavy-masked bucket wants the worst track that still shows
    the owner something to correct.
    """
    picks = []
    for source in tp_record.THIRD_PARTY_SOURCES:
        rows = [s for s in stats
                if s["source"] == source and seed_lines(s["mapped"])]
        chosen = []
        for key, bucket in (("drops", "chorus_to_drop"),
                            ("breakdowns", "chorus_to_breakdown"),
                            ("masked_frac", "heavy_masked")):
            taken = 0
            for row in sorted(rows, key=lambda s: (-s[key], s["id"])):
                if taken >= per_bucket:
                    break
                if row["id"] in {c["id"] for c in chosen}:
                    continue
                chosen.append(dict(row, bucket=bucket))
                taken += 1
        picks.extend(chosen)
    return picks


# --------------------------------------------------------------------------- #
# The batch
# --------------------------------------------------------------------------- #


def _below_normal_priority():
    if sys.platform != "win32":
        return
    import ctypes
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)


def admitted_rows(corpus: Path) -> list:
    manifest = tp_record.third_party_dir(corpus) / "clean_manifest.csv"
    with open(manifest, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["status"] == "ok"]


def load_feature_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("tracks", {})
    except ValueError:
        return {}


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(path)


def feature_pass(rows: list, records: dict, cache: dict, workers: int) -> tuple:
    fresh, skips = {}, []

    def work(row):
        track_id = row["track_id"]
        audio = Path(row["audio_path"])
        record = records[track_id]
        if not audio.exists():
            return track_id, None, f"audio missing: {audio}"
        sections = sorted(record["sections"], key=lambda s: float(s["start"]))
        held = cache.get(track_id)
        if held and held.get("audio_size") == audio.stat().st_size \
                and held.get("audio_mtime_ns") == audio.stat().st_mtime_ns \
                and held.get("spans") == _spans_of(sections):
            return track_id, held, None
        try:
            return track_id, measure_track(record, audio), None
        except Exception as exc:
            return track_id, None, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (track_id, features, reason) in enumerate(
                pool.map(work, rows), start=1):
            if reason:
                skips.append({"id": track_id, "reason": reason})
                print(f"[{index}/{len(rows)}] SKIP {track_id}: {reason}")
            else:
                fresh[track_id] = features
                if index % 50 == 0 or index == len(rows):
                    print(f"[{index}/{len(rows)}] measured through {track_id}")
    return fresh, skips


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage-spot-check", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args(argv)

    _below_normal_priority()
    corpus = corpus_dir()
    tp = tp_record.third_party_dir(corpus)
    ann = annotations_dir(corpus)

    rows = admitted_rows(corpus)
    if args.limit:
        rows = rows[: args.limit]
    records = {}
    for row in rows:
        path = ann / f"{row['track_id']}.{row['source']}.json"
        records[row["track_id"]] = tp_record.read_record(path)
    print(f"{len(rows)} admitted tracks to map")

    features_path = tp / "mapped_features.json"
    cache = {} if args.force_features else load_feature_cache(features_path)
    features, skips = feature_pass(rows, records, cache, args.workers)
    write_json(features_path, {
        "schema": 1,
        "decode": f"ffmpeg_f32le_{DECODE_RATE}_mono",
        "entry_window_sec": ENTRY_WINDOW_SEC,
        "tracks": features})

    chorus_entries = {}
    for row in rows:
        track_id = row["track_id"]
        if track_id not in features:
            continue
        sections = sorted(records[track_id]["sections"],
                          key=lambda s: float(s["start"]))
        for span, energy in zip(sections, features[track_id]["sections"]):
            if normalise_label(span["name"]) not in CHORUS_FAMILY:
                continue
            if energy["pre_db"] is None or energy["head_db"] is None:
                continue
            chorus_entries.setdefault(row["source"], []).append(
                round(energy["head_db"] - energy["pre_db"], 3))

    thresholds = derive_thresholds(chorus_entries)
    write_json(tp / "mapped_thresholds.json", thresholds)
    threshold_db = thresholds["chorus_drop_threshold_db"]
    print(f"registered chorus drop threshold: entry contrast >= "
          f"{threshold_db} dB (pooled Q3, "
          f"n={thresholds['chorus_entry_contrast_db']['pooled']['n']})")

    stats, in_counts, out_counts, rule_counts = [], {}, {}, {}
    for row in rows:
        track_id = row["track_id"]
        if track_id not in features:
            continue
        mapped = map_track(records[track_id], features[track_id], threshold_db)
        write_json(ann / f"{track_id}{MAPPED_SUFFIX}", mapped)
        source = row["source"]
        drops = breakdowns = 0
        for section in mapped["sections"]:
            evidence = section["evidence"]
            in_counts.setdefault(source, {}).setdefault(evidence["canonical"], 0)
            in_counts[source][evidence["canonical"]] += 1
            out_counts.setdefault(source, {}).setdefault(section["label"], 0)
            out_counts[source][section["label"]] += 1
            rule_counts.setdefault(source, {}).setdefault(evidence["rule"], 0)
            rule_counts[source][evidence["rule"]] += 1
            if evidence["rule"] == RULE_CHORUS_DROP:
                drops += 1
            elif evidence["rule"] in (RULE_CHORUS_BREAKDOWN,
                                      RULE_CHORUS_DEFAULT):
                breakdowns += 1
        stats.append({
            "id": track_id, "source": source,
            "audio_path": row["audio_path"], "mapped": mapped,
            "drops": drops, "breakdowns": breakdowns,
            "masked_frac": round(masked_fraction(mapped), 4)})

    late_quiet = sum(
        1 for s in stats for section in s["mapped"]["sections"]
        if section["evidence"]["rule"] == RULE_CHORUS_BREAKDOWN
        and s["mapped"]["duration"]
        and float(section["start"]) / float(s["mapped"]["duration"]) >= 0.7)

    sources = sorted({s["source"] for s in stats})
    summary = {
        "schema": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tracks_admitted": len(rows),
        "tracks_mapped": len(stats),
        "tracks_skipped": skips,
        "thresholds": thresholds,
        "labels_in_by_source": in_counts,
        "labels_out_by_source": out_counts,
        "rules_by_source": rule_counts,
        "chorus_split": {
            source: {
                "drop": rule_counts.get(source, {}).get(RULE_CHORUS_DROP, 0),
                "breakdown":
                    rule_counts.get(source, {}).get(RULE_CHORUS_BREAKDOWN, 0)
                    + rule_counts.get(source, {}).get(RULE_CHORUS_DEFAULT, 0)}
            for source in sources},
        "masked_fraction_by_source": {
            source: round(float(np.mean([s["masked_frac"] for s in stats
                                         if s["source"] == source])), 4)
            for source in sources},
        "noted_alternative": {
            "late_quiet_chorus_to_cooldown_candidates": late_quiet,
            "note": ("choruses mapped breakdown whose entry sits past 70% of "
                     "the track; a cooldown reading is defensible but "
                     "unmeasured against the show, so the conservative rule "
                     "ships and this is recorded only")},
    }
    write_json(tp / "mapped_summary.json", summary)
    write_summary_md(tp / "mapped_summary.md", summary)
    print(f"mapped {len(stats)} tracks; {len(skips)} skipped; "
          f"summary at {tp / 'mapped_summary.json'}")

    if args.stage_spot_check:
        picks = choose_sample(stats)
        seeded, skipped = stage_spot_check(picks, corpus)
        write_spot_check_md(tp / "spot_check_sample.md", seeded, skipped)
        print(f"spot-check: {len(seeded)} seeded, {len(skipped)} skipped; "
              f"doc at {tp / 'spot_check_sample.md'}")
    return 0


def write_summary_md(path: Path, summary: dict) -> None:
    lines = ["# Raw-9 mapping summary (drafts, quarantined)", "",
             f"Generated {summary['generated_utc']} -- "
             f"{summary['tracks_mapped']} of {summary['tracks_admitted']} "
             f"admitted tracks mapped, {len(summary['tracks_skipped'])} skipped.",
             "",
             "## Registered threshold",
             "",
             f"chorus -> drop iff entry contrast >= "
             f"**{summary['thresholds']['chorus_drop_threshold_db']} dB** "
             f"({summary['thresholds']['basis']}).", "",
             "Measured chorus entry-contrast distribution (dB):", ""]
    for source, dist in summary["thresholds"]["chorus_entry_contrast_db"].items():
        lines.append(f"- {source}: {json.dumps(dist)}")
    lines += ["", "## Chorus split", ""]
    for source, split in summary["chorus_split"].items():
        lines.append(f"- {source}: drop {split['drop']}, "
                     f"breakdown {split['breakdown']}")
    lines += ["", "## Masked fraction (mean per track)", ""]
    for source, frac in summary["masked_fraction_by_source"].items():
        lines.append(f"- {source}: {frac:.1%}")
    lines += ["", "## Labels out", ""]
    for source, counts in summary["labels_out_by_source"].items():
        ordered = ", ".join(f"{label} {count}" for label, count in
                            sorted(counts.items(), key=lambda kv: -kv[1]))
        lines.append(f"- {source}: {ordered}")
    if summary["tracks_skipped"]:
        lines += ["", "## Skipped", ""]
        for skip in summary["tracks_skipped"]:
            lines.append(f"- {skip['id']}: {skip['reason']}")
    lines += ["", summary["noted_alternative"]["note"] + " (count: "
              + str(summary["noted_alternative"]
                    ["late_quiet_chorus_to_cooldown_candidates"]) + ")", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_spot_check_md(path: Path, seeded: list, skipped: list) -> None:
    lines = ["# Spot-check sample -- raw-9 drafts staged for the owner's audit",
             "",
             "Each track below has its mapped draft seeded into "
             "`<corpus>/tmp_labels/` in the labelling tool's working format. "
             "Open one with the command shown (from the main checkout); "
             "correct and commit as usual.",
             "",
             "The tool's vocabulary cannot express a masked span, so masked "
             "spans (verse/solo/inst/...) carry NO boundary in the seed and "
             "visually inherit the previous section's label; a track opening "
             "masked starts with a placeholder `intro` at 0.0. The full masked "
             "spans are in the track's `annotations/<id>.mapped.json`.", ""]
    for pick in seeded:
        mapped = pick["mapped"]
        masked_spans = [f"{s['start']:.1f}-{s['end']:.1f}s "
                        f"({s['evidence']['source_label']})"
                        for s in mapped["sections"] if s["label"] == MASKED]
        lines += [f"## {pick['id']} ({pick['source']}, {pick['bucket']})", "",
                  f"- audio: `{pick['audio_path']}`",
                  f"- chorus->drop {pick['drops']}, chorus->breakdown "
                  f"{pick['breakdowns']}, masked {pick['masked_frac']:.0%}",
                  "- masked spans needing the owner's ear: "
                  + ("; ".join(masked_spans) if masked_spans else "none"),
                  f"- open: `python auto_pilot label \"{pick['audio_path']}\"`",
                  ""]
    if skipped:
        lines += ["## Not seeded", ""]
        for track_id, target, reason in skipped:
            lines.append(f"- {track_id}: {reason} ({target})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
