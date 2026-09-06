"""Recover title-named SALAMI audio by joining filenames to SALAMI ids.

Two independent signals must agree before a file is adopted: a normalised
title match against `metadata.csv`, and the mp3's real decoded duration
against the SALAMI length. Originals are never moved or modified.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

TITLE_MIN = 0.90
MARGIN_MIN = 0.05
DUR_FLOOR = 10.0
DUR_FRAC = 0.03
TOP_K = 25
TITLE_DIRS = ("youtube_dl", "youtube_dl_mp3")
MEDIA_EXT = {
    ".mp3", ".wmv", ".mp4", ".m4a", ".webm", ".flv", ".avi", ".wav",
    ".ogg", ".mkv", ".aac", ".opus",
}
TRACK_NO = re.compile(r"^\d{1,3}\s*[-._)\]]*\s+")

# ffprobe is a subprocess per file; the box is shared.
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


class Meta(NamedTuple):
    song_id: str
    source: str
    klass: str
    title: str
    artist: str
    duration: float | None


class Variant(NamedTuple):
    song_id: str
    text: str
    tokens: frozenset
    chars: Counter


class Probe(NamedTuple):
    duration: float | None
    error: str | None


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalise(text: str) -> str:
    text = strip_accents(text.lower())
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return " ".join(text.split())


def strip_media_ext(name: str) -> str:
    stem = name
    while True:
        root, ext = os.path.splitext(stem)
        if ext.lower() in MEDIA_EXT and root:
            stem = root
            continue
        return stem


def file_forms(name: str) -> list:
    base = normalise(strip_media_ext(name))
    forms = [base]
    stripped = TRACK_NO.sub("", base).strip()
    if stripped and stripped != base:
        forms.append(stripped)
    return forms


def token_set_ratio(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def char_bound(a: Counter, b: Counter, la: int, lb: int) -> float:
    """Upper bound on SequenceMatcher.ratio (difflib's quick_ratio)."""
    if not la or not lb:
        return 0.0
    shared = sum((a & b).values())
    return 2.0 * shared / (la + lb)


def read_metadata(path: Path) -> dict:
    out: dict = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            song_id = (row.get("SONG_ID") or "").strip()
            if not song_id:
                continue
            raw = (row.get("SONG_DURATION") or "").strip()
            try:
                duration = float(raw) if raw else None
            except ValueError:
                duration = None
            out[song_id] = Meta(
                song_id=song_id,
                source=(row.get("SOURCE") or "").strip(),
                klass=(row.get("CLASS") or "").strip(),
                title=(row.get("SONG_TITLE") or "").strip(),
                artist=(row.get("ARTIST") or "").strip(),
                duration=duration,
            )
    return out


def read_pairing_lengths(path: Path) -> dict:
    if not path.is_file():
        return {}
    out: dict = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                out[row["salami_id"].strip()] = float(row["salami_length"])
            except (KeyError, ValueError, AttributeError):
                continue
    return out


def build_variants(meta: dict) -> list:
    out: list = []
    for song_id, row in meta.items():
        title = normalise(row.title)
        artist = normalise(row.artist)
        texts = {title}
        if artist:
            texts.add(f"{artist} {title}".strip())
            texts.add(f"{title} {artist}".strip())
        for text in texts:
            if not text:
                continue
            out.append(
                Variant(
                    song_id=song_id,
                    text=text,
                    tokens=frozenset(text.split()),
                    chars=Counter(text),
                )
            )
    return out


def score_file(forms: list, variants: list) -> dict:
    """Best and runner-up score per file, over all metadata ids."""
    form_data = [(f, frozenset(f.split()), Counter(f), len(f)) for f in forms]

    bounds: dict = {}
    for var in variants:
        lv = len(var.text)
        best = 0.0
        for _, tokens, chars, lf in form_data:
            bound = max(
                token_set_ratio(tokens, var.tokens),
                char_bound(chars, var.chars, lf, lv),
            )
            if bound > best:
                best = bound
        if bound_gt(best, bounds.get(var.song_id)):
            bounds[var.song_id] = best

    ranked = sorted(bounds.items(), key=lambda kv: -kv[1])
    top = [song_id for song_id, _ in ranked[:TOP_K]]
    top_set = set(top)

    exact: dict = {}
    matchers = {f: difflib.SequenceMatcher(None, "", f) for f in forms}
    for var in variants:
        if var.song_id not in top_set:
            continue
        best = 0.0
        for form, tokens, _, _ in form_data:
            matcher = matchers[form]
            matcher.set_seq1(var.text)
            value = max(
                token_set_ratio(tokens, var.tokens),
                matcher.ratio(),
            )
            if value > best:
                best = value
        if bound_gt(best, exact.get(var.song_id)):
            exact[var.song_id] = best

    # Outside the top-K only the (always >=) bound is known; using it as a
    # runner-up can only make acceptance stricter.
    merged = [(sid, exact.get(sid, val)) for sid, val in ranked]
    merged.sort(key=lambda kv: -kv[1])
    best_id, best_score = merged[0] if merged else (None, 0.0)
    runner = merged[1][1] if len(merged) > 1 else 0.0
    return {"id": best_id, "score": best_score, "runner_up": runner}


def bound_gt(value: float, current: float | None) -> bool:
    return current is None or value > current


def probe_duration(path: Path) -> Probe:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            creationflags=BELOW_NORMAL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Probe(None, f"ffprobe_failed: {exc}")
    text = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not text:
        return Probe(None, f"ffprobe_rc{proc.returncode}")
    try:
        return Probe(float(text[0]), None)
    except ValueError:
        return Probe(None, "ffprobe_unparsable")


def collect_candidates(audio_dir: Path) -> dict:
    groups: dict = {}
    for sub in TITLE_DIRS:
        directory = audio_dir / sub
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.mp3")):
            groups.setdefault(path.name, []).append(path)
    return groups


def tolerance(salami_length: float) -> float:
    return max(DUR_FLOOR, DUR_FRAC * salami_length)


def decide(record: dict, audio_dir: Path) -> None:
    score = record["title_score"]
    runner = record["runner_up_score"]
    song_id = record["salami_id"]

    if record["reason"]:
        record["decision"] = "REJECTED"
        return
    if song_id is None or score < TITLE_MIN:
        record["decision"] = "REJECTED"
        record["reason"] = "title_score_too_low"
        record["salami_id"] = None
        return
    if score - runner < MARGIN_MIN:
        record["decision"] = "REJECTED"
        record["reason"] = "ambiguous"
        return
    if (audio_dir / f"{song_id}.mp3").exists():
        record["decision"] = "REJECTED"
        record["reason"] = "id_already_on_disk"
        return
    if record["ffprobe_duration"] is None:
        record["decision"] = "REJECTED"
        record["reason"] = "ffprobe_failed"
        return
    if record["salami_length"] is None:
        record["decision"] = "REJECTED"
        record["reason"] = "no_salami_length"
        return
    if record["duration_delta"] > tolerance(record["salami_length"]):
        record["decision"] = "REJECTED"
        record["reason"] = "duration_mismatch"
        return
    record["decision"] = "ACCEPTED"
    record["reason"] = "title_and_duration_agree"


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("training/data/salami-data-public"),
    )
    parser.add_argument("--pairings", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    audio_dir = data_dir / "audio"
    meta = read_metadata(data_dir / "metadata" / "metadata.csv")
    pairing_len = read_pairing_lengths(args.pairings) if args.pairings else {}
    annotated = {
        p.name for p in (data_dir / "annotations").glob("*") if p.is_dir()
    }
    variants = build_variants(meta)

    groups = collect_candidates(audio_dir)
    all_paths = [p for paths in groups.values() for p in paths]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        probes = dict(zip(all_paths, pool.map(probe_duration, all_paths)))

    records: list = []
    for name, paths in sorted(groups.items()):
        chosen = paths[0]
        probe = probes[chosen]
        conflict = None
        if len(paths) > 1:
            seen = [probes[p].duration for p in paths]
            good = [d for d in seen if d is not None]
            if len(good) > 1 and max(good) - min(good) > 1.0:
                conflict = "duplicate_duration_conflict"

        forms = file_forms(name)
        hit = score_file(forms, variants)
        song_id = hit["id"]
        row = meta.get(song_id) if song_id else None
        length = None
        length_src = None
        if row is not None and row.duration:
            length, length_src = row.duration, "metadata"
        elif song_id and song_id in pairing_len:
            length, length_src = pairing_len[song_id], "pairings"

        delta = None
        if probe.duration is not None and length is not None:
            delta = abs(probe.duration - length)

        record = {
            "source_path": str(chosen),
            "duplicate_paths": [str(p) for p in paths[1:]],
            "file_name": name,
            "salami_id": song_id,
            "title_score": round(hit["score"], 4),
            "runner_up_score": round(hit["runner_up"], 4),
            "ffprobe_duration": (
                None if probe.duration is None else round(probe.duration, 2)
            ),
            "salami_length": length,
            "salami_length_source": length_src,
            "duration_delta": None if delta is None else round(delta, 2),
            "duration_tolerance": (
                None if length is None else round(tolerance(length), 2)
            ),
            "metadata_title": row.title if row else None,
            "metadata_artist": row.artist if row else None,
            "metadata_class": row.klass if row else None,
            "metadata_source": row.source if row else None,
            "has_annotation": bool(song_id) and song_id in annotated,
            "decision": None,
            "reason": conflict or probe.error,
        }
        decide(record, audio_dir)
        records.append(record)

    claims: dict = {}
    for record in records:
        if record["decision"] == "ACCEPTED":
            claims.setdefault(record["salami_id"], []).append(record)
    for song_id, group in claims.items():
        if len(group) > 1:
            for record in group:
                record["decision"] = "REJECTED"
                record["reason"] = "id_claimed_twice"

    copied = 0
    for record in records:
        if record["decision"] != "ACCEPTED":
            continue
        dest = audio_dir / f"{record['salami_id']}.mp3"
        record["dest_path"] = str(dest)
        if dest.exists():
            record["copied"] = False
            continue
        if args.dry_run:
            record["copied"] = False
            continue
        shutil.copy2(record["source_path"], dest)
        record["copied"] = True
        copied += 1

    reasons = Counter(
        r["reason"] for r in records if r["decision"] == "REJECTED"
    )
    accepted = [r for r in records if r["decision"] == "ACCEPTED"]
    payload = {
        "summary": {
            "files_scanned": len(all_paths),
            "unique_candidates": len(groups),
            "accepted": len(accepted),
            "copied": copied,
            "rejected": len(records) - len(accepted),
            "rejected_by_reason": dict(sorted(reasons.items())),
            "accepted_with_annotation": sum(
                1 for r in accepted if r["has_annotation"]
            ),
            "accepted_class_distribution": dict(
                sorted(Counter(r["metadata_class"] for r in accepted).items())
            ),
            "thresholds": {
                "title_min": TITLE_MIN,
                "margin_min": MARGIN_MIN,
                "duration_floor_sec": DUR_FLOOR,
                "duration_fraction": DUR_FRAC,
            },
            "dry_run": args.dry_run,
        },
        "candidates": records,
    }
    report = args.report or (data_dir / "title_join_report.json")
    write_json(report, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
