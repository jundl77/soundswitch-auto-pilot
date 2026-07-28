#!/usr/bin/env python
"""Build the raveform track manifest and print the label statistics.

Consumes ``<data-dir>/annotations/`` (written by ``raveform_fetch_annotations.py``)
and writes ``<data-dir>/manifest.csv`` -- one row per annotated track::

    track_id,youtube_id,n_sections,total_sec

``track_id`` is the dataset's ``key`` (``<index>.<youtube_id>``, also the beat-CSV
basename), ``youtube_id`` is the 11-character YouTube video ID the downloader
fetches, ``n_sections`` is the RAW published section count, and ``total_sec`` is
the record's ``duration`` field (bit-equal to ``sections[-1].end`` on every track).

Two views of the same annotation are reported, because they answer different
questions:

* **RAW** -- labels exactly as published, sections exactly as published. This is
  what the manifest records and what any claim about the dataset must cite.
  Note that the corpus does *not* merge adjacent same-label sections, so raw
  section counts and median durations describe annotation events, not musical
  sections.
* **CANONICAL** -- the training vocabulary. The published vocabulary is a
  superset of the documented seven labels; the extras are folded per the
  project's ruling (see ``CANONICAL_*`` below), and adjacent same-label runs are
  then merged so a "section" means one contiguous stretch of one label. These
  are the numbers that preview the HSMM duration and transition priors.

Stdlib only.  Reuses the Task 2 parse helpers rather than re-parsing the JSON.

Usage::

    uv run python training/raveform/raveform_manifest.py \
        --data-dir C:\\Users\\Julian\\Projects\\soundswitch-auto-pilot\\training\\data\\raveform
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raveform_fetch_annotations import (  # noqa: E402  (needs the path insert above)
    annotations_dir,
    load_tracks,
    parse_sections,
    youtube_id,
)

MANIFEST_FILE = "manifest.csv"
MANIFEST_HEADER = ("track_id", "youtube_id", "n_sections", "total_sec")

# --------------------------------------------------------------------------- #
# Canonical label vocabulary
# --------------------------------------------------------------------------- #
#
# The published vocabulary has ten labels; the documented seven plus `end`,
# `altintro` and `bridge`.  Project ruling (recorded in the Task 2 report):
#
#   end       tail sentinel, not a musical section -- median 4.8 s and always
#             the final section.  DROPPED entirely; it would otherwise teach a
#             spurious 1,249-example class.
#   altintro  variant marker for the same structural role as `intro` (mirror of
#             the already-documented `altoutro`).  Folded into `intro`.
#   bridge    conventional mid-track section, only 54 examples -- too few to
#             learn as its own class.  Folded into `breakdown`.
#   altoutro  KEPT: already part of the documented seven-label vocabulary.
#
CANONICAL_DROP = frozenset({"end"})
CANONICAL_MAP = {"altintro": "intro", "bridge": "breakdown"}

# Musical order, for stable and readable table/matrix rows.
CANONICAL_ORDER = (
    "intro",
    "buildup",
    "drop",
    "breakdown",
    "cooldown",
    "outro",
    "altoutro",
)


# --------------------------------------------------------------------------- #
# Section helpers
# --------------------------------------------------------------------------- #


def section_length(start: float, end: float) -> float:
    """Length of one section, clamped at 0.

    One track (``1020.c1VBubZ2w3M``) has a final section whose ``start`` exceeds
    its ``end`` by 0.6 ms.  Clamping keeps every aggregate well-defined instead
    of letting a sub-millisecond annotation slip subtract from a total; the
    anomaly is reported separately rather than silently absorbed.
    """
    return max(0.0, end - start)


def canonical_runs(sections: list) -> list:
    """RAW sections -> canonical merged runs ``[(start, end, label, duration)]``.

    Drops the dropped labels, applies the fold map, then merges adjacent
    same-label runs.  Merging happens *after* the drop and the fold, so runs
    join across a removed sentinel and across a folded variant
    (``altintro`` + ``intro`` is one ``intro``).

    A merged run's duration is the SUM of its members' clamped lengths, not
    ``end - start``: those differ exactly when a dropped section sits between
    two merged members, and the dropped time must not be re-attributed.
    """
    runs: list = []
    for start, end, label in sections:
        if label in CANONICAL_DROP:
            continue
        label = CANONICAL_MAP.get(label, label)
        duration = section_length(start, end)
        if runs and runs[-1][2] == label:
            runs[-1][1] = end
            runs[-1][3] += duration
        else:
            runs.append([start, end, label, duration])
    return [tuple(run) for run in runs]


def raw_runs(sections: list) -> list:
    """RAW sections in the same ``(start, end, label, duration)`` shape."""
    return [(start, end, label, section_length(start, end)) for start, end, label in sections]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _percentile(values_sorted: list, pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty list."""
    if not values_sorted:
        return float("nan")
    if len(values_sorted) == 1:
        return values_sorted[0]
    rank = (len(values_sorted) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(values_sorted) - 1)
    return values_sorted[low] + (values_sorted[high] - values_sorted[low]) * (rank - low)


def label_stats(per_track_runs: list) -> dict:
    """Per-label section count, track count, total seconds and duration list.

    ``per_track_runs`` is one ``[(start, end, label, duration), ...]`` list per
    track.
    """
    counts = collections.Counter()
    totals = collections.defaultdict(float)
    tracks = collections.defaultdict(set)
    durations = collections.defaultdict(list)
    for index, runs in enumerate(per_track_runs):
        for _start, _end, label, duration in runs:
            counts[label] += 1
            totals[label] += duration
            tracks[label].add(index)
            durations[label].append(duration)
    return {
        label: {
            "count": counts[label],
            "tracks": len(tracks[label]),
            "total_sec": totals[label],
            "durations": durations[label],
        }
        for label in counts
    }


def transition_counts(per_track_runs: list) -> collections.Counter:
    """Count every adjacent ``from -> to`` label pair, within tracks only."""
    pairs = collections.Counter()
    for runs in per_track_runs:
        for before, after in zip(runs, runs[1:]):
            pairs[(before[2], after[2])] += 1
    return pairs


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def _print_label_table(stats: dict, order: list) -> None:
    print(
        f"  {'label':<12}{'sections':>10}{'tracks':>8}"
        f"{'total_h':>10}{'median_s':>10}{'mean_s':>9}{'max_s':>9}"
    )
    total_sections = 0
    total_seconds = 0.0
    for label in order:
        entry = stats.get(label)
        if entry is None:
            continue
        durations = entry["durations"]
        total_sections += entry["count"]
        total_seconds += entry["total_sec"]
        print(
            f"  {label:<12}{entry['count']:>10}{entry['tracks']:>8}"
            f"{entry['total_sec'] / 3600.0:>10.1f}"
            f"{statistics.median(durations):>10.1f}"
            f"{entry['total_sec'] / entry['count']:>9.1f}"
            f"{max(durations):>9.1f}"
        )
    print(f"  {'TOTAL':<12}{total_sections:>10}{'':>8}{total_seconds / 3600.0:>10.1f}")


def _print_transitions(pairs: collections.Counter, order: list, title: str) -> None:
    print(title)
    width = max(6, max((len(label) for label in order), default=6))
    corner = "from \\ to".ljust(12)  # f-strings below cannot carry the backslash
    print("  " + corner + "".join(f"{label:>{width + 2}}" for label in order))
    for source in order:
        row = f"  {source:<12}"
        for target in order:
            count = pairs.get((source, target), 0)
            row += f"{count if count else '.':>{width + 2}}"
        print(row)
    self_pairs = sum(count for (a, b), count in pairs.items() if a == b)
    print(f"  total pairs: {sum(pairs.values())}   self-transitions (X->X): {self_pairs}")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def build_manifest_rows(tracks: list) -> list:
    """One ``(track_id, youtube_id, n_sections, total_sec)`` row per track.

    Sorted by ``track_id`` so the file is byte-identical run to run regardless
    of the order the records happen to appear in.
    """
    rows = []
    for track in tracks:
        sections = parse_sections(track)
        rows.append(
            (
                str(track["key"]),
                youtube_id(track),
                len(sections),
                f"{float(track['duration']):.3f}",
            )
        )
    rows.sort(key=lambda row: row[0])
    return rows


def write_manifest(data_dir: Path, rows: list) -> Path:
    """Write ``manifest.csv`` atomically; returns its path."""
    path = data_dir / MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(MANIFEST_HEADER)
            writer.writerows(rows)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def report(data_dir: Path, tracks: list) -> None:
    """Print every statistic the plan asks for. Never raises on bad data."""
    per_track_raw = []
    per_track_canonical = []
    lead_ins = []
    negatives = []
    short_sections = collections.Counter()
    duration_mismatch = 0
    max_duration_delta = 0.0
    boundary_breaks = 0
    empty_tracks = 0

    for track in tracks:
        sections = parse_sections(track)
        key = str(track["key"])
        if not sections:
            empty_tracks += 1
            per_track_raw.append([])
            per_track_canonical.append([])
            continue

        for index, (start, end, label) in enumerate(sections):
            if end < start:
                negatives.append((key, index + 1, len(sections), label, start, end))
            elif end - start < 0.5:
                short_sections[label] += 1
        for (_s0, end0, _l0), (start1, _e1, _l1) in zip(sections, sections[1:]):
            if end0 != start1:
                boundary_breaks += 1

        delta = abs(float(track["duration"]) - sections[-1][1])
        max_duration_delta = max(max_duration_delta, delta)
        if delta != 0.0:
            duration_mismatch += 1

        lead_ins.append(sections[0][0])
        per_track_raw.append(raw_runs(sections))
        per_track_canonical.append(canonical_runs(sections))

    raw_stats = label_stats(per_track_raw)
    canonical_stats = label_stats(per_track_canonical)
    raw_sections = sum(entry["count"] for entry in raw_stats.values())
    canonical_sections = sum(entry["count"] for entry in canonical_stats.values())
    raw_seconds = sum(entry["total_sec"] for entry in raw_stats.values())
    canonical_seconds = sum(entry["total_sec"] for entry in canonical_stats.values())
    track_seconds = sum(float(track["duration"]) for track in tracks)
    per_track_counts = sorted(len(runs) for runs in per_track_raw)

    raw_pairs = transition_counts(per_track_raw)
    canonical_pairs = transition_counts(per_track_canonical)
    raw_adjacent_same = sum(count for (a, b), count in raw_pairs.items() if a == b)
    dropped_sections = sum(
        1
        for runs in per_track_raw
        for _s, _e, label, _d in runs
        if label in CANONICAL_DROP
    )

    # --- corpus ----------------------------------------------------------- #
    print()
    print("corpus")
    print(f"  tracks                 : {len(tracks)}")
    print(f"  total track duration   : {track_seconds / 3600.0:.1f} h  (sum of the 'duration' field)")
    print(f"  annotated section time : {raw_seconds / 3600.0:.1f} h  (raw sections)")
    print(f"  canonical section time : {canonical_seconds / 3600.0:.1f} h  ('end' sentinel removed)")
    print(
        f"  sections per track     : min {per_track_counts[0]}  "
        f"median {statistics.median(per_track_counts):.0f}  "
        f"mean {statistics.mean(per_track_counts):.1f}  max {per_track_counts[-1]}  (raw)"
    )

    # --- raw --------------------------------------------------------------- #
    print()
    print("RAW label statistics -- labels exactly as published, sections NOT merged.")
    print(
        f"  {raw_adjacent_same} adjacent same-label pairs are counted here as separate"
        " sections, so"
    )
    print("  these medians describe annotation events, not contiguous musical sections.")
    raw_order = sorted(raw_stats, key=lambda label: -raw_stats[label]["count"])
    _print_label_table(raw_stats, raw_order)

    # --- canonical --------------------------------------------------------- #
    fold = ", ".join(f"{src}->{dst}" for src, dst in sorted(CANONICAL_MAP.items()))
    print()
    print(
        "CANONICAL label statistics -- "
        f"{'/'.join(sorted(CANONICAL_DROP))} dropped; {fold}; "
        "adjacent same-label runs merged."
    )
    print(
        f"  merge effect: {raw_sections} raw sections"
        f" -> {dropped_sections} dropped ('{'/'.join(sorted(CANONICAL_DROP))}')"
        f" -> {raw_sections - dropped_sections} remaining"
        f" -> {raw_sections - dropped_sections - canonical_sections} merged away"
        f" -> {canonical_sections} canonical sections"
    )
    canonical_order = [label for label in CANONICAL_ORDER if label in canonical_stats]
    unmapped = sorted(set(canonical_stats) - set(CANONICAL_ORDER))
    canonical_order += unmapped
    _print_label_table(canonical_stats, canonical_order)
    if unmapped:
        # Fail loud, not open: a label the ruling never considered has appeared,
        # and it is now silently a training class. Someone must decide on it.
        print(
            f"  WARNING: {len(unmapped)} label(s) outside the canonical vocabulary "
            f"passed through unmapped: {', '.join(unmapped)}"
        )

    # --- transitions ------------------------------------------------------- #
    print()
    _print_transitions(
        raw_pairs,
        raw_order,
        "RAW transition counts (adjacent published sections; self-transitions included)",
    )
    print()
    _print_transitions(
        canonical_pairs,
        canonical_order,
        "CANONICAL transition counts (after drop + fold + merge; X->X is 0 by construction)",
    )

    first_labels = collections.Counter(runs[0][2] for runs in per_track_canonical if runs)
    last_labels = collections.Counter(runs[-1][2] for runs in per_track_canonical if runs)
    print()
    print("  canonical first-section labels : " + _counter_line(first_labels))
    print("  canonical last-section labels  : " + _counter_line(last_labels))

    # --- leading offset ---------------------------------------------------- #
    lead_sorted = sorted(lead_ins)
    print()
    print("leading UNANNOTATED offset -- sections[0].start, audio before the first label")
    if not lead_sorted:
        print("  (no track has any section -- nothing to summarise)")
    else:
        print(
            f"  min {lead_sorted[0]:.3f}  p25 {_percentile(lead_sorted, 25):.3f}  "
            f"median {_percentile(lead_sorted, 50):.3f}  p75 {_percentile(lead_sorted, 75):.3f}  "
            f"p90 {_percentile(lead_sorted, 90):.3f}  p99 {_percentile(lead_sorted, 99):.3f}  "
            f"max {lead_sorted[-1]:.3f}  mean {statistics.mean(lead_sorted):.3f}"
        )
        for threshold in (0.5, 1.0, 5.0, 10.0):
            count = sum(1 for value in lead_sorted if value > threshold)
            print(f"  tracks with lead-in > {threshold:>4.1f} s : {count}")
        print(f"  tracks with lead-in == 0.0 s : {sum(1 for v in lead_sorted if v == 0.0)}")
        print(
            f"  total unannotated lead-in    : {sum(lead_sorted) / 60.0:.2f} min "
            "-- must NOT be attributed to the first section when slicing training audio"
        )

    # --- anomalies --------------------------------------------------------- #
    print()
    print("data anomalies")
    print(f"  tracks with no sections       : {empty_tracks}")
    print(f"  negative-length sections      : {len(negatives)} (clamped to 0 in every statistic)")
    for key, index, total, label, start, end in negatives:
        print(
            f"      {key}  section {index}/{total}  {label!r}  "
            f"start {start!r} > end {end!r}  ({end - start:+.6f} s)"
        )
    short_total = sum(short_sections.values())
    breakdown = ", ".join(f"{label} x{n}" for label, n in short_sections.most_common())
    print(f"  sections under 0.5 s          : {short_total}" + (f"  ({breakdown})" if breakdown else ""))
    print(
        f"  duration != sections[-1].end  : {duration_mismatch}/{len(tracks)} tracks "
        f"(max abs delta {max_duration_delta!r})"
    )
    print(f"  boundary gaps/overlaps        : {boundary_breaks}")

    ids = [youtube_id(track) for track in tracks]
    keys = [str(track["key"]) for track in tracks]
    print(f"  duplicate youtube ids         : {len(ids) - len(set(ids))}")
    print(f"  duplicate track ids           : {len(keys) - len(set(keys))}")
    print(f"  youtube ids not 11 chars      : {sum(1 for i in ids if len(i) != 11)}")


def _counter_line(counter: collections.Counter) -> str:
    return "  ".join(f"{label} {count}" for label, count in counter.most_common())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_data_dir() -> Path:
    # parents[2] is the repo root: this file sits in training/raveform/.
    return Path(__file__).resolve().parents[2] / "training" / "data" / "raveform"


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="corpus root; reads <data-dir>/annotations, writes <data-dir>/manifest.csv "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="print the statistics without writing manifest.csv",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    print("raveform manifest + label statistics")
    print(f"data dir: {data_dir}")
    print(f"annotations: {annotations_dir(data_dir)}")

    tracks = load_tracks(data_dir)
    if not tracks:
        raise RuntimeError(f"no track records in {annotations_dir(data_dir)} -- run the fetch first")

    rows = build_manifest_rows(tracks)
    if args.stats_only:
        print(f"manifest: not written (--stats-only); {len(rows)} rows would be emitted")
    else:
        path = write_manifest(data_dir, rows)
        print()
        print(f"manifest: {path}")
        print(f"  columns : {','.join(MANIFEST_HEADER)}")
        print(f"  rows    : {len(rows)}")
        print(f"  first   : {','.join(str(field) for field in rows[0])}")
        print(f"  last    : {','.join(str(field) for field in rows[-1])}")

    report(data_dir, tracks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
