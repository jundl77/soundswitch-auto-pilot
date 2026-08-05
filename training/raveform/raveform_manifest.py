#!/usr/bin/env python
"""Build the raveform track manifest and print the label statistics."""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.label_space import DROPPED_LABELS, SECTION_LABELS  # noqa: E402
from raveform_fetch_annotations import (  # noqa: E402
    annotations_dir,
    load_all_tracks,
    parse_sections,
    youtube_id,
)

MANIFEST_FILE = "manifest.csv"
MANIFEST_HEADER = ("track_id", "youtube_id", "n_sections", "total_sec")


def section_length(start: float, end: float) -> float:
    # One published track has start > end by 0.6 ms; clamping keeps aggregates
    # well-defined.  The anomaly is reported separately, not absorbed.
    return max(0.0, end - start)


def section_runs(sections: list) -> list:
    # A merged run's duration sums its members' clamped lengths rather than
    # taking end - start, so time inside a dropped section is not re-attributed.
    runs: list = []
    for start, end, label in sections:
        if label in DROPPED_LABELS:
            continue
        duration = section_length(start, end)
        if runs and runs[-1][2] == label:
            runs[-1][1] = end
            runs[-1][3] += duration
        else:
            runs.append([start, end, label, duration])
    return [tuple(run) for run in runs]


def raw_runs(sections: list) -> list:
    return [(start, end, label, section_length(start, end)) for start, end, label in sections]


def _percentile(values_sorted: list, pct: float) -> float:
    if not values_sorted:
        return float("nan")
    if len(values_sorted) == 1:
        return values_sorted[0]
    rank = (len(values_sorted) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(values_sorted) - 1)
    return values_sorted[low] + (values_sorted[high] - values_sorted[low]) * (rank - low)


def label_stats(per_track_runs: list) -> dict:
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
    pairs = collections.Counter()
    for runs in per_track_runs:
        for before, after in zip(runs, runs[1:]):
            pairs[(before[2], after[2])] += 1
    return pairs


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
    corner = "from \\ to".ljust(12)  # the f-strings below cannot carry a backslash
    print("  " + corner + "".join(f"{label:>{width + 2}}" for label in order))
    for source in order:
        row = f"  {source:<12}"
        for target in order:
            count = pairs.get((source, target), 0)
            row += f"{count if count else '.':>{width + 2}}"
        print(row)
    self_pairs = sum(count for (a, b), count in pairs.items() if a == b)
    print(f"  total pairs: {sum(pairs.values())}   self-transitions (X->X): {self_pairs}")


def build_manifest_rows(tracks: list) -> list:
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


def report(data_dir: Path, tracks: list) -> None:
    per_track_raw = []
    per_track_merged = []
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
            per_track_merged.append([])
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
        per_track_merged.append(section_runs(sections))

    raw_stats = label_stats(per_track_raw)
    merged_stats = label_stats(per_track_merged)
    raw_sections = sum(entry["count"] for entry in raw_stats.values())
    merged_sections = sum(entry["count"] for entry in merged_stats.values())
    raw_seconds = sum(entry["total_sec"] for entry in raw_stats.values())
    merged_seconds = sum(entry["total_sec"] for entry in merged_stats.values())
    track_seconds = sum(float(track["duration"]) for track in tracks)
    per_track_counts = sorted(len(runs) for runs in per_track_raw)

    raw_pairs = transition_counts(per_track_raw)
    merged_pairs = transition_counts(per_track_merged)
    raw_adjacent_same = sum(count for (a, b), count in raw_pairs.items() if a == b)
    dropped_sections = sum(
        1
        for runs in per_track_raw
        for _s, _e, label, _d in runs
        if label in DROPPED_LABELS
    )

    print()
    print("corpus")
    print(f"  tracks                 : {len(tracks)}")
    print(f"  total track duration   : {track_seconds / 3600.0:.1f} h  (sum of the 'duration' field)")
    print(f"  annotated section time : {raw_seconds / 3600.0:.1f} h  (raw sections)")
    print(f"  merged section time    : {merged_seconds / 3600.0:.1f} h  ('end' sentinel removed)")
    print(
        f"  sections per track     : min {per_track_counts[0]}  "
        f"median {statistics.median(per_track_counts):.0f}  "
        f"mean {statistics.mean(per_track_counts):.1f}  max {per_track_counts[-1]}  (raw)"
    )

    print()
    print("RAW label statistics -- labels exactly as published, sections NOT merged.")
    print(
        f"  {raw_adjacent_same} adjacent same-label pairs are counted here as separate"
        " sections, so"
    )
    print("  these medians describe annotation events, not contiguous musical sections.")
    raw_order = sorted(raw_stats, key=lambda label: -raw_stats[label]["count"])
    _print_label_table(raw_stats, raw_order)

    print()
    print(
        "MERGED label statistics -- "
        f"{'/'.join(sorted(DROPPED_LABELS))} dropped; "
        "adjacent same-label runs merged."
    )
    print(
        f"  merge effect: {raw_sections} raw sections"
        f" -> {dropped_sections} dropped ('{'/'.join(sorted(DROPPED_LABELS))}')"
        f" -> {raw_sections - dropped_sections} remaining"
        f" -> {raw_sections - dropped_sections - merged_sections} merged away"
        f" -> {merged_sections} merged sections"
    )
    merged_order = [label for label in SECTION_LABELS if label in merged_stats]
    unknown = sorted(set(merged_stats) - set(SECTION_LABELS))
    merged_order += unknown
    _print_label_table(merged_stats, merged_order)
    if unknown:
        print(
            f"  WARNING: {len(unknown)} label(s) outside the vocabulary "
            f"passed through: {', '.join(unknown)}"
        )

    print()
    _print_transitions(
        raw_pairs,
        raw_order,
        "RAW transition counts (adjacent published sections; self-transitions included)",
    )
    print()
    _print_transitions(
        merged_pairs,
        merged_order,
        "MERGED transition counts (after drop + fold + merge; X->X is 0 by construction)",
    )

    first_labels = collections.Counter(runs[0][2] for runs in per_track_merged if runs)
    last_labels = collections.Counter(runs[-1][2] for runs in per_track_merged if runs)
    print()
    print("  first-section labels : " + _counter_line(first_labels))
    print("  last-section labels  : " + _counter_line(last_labels))

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


def default_data_dir() -> Path:
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

    tracks = load_all_tracks(data_dir)
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
