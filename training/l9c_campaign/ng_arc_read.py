"""#348: the #344 arc read over every banked probe report, in one table.

The per-chain probe reads are prose tables, one file per (chain, track).  What
the trade-off table needs is one row per chain: how many buildup->drop ARCS the
chain entered and held, and whether any drop the l9b baseline lands moved
outside the strict +-1.5 s clause.  Both are recomputed from the report JSONs
rather than parsed out of the markdown, so the arithmetic is the same on every
chain including the ones minted tonight.

An ARC is a labeled buildup section whose next labeled section is a drop.  It
PASSES the #344 floor when the chain is in BUILDUP over the labeled climb AND
the DROP the drop section is lit with was immediately preceded by BUILDUP --
"entered and held into the drop", the owner's clause, with the entry lag
reported beside it and never charged.

The DROP-STRICT clause is separate and reported loudly: for each labeled drop
the baseline chain lands within +-1.5 s, the candidate must also land within
+-1.5 s of the LABEL.  A baseline-landed drop the candidate loses is a DQ fact.

TWO READINGS, both emitted, because the campaign banked digits from the first
one.  ``first_entry_*`` is that first reading: a block STARTING inside the
window.  It is wrong, and #348's opus row is the proof -- a chain that entered
DROP at 292.8 s and held it across the whole 342.3-386.6 s labeled drop starts
no block in the window and was scored as having lost a drop it lit completely.
Intent blocks are SPANS: a block holds until the next block's ``song_t``, and
the last block in a report holds to the end.

``in_force_*`` is the state reading that replaces it, and its primary quantity
is SECONDS, not a boolean.  The opposite error is equally cheap: NMK on opus
and BC/NIWDW15 on inyathi each left a labeled drop dark for tens of seconds
while committing a DROP block on the label, so "in force anywhere in the span"
would pass a chain that lit one second of forty-four.  Every row therefore
carries seconds/fraction/dark-seconds over the LABELED span, and the boolean
``in_force_kept`` is those seconds against KEPT_FRACTION -- a stated reporting
convention, not a measured threshold, which is why the seconds sit beside it
for a caller that wants its own.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MAIN = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot")
DATA = MAIN / "training" / "data" / "raveform"
CAMP = DATA / "models" / "l9c_campaign"
L9B_CAMP = DATA / "models" / "l9b_campaign"

LABEL_IDS = {"opus": "hand-65cb8c94812d", "dwmu": "hand-8339586c555a",
             "inyathi": "3TfteM8_2o8", "yai": "9CpH_AljPyk",
             "skylark": "doedi2MI-iM"}
SEARCH_LEAD_SEC = 8.0
STRICT_SEC = 1.5
# fraction of a labeled span that must be in force before the row calls the
# intent KEPT.  A reporting convention: the seconds are printed beside it.
KEPT_FRACTION = 0.75
EPSILON_SEC = 0.01
# the l9b/H baseline read per track: opus and dwmu were probed in the l9b
# campaign as chain H; the three fit tracks were probed here as "shipped",
# which resolves the same l9b generation.
BASELINE = {"opus": L9B_CAMP / "probe_reports" / "H_opus.json",
            "dwmu": L9B_CAMP / "probe_reports" / "H_dwmu.json",
            "inyathi": CAMP / "probe_reports" / "shipped_inyathi.json",
            "yai": CAMP / "probe_reports" / "shipped_yai.json",
            "skylark": CAMP / "probe_reports" / "shipped_skylark.json"}


def sections(track_key):
    record = json.loads((DATA / "annotations"
                         / f"{LABEL_IDS[track_key]}.hand.json")
                        .read_text(encoding="utf-8"))
    return [(s["name"], float(s["start"]), float(s["end"]))
            for s in record["sections"]]


def blocks(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    out = [b for b in report["intents"]
           if "song_t" in b and b.get("trigger") != "silence"]
    out.sort(key=lambda b: b["song_t"])
    return out


def first_entry(items, intent, lo, hi):
    return next((b for b in items
                 if lo <= b["song_t"] < hi and b["intent"] == intent), None)


def intent_at(items, when):
    current = None
    for block in items:
        if block["song_t"] <= when:
            current = block
        else:
            break
    return current


def block_end(items, index):
    # a block holds until the next one starts; the last one holds to the end.
    return (items[index + 1]["song_t"] if index + 1 < len(items)
            else math.inf)


def covering(items, intent, lo, hi):
    """Every (index, block) of `intent` overlapping [lo, hi), in order."""
    return [(index, block) for index, block in enumerate(items)
            if block["intent"] == intent
            and block["song_t"] < hi and block_end(items, index) > lo]


def run_start(items, index):
    """The first block of the contiguous same-intent run containing `index`."""
    intent = items[index]["intent"]
    while index > 0 and items[index - 1]["intent"] == intent:
        index -= 1
    return items[index]


def force_seconds(items, intent, lo, hi) -> float:
    """Seconds of [lo, hi) the chain spent in `intent`."""
    total = 0.0
    for index, block in enumerate(items):
        if block["intent"] != intent:
            continue
        opens, closes = max(block["song_t"], lo), min(block_end(items, index),
                                                      hi)
        if closes > opens:
            total += closes - opens
    return total


def state_read(items, intent, start, end) -> dict:
    """The state reading of one labeled span: how long `intent` was in force.

    The state reading needs no search lead -- "in force" already reaches
    backwards, and a block that opened inside the lead but was superseded
    before the label lit none of it.  So every field here is measured against
    the LABELED span, and the landing block is the start of the earliest run of
    `intent` that covers any of it.
    """
    blocks_covering = covering(items, intent, start, end)
    seconds = force_seconds(items, intent, start, end)
    span = end - start
    first = (run_start(items, blocks_covering[0][0]) if blocks_covering
             else None)
    before = (intent_at(items, first["song_t"] - EPSILON_SEC) if first
              else None)
    return {
        "in_force": first is not None,
        "in_force_seconds": round(seconds, 2),
        "in_force_dark_seconds": round(span - seconds, 2),
        "in_force_fraction": round(seconds / span, 3) if span > 0 else None,
        "in_force_kept": bool(span > 0 and seconds >= KEPT_FRACTION * span),
        "in_force_song_t": round(first["song_t"], 2) if first else None,
        # negative == the chain was already in this intent when the span
        # opened, which is a held entry rather than a missing one.
        "in_force_lag": round(first["song_t"] - start, 2) if first else None,
        "in_force_held_from_before_window": bool(first
                                                 and first["song_t"] < start),
        # any covering block satisfies the clause: taking only the first would
        # call a drop lost because an earlier block sat in the lead.
        "in_force_strict": any(abs(b["song_t"] - start) <= STRICT_SEC
                               for _index, b in blocks_covering),
        "in_force_preceded_by": before["intent"] if before else None,
        "_first_block": first,
    }


def arc_read(items, spans) -> dict:
    arcs, drops = [], []
    for index, (name, start, end) in enumerate(spans):
        if name != "drop":
            continue
        entry = first_entry(items, "drop", start - SEARCH_LEAD_SEC, end)
        before = (intent_at(items, entry["song_t"] - EPSILON_SEC)
                  if entry else None)
        state = state_read(items, "drop", start, end)
        drop_first = state.pop("_first_block")
        drop_before = state["in_force_preceded_by"]
        drops.append({
            "onset": round(start, 2),
            **state,
            "first_entry_landed": entry is not None,
            "first_entry_delta": (round(entry["song_t"] - start, 2)
                                  if entry else None),
            "first_entry_strict": bool(
                entry and abs(entry["song_t"] - start) <= STRICT_SEC),
            "first_entry_preceded_by": (before["intent"] if before else None),
        })
        previous = spans[index - 1] if index else None
        if previous is None or previous[0] != "buildup":
            continue
        _pname, pstart, pend = previous
        buildup = first_entry(items, "buildup", pstart - SEARCH_LEAD_SEC, pend)
        climb = state_read(items, "buildup", pstart, pend)
        climb_first = climb.pop("_first_block")
        arcs.append({
            "buildup_onset": round(pstart, 2),
            "drop_onset": round(start, 2),
            "in_force_entered": climb["in_force"],
            "in_force_entry_lag": climb["in_force_lag"],
            "in_force_entered_before_window":
                climb["in_force_held_from_before_window"],
            "in_force_seconds": climb["in_force_seconds"],
            "in_force_fraction": climb["in_force_fraction"],
            "in_force_kept": climb["in_force_kept"],
            "in_force_drop_lit": drop_first is not None,
            "in_force_drop_kept": drops[-1]["in_force_kept"],
            "in_force_held_into_drop": bool(climb_first and drop_first
                                            and drop_before == "buildup"),
            "first_entry_entered": buildup is not None,
            "first_entry_entry_lag": (round(buildup["song_t"] - pstart, 2)
                                      if buildup else None),
            "first_entry_drop_landed": entry is not None,
            "first_entry_held_into_drop": bool(
                buildup and before and before["intent"] == "buildup"),
        })
    return {
        "arcs": arcs, "drops": drops,
        "arcs_total": len(arcs),
        "arcs_entered_in_force": sum(1 for a in arcs
                                     if a["in_force_entered"]),
        "arcs_held_in_force": sum(1 for a in arcs
                                  if a["in_force_held_into_drop"]),
        "arcs_entered_first_entry": sum(1 for a in arcs
                                        if a["first_entry_entered"]),
        "arcs_held_first_entry": sum(1 for a in arcs
                                     if a["first_entry_held_into_drop"]),
        "drops_total": len(drops),
        "drops_kept_in_force": sum(1 for d in drops if d["in_force_kept"]),
        "drops_landed_first_entry": sum(1 for d in drops
                                        if d["first_entry_landed"]),
    }


def track_read(path: Path, track_key: str) -> dict:
    return arc_read(blocks(path), sections(track_key))


def report_path(chain: str, key: str) -> Path:
    # the l9b baseline's own arc row: opus/dwmu were probed in the l9b campaign
    # as chain H and the three fit tracks here as "shipped", so the baseline
    # reads from BASELINE rather than from one campaign's probe_reports.
    if chain == "H":
        return BASELINE[key]
    return CAMP / "probe_reports" / f"{chain}_{key}.json"


def chain_read(chain: str, tracks) -> dict:
    per_track, missing = {}, []
    for key in tracks:
        path = report_path(chain, key)
        if not path.exists():
            missing.append(key)
            continue
        per_track[key] = track_read(path, key)

    lost_in_force, lost_first_entry, underlit = [], [], []
    for key, read in per_track.items():
        base = BASELINE.get(key)
        if base is None or not base.exists():
            continue
        baseline = track_read(base, key)
        for mine, theirs in zip(read["drops"], baseline["drops"]):
            if theirs["in_force_strict"] and not mine["in_force_strict"]:
                lost_in_force.append(
                    {"track": key, "onset": theirs["onset"],
                     "baseline_lag": theirs["in_force_lag"],
                     "candidate_lag": mine["in_force_lag"]})
            if theirs["first_entry_strict"] and not mine["first_entry_strict"]:
                lost_first_entry.append(
                    {"track": key, "onset": theirs["onset"],
                     "baseline_delta": theirs["first_entry_delta"],
                     "candidate_delta": mine["first_entry_delta"]})
            # the drop the baseline lit and the candidate leaves dark: the
            # failure the strict clause is structurally blind to, because a
            # chain can commit DROP on the label and abandon it a second later.
            if theirs["in_force_kept"] and not mine["in_force_kept"]:
                underlit.append(
                    {"track": key, "onset": theirs["onset"],
                     "baseline_dark_seconds": theirs["in_force_dark_seconds"],
                     "candidate_dark_seconds": mine["in_force_dark_seconds"]})

    def total(field):
        return sum(r[field] for r in per_track.values())

    return {
        "chain": chain,
        "tracks": sorted(per_track),
        "missing_tracks": missing,
        "arcs_total": total("arcs_total"),
        "arcs_entered_in_force": total("arcs_entered_in_force"),
        "arcs_held_in_force": total("arcs_held_in_force"),
        "arcs_entered_first_entry": total("arcs_entered_first_entry"),
        "arcs_held_first_entry": total("arcs_held_first_entry"),
        "drops_total": total("drops_total"),
        "drops_kept_in_force": total("drops_kept_in_force"),
        "drops_landed_first_entry": total("drops_landed_first_entry"),
        "drops_lost_vs_baseline_in_force": lost_in_force,
        "drops_lost_vs_baseline_first_entry": lost_first_entry,
        "drops_underlit_vs_baseline": underlit,
        "per_track": per_track,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chains", default=None,
                        help="comma list; default = every chain with a report")
    parser.add_argument("--tracks", default="opus,dwmu,inyathi,yai,skylark")
    parser.add_argument("--out", type=Path, default=CAMP / "ARCS_348.json")
    args = parser.parse_args()

    tracks = [t for t in args.tracks.split(",") if t]
    if args.chains:
        chains = [c for c in args.chains.split(",") if c]
    else:
        chains = ["H"] + sorted({path.name.rsplit("_", 1)[0] for path in
                                 (CAMP / "probe_reports").glob("*.json")
                                 if path.stem.rsplit("_", 1)[-1] in LABEL_IDS})

    rows = {}
    # both readings on every line: the state one is the answer, the first-entry
    # one is what the banked artifacts carry, and printing one without the
    # other is how a banked digit gets compared against a different quantity.
    print(f"{'chain':<10} {'arcs':>5}  {'entered':>7} {'held':>4}  "
          f"{'(1st: ent':>9} {'held)':>5}  {'drops':>5} {'kept':>4}  "
          f"drops lost / under-lit vs l9b baseline")
    for chain in chains:
        read = chain_read(chain, tracks)
        rows[chain] = read
        note = " / ".join(
            ("none" if not lost else
             ", ".join(f"{d['track']}@{d['onset']}s" for d in lost))
            for lost in (read["drops_lost_vs_baseline_in_force"],
                         read["drops_underlit_vs_baseline"]))
        print(f"{chain:<10} {read['arcs_total']:>5}  "
              f"{read['arcs_entered_in_force']:>7} "
              f"{read['arcs_held_in_force']:>4}  "
              f"{read['arcs_entered_first_entry']:>9} "
              f"{read['arcs_held_first_entry']:>5}  "
              f"{read['drops_total']:>5} {read['drops_kept_in_force']:>4}  "
              f"{note}")

    (args.out).write_text(json.dumps({
        "label": "#348 arc read: buildup->drop arcs entered/held per chain, "
                 "plus the strict +-1.5 s drop clause against the l9b "
                 "baseline",
        "readings": {
            "in_force_*": "the state reading: seconds of the LABELED span the "
                          "intent was in force, a block holding until the "
                          "next "
                          "block's song_t. in_force_kept is those seconds "
                          "against kept_fraction.",
            "first_entry_*": "the superseded reading a block STARTING inside "
                             "the window, kept verbatim because the campaign "
                             "banked its digits. It scores a held intent as "
                             "absent and an abandoned one as landed.",
        },
        "search_lead_sec": SEARCH_LEAD_SEC, "strict_sec": STRICT_SEC,
        "kept_fraction": KEPT_FRACTION,
        "baselines": {k: str(v) for k, v in BASELINE.items()},
        "chains": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
