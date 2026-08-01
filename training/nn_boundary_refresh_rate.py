"""What the boundary head fires at, and how often -- D9's threshold evidence.

The retired YAMNet detector re-rolled the lighting effect on a section change,
and that refresh is behaviour no class boundary can express, because the class
is the same either side.  The boundary head replaces the trigger, and the rule
this project applies to a new threshold is: match a measured rate, do not pick
a number.

**The rate YAMNet produced was never measured and cannot be recovered.**
`simulate/runner.py` stubbed `detect_change` out from the day fast simulation
landed, so YAMNet never fired in a report, a fixture or a training table; no
metric for it ever existed in the report schema, and no commit in the history
carries one.  Production left an uncounted log line.  What the retired
constants do bracket is `cooldown_time_window_sec = 10` -- at most six
refreshes a minute, with a hard ten-second floor, and fewer in practice because
firing required a majority of one second to be a MAD outlier.

So the governor moves across verbatim and the threshold is chosen to land the
realised rate well inside that bracket rather than at its ceiling.  This script
is what measures it: the live boundary stream of each track, and the refresh
rate a candidate threshold would produce once the cooldown is applied.

    uv run python training/nn_boundary_refresh_rate.py <mp3>... --write
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

RATE_FILE = REPO_ROOT / "training" / "nn_boundary_refresh_rate.json"
GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)


async def boundary_stream(mp3: str) -> dict:
    """Every cell's boundary score, off the real chain, with the real decoder."""
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from lib.engine.section_decoder import SectionDecoder
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import run_fast_simulation_components

    seen: list = []
    push = SectionDecoder.push_posterior

    def spy(self, at_sec, posterior, boundary):
        seen.append((float(at_sec), float(boundary)))
        return push(self, at_sec, posterior, boundary)

    SectionDecoder.push_posterior = spy
    try:
        components, queue = await run_fast_simulation_components(
            FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, mp3))
    finally:
        SectionDecoder.push_posterior = push
    report = components["event_buffer"].to_report(queue.get_timing_log())
    return {"cells": seen, "duration_sec": report["duration_sec"]}


def fires(cells: list, threshold: float, cooldown_sec: float) -> list:
    """The instants a refresh would be requested at, cooldown applied."""
    last = float("-inf")
    out = []
    for at, boundary in cells:
        if boundary < threshold or at - last < cooldown_sec:
            continue
        last = at
        out.append(round(at, 2))
    return out


def measure(mp3: str, cooldown_sec: float) -> dict:
    data = asyncio.run(boundary_stream(mp3))
    scores = sorted(score for _, score in data["cells"])
    count = len(scores)
    minutes = data["duration_sec"] / 60.0
    return {
        "duration_sec": round(data["duration_sec"], 1),
        "cells": count,
        "quantiles": {str(q): round(scores[min(count - 1, int(q * count))], 4)
                      for q in QUANTILES},
        "max": round(scores[-1], 4),
        "refreshes_per_minute": {
            str(threshold): round(len(fires(data["cells"], threshold,
                                            cooldown_sec)) / minutes, 3)
            for threshold in GRID},
    }


def main() -> None:
    from lib.engine.light_engine import (BOUNDARY_REFRESH_SCORE,
                                         REFRESH_COOLDOWN_SEC)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--write", nargs="?", const=str(RATE_FILE), default=None)
    args = ap.parse_args()

    from pipeline_digest import fixture_key

    tracks = {}
    for mp3 in args.tracks:
        name = fixture_key(mp3)
        print(f"--- {name}")
        tracks[name] = measure(mp3, REFRESH_COOLDOWN_SEC)
        print(json.dumps(tracks[name], indent=2, sort_keys=True))

    chosen = str(BOUNDARY_REFRESH_SCORE)
    realised = [t["refreshes_per_minute"][chosen] for t in tracks.values()]
    verdict = {
        "chosen_threshold": BOUNDARY_REFRESH_SCORE,
        "cooldown_sec": REFRESH_COOLDOWN_SEC,
        "retired_ceiling_per_minute": round(60.0 / REFRESH_COOLDOWN_SEC, 3),
        "realised_per_minute": {"min": min(realised), "max": max(realised),
                                "mean": round(sum(realised) / len(realised), 3)},
        "tracks": tracks,
    }
    print(f"\n{BOUNDARY_REFRESH_SCORE} gives "
          f"{verdict['realised_per_minute']['mean']}/min mean "
          f"({verdict['realised_per_minute']['min']}-"
          f"{verdict['realised_per_minute']['max']}), against a retired ceiling "
          f"of {verdict['retired_ceiling_per_minute']}/min")
    if args.write:
        Path(args.write).write_text(json.dumps(verdict, indent=2,
                                               sort_keys=True) + "\n",
                                    newline="\n")
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
