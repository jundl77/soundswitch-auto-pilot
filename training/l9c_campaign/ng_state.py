"""L9C campaign state machine (#346). Single writer: the coordinator session.

Usage:
  python ng_state.py --init
  python ng_state.py --stage NAME --note "text"
  python ng_state.py --note "text"

Every call stamps UTC time and the kernel nonpaged pool reading (the leak the
charter gates GPU launches on: STOP new GPU stages past 8 GB).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

CAMP = Path(__file__).resolve().parent
STATE = CAMP / "state.json"

STAGES = [
    "preflight",
    "instrument",
    "transition_axis",
    "priors_refit",
    "train_ng_N_w128_s1234",
    "train_ng_N_w128_s1235",
    "sweep_N_s1234",
    "sweep_N_s1235",
    "verdict",
    "probes",
    "conditional",
    "longcontext_probe",
    "bank",
    "done",
]


def pool_gb() -> float:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\\Memory\\Pool Nonpaged Bytes').CounterSamples[0].CookedValue"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return round(float(out) / 1e9, 3)
    except Exception:
        return -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--stage")
    ap.add_argument("--note")
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    pool = pool_gb()

    if args.init:
        state = {
            "campaign": "l9c_campaign (#346)",
            "stamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
            "status": "running",
            "stage": "preflight",
            "stages": STAGES,
            "started_utc": now,
            "updated_utc": now,
            "pool_nonpaged_gb": pool,
            "pool_gate_gb": 8.0,
            "notes": [f"{now}  init  pool={pool}GB"],
        }
    else:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if args.stage:
            if args.stage not in state["stages"]:
                raise SystemExit(f"unknown stage {args.stage!r}")
            state["stage"] = args.stage
            state["notes"].append(f"{now}  stage -> {args.stage}  pool={pool}GB")
            if args.stage == "done":
                state["status"] = "done"
        if args.note:
            state["notes"].append(f"{now}  {args.note}  pool={pool}GB")
        state["updated_utc"] = now
        state["pool_nonpaged_gb"] = pool

    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"stage={state['stage']} pool={pool}GB")
    if pool >= 8.0:
        print("POOL GATE TRIPPED: nonpaged pool >= 8 GB -- no new GPU stages")


if __name__ == "__main__":
    main()
