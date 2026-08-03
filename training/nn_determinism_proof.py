"""Two fresh processes, one file, one question: are the bytes the same?"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))

PROOF_FILE = REPO_ROOT / "training" / "nn_determinism_proof.json"


def sidecar_for(mp3: str) -> Path:
    from simulate.cell_cache import sidecar_path
    from simulate.fake_audio_client import FileAudioClient

    return sidecar_path(mp3, FileAudioClient.decode_path)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def run_once(mp3: str, mode: str) -> dict:
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from simulate.evaluator import report_checksum
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import run_fast_simulation

    sidecar = sidecar_for(mp3)
    if mode == "cold":
        sidecar.unlink(missing_ok=True)

    wall = time.monotonic()
    _, event_buffer, command_queue = await run_fast_simulation(
        FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, mp3))
    wall = time.monotonic() - wall

    report = event_buffer.to_report(command_queue.get_timing_log())
    return {
        "mode": mode,
        "checksum": report_checksum(report),
        "cells_sha256": (sha256_of(sidecar.read_bytes())
                         if sidecar.exists() else None),
        "intents": [block["intent"] for block in report["intents"]],
        "wall_sec": round(wall, 2),
    }


def _child(mp3: str, mode: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-one", mp3,
         "--mode", mode],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{mode} run of {mp3} failed:\n{out.stdout[-4000:]}"
                           f"\n{out.stderr[-4000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def prove(mp3: str) -> dict:
    runs = [_child(mp3, mode) for mode in ("cold", "cold", "warm", "warm")]
    cold_a, cold_b, warm_a, warm_b = runs

    def report_same(left, right):
        return left["checksum"] == right["checksum"]
    return {
        "runs": runs,
        "cold_across_processes": report_same(cold_a, cold_b),
        "warm_across_processes": report_same(warm_a, warm_b),
        "cold_matches_warm": report_same(cold_a, warm_a),
        "extractor_bytes_across_processes":
            cold_a["cells_sha256"] == cold_b["cells_sha256"],
        "checksum": cold_a["checksum"],
    }


def contract(proof: dict) -> str:
    if not all(t["warm_across_processes"] and t["cold_matches_warm"]
               for t in proof.values()):
        return "not byte-deterministic"
    if all(t["cold_across_processes"] for t in proof.values()):
        return "byte-deterministic, cold or warm"
    return "byte-deterministic given cached extractor cells"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tracks", nargs="*")
    ap.add_argument("--run-one", metavar="MP3")
    ap.add_argument("--mode", choices=("cold", "warm"))
    ap.add_argument("--write", nargs="?", const=str(PROOF_FILE), default=None)
    args = ap.parse_args()

    if args.run_one:
        print(json.dumps(asyncio.run(run_once(args.run_one, args.mode))))
        return

    from pipeline_digest import fixture_key

    proof = {}
    for track in args.tracks:
        name = fixture_key(track)
        print(f"--- {name}")
        proof[name] = prove(track)
        print(json.dumps(proof[name], indent=2, sort_keys=True))

    verdict = contract(proof)
    print(f"\nCONTRACT: {verdict}")
    if args.write:
        out = Path(args.write)
        out.write_text(json.dumps({"contract": verdict, "tracks": proof},
                                  indent=2, sort_keys=True) + "\n",
                       newline="\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
