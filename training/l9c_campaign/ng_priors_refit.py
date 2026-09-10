"""L9C priors refit (#346): arm N, the merged annotation view with tonight's labels.

The plain train-split fit -- the same computation as running nn.priors' CLI
directly, which l9b proved byte-identical to its in-process fit.  The l9c twist
is WHICH TREE the instrument imports from: the main repo, because the #345
transition knob lives there.  --verify-anchor proves tree parity by running the
label9 worktree's nn.priors CLI and the main repo's side by side on today's
data and byte-comparing (priors.py is committed-identical in both trees; this
proves the whole import graph agrees on the fit).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

MAIN = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot")
DATA = MAIN / "training" / "data" / "raveform"
CAMP = DATA / "models" / "l9c_campaign"
LABEL9 = Path(r"C:\Users\Julian\Projects\soundswitch-label9-worktree")
VENV_PY = MAIN / ".venv" / "Scripts" / "python.exe"

W = str(MAIN)
sys.path[:0] = [W, W + r"\training"]

try:
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception:  # noqa: BLE001
    pass

from lib.label_space import check_class_space  # noqa: E402


def run_priors_cli(tree: Path, out: Path, split: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tree), str(tree / "training")])
    result = subprocess.run(
        [str(VENV_PY), "-m", "nn.priors", "--data-dir", str(DATA),
         "--split", split, "--out", str(out)],
        capture_output=True, text=True, env=env, cwd=str(tree), timeout=3600)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"{tree} nn.priors CLI failed "
                         f"(exit {result.returncode})")


def floor_rows(priors, labels=("buildup", "drop", "breakdown", "bridge")) -> str:
    lines = []
    for label in labels:
        index = priors.index(label)
        stats = priors.corpus["duration_bars"][label]
        lines.append(f"  {label:<9} n={stats['n']:>4} median={stats['median']:>5.1f} "
                     f"floor={int(priors.floor_bars[index]):>2} "
                     f"hazard={priors.hazard[index]:.4f} "
                     f"E[bars]={stats['expected_bars']:.1f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=CAMP / "priors_N.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--verify-anchor", action="store_true")
    args = parser.parse_args()

    CAMP.mkdir(parents=True, exist_ok=True)
    anchor_dir = CAMP / "anchor"

    if args.verify_anchor:
        anchor_dir.mkdir(parents=True, exist_ok=True)
        label9_out = anchor_dir / "priors_label9_cli.json"
        main_out = anchor_dir / "priors_main_cli.json"
        print(f"label9 CLI -> {label9_out}", flush=True)
        run_priors_cli(LABEL9, label9_out, args.split)
        print(f"main  CLI -> {main_out}", flush=True)
        run_priors_cli(MAIN, main_out, args.split)
        a, b = label9_out.read_bytes(), main_out.read_bytes()
        if a == b:
            print(f"ANCHOR OK: label9 and main trees fit byte-identical "
                  f"priors on today's data ({len(a)} bytes)")
            return 0
        print("ANCHOR FAILED: the two trees' fits differ", file=sys.stderr)
        return 1

    run_priors_cli(MAIN, args.out, args.split)

    from nn.priors import Priors  # noqa: PLC0415
    priors = Priors.load(args.out)
    check_class_space(priors.classes, str(args.out))
    if len(priors.classes) != 9:
        raise SystemExit(f"{args.out}: {len(priors.classes)} classes, not the "
                         f"full vocabulary")
    corpus = priors.corpus
    print(f"arm N: fitted {corpus['tracks']} tracks / {corpus['runs']} runs / "
          f"{corpus['bars']} bars (split={args.split})")
    print(floor_rows(priors))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
