"""Prove two training runs are the same run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import _TRAINING_DIR  # noqa: F401

from build_training_table import default_data_dir  # noqa: E402

MODELS_DIR = "models"
REPORT_FILE = "training_report.json"

VOLATILE = frozenset({
    "wall_seconds", "steps_per_second", "seconds", "started_at", "run_name",
    "run_dir", "gpu_peak_alloc_mb", "gpu_peak_reserved_mb",
})


def strip(node):
    if isinstance(node, dict):
        return {key: strip(value) for key, value in node.items() if key not in VOLATILE}
    if isinstance(node, list):
        return [strip(value) for value in node]
    return node


def differences(left, right, path: str = "") -> list:
    if type(left) is not type(right):
        return [f"TYPE  {path}: {type(left).__name__} vs {type(right).__name__}"]
    if isinstance(left, dict):
        found: list = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                found.append(f"KEY   {path}.{key}: present in only one run")
            else:
                found += differences(left[key], right[key], f"{path}.{key}")
        return found
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"LEN   {path}: {len(left)} vs {len(right)}"]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, f"{path}[{index}]")]
    return [] if left == right else [f"VALUE {path}: {left!r} vs {right!r}"]


def load_report(data_dir: Path, model_version: str, run: str) -> dict:
    path = Path(data_dir) / MODELS_DIR / model_version / run / REPORT_FILE
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist -- a run that died before writing its report "
            f"cannot be compared (that is what the crash drill looks like)")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare(data_dir: Path, model_version: str, first: str, second: str,
            show: int = 20) -> int:
    left = load_report(data_dir, model_version, first)
    right = load_report(data_dir, model_version, second)

    print(f"weight_hash  {first}: {left['weight_hash']}")
    print(f"weight_hash  {second}: {right['weight_hash']}")
    matched = left["weight_hash"] == right["weight_hash"]
    print(f"MATCH        {matched}")

    steps = len(left["history"]["steps"])
    epochs = len(left["history"]["epochs"])
    found = differences(strip(left), strip(right))
    print(f"compared {steps} steps and {epochs} epochs, plus config and summary")
    print(f"non-volatile differences: {len(found)}")
    for line in found[:show]:
        print(f"  {line}")
    if len(found) > show:
        print(f"  ... and {len(found) - show} more")
    return 0 if matched and not found else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--model-version", default="downbeat_v1",
                        help="the <data-dir>/models/<version> the runs live under")
    parser.add_argument("runs", nargs=2, metavar="RUN")
    parser.add_argument("--show", type=int, default=20,
                        help="how many differences to print")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    return compare(args.data_dir, args.model_version, args.runs[0], args.runs[1],
                   args.show)


if __name__ == "__main__":
    raise SystemExit(main())
