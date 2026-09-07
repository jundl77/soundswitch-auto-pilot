"""PREP 2: pre-seed the six new hand tracks into TRAIN and rebuild the splits.

Ruling D1 (decision #341): all six are frozen into train -- two would hash to
test otherwise -- and afterwards no new hand id may sit in val or test, and the
13 relabel youtube-ids must not move.  Default is a dry-run; --apply backs the
frozen file up to CAMP, pre-seeds, runs training.nn.dataset.make_splits (the
additive rebuild with the eval-set and artist guards), and asserts the outcome
against the backup.  Any assertion failure restores the backup and exits 1.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ng_common import CAMP, CORPUS, NEW_HAND_IDS, inject_repo_paths, relabel_ids

inject_repo_paths()

from nn.dataset import assign_split, make_splits  # noqa: E402

BACKUP_NAME = "splits.json.pre_nextgen.bak"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def placement(doc: dict, track_id: str) -> str | None:
    for split in ("train", "val", "test"):
        if track_id in doc[split]:
            return split
    return None


def verify(doc: dict, backup: dict, relabels: list) -> list:
    failures = []
    for track_id in NEW_HAND_IDS:
        if placement(doc, track_id) != "train":
            failures.append(f"{track_id} is in {placement(doc, track_id)!r}, not train")
    for split in ("val", "test"):
        before, after = set(backup[split]), set(doc[split])
        if before != after:
            failures.append(
                f"{split} membership moved: +{sorted(after - before)} "
                f"-{sorted(before - after)}")
    for track_id in relabels:
        if placement(doc, track_id) != "train":
            failures.append(f"relabel {track_id} left train "
                            f"(now {placement(doc, track_id)!r})")
    for key in ("excluded_eval_set", "excluded_artist"):
        before, after = set(backup[key]), set(doc[key])
        if before != after:
            failures.append(f"{key} changed: +{sorted(after - before)} "
                            f"-{sorted(before - after)}")
    return failures


def summarise(doc: dict, backup: dict) -> None:
    for split in ("train", "val", "test"):
        delta = len(doc[split]) - len(backup[split])
        print(f"  {split}: {len(backup[split])} -> {len(doc[split])} ({delta:+d})")
    print(f"  excluded_eval_set {len(doc['excluded_eval_set'])}, "
          f"excluded_artist {len(doc['excluded_artist'])}, "
          f"retired {len(doc['retired'])}, candidates {doc['candidates']}")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=CORPUS)
    parser.add_argument("--camp", type=Path, default=CAMP)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    corpus = args.data_dir.resolve()
    splits_path = corpus / "splits.json"
    doc = load(splits_path)
    relabels = relabel_ids(corpus)
    if len(relabels) != 13:
        raise SystemExit(f"expected 13 relabel youtube-ids, found {len(relabels)}: "
                         f"{relabels}")

    missing = [i for i in NEW_HAND_IDS if placement(doc, i) is None]
    print("new hand ids:")
    for track_id in NEW_HAND_IDS:
        print(f"  {track_id}: placed={placement(doc, track_id) or 'ABSENT'}  "
              f"hash-would-say={assign_split(track_id)}")
    print(f"relabels in train: "
          f"{sum(1 for i in relabels if placement(doc, i) == 'train')}/{len(relabels)}")

    if not args.apply:
        print(f"\nDRY RUN: --apply would pre-seed {missing} into train and rebuild "
              f"via make_splits; val/test membership must not move")
        return 0

    args.camp.mkdir(parents=True, exist_ok=True)
    backup_path = args.camp / BACKUP_NAME
    if not backup_path.exists():
        shutil.copy2(splits_path, backup_path)
        print(f"\nbacked up to {backup_path}")
    else:
        print(f"\nkeeping existing backup {backup_path}")
    backup = load(backup_path)

    seeded = dict(doc)
    seeded["train"] = list(doc["train"]) + missing
    splits_path.write_text(json.dumps(seeded, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    try:
        rebuilt = make_splits(corpus)
    except BaseException:
        shutil.copy2(backup_path, splits_path)
        print(f"make_splits raised -- restored {splits_path} from backup",
              file=sys.stderr)
        raise

    failures = verify(rebuilt, backup, relabels)
    if failures:
        shutil.copy2(backup_path, splits_path)
        print("ARMING FAILED -- backup restored:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("\narmed; all assertions hold:")
    summarise(rebuilt, backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
