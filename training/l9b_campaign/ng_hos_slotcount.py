"""#342 mechanism verification: duplicated ids in splits_hos.json multiply
sampler slots.

Builds the H-OS train dataset through the trainer's own build_datasets (the
exact argv arm H trained under, plus --splits-file splits_hos.json) and counts
the epoch slots per track id.  The claim under test: a track listed N times
yields N x crops_per_track slots -- oversampling by duplication, no dedup
anywhere on the path.  Run under the exp-ceiling venv; CPU only.
"""
import argparse
import collections
import ctypes
import json
import sys
from pathlib import Path

try:
    _k32 = ctypes.windll.kernel32
    _k32.GetCurrentProcess.restype = ctypes.c_void_p
    _k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _k32.SetPriorityClass(_k32.GetCurrentProcess(), 0x00004000)
except Exception:  # noqa: BLE001
    pass

PHASE_B = Path(r"C:\Users\Julian\Projects\soundswitch-phase-b-worktree")
DATA = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot"
            r"\training\data\raveform")
CAMP = DATA / "models" / "l9b_campaign"
SIX = ("hand-65cb8c94812d", "hand-8339586c555a", "hand-1b57bc38e8e4",
       "hand-33d3513481ac", "hand-b7d98ca02e86", "hand-52973d7b1767")

sys.path.insert(0, str(PHASE_B))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import module_source  # noqa: E402

# ``ceiling`` exists only in the phase-B checkout, so the rest of training.nn
# must come from beside it rather than from whichever copy an import reached
# first -- the two disagree, and nothing downstream would say which ran.
module_source.require("training.nn", PHASE_B)

from training.nn.ceiling import demote as DD  # noqa: E402
from training.nn.ceiling.train_head import build_datasets  # noqa: E402


def main() -> int:
    args = argparse.Namespace(
        data_dir=DATA,
        feature_dir=DATA / "features_stream" / "MERT-v1-330M_L6-22_F3_hop1",
        layers=[6, 22],
        label_space=DATA / "models" / "l9" / "priors.json",
        arm="online_crnn",
        splits_file=CAMP / "splits_hos.json",
        demote_drops=None, mask_labels=None, limit_tracks=None,
        teacher_dir=None,
        seed=1234, crop_sec=300.0, crops_per_track=3,
        cache_bytes=0,
    )
    sections = DD.merged_sections(args.data_dir)
    train, val, _, _ = build_datasets(args, sections)

    per_id = collections.Counter(train.ids[index] for index, _ in train._slots)
    baseline = collections.Counter(v for k, v in per_id.items()
                                   if k not in SIX).most_common(1)[0][0]
    print(f"train entries: {len(train.ids)}   slots/epoch: {len(train._slots)}"
          f"   val tracks: {len(val.ids)}")
    print(f"baseline slots per non-oversampled track: {baseline}")
    ok = True
    for track_id in SIX:
        n = per_id[track_id]
        factor = n / baseline
        verdict = "OK" if factor == 16 else "WRONG"
        ok = ok and factor == 16
        print(f"  {track_id}: {n} slots ({factor:.0f}x baseline) {verdict}")
    document = json.loads((CAMP / "splits_hos.json").read_text("utf-8"))
    print(f"splits_hos train list: {len(document['train'])} entries "
          f"(969 unique + 6 ids x15 extra copies)")
    print("MECHANISM VERIFIED: duplicated ids multiply sampler slots"
          if ok else "MECHANISM NOT VERIFIED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
