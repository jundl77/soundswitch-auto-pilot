"""ARM H-MK mechanical proof (#343): the masked spans contribute ZERO label
loss terms, and nothing else moves.

Runs BEFORE any training, CPU only, through the trainer's own modules
(``mask.apply_supervision_mask`` + ``dataset.track_targets`` under the l9
class space, frames counted exactly as ``FullTrackDataset`` counts them).
Checks, per overlay span:

  1. every frame whose time falls inside [start, end) carried the masked
     section's class pre-mask and carries -1 (IGNORE_INDEX at pooling)
     post-mask;
  2. every frame outside the span carries an identical label pre vs post;
  3. every pooled cell all of whose in-track frames sit inside the span has
     ``label_pooled_mask`` False post-mask (zero focal, zero TV);
  4. the boundary channel the dataset will train on (post-mask targets with
     the pre-mask boundary override, exactly ``FullTrackDataset._targets``)
     is array-equal to the pre-mask gold boundary.

Aggregates the masked frame count against the overlay's own seconds and
writes CAMP/mask_proof.json.  Any violation exits nonzero and the arm holds.
"""
import copy
import ctypes
import datetime
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

sys.path.insert(0, str(PHASE_B))

import numpy as np  # noqa: E402

from training.nn.ceiling import demote as DD  # noqa: E402
from training.nn.ceiling import mask as MK  # noqa: E402
from training.nn.dataset import (  # noqa: E402
    FEATURES_DIR, FRAME_SEC, LABEL_POOL, load_label_space, sidecar_shape,
    track_targets)
from training.nn.infer import usable_frames  # noqa: E402

FEATURE_DIR = DATA / "features_stream" / "MERT-v1-330M_L6-22_F3_hop1"


def frames_for(track_id: str) -> int:
    mel = DATA / FEATURES_DIR / f"{track_id}.npz"
    if mel.exists():
        return usable_frames(sidecar_shape(mel)[0])
    with np.load(FEATURE_DIR / f"{track_id}.npz") as archive:
        return usable_frames(int(archive["n_frames"]))


def main() -> int:
    classes = load_label_space(DATA / "models" / "l9" / "priors.json")
    class_index = {label: i for i, label in enumerate(classes)}
    splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))

    sections = DD.merged_sections(DATA)
    pre_sections = copy.deepcopy(sections)
    pre_by_id, stats = MK.apply_supervision_mask(
        CAMP / "supervision_mask_overlay.json", sections,
        train_ids=splits["train"], val_ids=splits["val"], data_dir=DATA)
    overlay = MK.load_overlay(CAMP / "supervision_mask_overlay.json")
    print(f"applied: {stats}")

    for track_id, rows in pre_by_id.items():
        assert list(rows) == list(map(tuple, pre_sections[track_id])), \
            f"{track_id}: pre_by_id is not the pre-mask section list"

    spans_by_id: dict = {}
    for entry in overlay["sections"]:
        spans_by_id.setdefault(str(entry["id"]), []).append(
            (float(entry["start"]), float(entry["end"]), str(entry["label"])))

    masked_frames = expected_frames = 0
    cells_masked_out = 0
    failures: list = []
    for n, (track_id, spans) in enumerate(sorted(spans_by_id.items()), 1):
        n_frames = frames_for(track_id)
        pre = track_targets(pre_sections[track_id], n_frames, classes=classes)
        post = track_targets(sections[track_id], n_frames, classes=classes)
        times = FRAME_SEC + np.arange(n_frames, dtype=np.float64) * FRAME_SEC

        in_any_span = np.zeros(n_frames, dtype=bool)
        for start, end, label in spans:
            inside = (times >= start) & (times < end)
            in_any_span |= inside
            want = class_index[label]
            if not np.all(pre.label_frame[inside] == want):
                failures.append(f"{track_id} [{start},{end}]: pre-mask frames "
                                f"are not all {label}")
            if not np.all(post.label_frame[inside] == -1):
                failures.append(f"{track_id} [{start},{end}]: post-mask frames "
                                f"are not all -1")
            masked_frames += int(np.sum(post.label_frame[inside] == -1))
            expected_frames += int(np.sum(inside))

        outside = ~in_any_span
        if not np.array_equal(pre.label_frame[outside],
                              post.label_frame[outside]):
            failures.append(f"{track_id}: labels OUTSIDE the spans moved")

        usable = (n_frames // LABEL_POOL) * LABEL_POOL
        cell_all_in = in_any_span[:usable].reshape(-1, LABEL_POOL).all(axis=1)
        if post.label_pooled_mask[cell_all_in].any():
            failures.append(f"{track_id}: a pooled cell fully inside a masked "
                            f"span is still supervised")
        cells_masked_out += int(np.sum(pre.label_pooled_mask[cell_all_in]
                                       & ~post.label_pooled_mask[cell_all_in]))

        # The boundary channel the dataset trains on: post targets with the
        # pre-mask override -- FullTrackDataset._targets verbatim.
        gold = track_targets(pre_by_id[track_id], n_frames, classes=classes)
        final = post._replace(boundary=gold.boundary,
                              boundary_mask=gold.boundary_mask)
        if not (np.array_equal(final.boundary, pre.boundary)
                and np.array_equal(final.boundary_mask, pre.boundary_mask)):
            failures.append(f"{track_id}: trained boundary channel is not the "
                            f"pre-mask gold")
        if n % 50 == 0:
            print(f"  checked {n}/{len(spans_by_id)} tracks", flush=True)

    overlay_sec = sum(float(e["end"]) - float(e["start"])
                      for e in overlay["sections"])
    masked_sec = masked_frames * FRAME_SEC
    proof = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "overlay": str(CAMP / "supervision_mask_overlay.json"),
        "applied_stats": stats,
        "tracks_checked": len(spans_by_id),
        "spans_checked": len(overlay["sections"]),
        "masked_frames": masked_frames,
        "expected_frames_from_span_geometry": expected_frames,
        "pooled_cells_supervision_removed": cells_masked_out,
        "masked_seconds_at_frame_rate": round(masked_sec, 2),
        "overlay_seconds": round(overlay_sec, 2),
        "boundary_channel": "array-equal to pre-mask gold on every masked "
                            "track (post targets + pre-mask override, the "
                            "dataset's own _targets composition)",
        "failures": failures,
        "verdict": "MASK PROVEN: masked spans contribute zero label-loss "
                   "terms; everything else is byte-identical"
                   if not failures and masked_frames == expected_frames
                   else "MASK NOT PROVEN",
    }
    (CAMP / "mask_proof.json").write_text(json.dumps(proof, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"masked frames: {masked_frames} (geometry expects "
          f"{expected_frames}) = {masked_sec:.0f} s vs overlay "
          f"{overlay_sec:.0f} s")
    print(f"pooled cells with supervision removed: {cells_masked_out}")
    print(proof["verdict"])
    if failures:
        for line in failures[:20]:
            print(f"  FAIL: {line}")
    print(f"wrote {CAMP / 'mask_proof.json'}")
    return 0 if proof["verdict"].startswith("MASK PROVEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
