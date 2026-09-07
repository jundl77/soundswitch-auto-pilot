"""ARM H-AG8 mechanical proof (#343): every variant trains on its base's labels
over acoustically distinct features, through the trainer's own modules.

Runs BEFORE training, CPU only.  Checks, per variant:

  1. splits_ag8.json is corpus train (969) + the 42 variant ids, val
     byte-equal to the corpus record;
  2. the tier's sections are byte-equal to the base id's merged-view sections,
     and the trainer's own seam (``sections.setdefault``, train_head verbatim)
     leaves the variant carrying exactly those sections;
  3. the variant's F3 sidecar loads from the campaign feature root with the
     (causal, F3, hop1) stream geometry and the base's exact frame count
     (time preservation at the cell grid);
  4. ``track_targets`` for the variant is array-equal to the base's on every
     channel (label frames, pooled labels, pooled mask, boundary, boundary
     mask);
  5. a ``FullTrackDataset`` built exactly as the online_crnn arm builds it
     yields, for each variant, a Sample whose targets equal the base's Sample
     targets while its FEATURES differ from the base's (acoustic diversity is
     real, not a copy).

Writes CAMP/ag8_proof.json.  Any violation exits nonzero and the arm holds.
"""
from __future__ import annotations

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
sys.path.insert(0, str(PHASE_B))

import numpy as np  # noqa: E402

from ng_common import CAMP, CORPUS, NEW_HAND_IDS  # noqa: E402
from ng_ag8_variants import SHIFTS, variant_id  # noqa: E402

from training.nn.ceiling import data as D  # noqa: E402
from training.nn.ceiling import demote as DD  # noqa: E402
from training.nn.ceiling.stream_extract import read_stream_geometry  # noqa: E402
from training.nn.dataset import load_label_space, track_targets  # noqa: E402
from training.nn.infer import usable_frames  # noqa: E402

FEAT_ROOT = CAMP / "features_ag8_root" / "MERT-v1-330M_L6-22_F3_hop1"
LAYERS = [6, 22]


def frames_for(track_id: str) -> int:
    with np.load(FEAT_ROOT / f"{track_id}.npz") as archive:
        return usable_frames(int(archive["n_frames"]))


def main() -> int:
    failures: list = []
    classes = load_label_space(CORPUS / "models" / "l9" / "priors.json")

    corpus_splits = json.loads((CORPUS / "splits.json").read_text("utf-8"))
    splits = json.loads((CAMP / "splits_ag8.json").read_text("utf-8"))
    tier = json.loads((CAMP / "mapped_tier_ag8.json").read_text("utf-8"))
    variants = [variant_id(base, suffix)
                for base in NEW_HAND_IDS for suffix, _ in SHIFTS]

    if splits["train"] != list(map(str, corpus_splits["train"])) + variants:
        failures.append("splits_ag8 train is not corpus train + the 42 "
                        "variants in order")
    if splits["val"] != list(map(str, corpus_splits["val"])):
        failures.append("splits_ag8 val is not byte-equal to the corpus val")
    if sorted(tier["tracks"]) != sorted(variants):
        failures.append("tier ids != the 42 variant ids")

    # The trainer's own seam, train_head verbatim.
    sections = DD.merged_sections(CORPUS)
    pre_bases = {b: list(sections[b]) for b in NEW_HAND_IDS}
    for track_id, record in tier["tracks"].items():
        sections.setdefault(str(track_id), [
            (float(s["start"]), float(s["end"]), str(s["name"]))
            for s in record["sections"]])
    for base in NEW_HAND_IDS:
        if list(sections[base]) != pre_bases[base]:
            failures.append(f"{base}: the tier moved a corpus record")

    per_variant: dict = {}
    for base in NEW_HAND_IDS:
        base_frames = frames_for(base)
        base_targets = track_targets(sections[base], base_frames,
                                     classes=classes)
        for suffix, steps in SHIFTS:
            vid = variant_id(base, suffix)
            state: dict = {"base": base, "n_steps": steps}
            path = FEAT_ROOT / f"{vid}.npz"
            if not path.exists():
                failures.append(f"{vid}: no sidecar at {path}")
                continue
            geometry = read_stream_geometry(path)
            if (geometry is None or geometry["causal"] != 1
                    or geometry["margin_sec"] != 3.0
                    or geometry["hop_sec"] != 1.0):
                failures.append(f"{vid}: stream geometry {geometry} is not "
                                f"(causal, F3, hop1)")
            frames = frames_for(vid)
            state["n_frames"] = frames
            if frames != base_frames:
                failures.append(f"{vid}: {frames} frames vs base's "
                                f"{base_frames} -- time preservation broken")
                continue
            if list(map(tuple, sections[vid])) != \
                    list(map(tuple, sections[base])):
                failures.append(f"{vid}: post-seam sections != base sections")
            targets = track_targets(sections[vid], frames, classes=classes)
            for channel in ("label_frame", "label_pooled", "label_pooled_mask",
                            "boundary", "boundary_mask"):
                if not np.array_equal(getattr(targets, channel),
                                      getattr(base_targets, channel)):
                    failures.append(f"{vid}: {channel} != base's")
            state["supervised_cells"] = int(targets.label_pooled_mask.sum())
            per_variant[vid] = state

    # Through FullTrackDataset itself, exactly as the online_crnn arm builds it.
    common = dict(feature_dir=FEAT_ROOT, layers=LAYERS, augment=False,
                  sections_by_youtube_id=sections, classes=classes,
                  normalise=False, store_fp16=True, cache_bytes=0)
    base_set = D.FullTrackDataset(CORPUS, list(NEW_HAND_IDS), **common)
    var_set = D.FullTrackDataset(CORPUS, variants, **common)
    base_samples = {s.youtube_id: s for s in (base_set[i]
                                              for i in range(len(base_set)))}
    checked = distinct = 0
    for index in range(len(var_set)):
        sample = var_set[index]
        base = per_variant[sample.youtube_id]["base"]
        ref = base_samples[base]
        if not (np.array_equal(sample.labels, ref.labels)
                and np.array_equal(sample.label_mask, ref.label_mask)
                and np.array_equal(sample.boundary, ref.boundary)
                and np.array_equal(sample.boundary_mask, ref.boundary_mask)):
            failures.append(f"{sample.youtube_id}: dataset Sample targets != "
                            f"base's")
        if sample.x.shape != ref.x.shape:
            failures.append(f"{sample.youtube_id}: feature shape "
                            f"{sample.x.shape} != base's {ref.x.shape}")
        elif np.array_equal(sample.x, ref.x):
            failures.append(f"{sample.youtube_id}: features are byte-equal to "
                            f"the base's -- a copy, not a variant")
        else:
            distinct += 1
            per_variant[sample.youtube_id]["mean_abs_feature_delta"] = round(
                float(np.mean(np.abs(sample.x.astype(np.float32)
                                     - ref.x.astype(np.float32)))), 4)
        checked += 1
        if checked % 10 == 0:
            print(f"  dataset check {checked}/{len(var_set)}", flush=True)

    proof = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "feature_root": str(FEAT_ROOT),
        "variants_checked": checked,
        "features_distinct_from_base": distinct,
        "per_variant": per_variant,
        "failures": failures,
        "verdict": ("AG8 WIRING PROVEN: 42 variants, each trained on its "
                    "base's exact targets over acoustically distinct features"
                    if not failures and checked == 42 and distinct == 42
                    else "AG8 WIRING NOT PROVEN"),
    }
    (CAMP / "ag8_proof.json").write_text(json.dumps(proof, indent=2) + "\n",
                                         encoding="utf-8")
    print(proof["verdict"])
    for line in failures[:20]:
        print(f"  FAIL: {line}")
    print(f"wrote {CAMP / 'ag8_proof.json'}")
    return 0 if proof["verdict"].startswith("AG8 WIRING PROVEN") else 1


if __name__ == "__main__":
    sys.exit(main())
