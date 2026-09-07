"""ARM H-AG8 (#343 rung 5): wire the variants into trainer-consumable artifacts.

Three campaign-local outputs, nothing in the corpus touched:

1. ``mapped_tier_ag8.json`` -- each variant id mapped to its BASE track's
   merged-view sections, copied programmatically from the trainer's own
   ``demote.merged_sections`` so the variant trains on byte-identical section
   tuples.  Consumed via the trainer's EXISTING ``--mapped-tier`` additive
   seam (``sections.setdefault``; corpus records win) -- no trainer patch, and
   never a ``*.hand.json`` (which would poison the owner's-ear precedence).
2. ``splits_ag8.json`` -- corpus train (969, six base ids at 1x as they
   already are) + the 42 variant ids = 1011 entries; val byte-equal.  1011
   equals splits_hos8.json's count: the same content dose as the 8x arm with
   zero identical-content duplication.
3. ``features_ag8_root/MERT-v1-330M_L6-22_F3_hop1/`` -- every corpus F3
   sidecar HARDLINKED in (same NTFS volume, zero bytes), so the trainer's one
   ``--feature-dir`` resolves corpus and variant ids alike; stream_extract
   later writes the 42 variant sidecars into this same root via --out-root.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PHASE_B = Path(r"C:\Users\Julian\Projects\soundswitch-phase-b-worktree")
sys.path.insert(0, str(PHASE_B))

from ng_common import CAMP, CORPUS, F3_DIR, NEW_HAND_IDS  # noqa: E402
from ng_ag8_variants import SHIFTS, variant_id  # noqa: E402

from training.nn.ceiling import demote as DD  # noqa: E402

FEAT_ROOT = CAMP / "features_ag8_root" / "MERT-v1-330M_L6-22_F3_hop1"


def build_tier(sections: dict) -> dict:
    tracks = {}
    for base in NEW_HAND_IDS:
        rows = sections.get(base)
        if not rows:
            raise SystemExit(f"{base}: no sections in the merged view")
        for suffix, steps in SHIFTS:
            tracks[variant_id(base, suffix)] = {
                "base": base,
                "pitch_shift_semitones": steps,
                "sections": [{"start": float(s), "end": float(e),
                              "name": str(n)} for s, e, n in rows],
            }
    return {
        "kind": "nextgen_ag8_variant_tier",
        "note": "ARM H-AG8 (#343): time-preserving pitch-shift variants carry "
                "their base track's merged-view sections verbatim; consumed "
                "via the trainer's --mapped-tier additive seam "
                "(sections.setdefault, corpus records win)",
        "tracks": tracks,
    }


def build_splits(corpus_splits: dict) -> dict:
    train = list(map(str, corpus_splits["train"]))
    val = list(map(str, corpus_splits["val"]))
    if len(train) != 969 or len(val) != 215:
        raise SystemExit(f"corpus splits are {len(train)}/{len(val)}, "
                         f"expected 969/215 -- refusing on a surprise")
    for base in NEW_HAND_IDS:
        if train.count(base) != 1:
            raise SystemExit(f"{base} appears {train.count(base)}x in corpus "
                             f"train (expected exactly 1)")
    variants = [variant_id(base, suffix)
                for base in NEW_HAND_IDS for suffix, _ in SHIFTS]
    if set(variants) & set(train):
        raise SystemExit("variant id collides with a corpus train id")
    merged = train + variants
    if len(merged) != len(set(merged)) or len(merged) != 1011:
        raise SystemExit(f"merged train list is {len(merged)} entries with "
                         f"{len(set(merged))} unique -- expected 1011 unique")
    return {
        "kind": "nextgen_ag8_splits",
        "base": "corpus splits.json (read-only)",
        "note": "ARM H-AG8 (#343): 969 corpus train + 42 pitch-shift variant "
                "ids at 1x = 1011 entries (= splits_hos8.json's count; same "
                "content dose, zero identical-content duplication); val "
                "byte-equal to the corpus record",
        "variant_ids": variants,
        "train": merged,
        "val": val,
    }


def link_corpus_sidecars() -> tuple:
    FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    linked = kept = 0
    for source in sorted(F3_DIR.glob("*.npz")):
        target = FEAT_ROOT / source.name
        if target.exists():
            kept += 1
            continue
        os.link(source, target)
        linked += 1
    return linked, kept


def main() -> int:
    sections = DD.merged_sections(CORPUS)
    tier = build_tier(sections)
    (CAMP / "mapped_tier_ag8.json").write_text(
        json.dumps(tier, indent=2) + "\n", encoding="utf-8")
    print(f"wrote mapped_tier_ag8.json ({len(tier['tracks'])} variant "
          f"entries)")

    corpus_splits = json.loads((CORPUS / "splits.json").read_text("utf-8"))
    splits = build_splits(corpus_splits)
    (CAMP / "splits_ag8.json").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(f"wrote splits_ag8.json (train {len(splits['train'])}, "
          f"val {len(splits['val'])})")

    linked, kept = link_corpus_sidecars()
    print(f"feature root {FEAT_ROOT}: {linked} corpus sidecars hardlinked, "
          f"{kept} already present")

    missing = [i for i in splits["train"][:969] + splits["val"]
               if not (FEAT_ROOT / f"{i}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} corpus ids missing from the feature "
                         f"root: {missing[:5]}")
    print("every corpus train+val id resolves in the feature root")
    still = [v for v in splits["variant_ids"]
             if not (FEAT_ROOT / f"{v}.npz").exists()]
    print(f"{len(still)} variant sidecars still to extract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
