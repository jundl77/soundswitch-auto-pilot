"""Shared paths and identities for the nextgen retrain campaign's PREP stage."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "training" / "data" / "raveform"
CAMP = CORPUS / "models" / "nextgen_campaign"
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"

LABEL9_WORKTREE = Path(r"C:\Users\Julian\Projects\soundswitch-label9-worktree")

F3_DIR = CORPUS / "features_stream" / "MERT-v1-330M_L6-22_F3_hop1"

# Decision #341 / ruling D1: the six new hand tracks, pre-seeded into train.
NEW_HAND_IDS = (
    "hand-1b57bc38e8e4",
    "hand-33d3513481ac",
    "hand-52973d7b1767",
    "hand-65cb8c94812d",
    "hand-8339586c555a",
    "hand-b7d98ca02e86",
)

DEMOTION_THRESHOLD_DB = -16.5
DEMOTION_BASE_THRESHOLD_DB = -15.5
DEMOTION_MARGIN_DB = 1.0
DEMOTION_FEATURE = "mean_db"


def inject_repo_paths() -> None:
    for path in (str(REPO), str(REPO / "training"), str(REPO / "training" / "raveform")):
        if path not in sys.path:
            sys.path.insert(0, path)


def relabel_ids(corpus: Path) -> list:
    return sorted(
        p.name[: -len(".hand.json")]
        for p in (corpus / "annotations").glob("*.hand.json")
        if not p.name.startswith("hand-")
    )
