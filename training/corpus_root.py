"""Where the Raveform corpus is on this machine, and nothing else."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_ENV = "RAVEFORM_DATA_DIR"


def default_data_dir() -> Path:
    return REPO_ROOT / "training" / "data" / "raveform"


def corpus_dir() -> Path:
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).resolve()
    local = default_data_dir()
    if local.exists():
        return local
    main_checkout_git_dir = _git("rev-parse", "--path-format=absolute",
                                 "--git-common-dir")
    if main_checkout_git_dir:
        shared = Path(main_checkout_git_dir).parent / local.relative_to(REPO_ROOT)
        if shared.exists():
            return shared
    return local


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None
