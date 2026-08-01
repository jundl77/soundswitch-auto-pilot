"""Where the Raveform corpus is on this machine, and nothing else.

Split out so a *show* can ask.  `lib/section_chain.py` needs the corpus root to
find the shipped model, and reaching it through `run_eval_set` imported the
whole benchmark harness -- the table builder, the label evaluator, the raveform
acquisition scripts, three `sys.path` inserts and a git subprocess -- into
production startup, to resolve a path.  Worse, any failure in that chain read as
"this machine has no model" and the show quietly ran the degradation state.

Stdlib only, and one definition: `run_eval_set` and `build_training_table`
re-export from here rather than keeping copies.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_ENV = "RAVEFORM_DATA_DIR"


def default_data_dir() -> Path:
    return REPO_ROOT / "training" / "data" / "raveform"


def corpus_dir() -> Path:
    """``$RAVEFORM_DATA_DIR``, else the repo's, else the main worktree's.

    The corpus is gitignored, so there is exactly ONE copy of it on a machine
    and it does not follow ``git worktree add``: branch work in a linked
    worktree still has to reach the audio sitting in the main checkout.
    Resolving that here -- rather than making every caller pass ``--data-dir``
    -- is what lets the integration suite run green from any worktree.  The
    environment variable wins for the case this cannot guess (a corpus on
    another drive).
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).resolve()
    local = default_data_dir()
    if local.exists():
        return local
    # `--git-common-dir` is the main checkout's .git even from a linked worktree.
    common = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
    if common:
        shared = Path(common).parent / local.relative_to(REPO_ROOT)
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
