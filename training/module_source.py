"""Which checkout an offline script's imports came from -- checked, never assumed.

Several scripts here put another checkout on ``sys.path``: the phase-B worktree
holds the training code the shipped model generations were built with, and some
of that code exists nowhere else.  Whether ``training.nn`` then resolves there
or in this repository is decided by path order and by whatever imported first,
and neither is visible in the output -- a measurement taken through the wrong
decoder looks exactly like one taken through the right one.  So each script
states which checkout it means and is held to it.

``training.nn`` carries an ``__init__.py``, so it is a regular package and every
submodule under it comes from the one directory the package resolved to.  One
check on the package therefore covers the whole of it.
"""
from __future__ import annotations

import importlib
from pathlib import Path


def require(package: str, root: Path) -> None:
    where = Path(importlib.import_module(package).__file__).resolve()
    if not where.is_relative_to(Path(root).resolve()):
        raise RuntimeError(
            f"{package} resolved to {where}, which is not under {root}. "
            f"Something on sys.path -- or an earlier import -- decided that, "
            f"not this script; fix the path setup rather than the symptom.")
