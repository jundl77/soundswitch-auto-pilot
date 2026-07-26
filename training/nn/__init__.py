"""Neural section classifier: dataset, model, export, decoder.

Committed code only -- every artifact this package produces (splits, model
checkpoints, ONNX graphs, posterior sidecars) lives in the gitignored data
directory, never in the repo.

The eval-pipeline scripts next door (``build_training_table``,
``select_eval_set``, ``raveform_manifest``) are plain modules in ``training/``
rather than an installed package, so importing them requires their directory on
``sys.path``.  Doing it here, once, keeps every module in this package -- and
every test that imports one -- from repeating the incantation, and guarantees
the label vocabulary, the mel geometry and the artist parser are the *same*
definitions the corpus was built with rather than copies that can drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Must precede the first `import torch`, and `dataset` imports torch -- so this
# package's __init__ is the only place guaranteed to run first.  Not strictly
# required on the validated stack (torch 2.11 / CUDA 12.8 stayed deterministic
# without it), set defensively: it costs nothing and keeps determinism if the
# CUDA version moves or a cuBLAS-heavy op is added.  See the CUDA pre-flight.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_TRAINING_DIR = str(Path(__file__).resolve().parents[1])
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)
