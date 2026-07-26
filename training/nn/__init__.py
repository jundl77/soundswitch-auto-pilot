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

import sys
from pathlib import Path

_TRAINING_DIR = str(Path(__file__).resolve().parents[1])
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)
