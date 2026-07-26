"""Freeze a trained ``SectionCRNN`` checkpoint into the ONNX graph everything
downstream reads.

    uv run python -m training.nn.export_onnx --checkpoint <run>/best.pt

The checkpoint is a *training* artifact -- weights plus an optimiser, a
scheduler and a config nobody downstream cares about, loadable only by a
process that has torch and CUDA wheels installed.  ``model.onnx`` is the
*inference* artifact: one file, one runtime (onnxruntime CPU), and a SHA-256
that every posterior sidecar carries so a stale cache can never be mistaken for
a fresh one.

Three things here are load-bearing rather than boilerplate:

**`dynamo=False`.**  The pre-flight bisected it: the TorchDynamo exporter
exports this architecture *without error* and silently bakes ``time = 348``
into the graph -- `nn.GRU` and `F.avg_pool1d` both specialize the dim -- so
onnxruntime then rejects every other length.  The legacy TorchScript exporter
is deprecated (hence the torch 2.11 pin in ``pyproject.toml``) and is the only
path that keeps the axis symbolic today.  Because the failure is silent, the
declared dimensions are **asserted after export**, not assumed: an export that
loses the dynamic time axis fails here rather than in Task 5's decoder.

**Verify-then-build.**  The model is constructed from the checkpoint's own
``arch`` block, never from this module's defaults, and the constructed model's
``arch()`` is checked back against it.  A ``label_pool`` mismatch changes no
tensor shape, so ``load_state_dict`` would accept it cleanly and the decoder
would run at the wrong frame rate with nothing to see.

**Single-threaded sessions.**  ``session()`` is the only place an
``InferenceSession`` is built, so the determinism contract (CPU EP,
``intra_op_num_threads = 1``, ``inter_op_num_threads = 1``) is one definition
rather than a convention every caller has to remember.  Parallelism downstream
is per-*track*, across sessions -- which changes nothing about any single
track's numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .dataset import WINDOW_FRAMES
from .model import SectionCRNN, count_parameters
from .train import MODEL_VERSION, MODELS_DIR, weight_hash

from build_training_table import default_data_dir  # noqa: E402

MODEL_FILE = "model.onnx"
MODEL_META_FILE = "model.onnx.json"

# The Task 2b verdict: `v1b` epoch 30, val macro-F1 0.5211.  Named here rather
# than left to the caller so "the v1 model" means one file, not whichever run
# directory someone typed last.
DEFAULT_RUN = "v1b"
CHECKPOINT_FILE = "best.pt"

OPSET = 17
INPUT_NAME = "mel"
LABEL_OUTPUT = "label_logits"
BOUNDARY_OUTPUT = "boundary_logits"
# The symbolic dimension names asserted back out of the exported graph.
BATCH_AXIS = "batch"
TIME_AXIS = "time"
POOLED_TIME_AXIS = "time_pooled"


def model_dir(data_dir) -> Path:
    """``<data-dir>/models/v1`` -- where the exported graph lives."""
    return Path(data_dir) / MODELS_DIR / MODEL_VERSION


def default_checkpoint(data_dir) -> Path:
    return model_dir(data_dir) / DEFAULT_RUN / CHECKPOINT_FILE


def sha256_file(path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Checkpoint -> module
# --------------------------------------------------------------------------- #


def build_from_checkpoint(checkpoint: dict) -> SectionCRNN:
    """The exact module the checkpoint's weights belong to, in eval mode.

    Reads geometry from the checkpoint's ``arch`` block and checks the built
    module reports that same geometry back before any weight is loaded.  The
    round-trip is not paranoia about typos: ``arch`` is JSON (lists, not
    tuples) and a constructor that quietly reinterpreted one of its fields
    would produce a model that loads and runs and is wrong.
    """
    arch = checkpoint.get("arch")
    if not isinstance(arch, dict) or not arch:
        raise RuntimeError(
            "checkpoint carries no `arch` block -- it predates self-describing "
            "checkpoints and its geometry cannot be recovered from the weights "
            "alone (a label_pool mismatch is invisible to state_dict)"
        )
    model = SectionCRNN(**arch)
    if model.arch() != arch:
        raise RuntimeError(
            f"built model reports {model.arch()} from arch block {arch} -- the "
            f"constructor reinterpreted a field, so the weights would load into "
            f"a differently shaped network"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def load_checkpoint(path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def declared_axes(path) -> dict:
    """``{tensor name: [dim names or sizes]}`` read back out of the graph file.

    The only way to know the time axis survived: ``torch.onnx.export`` reports
    success either way.
    """
    graph = onnx.load(str(path)).graph
    axes: dict = {}
    for value in list(graph.input) + list(graph.output):
        axes[value.name] = [
            dim.dim_param or dim.dim_value
            for dim in value.type.tensor_type.shape.dim
        ]
    return axes


def export_model(model: SectionCRNN, path, *, window_frames: int = WINDOW_FRAMES) -> dict:
    """Write ``path`` and return its declared axes, having verified them.

    ``window_frames`` only decides the shape of the tracing input; the point of
    the assertions below is that it does not decide the shape of the graph.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, int(window_frames), model.n_mels)

    tmp = path.with_suffix(path.suffix + ".part")
    try:
        torch.onnx.export(
            model, (dummy,), str(tmp),
            dynamo=False,                 # see the module docstring; not a default
            opset_version=OPSET,
            input_names=[INPUT_NAME],
            output_names=[LABEL_OUTPUT, BOUNDARY_OUTPUT],
            dynamic_axes={
                INPUT_NAME: {0: BATCH_AXIS, 1: TIME_AXIS},
                LABEL_OUTPUT: {0: BATCH_AXIS, 1: POOLED_TIME_AXIS},
                BOUNDARY_OUTPUT: {0: BATCH_AXIS, 1: TIME_AXIS},
            },
        )
        onnx.checker.check_model(onnx.load(str(tmp)))
        axes = declared_axes(tmp)
        expected = {
            INPUT_NAME: [BATCH_AXIS, TIME_AXIS, model.n_mels],
            LABEL_OUTPUT: [BATCH_AXIS, POOLED_TIME_AXIS, model.n_classes],
            BOUNDARY_OUTPUT: [BATCH_AXIS, TIME_AXIS],
        }
        if axes != expected:
            raise RuntimeError(
                f"exported graph declares {axes}, expected {expected} -- a "
                f"specialized time axis means the graph only runs at the length "
                f"it was traced at"
            )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return axes


def export(checkpoint_path, out_path=None, *, window_frames: int = WINDOW_FRAMES) -> dict:
    """Checkpoint file -> ``model.onnx`` + ``model.onnx.json``; returns the metadata."""
    checkpoint_path = Path(checkpoint_path)
    out_path = Path(out_path) if out_path else checkpoint_path.parent.parent / MODEL_FILE

    state = load_checkpoint(checkpoint_path)
    model = build_from_checkpoint(state)
    axes = export_model(model, out_path, window_frames=window_frames)

    meta = {
        "checkpoint": str(checkpoint_path),
        "arch": model.arch(),
        "param_count": count_parameters(model),
        "epoch": state.get("epoch"),
        "metrics": state.get("metrics", {}).get("metrics", state.get("metrics")),
        "weight_hash": weight_hash(state["model"]),
        "model_sha": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
        "opset": OPSET,
        "dynamo": False,
        "declared_axes": axes,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }
    meta_path = out_path.with_name(out_path.name + ".json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return meta


# --------------------------------------------------------------------------- #
# Inference session
# --------------------------------------------------------------------------- #


def session_options() -> ort.SessionOptions:
    """The one pinned session configuration -- one thread, no tuning.

    ``intra_op_num_threads = 1`` is the determinism contract, not a performance
    choice: a multi-threaded reduction sums in whatever order the pool finishes
    in, and float addition is not associative.  Graph optimisations are left at
    the default level so the same onnxruntime version always builds the same
    plan from the same file.

    Split out from ``session`` because a constructed ``InferenceSession`` does
    not report the options it was built with, so this is the only object a test
    can actually assert on.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def session(path) -> ort.InferenceSession:
    """An onnxruntime session on the CPU EP under the pinned options."""
    return ort.InferenceSession(str(path), session_options(),
                                providers=["CPUExecutionProvider"])


def run_window(sess: ort.InferenceSession, mel: np.ndarray) -> tuple:
    """``mel [batch, time, n_mels] float32`` -> ``(label_logits, boundary_logits)``."""
    if mel.ndim != 3:
        raise ValueError(f"mel must be [batch, time, n_mels], got {mel.shape}")
    label, boundary = sess.run(
        [LABEL_OUTPUT, BOUNDARY_OUTPUT],
        {INPUT_NAME: np.ascontiguousarray(mel, dtype=np.float32)},
    )
    return label, boundary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help=f"training checkpoint (default: "
                             f"<data-dir>/{MODELS_DIR}/{MODEL_VERSION}/{DEFAULT_RUN}/"
                             f"{CHECKPOINT_FILE})")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"output graph (default: <data-dir>/{MODELS_DIR}/"
                             f"{MODEL_VERSION}/{MODEL_FILE})")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint or default_checkpoint(args.data_dir)
    out = args.out or model_dir(args.data_dir) / MODEL_FILE

    meta = export(checkpoint, out)
    print(f"{out}  {meta['bytes'] / 1024:.0f} KiB")
    print(f"  sha256      {meta['model_sha']}")
    print(f"  weights     {meta['weight_hash'][:16]}  epoch {meta['epoch']}")
    print(f"  arch        {meta['arch']}  ({meta['param_count']} params)")
    for name, axes in meta["declared_axes"].items():
        print(f"  {name:16s} {axes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
