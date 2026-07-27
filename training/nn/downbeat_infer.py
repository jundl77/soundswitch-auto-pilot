"""Downbeat activation sidecars: checkpoint -> ONNX -> one npz per track.

    uv run python -m training.nn.downbeat_infer --data-dir <corpus> --export
    uv run python -m training.nn.downbeat_infer --data-dir <corpus> --workers 6

This is the offline stand-in for the runtime, and the *only* thing Task 4 reads.
It exports the trained head to the one inference artifact (a graph plus a
SHA-256 every sidecar carries), slides it over each track exactly as the runtime
would, and aggregates the per-frame activation onto beat instants from **both**
input conditions:

* ``aubio`` -- the production pipeline's own beat stream, lifted bit for bit out
  of each track's cached sim report.  This is the deployment condition and the
  one the plan's gates bind to.
* ``expert`` -- the annotator's beat grid, the diagnostic upper bound.  The gap
  between the two is the aubio-degradation cost.

**The sidecar carries beat *instants*, never bar phase.**  The phase is truth,
and truth does not ride into an artifact generated for the test split.  The
expert beat times are an *input* (they define the diagnostic condition); the
phase they carry is what Task 4 scores against, read from the annotations.

**`dynamo=False`.**  The section head's pre-flight bisected it: the TorchDynamo
exporter exports a GRU model *without error* and silently bakes the traced time
length into the graph.  This architecture has the same GRU, so the same rule
applies and the declared axes are asserted after export rather than assumed.

**The window is the model's whole world.**  ``DownbeatCRNN`` is bidirectional
over a 16 s window, so a frame's activation depends on where in a window it was
read.  The runtime slides that window on its 100 ms callback and averages every
window covering a frame; so does this, on the section chain's geometry, because
one cadence in the runtime is the point.  A whole-track single pass would be two
orders of magnitude cheaper and would not be the same model.  ``EDGE_SEC`` of
each window is dropped before aggregation -- those frames have no context on one
side and a cold recurrent state -- except at the two ends of the track, which no
window's interior can reach; ``coverage`` records how many windows voted, so the
thin ends are visible rather than indistinguishable from the confident middle.

**Note that these activations are not the ones Task 2's 0.5512 describes.**  That
number was measured under the trainer's non-overlapping eval tiling.  The sliding
window is a better estimate of the same quantity and a different one; the naive
peak-picking floor has to be re-measured on these sidecars before any decoder
gain is attributed to the decoder.

**The aggregation window was measured, not assumed.**  Task 1 left it open with
symmetric ``[-1, +1]`` as the null hypothesis, explicitly refusing to infer it
from where input flux peaks.  ``lag_profile`` is the instrument: it reports mean
activation against frame offset from the annotated instants, so where a *trained*
activation actually peaks is a measurement in the report rather than an argument.
``--lag-profile`` runs it over existing sidecars.

The window itself, and the arithmetic that applies it, live in
``downbeat_decoder``: a half-beat candidate grid has to be aggregated at decode
time off the stored curve, and in the runtime that aggregation is decode-path
work.  This module owns the *curve* and caches the per-beat case beside it.

**Determinism is a written contract.**  One pinned single-threaded CPU session
(``export_onnx.session`` -- one definition, not a convention), windows summed in
a fixed order into float64, parallelism per *track* so no track's numbers depend
on the worker count, and the archive written with a fixed member order and epoch
by ``infer.save_posteriors`` rather than by ``np.savez``, whose reproducibility
is a CPython default rather than an API promise.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .dataset import FEATURES_DIR, FRAME_SEC, WINDOW_FRAMES, load_sidecar, make_splits
# The aggregation window and its arithmetic live in the decoder: in the runtime
# putting an activation onto a beat stream is decode-path work, and a half-beat
# candidate grid has to be aggregated at decode time off the stored curve.  This
# module owns the *curve*; it borrows the window so the two cannot disagree.
from .downbeat_decoder import (
    AGG_HI_FRAMES,
    AGG_LO_FRAMES,
    aggregate_at_beats,
    nearest_frames,
)
from .downbeat_model import PARAM_BUDGET, DownbeatCRNN
from .downbeat_train import MODEL_VERSION
from .export_onnx import OPSET, declared_axes, session, sha256_file
from .model import count_parameters
from .train import BEST_CHECKPOINT, MODELS_DIR, weight_hash
# ``_sigmoid`` and ``save_posteriors`` are the section chain's own: the numerically
# stable sigmoid and the byte-reproducible archive writer are exactly the same
# problem here, and a second copy of either is a second thing to keep true.
from .infer import (
    EDGE_FRAMES,
    HOP_FRAMES,
    _sigmoid,
    contribution_span,
    save_posteriors as save_sidecar,
    usable_frames,
    window_offsets,
)

# ``_read_json_gz`` is imported private and deliberately: it is the reader that
# wrote the cache, and a second one that disagrees about the envelope is how a
# report cache comes to be read as something it is not.  Same trade as the
# downbeat dataset's private ``_take``.
from build_training_table import (  # noqa: E402
    _read_json_gz,
    default_data_dir,
    report_path,
)
from raveform_fetch_annotations import beat_csv_path, load_tracks  # noqa: E402

MODEL_FILE = "downbeat.onnx"
MODEL_META_FILE = "downbeat.onnx.json"
SIDECAR_DIR = "downbeat_posteriors"
MANIFEST_FILE = "manifest.json"

# The Task 2 verdict: `downbeat_v1` epoch 20, val peak F1 0.5512.  Named here so
# "the downbeat model" is one file rather than whichever run someone typed last.
DEFAULT_RUN = "downbeat_v1"
CHECKPOINT_FILE = BEST_CHECKPOINT

INPUT_NAME = "mel"
OUTPUT_NAME = "downbeat_logits"
BATCH_AXIS = "batch"
TIME_AXIS = "time"

# The two input conditions, in the order they are written and reported.
CONDITIONS = ("aubio", "expert")


def model_dir(data_dir) -> Path:
    """``<data-dir>/models/downbeat_v1`` -- where the exported graph lives."""
    return Path(data_dir) / MODELS_DIR / MODEL_VERSION


def default_checkpoint(data_dir) -> Path:
    return model_dir(data_dir) / DEFAULT_RUN / CHECKPOINT_FILE


# --------------------------------------------------------------------------- #
# Checkpoint -> module
# --------------------------------------------------------------------------- #


def load_downbeat_checkpoint(path) -> dict:
    """Read a downbeat ``best.pt`` and check it is shaped like one.

    Deliberately *not* shared with the section head's loader.  The two payloads
    differ: this one is flat (``["f1"]`` and ``["metrics"]["f1"]`` are both
    floats), the section head's nests its metric block one level deeper, and a
    "shared" loader would have to guess which it was holding.  The fields are
    checked by name because the alternative -- a ``KeyError`` three functions
    later, or worse a ``.get`` that returns ``None`` and exports a graph with no
    ``pos_weight`` recorded -- costs more than this does.
    """
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    for field in ("model", "arch", "pos_weight"):
        if field not in state:
            raise RuntimeError(
                f"{path}: checkpoint carries no `{field}` -- it is not a "
                f"downbeat_train best.pt, or it predates self-describing "
                f"checkpoints and cannot be exported safely")
    if not isinstance(state["arch"], dict) or not state["arch"]:
        raise RuntimeError(f"{path}: `arch` is not a geometry block")
    return state


def build_from_checkpoint(state: dict) -> DownbeatCRNN:
    """The exact module the checkpoint's weights belong to, in eval mode.

    Geometry comes from the checkpoint's own ``arch`` block and the built
    module's ``arch()`` is checked back against it *before* any weight loads.
    The round trip is not typo paranoia: ``arch`` survives JSON as lists rather
    than tuples, and a constructor that quietly reinterpreted one of its fields
    would produce a model that loads, runs, and is wrong.
    """
    arch = state["arch"]
    model = DownbeatCRNN(**arch)
    if model.arch() != arch:
        raise RuntimeError(
            f"built model reports {model.arch()} from arch block {arch} -- the "
            f"constructor reinterpreted a field, so the weights would load into "
            f"a differently shaped network")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def export_model(model: DownbeatCRNN, path, *,
                 window_frames: int = WINDOW_FRAMES) -> dict:
    """Write ``path`` and return its declared axes, having verified them.

    ``window_frames`` decides the shape of the tracing input only; the whole
    point of the assertion below is that it does not decide the shape of the
    graph.
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
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: BATCH_AXIS, 1: TIME_AXIS},
                          OUTPUT_NAME: {0: BATCH_AXIS, 1: TIME_AXIS}},
        )
        onnx.checker.check_model(onnx.load(str(tmp)))
        axes = declared_axes(tmp)
        expected = {INPUT_NAME: [BATCH_AXIS, TIME_AXIS, model.n_mels],
                    OUTPUT_NAME: [BATCH_AXIS, TIME_AXIS]}
        if axes != expected:
            raise RuntimeError(
                f"exported graph declares {axes}, expected {expected} -- a "
                f"specialized time axis means the graph only runs at the length "
                f"it was traced at")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return axes


def export(checkpoint_path, out_path=None, *,
           window_frames: int = WINDOW_FRAMES) -> dict:
    """Checkpoint file -> ``downbeat.onnx`` + its metadata sidecar."""
    checkpoint_path = Path(checkpoint_path)
    out_path = Path(out_path) if out_path else checkpoint_path.parent.parent / MODEL_FILE

    state = load_downbeat_checkpoint(checkpoint_path)
    model = build_from_checkpoint(state)
    params = count_parameters(model)
    if params > PARAM_BUDGET:
        raise RuntimeError(
            f"{params} parameters exceeds the {PARAM_BUDGET} budget the spec "
            f"sets for this head -- this is not the model that was validated")
    axes = export_model(model, out_path, window_frames=window_frames)

    meta = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha": sha256_file(checkpoint_path),
        "arch": model.arch(),
        "param_count": params,
        "epoch": state.get("epoch"),
        "f1": state.get("f1"),
        "pos_weight": float(state["pos_weight"]),
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
    with open(out_path.with_name(out_path.name + ".json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return meta


def run_window(sess, mel: np.ndarray) -> np.ndarray:
    """``mel [batch, time, n_mels] float32`` -> ``downbeat_logits [batch, time]``."""
    if mel.ndim != 3:
        raise ValueError(f"mel must be [batch, time, n_mels], got {mel.shape}")
    (logits,) = sess.run([OUTPUT_NAME],
                         {INPUT_NAME: np.ascontiguousarray(mel, dtype=np.float32)})
    return logits


# --------------------------------------------------------------------------- #
# One track's activation
# --------------------------------------------------------------------------- #


class TrackActivation(NamedTuple):
    """One track's whole-track activation and the geometry it was produced with.

    The geometry travels with the result rather than being re-read from the
    module constants at write time: ``infer_track`` takes window, hop and edge as
    arguments, and a sidecar that recorded the defaults while holding
    non-default numbers would be worse than an unlabelled one -- the cache key
    built on it would accept it.
    """

    activation: np.ndarray      # [n] float32, mean of the per-window sigmoids
    coverage: np.ndarray        # [n] uint16, windows that voted on each frame
    n_frames: int
    windows: int
    window_frames: int
    hop_frames: int
    edge_frames: int


def infer_track(sess, mel: np.ndarray, *, window_frames: int = WINDOW_FRAMES,
                hop_frames: int = HOP_FRAMES,
                edge_frames: int = EDGE_FRAMES) -> TrackActivation:
    """Whole-track downbeat activation by sliding ``sess``'s graph over ``mel``.

    ``mel`` is ``[n, n_mels]`` as written by the batch sim.  The frame count is
    truncated exactly as ``WindowDataset`` truncates it, so the activation array
    is the same length as the targets Task 1 built and the two can be indexed
    against one another without an off-by-one.
    """
    n_frames = usable_frames(len(mel))
    if n_frames < 1:
        raise RuntimeError(f"track has {len(mel)} mel frames -- nothing to infer")
    if hop_frames > window_frames - 2 * edge_frames:
        raise ValueError(
            f"hop {hop_frames} exceeds the usable window interior "
            f"{window_frames - 2 * edge_frames} -- frames would go uncovered")
    mel = np.ascontiguousarray(mel[:n_frames], dtype=np.float32)

    padded = mel
    if n_frames < window_frames:
        padded = np.zeros((window_frames, mel.shape[1]), dtype=np.float32)
        padded[:n_frames] = mel

    total = np.zeros(n_frames, dtype=np.float64)
    coverage = np.zeros(n_frames, dtype=np.int32)

    offsets = window_offsets(n_frames, window_frames=window_frames,
                             hop_frames=hop_frames)
    for index, offset in enumerate(offsets):
        logits = run_window(sess, padded[offset:offset + window_frames][None])
        activation = _sigmoid(logits[0])

        lo, hi = contribution_span(
            offset, n_frames=n_frames, first=index == 0,
            last=index == len(offsets) - 1,
            window_frames=window_frames, edge_frames=edge_frames)
        if hi <= lo:                                   # pragma: no cover
            continue
        total[lo:hi] += activation[lo - offset:hi - offset]
        coverage[lo:hi] += 1

    if not coverage.all():                             # pragma: no cover
        raise RuntimeError(
            f"{int((coverage == 0).sum())} frames were covered by no window -- "
            f"the hop/edge geometry leaves holes")

    return TrackActivation((total / coverage).astype(np.float32),
                           coverage.astype(np.uint16), n_frames, len(offsets),
                           int(window_frames), int(hop_frames), int(edge_frames))


# --------------------------------------------------------------------------- #
# Where the activation sits against the grid
# --------------------------------------------------------------------------- #


def lag_profile(activation: np.ndarray, frames, radius: int = 4) -> np.ndarray:
    """Mean activation at each frame offset from a set of reference frames.

    The instrument the aggregation window is chosen with.  Task 1 refused to
    infer the window from where input flux peaks -- that says where the
    *evidence* arrives, not where a model fitted to a target sitting on the
    instant puts its output -- so this measures the trained activation directly.
    Index ``radius`` is offset 0.
    """
    activation = np.asarray(activation, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    profile = np.full(2 * int(radius) + 1, np.nan, dtype=np.float64)
    for index, offset in enumerate(range(-int(radius), int(radius) + 1)):
        shifted = frames + offset
        keep = (shifted >= 0) & (shifted < len(activation))
        if keep.any():
            profile[index] = activation[shifted[keep]].mean()
    return profile


# --------------------------------------------------------------------------- #
# Beat streams
# --------------------------------------------------------------------------- #


def aubio_beat_times(data_dir, youtube_id: str) -> np.ndarray:
    """The production pipeline's own beat instants, out of the cached sim report.

    Not a re-derivation: this is the exact stream the live engine produces, which
    is what makes the primary evaluation condition the deployment condition.
    """
    path = report_path(Path(data_dir), youtube_id)
    if not path.exists():
        raise RuntimeError(
            f"no cached sim report for {youtube_id} at {path} -- the live "
            f"condition has no beat stream without it")
    beats = _read_json_gz(path)["report"]["beats"]
    return np.asarray([float(beat["t"]) for beat in beats], dtype=np.float64)


def expert_beat_times(grid) -> np.ndarray:
    """The annotator's beat instants -- the diagnostic condition's input.

    Only the instants.  The phase is truth and stays in the annotations.
    """
    return np.asarray(grid.times, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Sidecar I/O
# --------------------------------------------------------------------------- #


def sidecar_arrays(track: TrackActivation, beats: dict, model_sha: str,
                   pos_weight: float, *, lo: int = AGG_LO_FRAMES,
                   hi: int = AGG_HI_FRAMES) -> dict:
    """Everything one sidecar records, in write order.

    The frame time base is stated rather than implied: ``activation[k]`` is at
    ``t0 + k * frame_sec`` with ``t0 == frame_sec``, the convention the mel
    sidecars and Task 1's targets share.  ``pos_weight`` rides along because the
    activation is a *ranking* score inflated by it, and a consumer that wants a
    probability-shaped input needs the number to undo the shift with.
    """
    arrays: dict = {"activation": track.activation, "coverage": track.coverage}
    for condition in CONDITIONS:
        times, scores, counts = beats[condition]
        arrays[f"{condition}_beat_time"] = np.asarray(times, dtype=np.float64)
        arrays[f"{condition}_beat_score"] = np.asarray(scores, dtype=np.float64)
        arrays[f"{condition}_beat_frames"] = np.asarray(counts, dtype=np.int32)
    arrays.update({
        "frame_sec": np.float64(FRAME_SEC),
        "t0": np.float64(FRAME_SEC),
        "n_frames": np.int32(track.n_frames),
        "windows": np.int32(track.windows),
        "window_frames": np.int32(track.window_frames),
        "hop_frames": np.int32(track.hop_frames),
        "edge_frames": np.int32(track.edge_frames),
        "agg_lo_frames": np.int32(lo),
        "agg_hi_frames": np.int32(hi),
        "pos_weight": np.float64(pos_weight),
        "model_sha": np.str_(model_sha),
    })
    return arrays


def sidecar_is_current(path, model_sha: str) -> bool:
    """Is ``path`` already this model's answer, on this geometry?

    (model hash, track) as the spec asks, plus the window *and* aggregation
    geometry: either produces different numbers from the same graph and would
    otherwise be silently reused.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as archive:
            return (str(archive["model_sha"]) == model_sha
                    and int(archive["window_frames"]) == WINDOW_FRAMES
                    and int(archive["hop_frames"]) == HOP_FRAMES
                    and int(archive["edge_frames"]) == EDGE_FRAMES
                    and int(archive["agg_lo_frames"]) == AGG_LO_FRAMES
                    and int(archive["agg_hi_frames"]) == AGG_HI_FRAMES)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


# --------------------------------------------------------------------------- #
# Corpus run
# --------------------------------------------------------------------------- #


def split_ids(data_dir, splits=("val", "test")) -> list:
    """Every id in the named splits, sorted.

    Val and test by default and nothing else: the decoder is tuned on val and the
    verdict reads test once, and 962 train tracks would cost hours of inference
    that nothing downstream reads.  Generating the test sidecars now is
    inputs-only -- no label is read here and none is written -- and regenerating
    them later at a different code revision is the larger risk.
    """
    assignment = make_splits(Path(data_dir), write=False)
    ids: set = set()
    for name in splits:
        ids.update(str(i) for i in assignment[name])
    return sorted(ids)


def _beat_streams(data_dir: Path, youtube_id: str, grid,
                  activation: np.ndarray) -> dict:
    streams: dict = {}
    for condition, times in (("aubio", aubio_beat_times(data_dir, youtube_id)),
                             ("expert", expert_beat_times(grid))):
        scores, counts = aggregate_at_beats(activation, times, FRAME_SEC, FRAME_SEC)
        streams[condition] = (times, scores, counts)
    return streams


def generate(data_dir, *, model_path=None, out_dir=None, ids=None,
             workers: int = 1, force: bool = False,
             progress_every: int = 25) -> dict:
    """Write an activation sidecar for every id; returns the run manifest."""
    from .downbeat_dataset import load_beat_grid

    data_dir = Path(data_dir)
    model_path = Path(model_path) if model_path else model_dir(data_dir) / MODEL_FILE
    out_dir = Path(out_dir) if out_dir else data_dir / SIDECAR_DIR
    ids = list(ids) if ids is not None else split_ids(data_dir)

    model_sha = sha256_file(model_path)
    meta_path = model_path.with_name(model_path.name + ".json")
    pos_weight = float(json.loads(meta_path.read_text(encoding="utf-8"))["pos_weight"])
    features = data_dir / FEATURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # One shared session: onnxruntime releases the GIL inside `Run`, so N python
    # threads give N single-threaded inferences in parallel without the per-process
    # model copy a ProcessPool needs.  Each track is handled start to finish by one
    # task, so nothing about a track's output depends on how many threads run.
    sess = session(model_path)
    records_by_id = {str(track.get("id")): track for track in load_tracks(data_dir)}

    started = time.perf_counter()
    done = {"count": 0}
    records: list = []

    def one(youtube_id: str) -> dict:
        path = out_dir / f"{youtube_id}.npz"
        if not force and sidecar_is_current(path, model_sha):
            return {"youtube_id": youtube_id, "cached": True,
                    "bytes": path.stat().st_size}
        clock = time.perf_counter()
        try:
            record = records_by_id.get(youtube_id)
            grid_path = beat_csv_path(data_dir, record) if record is not None else None
            if grid_path is None or not Path(grid_path).exists():
                raise RuntimeError(f"no expert beat grid for {youtube_id}")
            grid = load_beat_grid(grid_path)
            track = infer_track(sess, load_sidecar(features / f"{youtube_id}.npz"))
            streams = _beat_streams(data_dir, youtube_id, grid, track.activation)
            save_sidecar(path, sidecar_arrays(track, streams, model_sha, pos_weight))
        except Exception as error:
            # A corpus run is long and the sidecars are cached, so one unreadable
            # track must not throw the rest away.  A sidecar left over from an
            # older model or geometry is deleted on the way out: otherwise the
            # manifest says "failed" while the file still answers to np.load and
            # the next reader consumes last week's numbers believing they are
            # this model's.  A file that *is* current is left alone -- a transient
            # read error must not destroy a good artifact.
            stale = path.exists() and not sidecar_is_current(path, model_sha)
            if stale:
                path.unlink(missing_ok=True)
            print(f"  FAILED {youtube_id}: {error!r}"
                  f"{' (removed stale sidecar)' if stale else ''}", flush=True)
            return {"youtube_id": youtube_id, "cached": False, "removed_stale": stale,
                    "error": f"{type(error).__name__}: {error}"}
        return {"youtube_id": youtube_id, "cached": False, "frames": track.n_frames,
                "windows": track.windows, "bytes": path.stat().st_size,
                "beats": {name: int(len(streams[name][0])) for name in CONDITIONS},
                "seconds": round(time.perf_counter() - clock, 3)}

    def report(record: dict) -> dict:
        done["count"] += 1
        index = done["count"]
        if progress_every and (index % progress_every == 0 or index == len(ids)):
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(ids) - index) / rate if rate else 0.0
            print(f"  {index}/{len(ids)}  {elapsed / 60:.1f} min elapsed, "
                  f"{eta / 60:.1f} min left  ({rate * 60:.1f} tracks/min)", flush=True)
        return record

    if workers > 1:
        with ThreadPoolExecutor(int(workers)) as pool:
            for record in pool.map(one, ids):
                records.append(report(record))
    else:
        for youtube_id in ids:
            records.append(report(one(youtube_id)))

    wall = time.perf_counter() - started
    failed = [r for r in records if r.get("error")]
    computed = [r for r in records if not r.get("cached") and not r.get("error")]
    manifest = {
        "failed": len(failed),
        "model": str(model_path),
        "model_sha": model_sha,
        "pos_weight": pos_weight,
        "frame_sec": FRAME_SEC,
        "window_frames": WINDOW_FRAMES,
        "hop_frames": HOP_FRAMES,
        "hop_sec": HOP_FRAMES * FRAME_SEC,
        "edge_frames": EDGE_FRAMES,
        "edge_sec": EDGE_FRAMES * FRAME_SEC,
        "agg_lo_frames": AGG_LO_FRAMES,
        "agg_hi_frames": AGG_HI_FRAMES,
        "conditions": list(CONDITIONS),
        "tracks": len(records),
        "computed": len(computed),
        "cached": len(records) - len(computed) - len(failed),
        "frames": sum(r.get("frames", 0) for r in computed),
        "windows": sum(r.get("windows", 0) for r in computed),
        "bytes": sum(r.get("bytes", 0) for r in records),
        "wall_seconds": round(wall, 1),
        "workers": int(workers),
        "records": sorted(records, key=lambda r: r["youtube_id"]),
    }
    with open(out_dir / MANIFEST_FILE, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


# --------------------------------------------------------------------------- #
# The aggregation-window measurement
# --------------------------------------------------------------------------- #


def measure_lag(data_dir, ids, *, radius: int = 4, sidecar_dir=None) -> dict:
    """Where the trained activation peaks relative to annotated downbeats.

    Pooled over tracks, on existing sidecars, at frame resolution.  A profile
    that peaks anywhere but 0 says the aggregation window has to move; adopting a
    shift on any other evidence would bake it in (Task 1 §8 F2).
    """
    from .downbeat_dataset import load_beat_grid

    data_dir = Path(data_dir)
    sidecar_dir = Path(sidecar_dir) if sidecar_dir else data_dir / SIDECAR_DIR
    records_by_id = {str(track.get("id")): track for track in load_tracks(data_dir)}

    downbeat = np.zeros(2 * radius + 1, dtype=np.float64)
    control = np.zeros(2 * radius + 1, dtype=np.float64)
    peaks = np.zeros(2 * radius + 1, dtype=np.int64)
    tracks = 0
    for youtube_id in ids:
        path = sidecar_dir / f"{youtube_id}.npz"
        if not path.exists():
            continue
        with np.load(path) as archive:
            activation = np.asarray(archive["activation"], dtype=np.float64)
        grid = load_beat_grid(beat_csv_path(data_dir, records_by_id[youtube_id]))
        frames = nearest_frames(grid.downbeat_times, len(activation),
                                FRAME_SEC, FRAME_SEC)
        frames = frames[frames >= 0]
        others = nearest_frames(grid.times[grid.phases != 1], len(activation),
                                FRAME_SEC, FRAME_SEC)
        if not frames.size:
            continue
        downbeat += lag_profile(activation, frames, radius)
        control += lag_profile(activation, others[others >= 0], radius)
        # Per-downbeat argmax as well as the mean: a mean can be dragged by the
        # shoulders of a peak that is exactly centred, and the modal offset
        # cannot.
        window = np.clip(frames[:, None] + np.arange(-radius, radius + 1),
                         0, len(activation) - 1)
        peaks += np.bincount(activation[window].argmax(axis=1),
                             minlength=2 * radius + 1)
        tracks += 1

    offsets = list(range(-radius, radius + 1))
    return {"tracks": tracks, "offsets": offsets,
            "downbeat_mean": (downbeat / max(tracks, 1)).tolist(),
            "offbeat_mean": (control / max(tracks, 1)).tolist(),
            "peak_histogram": peaks.tolist(),
            "argmax_offset": offsets[int(np.argmax(downbeat))],
            "modal_peak_offset": offsets[int(np.argmax(peaks))]}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help=f"training checkpoint (default: <data-dir>/{MODELS_DIR}/"
                             f"{MODEL_VERSION}/{DEFAULT_RUN}/{CHECKPOINT_FILE})")
    parser.add_argument("--model", type=Path, default=None,
                        help=f"exported graph (default: <data-dir>/{MODELS_DIR}/"
                             f"{MODEL_VERSION}/{MODEL_FILE})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"default: <data-dir>/{SIDECAR_DIR}")
    parser.add_argument("--export", action="store_true",
                        help="export the graph from --checkpoint and stop")
    parser.add_argument("--lag-profile", action="store_true",
                        help="measure where the activation peaks against the "
                             "annotated downbeats, on existing sidecars, and stop")
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--ids", nargs="*", default=None,
                        help="youtube ids (default: every id in --splits)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N ids")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--force", action="store_true",
                        help="recompute even where the sidecar matches this model")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.model or model_dir(args.data_dir) / MODEL_FILE

    if args.export:
        meta = export(args.checkpoint or default_checkpoint(args.data_dir), out)
        print(f"{out}  {meta['bytes'] / 1024:.0f} KiB")
        print(f"  sha256      {meta['model_sha']}")
        print(f"  checkpoint  {meta['checkpoint_sha'][:16]}  epoch {meta['epoch']} "
              f"F1 {meta['f1']:.4f}  pos_weight {meta['pos_weight']:.3f}")
        print(f"  arch        {meta['arch']}  ({meta['param_count']} params)")
        for name, axes in meta["declared_axes"].items():
            print(f"  {name:16s} {axes}")
        return 0

    ids = args.ids if args.ids is not None else split_ids(args.data_dir, args.splits)
    if args.limit:
        ids = ids[:args.limit]

    if args.lag_profile:
        profile = measure_lag(args.data_dir, ids, sidecar_dir=args.out_dir)
        print(f"aggregation-window measurement over {profile['tracks']} tracks")
        print(f"  offset {'':>4}" + "".join(f"{o:>9d}" for o in profile["offsets"]))
        print(f"  downbeat  " + "".join(f"{v:9.4f}" for v in profile["downbeat_mean"]))
        print(f"  off-beat  " + "".join(f"{v:9.4f}" for v in profile["offbeat_mean"]))
        print(f"  peak at   " + "".join(f"{v:9d}" for v in profile["peak_histogram"]))
        print(f"  mean argmax {profile['argmax_offset']:+d} frames | "
              f"modal peak {profile['modal_peak_offset']:+d} frames "
              f"({profile['modal_peak_offset'] * FRAME_SEC * 1000:+.1f} ms)")
        return 0

    manifest = generate(args.data_dir, model_path=args.model, out_dir=args.out_dir,
                        ids=ids, workers=args.workers, force=args.force,
                        progress_every=args.progress_every)
    print(f"{manifest['tracks']} tracks ({manifest['computed']} computed, "
          f"{manifest['cached']} cached) in {manifest['wall_seconds'] / 60:.1f} min")
    print(f"  {manifest['windows']} windows over {manifest['frames']} frames "
          f"({manifest['frames'] * FRAME_SEC / 3600:.1f} audio hours)")
    print(f"  {manifest['bytes'] / (1 << 20):.1f} MiB of sidecars, "
          f"model {manifest['model_sha'][:16]}")
    if manifest["failed"]:
        print(f"  {manifest['failed']} FAILED -- see {MANIFEST_FILE}; rerun to "
              f"retry (finished tracks are cached)")
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
