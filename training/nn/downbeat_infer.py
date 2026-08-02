"""Downbeat activation sidecars: checkpoint -> ONNX -> one npz per track."""
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
from .infer import (
    EDGE_FRAMES,
    HOP_FRAMES,
    _sigmoid,
    contribution_span,
    save_posteriors as save_sidecar,
    usable_frames,
    window_offsets,
)

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

DEFAULT_RUN = "downbeat_v1"
CHECKPOINT_FILE = BEST_CHECKPOINT

INPUT_NAME = "mel"
OUTPUT_NAME = "downbeat_logits"
BATCH_AXIS = "batch"
TIME_AXIS = "time"

CONDITIONS = ("live", "expert")


def model_dir(data_dir) -> Path:
    return Path(data_dir) / MODELS_DIR / MODEL_VERSION


def default_checkpoint(data_dir) -> Path:
    return model_dir(data_dir) / DEFAULT_RUN / CHECKPOINT_FILE


def load_downbeat_checkpoint(path) -> dict:
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


def export_model(model: DownbeatCRNN, path, *,
                 window_frames: int = WINDOW_FRAMES) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, int(window_frames), model.n_mels)

    tmp = path.with_suffix(path.suffix + ".part")
    try:
        # torch's dynamo exporter bakes the traced time length into a GRU graph
        # without erroring, so the declared axes are asserted after export.
        torch.onnx.export(
            model, (dummy,), str(tmp),
            dynamo=False,
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
    if mel.ndim != 3:
        raise ValueError(f"mel must be [batch, time, n_mels], got {mel.shape}")
    (logits,) = sess.run([OUTPUT_NAME],
                         {INPUT_NAME: np.ascontiguousarray(mel, dtype=np.float32)})
    return logits


class TrackActivation(NamedTuple):
    activation: np.ndarray
    coverage: np.ndarray
    n_frames: int
    windows: int
    window_frames: int
    hop_frames: int
    edge_frames: int


def infer_track(sess, mel: np.ndarray, *, window_frames: int = WINDOW_FRAMES,
                hop_frames: int = HOP_FRAMES,
                edge_frames: int = EDGE_FRAMES) -> TrackActivation:
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


def lag_profile(activation: np.ndarray, frames, radius: int = 4) -> np.ndarray:
    activation = np.asarray(activation, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    profile = np.full(2 * int(radius) + 1, np.nan, dtype=np.float64)
    for index, offset in enumerate(range(-int(radius), int(radius) + 1)):
        shifted = frames + offset
        keep = (shifted >= 0) & (shifted < len(activation))
        if keep.any():
            profile[index] = activation[shifted[keep]].mean()
    return profile


def live_beat_times(data_dir, youtube_id: str) -> np.ndarray:
    path = report_path(Path(data_dir), youtube_id)
    if not path.exists():
        raise RuntimeError(
            f"no cached sim report for {youtube_id} at {path} -- the live "
            f"condition has no beat stream without it")
    beats = _read_json_gz(path)["report"]["beats"]
    return np.asarray([float(beat["t"]) for beat in beats], dtype=np.float64)


def expert_beat_times(grid) -> np.ndarray:
    return np.asarray(grid.times, dtype=np.float64)


def sidecar_arrays(track: TrackActivation, beats: dict, model_sha: str,
                   pos_weight: float, *, lo: int = AGG_LO_FRAMES,
                   hi: int = AGG_HI_FRAMES) -> dict:
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


def split_ids(data_dir, splits=("val", "test")) -> list:
    assignment = make_splits(Path(data_dir), write=False)
    ids: set = set()
    for name in splits:
        ids.update(str(i) for i in assignment[name])
    return sorted(ids)


def _beat_streams(data_dir: Path, youtube_id: str, grid,
                  activation: np.ndarray) -> dict:
    streams: dict = {}
    for condition, times in (("live", live_beat_times(data_dir, youtube_id)),
                             ("expert", expert_beat_times(grid))):
        scores, counts = aggregate_at_beats(activation, times, FRAME_SEC, FRAME_SEC)
        streams[condition] = (times, scores, counts)
    return streams


def generate(data_dir, *, model_path=None, out_dir=None, ids=None,
             workers: int = 1, force: bool = False,
             progress_every: int = 25) -> dict:
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

    # Threads share one single-threaded session: onnxruntime frees the GIL in `Run`,
    # and a threaded reduction sums in completion order -- float add is not associative.
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


def measure_lag(data_dir, ids, *, radius: int = 4, sidecar_dir=None) -> dict:
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
                             "annotated downbeats, on existing sidecars, and stop; "
                             "val only, because it reads bar phase")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="default: val+test for generation, val alone for "
                             "--lag-profile (which reads a tuning quantity)")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="youtube ids (default: every id in --splits)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N ids")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--force", action="store_true",
                        help="recompute even where the sidecar matches this model")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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

    splits = args.splits if args.splits is not None else (
        ["val"] if args.lag_profile else ["val", "test"])
    ids = args.ids if args.ids is not None else split_ids(args.data_dir, splits)
    if args.limit:
        ids = ids[:args.limit]

    if args.lag_profile:
        stray = sorted(set(ids) - set(split_ids(args.data_dir, ["val"])))
        if stray:
            parser.error(
                f"--lag-profile reads annotated bar phase to choose a decoder "
                f"parameter, so it is val's alone; {len(stray)} of the requested "
                f"ids are not in val ({stray[:3]}). The test split is the "
                f"verdict's to read once.")
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
