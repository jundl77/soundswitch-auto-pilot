"""Sliding-window ONNX inference -> one posterior sidecar per track.

    uv run python -m training.nn.infer --data-dir <corpus> --workers 12

This is the offline stand-in for the runtime: the same graph, the same window,
the same cadence, the same aggregation.  Tasks 4-6 (priors, decoder, sweeps,
evaluation) then never touch a neural network again -- they read
``posteriors/<youtube_id>.npz`` and are pure numpy over a cached array, which is
what makes a decoder parameter sweep cost seconds instead of GPU-hours.

**The window is the model's whole world.**  ``SectionCRNN`` is bidirectional
over a 16 s window, so a frame's posterior depends on which window it was read
in -- 8 s from the left edge is a different prediction than 1 s from the right
edge of the next window.  The runtime slides that window every 100 ms and
averages every posterior covering a frame (pyannote's fix for window flicker),
so this does too.  A whole-track single pass would be 150x cheaper and would
not be the same model.

**Never read the edge.**  The design spec forbids consuming a window's outermost
frames: they have no context on one side and a cold GRU state, and they are the
most miscalibrated position in the window.  ``EDGE_SEC`` of every window is
therefore dropped before aggregation.  The one exception is arithmetic rather
than policy: the first ``EDGE_SEC`` of a track and its last ``EDGE_SEC`` are
reachable by *no* window's interior, so the first and last window donate their
outer margin there rather than leave the decoder with an undefined posterior.
``coverage`` records how many windows voted on each frame, so those thin ends
are visible to whoever reads the file instead of being indistinguishable from
the confident middle.

**The final window deliberately re-overlaps its predecessor** (a track length is
not a whole number of hops), exactly as ``WindowDataset`` does in eval mode --
which is why aggregation *averages* per frame and never concatenates window
outputs end to end.  Concatenation would double-count the tail and shift every
frame after it.

**The boundary head is stored raw.**  Task 2b measured per-window normalisation
(z-score, min-max) destroying 65 % of its PR-AUC: the sigmoid *is* cross-window
comparable at this corpus size.  So the stored value is the plain mean of the
per-window sigmoids -- a ranking score, not a probability (``pos_weight`` was
44.8 during training), and nothing downstream may read it as P(boundary).

**Determinism is a written contract, and npz is a zip.**  Two identical runs
must produce byte-identical files, and half of that guarantee is about the
container rather than the numbers.  ``np.savez`` happens to be reproducible on
CPython 3.11 -- ``ZipInfo`` defaults to the 1980 epoch when ``ZipFile.open``
creates one -- but that is an implementation default, not an API promise, and
``savez_compressed`` additionally folds the zlib build into the bytes.
``save_posteriors`` therefore writes the archive itself: fixed member order,
fixed timestamp, no compression, so the file depends on nothing but the array
contents.  The rest follows the session pinning in ``export_onnx.session``:
windows are summed in a fixed order into float64, and parallelism is per
*track* -- no track's numbers depend on how many workers ran.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .dataset import (
    FEATURES_DIR,
    FRAME_SEC,
    LABEL_POOL,
    NUM_CLASSES,
    WINDOW_FRAMES,
    candidate_tracks,
    load_sidecar,
)
from .export_onnx import MODEL_FILE, model_dir, run_window, session, sha256_file

from build_training_table import default_data_dir  # noqa: E402

POSTERIORS_DIR = "posteriors"
MANIFEST_FILE = "manifest.json"

# The runtime's 100 ms callback cadence, on the mel grid.  46.44 ms frames put
# the nearest legal hop at 2 frames = 92.9 ms; legal because the label head
# speaks at half the frame rate, so an odd hop would land its output between two
# cells of the track-wide pooled grid and force a resample nobody wants.
HOP_SEC = 0.100
# The spec's never-read-the-edge margin, rounded UP to the pooled grid.
EDGE_SEC = 1.0


def _aligned(seconds: float, *, up: bool) -> int:
    """Seconds -> whole mel frames, snapped to a multiple of ``LABEL_POOL``.

    The snap happens in *pooled groups*, not in frames: rounding to frames and
    then flooring to a multiple would round a margin back DOWN below the second
    the spec asks for, which is the one direction that must not happen.
    """
    groups = seconds / FRAME_SEC / LABEL_POOL
    return LABEL_POOL * (math.ceil(groups) if up else max(1, round(groups)))


HOP_FRAMES = _aligned(HOP_SEC, up=False)        # 2 frames, 92.9 ms
EDGE_FRAMES = _aligned(EDGE_SEC, up=True)       # 22 frames, 1.02 s

# A window contributes only its interior; a hop wider than that interior would
# leave frames no window votes on.
if HOP_FRAMES > WINDOW_FRAMES - 2 * EDGE_FRAMES:  # pragma: no cover - constant
    raise RuntimeError(
        f"hop {HOP_FRAMES} exceeds the usable window interior "
        f"{WINDOW_FRAMES - 2 * EDGE_FRAMES} -- frames would go uncovered"
    )

# Zip epoch: the earliest timestamp the format can represent.  Any fixed value
# works; this one is the convention reproducible-build tooling settled on.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def usable_frames(n_frames: int) -> int:
    """Frames truncated to a whole number of pooled groups.

    The same truncation ``WindowDataset`` applies, for the same reason: a lone
    trailing frame has no partner in the label head's pooled grid, and the
    decoder reads both grids against one another.
    """
    return (int(n_frames) // LABEL_POOL) * LABEL_POOL


def window_offsets(n_frames: int, *, window_frames: int = WINDOW_FRAMES,
                   hop_frames: int = HOP_FRAMES) -> list:
    """Start frame of every window, ending exactly at the end of the track.

    The last offset is clamped to ``n_frames - window_frames`` whether or not
    the hop lands there, so the tail is covered by a window that *re-overlaps*
    its predecessor.  A track shorter than one window yields a single offset 0
    and is zero-padded on the right.
    """
    limit = max(0, int(n_frames) - int(window_frames))
    offsets = list(range(0, limit + 1, int(hop_frames)))
    if offsets[-1] != limit:
        offsets.append(limit)
    return offsets


def contribution_span(offset: int, *, n_frames: int, first: bool, last: bool,
                      window_frames: int = WINDOW_FRAMES,
                      edge_frames: int = EDGE_FRAMES) -> tuple:
    """``[lo, hi)`` frames of the track this window is allowed to vote on.

    The interior of the window, except at the two ends of the track where no
    other window's interior can reach -- there the outer margin is the only
    evidence that exists, so it is used rather than leaving a hole.
    """
    lo = int(offset) if first else int(offset) + int(edge_frames)
    hi = int(offset) + int(window_frames)
    if not last:
        hi -= int(edge_frames)
    return lo, min(hi, int(n_frames))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    # Branch-free stable form: exp() never sees a large positive argument.
    positive = x >= 0.0
    exp_neg = np.exp(-np.abs(x))
    return np.where(positive, 1.0 / (1.0 + exp_neg), exp_neg / (1.0 + exp_neg))


# --------------------------------------------------------------------------- #
# One track
# --------------------------------------------------------------------------- #


class TrackPosteriors(NamedTuple):
    """The arrays one sidecar holds, the counts worth reporting, and the
    geometry they were actually produced with.

    The geometry travels with the result rather than being re-read from the
    module constants at write time.  ``infer_track`` takes window, hop and edge
    as arguments -- a sweep or a test may pass anything -- and a sidecar that
    recorded the defaults while holding non-default numbers would be worse than
    an unlabelled one: ``sidecar_is_current`` would accept it, and a decoder
    would read a 200 ms hop as a 93 ms one with nothing to notice.
    """

    label_post: np.ndarray      # [n // LABEL_POOL, NUM_CLASSES] float32
    boundary: np.ndarray        # [n] float32, mean of the per-window sigmoids
    coverage: np.ndarray        # [n] uint16, windows that voted on each frame
    n_frames: int
    windows: int
    window_frames: int
    hop_frames: int
    edge_frames: int


def infer_track(sess, mel: np.ndarray, *, window_frames: int = WINDOW_FRAMES,
                hop_frames: int = HOP_FRAMES,
                edge_frames: int = EDGE_FRAMES) -> TrackPosteriors:
    """Whole-track posteriors by sliding ``sess``'s graph over ``mel``.

    ``mel`` is ``[n, n_mels]`` as written by the batch sim.  Returns the label
    posteriors on the pooled grid, the mean boundary score at frame rate, the
    per-frame window count, and the geometry all three were produced with.
    """
    n_frames = usable_frames(len(mel))
    if n_frames < LABEL_POOL:
        raise RuntimeError(f"track has {len(mel)} mel frames -- too short to pool")
    # Pool alignment is what lets the pooled label slice be taken straight out
    # of the window's output instead of resampled, so it is checked rather than
    # asserted in a comment: an unaligned window or hop silently shifts every
    # label posterior by half a pooled frame.
    for name, value in (("window_frames", window_frames), ("hop_frames", hop_frames),
                        ("edge_frames", edge_frames)):
        if int(value) % LABEL_POOL:
            raise ValueError(
                f"{name}={value} is not a multiple of the label pooling factor "
                f"{LABEL_POOL} -- the pooled label grid would not line up"
            )
    if hop_frames > window_frames - 2 * edge_frames:
        raise ValueError(
            f"hop {hop_frames} exceeds the usable window interior "
            f"{window_frames - 2 * edge_frames} -- frames would go uncovered"
        )
    mel = np.ascontiguousarray(mel[:n_frames], dtype=np.float32)

    padded = mel
    if n_frames < window_frames:
        padded = np.zeros((window_frames, mel.shape[1]), dtype=np.float32)
        padded[:n_frames] = mel

    pooled_frames = n_frames // LABEL_POOL
    label_sum = np.zeros((pooled_frames, NUM_CLASSES), dtype=np.float64)
    boundary_sum = np.zeros(n_frames, dtype=np.float64)
    coverage = np.zeros(n_frames, dtype=np.int32)

    offsets = window_offsets(n_frames, window_frames=window_frames,
                             hop_frames=hop_frames)
    if any(offset % LABEL_POOL for offset in offsets):   # pragma: no cover
        raise ValueError(
            f"window offsets {[o for o in offsets if o % LABEL_POOL][:4]} are not "
            f"pool-aligned despite aligned inputs -- window_offsets has changed"
        )
    for index, offset in enumerate(offsets):
        label_logits, boundary_logits = run_window(
            sess, padded[offset:offset + window_frames][None])
        label = _softmax(label_logits[0])
        boundary = _sigmoid(boundary_logits[0])

        lo, hi = contribution_span(
            offset, n_frames=n_frames, first=index == 0,
            last=index == len(offsets) - 1,
            window_frames=window_frames, edge_frames=edge_frames)
        if hi <= lo:                                   # pragma: no cover
            continue
        boundary_sum[lo:hi] += boundary[lo - offset:hi - offset]
        coverage[lo:hi] += 1
        # Frame spans are pool-aligned because offset, edge and window all are
        # (checked above), so the pooled slice is exact rather than rounded.
        label_sum[lo // LABEL_POOL:hi // LABEL_POOL] += \
            label[(lo - offset) // LABEL_POOL:(hi - offset) // LABEL_POOL]

    if not coverage.all():                             # pragma: no cover
        raise RuntimeError(
            f"{int((coverage == 0).sum())} frames were covered by no window -- "
            f"the hop/edge geometry leaves holes"
        )

    pooled_coverage = coverage[::LABEL_POOL].astype(np.float64)
    label_post = (label_sum / pooled_coverage[:, None]).astype(np.float32)
    boundary_mean = (boundary_sum / coverage).astype(np.float32)
    return TrackPosteriors(label_post, boundary_mean, coverage.astype(np.uint16),
                           n_frames, len(offsets), int(window_frames),
                           int(hop_frames), int(edge_frames))


# --------------------------------------------------------------------------- #
# Sidecar I/O
# --------------------------------------------------------------------------- #


def posterior_arrays(track: TrackPosteriors, model_sha: str) -> dict:
    """Everything the sidecar records, in write order.

    Two time bases, both stated rather than implied.  ``boundary[k]`` is at
    ``t0 + k * frame_sec``; ``label_post[j]`` is a pooled group stamped at the
    song time of its LAST frame (``label_t0 + j * label_frame_sec``), the
    convention ``WindowDataset`` pooled its targets on.  A decoder that assumed
    both grids share an origin would read every label ~46 ms early.

    The window geometry comes from ``track``, not from this module's constants:
    the file has to describe the run that produced it, or the cache key built on
    it is a statement about the defaults rather than about the contents.
    """
    return {
        "label_post": track.label_post,
        "boundary": track.boundary,
        "coverage": track.coverage,
        "frame_sec": np.float64(FRAME_SEC),
        "t0": np.float64(FRAME_SEC),
        "label_frame_sec": np.float64(FRAME_SEC * LABEL_POOL),
        "label_t0": np.float64(FRAME_SEC + (LABEL_POOL - 1) * FRAME_SEC),
        "label_pool": np.int32(LABEL_POOL),
        "n_frames": np.int32(track.n_frames),
        "windows": np.int32(track.windows),
        "window_frames": np.int32(track.window_frames),
        "hop_frames": np.int32(track.hop_frames),
        "edge_frames": np.int32(track.edge_frames),
        "model_sha": np.str_(model_sha),
    }


def save_posteriors(path, arrays: dict) -> None:
    """Write an npz whose bytes are a pure function of its contents.

    Not ``np.savez``: its reproducibility is inherited from ``ZipInfo``'s
    default timestamp, which is a CPython default rather than a documented
    guarantee, and ``savez_compressed`` makes the bytes a function of the zlib
    build as well.  A determinism claim that rests on either is a claim about
    this machine, not about the pipeline -- so the members are written here
    with an explicit order, an explicit epoch and no compression.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, value in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(value),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue())

    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "wb") as handle:
            handle.write(buffer.getvalue())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def sidecar_is_current(path, model_sha: str) -> bool:
    """Is ``path`` already this model's answer, on this geometry?

    The cache key the spec asks for -- (model hash, track) -- plus the window
    geometry, because a hop or edge change produces different numbers from the
    same graph and would otherwise silently reuse the old ones.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as archive:
            return (str(archive["model_sha"]) == model_sha
                    and int(archive["window_frames"]) == WINDOW_FRAMES
                    and int(archive["hop_frames"]) == HOP_FRAMES
                    and int(archive["edge_frames"]) == EDGE_FRAMES)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


# --------------------------------------------------------------------------- #
# Corpus run
# --------------------------------------------------------------------------- #


def track_ids(data_dir) -> list:
    """Every clean, sidecar-carrying, labelled track -- train, val and test.

    Test-split ids are generated too: the decoder sweeps (Task 5) need val, the
    evaluation (Task 6) needs test, and regenerating 100 tracks later at a
    different code revision is a worse risk than generating them now.  Nothing
    here reads a label, so producing them leaks nothing.
    """
    candidates, *_ = candidate_tracks(Path(data_dir))
    return sorted(ref.youtube_id for ref in candidates)


def generate(data_dir, *, model_path=None, out_dir=None, ids=None,
             workers: int = 1, force: bool = False,
             progress_every: int = 25) -> dict:
    """Write a posterior sidecar for every id; returns the run manifest."""
    data_dir = Path(data_dir)
    model_path = Path(model_path) if model_path else model_dir(data_dir) / MODEL_FILE
    out_dir = Path(out_dir) if out_dir else data_dir / POSTERIORS_DIR
    ids = list(ids) if ids is not None else track_ids(data_dir)

    model_sha = sha256_file(model_path)
    features = data_dir / FEATURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # One session, shared: onnxruntime releases the GIL inside `Run`, so N
    # Python threads give N single-threaded inferences in parallel without the
    # per-process model copy a ProcessPool would need.  Each track is handled
    # start to finish by one task, so nothing about a track's output depends on
    # how many threads are running.
    sess = session(model_path)

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
            mel = load_sidecar(features / f"{youtube_id}.npz")
            track = infer_track(sess, mel)
            save_posteriors(path, posterior_arrays(track, model_sha))
        except Exception as error:
            # A corpus-wide run is hours long and the sidecars are cached, so one
            # unreadable track must not throw away every other track's work.  The
            # failure is recorded and re-reported at the end rather than logged
            # and forgotten: a silently short posterior set would show up in Task
            # 6 as an unexplained corpus size.
            #
            # A sidecar left over from an older model or geometry is deleted on
            # the way out.  Otherwise the manifest says "failed" while the file
            # on disk still answers to `np.load`, and the next reader consumes
            # last week's posteriors believing they are this model's.  A file
            # that *is* current (only reachable under --force, where the recompute
            # was discretionary) is left alone -- a transient read error must not
            # destroy a good artifact.
            stale = path.exists() and not sidecar_is_current(path, model_sha)
            if stale:
                path.unlink(missing_ok=True)
            print(f"  FAILED {youtube_id}: {error!r}"
                  f"{' (removed stale sidecar)' if stale else ''}", flush=True)
            return {"youtube_id": youtube_id, "cached": False,
                    "removed_stale": stale,
                    "error": f"{type(error).__name__}: {error}"}
        return {"youtube_id": youtube_id, "cached": False,
                "frames": track.n_frames, "windows": track.windows,
                "bytes": path.stat().st_size,
                "seconds": round(time.perf_counter() - clock, 3)}

    def report(record: dict) -> dict:
        done["count"] += 1
        index = done["count"]
        if progress_every and (index % progress_every == 0 or index == len(ids)):
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(ids) - index) / rate if rate else 0.0
            print(f"  {index}/{len(ids)}  {elapsed / 60:.1f} min elapsed, "
                  f"{eta / 60:.1f} min left  ({rate * 60:.1f} tracks/min)",
                  flush=True)
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
        "frame_sec": FRAME_SEC,
        "window_frames": WINDOW_FRAMES,
        "hop_frames": HOP_FRAMES,
        "hop_sec": HOP_FRAMES * FRAME_SEC,
        "edge_frames": EDGE_FRAMES,
        "edge_sec": EDGE_FRAMES * FRAME_SEC,
        "label_pool": LABEL_POOL,
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
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--model", type=Path, default=None,
                        help=f"exported graph (default: <data-dir>/models/v1/{MODEL_FILE})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"default: <data-dir>/{POSTERIORS_DIR}")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="youtube ids (default: every clean track)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N ids (0 = all)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4),
                        help="concurrent single-threaded sessions (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="recompute even where the sidecar already matches "
                             "this model and geometry")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    ids = args.ids if args.ids is not None else track_ids(args.data_dir)
    if args.limit:
        ids = ids[:args.limit]

    manifest = generate(args.data_dir, model_path=args.model, out_dir=args.out_dir,
                        ids=ids, workers=args.workers, force=args.force,
                        progress_every=args.progress_every)
    print(f"{manifest['tracks']} tracks "
          f"({manifest['computed']} computed, {manifest['cached']} cached) "
          f"in {manifest['wall_seconds'] / 60:.1f} min")
    print(f"  {manifest['windows']} windows over {manifest['frames']} frames "
          f"({manifest['frames'] * FRAME_SEC / 3600:.1f} audio hours)")
    print(f"  {manifest['bytes'] / (1 << 20):.1f} MiB of sidecars, "
          f"model {manifest['model_sha'][:16]}")
    if manifest["failed"]:
        print(f"  {manifest['failed']} FAILED -- see "
              f"{MANIFEST_FILE}; rerun to retry (finished tracks are cached)")
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
