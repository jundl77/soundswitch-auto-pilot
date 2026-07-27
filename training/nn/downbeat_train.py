"""Train ``DownbeatCRNN`` on the expert beat grids.

    uv run python -m training.nn.downbeat_train --data-dir <corpus> \
        --run-name downbeat_v1 --epochs 40 --batch-size 128

One head, one loss: masked BCE against the Gaussian downbeat target, with
``pos_weight`` measured off the train split rather than guessed.  Almost
everything else -- seeding, the loader policy, the checkpoint/resume contract,
the RNG capture, the weight hash, TensorBoard, PR-AUC, ECE -- is imported from
``train`` rather than restated.  The two heads share a determinism contract, and
a second copy of it is a second thing to keep true.

**The metric is peak-based, and that is not a stylistic choice.**  A frame-wise
PR-AUC scores a model on how well it ranks *frames*, but nothing downstream
consumes frames: Task 3's HMM consumes one activation value per beat instant and
the verdict counts downbeats matched within +-70 ms.  So validation stitches the
windowed activations back into whole-track curves, picks peaks, and matches them
to the annotated downbeats at the tolerance the component is actually scored at.
The peak picker here is deliberately naive -- non-maximum suppression and a
threshold, no tempo model, no phase state -- because it must not flatter the
model by doing the decoder's job.  The number it produces is a floor, and Task
3's decoder should beat it.

**Calibration is a metric, not a side effect, and it is reported twice.**
``pos_weight`` ~ 9 deliberately inflates every probability the head emits: that
is the price of learning a 10 %-mass target and it makes the raw sigmoid a
*ranking* score, not P(downbeat).  Reporting only the raw ECE would therefore
score the reweighting rather than the model.  The second number subtracts
``log(pos_weight)`` from every logit -- the exact analytic inverse of the
reweighting under a balanced-prior reading -- and answers the question that
matters: is the head's confidence *shaped* right, once the training-time prior
shift is undone?  Nothing in the objective pushes the model toward hard 0/1
activations, and nothing smooths them either; the decoder wants an honest ranked
curve and both numbers are logged every epoch so a drift into overconfidence is
visible rather than inferred.

Determinism follows the same validated recipe as the section head: seeded
everything, ``use_deterministic_algorithms(True)``, cuDNN benchmarking off, no
``gather`` and no boolean indexing anywhere in the loss path, and the loader
generator re-seeded from ``(seed, epoch)`` so a resumed run rejoins the exact
trajectory.  Two runs of one config in fresh processes produce bitwise-identical
weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    FEATURES_DIR,
    FRAME_SEC,
    LABEL_POOL,
    WINDOW_FRAMES,
    candidate_tracks,
    make_splits,
    sidecar_shape,
)
from .downbeat_dataset import (
    DOWNBEAT_SIGMA_SEC,
    DownbeatWindowDataset,
    load_beat_grids,
    track_downbeat_targets,
)
from .downbeat_model import PARAM_BUDGET, DownbeatCRNN
from .model import count_parameters
from .train import (
    CONFIG_FILE,
    ECE_BINS,
    LAST_CHECKPOINT,
    BEST_CHECKPOINT,
    MODELS_DIR,
    REPORT_FILE,
    TB_DIR,
    TENSORBOARD_PORT,
    # The section head's masked, positive-reweighted BCE over [B, T] logits is
    # *exactly* this head's loss, and its ratio-form weight is exactly this
    # head's weight.  Imported under neutral names rather than copied: two
    # implementations of one formula is two things to keep true.
    boundary_bce as masked_bce,
    boundary_pos_weight as sparsity_pos_weight,
    build_loader,
    capture_rng,
    config_fingerprint,
    ensure_tensorboard,
    per_class_ece,
    pr_auc,
    restore_rng,
    seed_everything,
    select_smoke_subset,
    weight_hash,
)

from build_training_table import default_data_dir  # noqa: E402
from raveform_fetch_annotations import load_tracks, parse_sections  # noqa: E402

MODEL_VERSION = "downbeat_v1"

# The tolerance the whole component is scored at, and the sigma the targets were
# built with -- imported rather than restated so the two cannot diverge.
TOLERANCE_SEC = DOWNBEAT_SIGMA_SEC

# The Gaussian target crosses 0.5 at ~1.18 sigma (~82 ms), i.e. just outside the
# match tolerance -- the natural place to binarise for a frame-wise PR curve.
POSITIVE_THRESHOLD = 0.5

# Non-maximum suppression radius for the naive peak picker.  A bar at the fastest
# tempo in the corpus (252 BPM, 4/4) is 0.95 s, so 0.70 s cannot merge two real
# downbeats at any tempo the corpus contains, while still collapsing the shoulder
# frames of a single activation bump.
MIN_PEAK_DISTANCE_SEC = 0.70

# Operating points swept on val to report the peak-picking F1.  Coarse on
# purpose: this is a floor for Task 3's decoder, not a tuned system.
PEAK_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))


# --------------------------------------------------------------------------- #
# Corpus statistics -> the loss weight
# --------------------------------------------------------------------------- #


class DownbeatStats(NamedTuple):
    """Target sparsity over a split, plus the spread that a single weight hides."""

    positive: float          # summed target mass over supervised frames
    valid: int               # supervised frames
    frames: int              # frames in total, supervised or not
    track_mass: np.ndarray   # [tracks] per-track mean target mass


def accumulate_downbeat_stats(targets_iterable) -> DownbeatStats:
    """Fold whole-track ``DownbeatTargets`` into the sparsity statistics."""
    positive = 0.0
    valid = 0
    frames = 0
    per_track: list = []
    for targets in targets_iterable:
        live = targets.mask.astype(np.float64)
        mass = float((targets.downbeat * live).sum())
        supervised = int(live.sum())
        positive += mass
        valid += supervised
        frames += int(len(targets.downbeat))
        per_track.append(mass / supervised if supervised else 0.0)
    return DownbeatStats(positive, valid, frames,
                         np.asarray(per_track, dtype=np.float64))


def load_downbeat_stats(data_dir, youtube_ids, grids_by_youtube_id) -> DownbeatStats:
    """``accumulate_downbeat_stats`` over a split, without decoding any mel.

    Reads each sidecar's *header* for its frame count and rebuilds the targets,
    so the weight for a 962-track split costs a couple of seconds rather than a
    gigabyte of decompressed spectrogram.  The frame count is truncated exactly
    as ``WindowDataset`` truncates it, so the statistic describes the frames the
    model is actually shown.
    """
    data_dir = Path(data_dir)

    def targets():
        for youtube_id in youtube_ids:
            frames = sidecar_shape(data_dir / FEATURES_DIR / f"{youtube_id}.npz")[0]
            usable = (frames // LABEL_POOL) * LABEL_POOL
            yield track_downbeat_targets(grids_by_youtube_id[youtube_id], usable,
                                         FRAME_SEC, FRAME_SEC)

    return accumulate_downbeat_stats(targets())


# --------------------------------------------------------------------------- #
# Peak picking and event matching (the metric the component is scored at)
# --------------------------------------------------------------------------- #


def peak_candidates(scores: np.ndarray, min_distance: int) -> np.ndarray:
    """Frame indices surviving greedy non-maximum suppression, in time order.

    Threshold-*free* by construction, and that is what makes the threshold sweep
    honest: suppression order depends only on the scores, so the set of peaks
    above a threshold ``t`` is exactly the subset of this list scoring >= ``t``.
    Picking peaks after thresholding instead would let a lower threshold change
    which peak wins a neighbourhood, and the sweep would be comparing different
    pickers rather than different operating points.

    Only local maxima can survive suppression -- a frame lower than its immediate
    neighbour is blocked by it at any ``min_distance >= 1`` -- so the candidate
    set is narrowed to those first, which is what keeps this cheap enough to run
    on every val track every epoch.

    **A plateau collapses to its centre, not to an edge.**  A saturating head
    emits runs of identical values, and every tie-break rule that picks an
    endpoint biases the reported instant by half the plateau -- up to a frame,
    which is 46 ms of the 70 ms budget.  Taking the midpoint is unbiased, and it
    also stops a flat activation from being mistaken for a comb of peaks: a
    constant curve has exactly one, in the middle, which is the right answer for
    a model that has said nothing.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0 or min_distance < 1:
        return np.zeros(0, dtype=np.int64)

    padded = np.concatenate(([-np.inf], scores, [-np.inf]))
    local = np.flatnonzero((scores >= padded[:-2]) & (scores >= padded[2:]))
    if local.size == 0:
        return np.zeros(0, dtype=np.int64)

    # Consecutive candidate indices are a plateau (an isolated maximum is a run
    # of one and passes through untouched); keep each run's middle frame.
    breaks = np.flatnonzero(np.diff(local) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [local.size - 1]))
    local = local[(starts + ends) // 2]

    # Stable sort: equal-scoring peaks resolve by frame index, so the picker is a
    # pure function of the activation and the determinism proof holds.
    order = local[np.argsort(-scores[local], kind="stable")]
    blocked = np.zeros(n, dtype=bool)
    taken: list = []
    for index in order:
        if blocked[index]:
            continue
        taken.append(int(index))
        blocked[max(0, index - min_distance):index + min_distance + 1] = True
    return np.array(sorted(taken), dtype=np.int64)


def match_events(predicted: np.ndarray, reference: np.ndarray,
                 tolerance: float = TOLERANCE_SEC) -> tuple:
    """``(tp, fp, fn)`` for one-to-one matching of two sorted time arrays.

    Two pointers, greedy in time.  Greedy is *optimal* here rather than merely
    convenient: both sequences are separated by far more than ``2 * tolerance``
    (a bar is >= 0.95 s at the corpus's fastest tempo, and the picker suppresses
    peaks within 0.70 s), so no reference instant is ever in range of two
    predictions and there is no matching for a bipartite solver to improve on.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    index = other = 0
    hits = 0
    while index < len(predicted) and other < len(reference):
        delta = predicted[index] - reference[other]
        if abs(delta) <= tolerance:
            hits += 1
            index += 1
            other += 1
        elif delta < 0:
            index += 1
        else:
            other += 1
    return hits, len(predicted) - hits, len(reference) - hits


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision/recall/F1 from a confusion triple, undefined cases at 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def frame_times(frames, frame_sec: float = FRAME_SEC,
                t0: float | None = None) -> np.ndarray:
    """Song time of mel frame ``k``: ``t0 + k * frame_sec`` with ``t0 == frame_sec``."""
    t0 = frame_sec if t0 is None else t0
    return t0 + np.asarray(frames, dtype=np.float64) * frame_sec


def sweep_peak_f1(per_track: list, thresholds=PEAK_THRESHOLDS,
                  tolerance: float = TOLERANCE_SEC) -> dict:
    """Micro-averaged peak F1 at every threshold, plus the best operating point.

    ``per_track`` is ``[(candidate_times, candidate_scores, reference_times)]``.
    Micro rather than macro: a track contributes in proportion to its bar count,
    which is how the corpus-wide verdict counts and keeps a 90-second track from
    weighing the same as a nine-minute one.
    """
    curve: dict = {}
    for threshold in thresholds:
        tp = fp = fn = 0
        for times, scores, reference in per_track:
            kept = times[scores >= threshold]
            hits, extra, missed = match_events(kept, reference, tolerance)
            tp += hits
            fp += extra
            fn += missed
        curve[float(threshold)] = prf(tp, fp, fn)
    best = max(curve.items(), key=lambda item: (item[1]["f1"], -item[0]))
    return {"curve": curve, "best_threshold": float(best[0]), **best[1]}


def binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error of a single sigmoid output.

    Delegates to the section head's ``per_class_ece`` on the two-column
    ``[1 - p, p]`` form: the positive column of that is exactly the textbook
    binary ECE, and one binning implementation is worth more than a second
    hand-rolled one that has to be trusted separately.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    if len(labels) == 0:
        return 0.0
    stacked = np.stack([1.0 - probs, probs], axis=1)
    return float(per_class_ece(stacked, labels, n_bins)[1])


def deweighted(probs: np.ndarray, pos_weight: float) -> np.ndarray:
    """Undo the ``pos_weight`` prior shift on a probability array.

    Reweighting the positive class by ``w`` is, at the optimum, exactly a
    ``+log w`` shift of every logit; subtracting it back recovers what the head
    would have said under the corpus's own prior.  This is a *reporting*
    transform -- nothing trains against it -- and it is what separates "the model
    is overconfident" from "we asked it to be".
    """
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    logits = np.log(probs / (1.0 - probs)) - math.log(max(pos_weight, 1e-12))
    return 1.0 / (1.0 + np.exp(-logits))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def stitch(dataset: DownbeatWindowDataset, rows: np.ndarray, start: int,
           sums: dict, counts: dict) -> int:
    """Accumulate a batch of window activations into whole-track curves.

    Windows tile a track from frame 0 in eval mode and the *last* one is clamped
    back so it re-overlaps its predecessor (``WindowDataset``'s tail fix), so the
    overlap is averaged rather than overwritten: taking the later window would
    silently prefer one of two equally valid estimates on every track's final
    ~16 s, which is where a peak-picking metric is most fragile.

    **Requires ``dataset.augment=False`` and a loader with ``shuffle=False``.**
    ``rows`` is matched to dataset indices by position from ``start``, and the
    offsets are read back from ``window_offset``, which under augmentation
    re-draws a *random* offset per epoch. Stitching an augmented dataset would
    therefore scatter each window to a position it was not cut from and produce a
    plausible-looking curve of noise -- no exception, no shape error.
    """
    index = start
    for row in rows:
        youtube_id = dataset.track_id_of(index)
        offset = dataset.window_offset(index)
        end = min(offset + dataset.window_frames, len(sums[youtube_id]))
        if end > offset:
            sums[youtube_id][offset:end] += row[:end - offset]
            counts[youtube_id][offset:end] += 1
        index += 1
    return index


def evaluate(model: nn.Module, loader: DataLoader, dataset: DownbeatWindowDataset,
             device: torch.device, *, pos_weight: float,
             min_distance: int) -> dict:
    """One pass over the val split -> the metric block logged each epoch.

    The loss is per window (that is the objective); every ranking number is read
    off the stitched whole-track curves instead, so an overlapped frame is
    counted once and the frame-wise metrics describe the same array the peak
    picker sees.
    """
    model.eval()
    sums = {}
    counts = {}
    for youtube_id in dataset.track_ids():
        n_frames = len(dataset.targets_for(youtube_id).downbeat)
        sums[youtube_id] = np.zeros(n_frames, dtype=np.float64)
        counts[youtube_id] = np.zeros(n_frames, dtype=np.int64)

    total = 0.0
    batches = 0
    index = 0
    with torch.no_grad():
        for mel, target, mask in loader:
            mel = mel.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(mel)
            total += float(masked_bce(logits, target, mask, pos_weight=pos_weight))
            batches += 1
            index = stitch(dataset, torch.sigmoid(logits).float().cpu().numpy(),
                           index, sums, counts)

    per_track: list = []
    scores: list = []
    labels: list = []
    candidates = 0
    references = 0
    for youtube_id in dataset.track_ids():
        targets = dataset.targets_for(youtube_id)
        seen = counts[youtube_id] > 0
        live = targets.mask & seen
        activation = np.zeros_like(sums[youtube_id])
        np.divide(sums[youtube_id], counts[youtube_id], out=activation, where=seen)

        scores.append(activation[live])
        labels.append(targets.downbeat[live] >= POSITIVE_THRESHOLD)

        # Frames outside the supervised region are not evidence of absence -- the
        # annotation simply stops -- so the picker must not be allowed to spend
        # peaks there, and the truth must not count downbeats there either.  The
        # sentinel is below every probability rather than 0.0, so an unsupervised
        # frame is processed last by the suppression and cannot delete a real
        # peak beside it.
        picked = peak_candidates(np.where(live, activation, -1.0), min_distance)
        picked = picked[live[picked]] if picked.size else picked
        is_downbeat = targets.beat_phase == 1
        beat_frame = targets.beat_frame
        on_grid = is_downbeat & (beat_frame >= 0)
        reference = targets.beat_time[on_grid & live[np.maximum(beat_frame, 0)]]
        per_track.append((frame_times(picked), activation[picked], reference))
        candidates += int(picked.size)
        references += int(reference.size)

    scores = np.concatenate(scores) if scores else np.zeros(0)
    labels = np.concatenate(labels) if labels else np.zeros(0, dtype=bool)
    swept = sweep_peak_f1(per_track)
    return {
        "loss": total / max(batches, 1),
        "f1": swept["f1"],
        "precision": swept["precision"],
        "recall": swept["recall"],
        "best_threshold": swept["best_threshold"],
        "f1_at_half": swept["curve"][0.5]["f1"],
        "f1_curve": {str(key): value["f1"] for key, value in swept["curve"].items()},
        "pr_auc": pr_auc(scores, labels),
        "ece": binary_ece(scores, labels),
        "ece_deweighted": binary_ece(deweighted(scores, pos_weight), labels),
        "mean_prob": float(scores.mean()) if scores.size else 0.0,
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "frames": int(scores.size),
        # Peaks *at the operating point*, not raw candidates: a valley of exactly
        # equal activation is a plateau and contributes a candidate at its centre,
        # so the raw count roughly doubles on a well-separated activation and
        # reading it as "predicted downbeats" would be wrong.
        "peaks": int(swept["tp"] + swept["fp"]),
        "candidates": candidates,
        "downbeats": references,
    }


def log_epoch(writer, epoch: int, record: dict, metrics: dict) -> None:
    """Everything the owner watches live, flushed so it is on screen at once."""
    if writer is None:
        return
    for key in ("f1", "precision", "recall", "f1_at_half", "pr_auc", "ece",
                "ece_deweighted", "best_threshold", "mean_prob", "loss"):
        writer.add_scalar(f"val/{key}", metrics[key], epoch)
    writer.add_scalar("epoch/train_loss", record["train"], epoch)
    writer.flush()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(config: dict) -> dict:
    """Run (or resume) one downbeat training job; returns the training report."""
    data_dir = Path(config["data_dir"])
    run_dir = data_dir / MODELS_DIR / MODEL_VERSION / config["run_name"]
    tb_dir = data_dir / MODELS_DIR / MODEL_VERSION / TB_DIR / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["seed"])
    device = torch.device(config["device"])

    splits = make_splits(data_dir, write=False)
    track_ids = {ref.youtube_id: ref.track_id for ref in candidate_tracks(data_dir)[0]}
    sections = {str(track.get("id")): parse_sections(track) for track in load_tracks(data_dir)}
    chosen = {name: select_smoke_subset(splits[name], track_ids, config["smoke_tracks"])
              for name in ("train", "val")}

    grids, missing = load_beat_grids(data_dir, chosen["train"] + chosen["val"])
    if missing:
        raise RuntimeError(
            f"{len(missing)} split track(s) have no beat grid ({missing[:5]}...) -- "
            f"they would train on nothing but masked frames"
        )

    config = dict(config)
    config["tracks"] = {name: [[track_ids[i], i] for i in ids]
                        for name, ids in chosen.items()}

    train_set = DownbeatWindowDataset(data_dir, chosen["train"], augment=True,
                                      seed=config["seed"],
                                      sections_by_youtube_id=sections,
                                      grids_by_youtube_id=grids)
    val_set = DownbeatWindowDataset(data_dir, chosen["val"], augment=False,
                                    seed=config["seed"],
                                    sections_by_youtube_id=sections,
                                    grids_by_youtube_id=grids)

    stats = load_downbeat_stats(data_dir, chosen["train"], grids)
    pos_weight = (float(config["pos_weight"]) if config["pos_weight"] > 0
                  else sparsity_pos_weight(stats.positive, stats.valid))
    min_distance = max(1, int(round(config["min_peak_distance"] / FRAME_SEC)))

    model = DownbeatCRNN(rnn_hidden=config["rnn_hidden"],
                         rnn_layers=config["rnn_layers"],
                         conv1d_channels=config["conv1d_channels"],
                         dropout=config["dropout"]).to(device)
    params = count_parameters(model)
    if params > PARAM_BUDGET:
        raise RuntimeError(
            f"{params} parameters exceeds the {PARAM_BUDGET} budget the spec sets "
            f"for this head -- widen deliberately, in the spec, not by accident"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    # Schedule length derived from the dataset, never hardcoded: the corpus is
    # still growing and a cosine tail computed for a stale window count decays to
    # zero before the run ends (or never gets there).
    steps_per_epoch = max(1, math.ceil(len(train_set) / config["batch_size"]))
    total_steps = steps_per_epoch * config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    fingerprint = config_fingerprint(config)
    history: dict = {"steps": [], "epochs": []}
    best = {"f1": -math.inf, "epoch": 0}
    start_epoch = 0

    checkpoint_path = run_dir / LAST_CHECKPOINT
    if config["resume"] and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["config_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"{checkpoint_path} was written by a different configuration "
                f"({state['config_fingerprint'][:12]} vs {fingerprint[:12]}) -- "
                f"resuming would produce a model no report describes"
            )
        # Before load_state_dict, not after: a geometry mismatch that changes no
        # tensor shape loads *cleanly* and is then invisible.
        if state.get("arch") != model.arch():
            raise RuntimeError(
                f"{checkpoint_path} was written for architecture "
                f"{state.get('arch')}, this run builds {model.arch()} -- the "
                f"weights would load into a differently shaped model"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best = state["best"]
        start_epoch = int(state["epoch"])
        restore_rng(state["rng"])
        print(f"resumed {config['run_name']} from epoch {start_epoch}", flush=True)
    elif config["resume"]:
        print(f"no checkpoint at {checkpoint_path}; starting from scratch", flush=True)

    config["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    config["param_count"] = params
    config["arch"] = model.arch()
    config["steps_per_epoch"] = steps_per_epoch
    config["total_steps"] = total_steps
    config["train_windows"] = len(train_set)
    config["val_windows"] = len(val_set)
    config["pos_weight_used"] = pos_weight
    with open(run_dir / CONFIG_FILE, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    writer = None
    if config["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(tb_dir))
    if config["launch_tensorboard"]:
        logdir = data_dir / MODELS_DIR
        started = ensure_tensorboard(logdir, config["tensorboard_port"],
                                     logdir / "tensorboard.log")
        print(f"tensorboard {'started' if started else 'already running'}: "
              f"http://localhost:{config['tensorboard_port']}", flush=True)

    mass = stats.track_mass
    print(f"{config['run_name']}: {params} params (budget {PARAM_BUDGET}) | "
          f"{len(train_set)} train windows / {len(val_set)} val windows | "
          f"{steps_per_epoch} steps/epoch x {config['epochs']} epochs | device {device}",
          flush=True)
    print(f"target mass {stats.positive / max(stats.valid, 1):.4f}/frame over "
          f"{stats.valid} supervised frames -> pos_weight {pos_weight:.3f} | "
          f"per-track mass p05 {np.percentile(mass, 5):.4f} median "
          f"{np.median(mass):.4f} p95 {np.percentile(mass, 95):.4f} "
          f"max {mass.max():.4f}", flush=True)

    generator = torch.Generator()
    val_loader = build_loader(val_set, batch_size=config["batch_size"], shuffle=False,
                              num_workers=config["num_workers"],
                              pin_memory=device.type == "cuda", generator=None)
    train_loader = build_loader(train_set, batch_size=config["batch_size"], shuffle=True,
                                num_workers=config["num_workers"],
                                pin_memory=device.type == "cuda", generator=generator)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    trained_steps = 0
    stopped_early = False

    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        train_set.set_epoch(epoch)
        # Re-seeded from (seed, epoch) rather than left to run on, so a resumed
        # run draws the same batch order as an uninterrupted one without having
        # to restore the sampler's internal state.
        generator.manual_seed(config["seed"] * 1_000_003 + epoch)

        model.train()
        epoch_start = time.perf_counter()
        running = 0.0
        index = -1
        for index, (mel, target, mask) in enumerate(train_loader):
            mel = mel.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            loss = masked_bce(model(mel), target, mask, pos_weight=pos_weight)
            step = (epoch - 1) * steps_per_epoch + index + 1
            value = float(loss.detach())
            # Before the step, not after: a NaN that has already been applied has
            # destroyed the weights, and the checkpoint written at the end of the
            # epoch would preserve the wreckage.
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite loss at step {step}")

            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            running += value
            history["steps"].append({"step": step, "epoch": epoch,
                                     "lr": learning_rate, "loss": value})
            trained_steps += 1
            if writer is not None:
                writer.add_scalar("train/loss", value, step)
                writer.add_scalar("train/lr", learning_rate, step)

        batches = max(1, index + 1)
        metrics = evaluate(model, val_loader, val_set, device,
                           pos_weight=pos_weight, min_distance=min_distance)
        record = {
            "epoch": epoch,
            "train": running / batches,
            "val": {key: value for key, value in metrics.items() if key != "f1_curve"},
            "f1_curve": metrics["f1_curve"],
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": time.perf_counter() - epoch_start,
        }
        history["epochs"].append(record)
        log_epoch(writer, epoch, record, metrics)

        print(f"epoch {epoch:3d} | train {record['train']:.4f} | "
              f"val {metrics['loss']:.4f} F1@{TOLERANCE_SEC * 1000:.0f}ms "
              f"{metrics['f1']:.4f} (P {metrics['precision']:.3f} R "
              f"{metrics['recall']:.3f} @{metrics['best_threshold']:.2f}) "
              f"PR-AUC {metrics['pr_auc']:.4f} ECE {metrics['ece']:.4f}"
              f"/{metrics['ece_deweighted']:.4f} | {record['seconds']:.1f}s",
              flush=True)

        if metrics["f1"] > best["f1"]:
            best = {"f1": float(metrics["f1"]), "epoch": epoch,
                    "metrics": {key: value for key, value in metrics.items()
                                if key != "f1_curve"}}
            # `metrics` is the flat metric block, not `best` -- nesting `best`
            # here would make the field `best.pt["metrics"]["metrics"]["f1"]`,
            # and a consumer that guessed the shorter path would read a dict
            # where it expected a float.
            torch.save({"model": model.state_dict(), "arch": model.arch(),
                        "config": config, "epoch": epoch, "f1": best["f1"],
                        "metrics": best["metrics"], "pos_weight": pos_weight},
                       run_dir / BEST_CHECKPOINT)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "arch": model.arch(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best": best,
            "rng": capture_rng(),
            "config": config,
            "config_fingerprint": fingerprint,
            "pos_weight": pos_weight,
        }, checkpoint_path)

        if config["crash_after_epoch"] and epoch >= config["crash_after_epoch"]:
            # Resume drill: die the way a killed process dies -- checkpoint on
            # disk, nothing flushed, no cleanup -- so `--resume` is proven against
            # a real interruption rather than a graceful shutdown.
            print(f"crash-after-epoch {epoch}: exiting hard", flush=True)
            sys.stdout.flush()
            os._exit(17)

        if epoch - best["epoch"] >= config["patience"]:
            print(f"early stop: no val F1 improvement in {config['patience']} epochs",
                  flush=True)
            stopped_early = True
            break

    wall = time.perf_counter() - wall_start
    report = {
        "config": config,
        "param_count": params,
        "arch": model.arch(),
        "pos_weight": pos_weight,
        "target_mass_per_frame": stats.positive / max(stats.valid, 1),
        "supervised_frames": stats.valid,
        "total_frames": stats.frames,
        "track_mass_quantiles": {
            "min": float(mass.min()) if mass.size else 0.0,
            "p05": float(np.percentile(mass, 5)) if mass.size else 0.0,
            "median": float(np.median(mass)) if mass.size else 0.0,
            "p95": float(np.percentile(mass, 95)) if mass.size else 0.0,
            "max": float(mass.max()) if mass.size else 0.0,
        },
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "history": history,
        "best": best,
        "final": history["epochs"][-1] if history["epochs"] else None,
        "stopped_early": stopped_early,
        "weight_hash": weight_hash(model.state_dict()),
        "wall_seconds": wall,
        "steps_per_second": trained_steps / wall if wall > 0 else 0.0,
        "gpu_peak_alloc_mb": (torch.cuda.max_memory_allocated(device) / 2 ** 20
                              if device.type == "cuda" else None),
        "gpu_peak_reserved_mb": (torch.cuda.max_memory_reserved(device) / 2 ** 20
                                 if device.type == "cuda" else None),
    }
    with open(run_dir / REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    if writer is not None:
        writer.close()

    print(f"done: best val F1 {best['f1']:.4f} @ epoch {best['epoch']} | "
          f"weights {report['weight_hash'][:16]} | {wall:.1f}s "
          f"({report['steps_per_second']:.1f} steps/s)", flush=True)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="corpus root; checkpoints land in "
                             f"<data-dir>/{MODELS_DIR}/{MODEL_VERSION}/<run-name> "
                             "(default: %(default)s)")
    parser.add_argument("--run-name", default="downbeat_v1",
                        help="run directory name; keep the downbeat_ prefix so "
                             "the shared TensorBoard logdir stays readable")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--rnn-hidden", type=int, default=96)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--conv1d-channels", type=int, default=64)
    parser.add_argument("--pos-weight", type=float, default=0.0,
                        help="override the weight derived from train-split target "
                             "sparsity (0 = derive it, the default)")
    parser.add_argument("--min-peak-distance", type=float, default=MIN_PEAK_DISTANCE_SEC,
                        help="non-maximum suppression radius, seconds")
    parser.add_argument("--patience", type=int, default=8,
                        help="early stop after N epochs without a val F1 gain")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; 0 (default) is fastest for this "
                             "corpus and the only mode that re-rolls augmentation "
                             "every epoch -- see train.build_loader")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-tracks", type=int, default=0,
                        help="limit train and val to the first N tracks of each "
                             "split by corpus track id (0 = the whole split)")
    parser.add_argument("--resume", action="store_true",
                        help=f"continue from <run-dir>/{LAST_CHECKPOINT}")
    parser.add_argument("--crash-after-epoch", type=int, default=0,
                        help="resume drill: exit hard right after epoch N's "
                             "checkpoint is written (0 = never)")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false",
                        help="skip the event writer entirely")
    parser.add_argument("--no-tensorboard-server", dest="launch_tensorboard",
                        action="store_false", help="do not start a TensorBoard process")
    parser.add_argument("--tensorboard-port", type=int, default=TENSORBOARD_PORT)
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = vars(args)
    config["data_dir"] = str(Path(config["data_dir"]).resolve())
    config["window_frames"] = WINDOW_FRAMES
    train(config)
    return 0


if __name__ == "__main__":   # spawn re-imports this module; the guard is required
    raise SystemExit(main())
