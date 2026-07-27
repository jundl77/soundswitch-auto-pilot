"""Train ``SectionCRNN`` on the windowed Raveform corpus.

    uv run python -m training.nn.train --data-dir <corpus> --run-name v1 \
        --epochs 60 --batch-size 128

The objective is three terms, and each exists to stop a specific failure the
other two cannot see:

**Focal loss (label head).**  ``drop`` outnumbers ``outro`` by an order of
magnitude in this corpus, so plain cross-entropy converges happily to a model
that predicts the majority class in the ambiguous middle.  Inverse-frequency
class weights fix the prior; the focal term (gamma) then stops the *easy*
frames -- the middle of a four-minute drop -- from drowning out the twenty
frames either side of a transition, which is the only region anyone cares
about.

**BCE (boundary head).**  Positives are ~1 % of frames even after Gaussian
smearing, so ``pos_weight`` is computed from the corpus rather than guessed: it
is the ratio of negative to positive *mass* over supervised frames.

**Total variation (label posteriors).**  Left alone, a frame-wise classifier
flickers between neighbouring classes, and flicker is exactly what the runtime
is being replaced for.  Penalising ``|p_t - p_{t-1}|`` buys smoothness, but a
blanket penalty also teaches the model to smear the section change itself --
the one step the decoder must see.  So the penalty is *boundary aware*: it is
weighted by ``1 - boundary_target``, stands down entirely where the annotation
says a transition is happening, and abstains where the boundary target was
deleted (merged-run joins) because there we do not know whether a step is
legitimate.

**Calibration is a metric, not a side effect.**  The decoder consumes
posteriors, not argmaxes, so a model that is 99 % confident and 80 % right is
worse than useless to it.  Per-class ECE is logged every epoch alongside
macro-F1 and is one of the plan's release gates.

Determinism follows the validated CUDA pre-flight recipe: seeded everything,
``use_deterministic_algorithms(True)``, cuDNN benchmarking off, and -- because
they are the two ops that would silently reintroduce nondeterminism -- neither
``gather`` nor boolean indexing appears anywhere in the loss path (both route
through ``scatter_add`` on the backward pass).  Masking is done by multiplying
with a float mask instead.  ``CUBLAS_WORKSPACE_CONFIG`` is set defensively in
this package's ``__init__`` (it must precede the first ``import torch``, which
``dataset`` performs).  Two runs of the same config in fresh processes produce
bitwise-identical weights; ``--resume`` restores optimiser, scheduler and RNG
state so an interrupted run rejoins that same trajectory.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    FEATURES_DIR,
    FRAME_SEC,
    IGNORE_INDEX,
    LABEL_POOL,
    NUM_CLASSES,
    V1_ORDER,
    WindowDataset,
    candidate_tracks,
    make_splits,
    sidecar_shape,
    track_targets,
)
from .model import PARAM_BUDGET, SectionCRNN, count_parameters

from build_training_table import default_data_dir  # noqa: E402
from raveform_fetch_annotations import load_tracks, parse_sections  # noqa: E402

MODELS_DIR = "models"
MODEL_VERSION = "v1"
TB_DIR = "tb"
REPORT_FILE = "training_report.json"
CONFIG_FILE = "config.json"
LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"

TENSORBOARD_PORT = 6006
ECE_BINS = 15
# The Gaussian boundary target reaches 0.5 at ~1.18 sigma (~0.59 s) -- close
# enough to the +-0.5 s tolerance the evaluator scores at that it is the natural
# place to binarise for a precision/recall curve.
BOUNDARY_POSITIVE_THRESHOLD = 0.5

# The plan specified lr 3e-4 at batch 32; the CUDA pre-flight then moved the
# batch to 128 and no scaling correction was applied to the learning rate.  That
# is a deliberate, recorded choice rather than an oversight, and it decides the
# triage order if the full run underperforms -- so it is written into every run
# config and report instead of living in someone's memory.
LR_REFERENCE_BATCH = 32


def lr_note(lr: float, batch_size: int) -> str:
    if batch_size == LR_REFERENCE_BATCH:
        return (f"lr {lr:g} at the plan's reference batch {LR_REFERENCE_BATCH} -- "
                f"no scaling question.")
    return (
        f"lr {lr:g} was specified in the plan at batch {LR_REFERENCE_BATCH} and is "
        f"running at batch {batch_size} ({batch_size / LR_REFERENCE_BATCH:g}x); no "
        f"linear- or sqrt-scaling correction was applied. TRIAGE: if val macro-F1 "
        f"stalls below the gate while the train loss is still falling, the run is "
        f"under-stepping -- raise lr FIRST (try {lr * 2:g}, then {lr * 4:g}) before "
        f"touching the architecture, the TV lambda or the class weights."
    )


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    """The pre-flight's validated block, verbatim in intent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      # benchmark=True is nondeterministic
    torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id: int) -> None:
    """Worker RNG is not covered by ``seed_everything`` in the parent."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, *,
               weight: torch.Tensor, gamma: float = 2.0,
               ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Class-weighted focal loss over ``[B, T, C]`` logits, masked positions out.

    Mean over *supervised* positions, so the value does not depend on how much
    of a batch happened to be unlabelled.  A fully masked batch returns a
    genuine zero that still carries a gradient edge (all-zero), rather than NaN.
    """
    flat = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = (flat_targets != ignore_index).to(flat.dtype)
    # IGNORE_INDEX is negative: one_hot would raise on it, so park masked
    # positions on class 0 and let `valid` delete their contribution.
    safe = torch.where(flat_targets == ignore_index,
                       torch.zeros_like(flat_targets), flat_targets)

    log_probs = F.log_softmax(flat, dim=-1)
    onehot = F.one_hot(safe, logits.shape[-1]).to(flat.dtype)
    log_pt = (log_probs * onehot).sum(dim=-1)
    pt = log_pt.exp()
    alpha = (weight.to(flat.dtype).to(flat.device) * onehot).sum(dim=-1)

    losses = -alpha * (1.0 - pt).pow(gamma) * log_pt
    return (losses * valid).sum() / valid.sum().clamp(min=1.0)


def boundary_bce(logits: torch.Tensor, targets: torch.Tensor,
                 mask: torch.Tensor, *, pos_weight: float) -> torch.Tensor:
    """Masked, positive-reweighted BCE over ``[B, T]`` boundary logits."""
    valid = mask.to(logits.dtype)
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype),
        pos_weight=torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device),
        reduction="none",
    )
    return (losses * valid).sum() / valid.sum().clamp(min=1.0)


def tv_penalty(label_logits: torch.Tensor, label_mask: torch.Tensor,
               boundary_targets: torch.Tensor, boundary_mask: torch.Tensor, *,
               pool: int = LABEL_POOL) -> torch.Tensor:
    """Boundary-aware total variation of the label posteriors.

    ``mean_c |p_t - p_{t-1}|`` averaged over pooled frame pairs, each pair
    weighted by ``1 - max(boundary target over the two pooled frames)`` and
    dropped entirely unless both frames are label-supervised and both carry a
    live boundary target.
    """
    probs = label_logits.softmax(dim=-1)
    variation = (probs[:, 1:] - probs[:, :-1]).abs().mean(dim=-1)   # [B, T-1]

    # Frame-rate boundary info -> pooled rate.  `max` for the target (a boundary
    # anywhere in the group makes the group a boundary) and `min` for the mask
    # (a group is trustworthy only if all of its frames are).
    frames = boundary_targets.shape[1] // pool * pool
    grouped = boundary_targets[:, :frames].reshape(boundary_targets.shape[0], -1, pool)
    pooled_target = grouped.max(dim=-1).values
    grouped_mask = boundary_mask[:, :frames].reshape(boundary_mask.shape[0], -1, pool)
    pooled_mask = grouped_mask.all(dim=-1).to(probs.dtype)

    live = label_mask.to(probs.dtype) * pooled_mask
    pair = live[:, 1:] * live[:, :-1]
    quiet = 1.0 - torch.maximum(pooled_target[:, 1:], pooled_target[:, :-1])
    weights = pair * quiet.clamp(min=0.0)

    return (variation * weights).sum() / weights.sum().clamp(min=1e-6)


# --------------------------------------------------------------------------- #
# Corpus statistics -> loss weights
# --------------------------------------------------------------------------- #


class TargetStats(NamedTuple):
    class_counts: np.ndarray     # [C] supervised pooled frames per class
    boundary_positive: float     # summed boundary target mass over live frames
    boundary_valid: int          # live boundary frames


def accumulate_target_stats(targets_iterable) -> TargetStats:
    """Fold whole-track ``TrackTargets`` into the two loss-weight statistics."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    positive = 0.0
    valid = 0
    for targets in targets_iterable:
        supervised = targets.label_pooled[targets.label_pooled_mask]
        counts += np.bincount(supervised, minlength=NUM_CLASSES).astype(np.int64)
        positive += float(targets.boundary[targets.boundary_mask].sum())
        valid += int(targets.boundary_mask.sum())
    return TargetStats(counts, positive, valid)


def load_target_stats(data_dir, youtube_ids, sections_by_youtube_id=None) -> TargetStats:
    """``accumulate_target_stats`` over a split, without decoding any mel.

    Reads each sidecar's *header* for its frame count (``sidecar_shape``) and
    rebuilds the targets, so computing the class histogram over 538 tracks costs
    a second rather than a GB of decompressed spectrogram.
    """
    data_dir = Path(data_dir)
    if sections_by_youtube_id is None:
        sections_by_youtube_id = {str(track.get("id")): parse_sections(track)
                                  for track in load_tracks(data_dir)}

    def targets():
        for youtube_id in youtube_ids:
            frames = sidecar_shape(data_dir / FEATURES_DIR / f"{youtube_id}.npz")[0]
            # Same truncation WindowDataset applies, so the histogram counts the
            # frames the model is actually shown.
            usable = (frames // LABEL_POOL) * LABEL_POOL
            yield track_targets(sections_by_youtube_id[youtube_id], usable,
                                FRAME_SEC, FRAME_SEC)

    return accumulate_target_stats(targets())


def class_weights(counts: np.ndarray) -> np.ndarray:
    """Inverse frequency, normalised to mean 1 over the classes that occur.

    Normalising keeps the focal term's magnitude comparable across splits (and
    so keeps a single ``--tv-lambda`` meaningful); a class with no supervised
    frames gets weight 0 -- it cannot contribute a gradient anyway, and any
    other value would be an invented prior.
    """
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    weights = np.zeros_like(counts)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    scale = weights[present].mean() if present.any() else 1.0
    if scale > 0:
        weights[present] /= scale
    return weights


def boundary_pos_weight(positive: float, valid: int) -> float:
    """``negatives / positives`` over supervised frames; 1.0 if undefined."""
    if positive <= 0.0 or valid <= 0:
        return 1.0
    return float(max(valid - positive, 0.0) / positive)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def confusion_matrix(true: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """``matrix[t, p]`` -- rows are truth, columns are prediction."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(true, dtype=np.int64),
                       np.asarray(pred, dtype=np.int64)), 1)
    return matrix


def per_class_f1(matrix: np.ndarray) -> np.ndarray:
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    denominator = predicted + actual
    return np.divide(2.0 * true_positive, denominator,
                     out=np.zeros_like(true_positive), where=denominator > 0)


def macro_f1(matrix: np.ndarray) -> float:
    """Mean F1 over classes that occur in truth *or* prediction.

    Averaging over all five classes when a val batch contains two would report
    0.4 for a perfect classifier, and this number is a release gate.
    """
    present = (matrix.sum(axis=0) > 0) | (matrix.sum(axis=1) > 0)
    if not present.any():
        return float("nan")
    return float(per_class_f1(matrix)[present].mean())


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision -- the interpolation-free area under the PR curve.

    NaN when there is no positive: a metric that cannot be computed must not
    look like a model that scored zero.
    """
    labels = np.asarray(labels).astype(bool)
    if not labels.any():
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    hits = labels[order]
    true_positive = np.cumsum(hits)
    precision = true_positive / np.arange(1, len(hits) + 1)
    recall = true_positive / hits.sum()
    return float((np.diff(recall, prepend=0.0) * precision).sum())


def per_class_ece(probs: np.ndarray, labels: np.ndarray,
                  n_bins: int = ECE_BINS) -> np.ndarray:
    """One-vs-rest expected calibration error per class.

    Per class, not just on the argmax: the decoder divides by class priors and
    multiplies posteriors along a path, so it reads every column of the softmax,
    and a class can be badly calibrated while never winning an argmax.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros(probs.shape[1], dtype=np.float64)
    if len(labels) == 0:
        return out
    for index in range(probs.shape[1]):
        confidence = probs[:, index]
        correct = (labels == index).astype(np.float64)
        # `right=True` on all but the first bin so 1.0 lands in the last bin.
        bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
        total = np.bincount(bins, minlength=n_bins).astype(np.float64)
        conf_sum = np.bincount(bins, weights=confidence, minlength=n_bins)
        acc_sum = np.bincount(bins, weights=correct, minlength=n_bins)
        occupied = total > 0
        gap = np.abs(conf_sum[occupied] - acc_sum[occupied])
        out[index] = float(gap.sum() / len(labels))
    return out


def confusion_image(matrix: np.ndarray, cell: int = 24) -> np.ndarray:
    """Row-normalised confusion matrix as a CHW greyscale image for TensorBoard.

    Rendered by hand rather than through matplotlib: the project does not depend
    on it, and a 5x5 heatmap does not justify adding one.
    """
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix.astype(np.float32), np.maximum(totals, 1))
    return np.kron(normalised, np.ones((cell, cell), dtype=np.float32))[None, :, :]


# --------------------------------------------------------------------------- #
# Run plumbing
# --------------------------------------------------------------------------- #


def select_smoke_subset(youtube_ids, track_id_by_youtube_id, count: int) -> list:
    """The first ``count`` ids of a split ordered by corpus track id.

    Ordered by ``track_id`` (``0002.kfJQCu-Jbec``) rather than by youtube id so
    the subset is a stable, human-checkable slice of the corpus, and recorded in
    the run config so a determinism re-run cannot silently pick a different ten.
    ``count <= 0`` means the whole split, still ordered.
    """
    ordered = sorted(youtube_ids, key=lambda youtube_id: track_id_by_youtube_id[youtube_id])
    return ordered if count <= 0 else ordered[:count]


def weight_hash(state_dict) -> str:
    """SHA-256 over every tensor in key order -- the determinism fingerprint."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        digest.update(key.encode("utf-8"))
        tensor = value.detach().cpu().contiguous() if torch.is_tensor(value) \
            else torch.as_tensor(value)
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def config_fingerprint(config: dict) -> str:
    """Hash of everything that changes the trajectory.

    Resume compares this: rejoining a run under a different learning rate or a
    different track list produces a model no report describes.
    """
    volatile = {"started_at", "crash_after_epoch", "resume", "launch_tensorboard",
                "tensorboard_port", "run_dir", "tb_dir"}
    trimmed = {key: value for key, value in config.items() if key not in volatile}
    return hashlib.sha256(
        json.dumps(trimmed, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with contextlib.suppress(OSError):
        with socket.create_connection((host, port), timeout=0.5):
            return True
    return False


def ensure_tensorboard(logdir: Path, port: int, log_path: Path) -> bool:
    """Start a detached TensorBoard on ``port`` unless one is already there.

    Detached on purpose: the owner watches training live, and a child that dies
    with the trainer would take the run's history off the screen the moment it
    finishes.
    """
    if port_is_open(port):
        return False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    handle = open(log_path, "ab")
    subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir),
         "--port", str(port), "--host", "127.0.0.1"],
        stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=flags, close_fds=True,
    )
    return True


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def build_loader(dataset: WindowDataset, *, batch_size: int, shuffle: bool,
                 num_workers: int, pin_memory: bool,
                 generator: torch.Generator | None) -> DataLoader:
    """A DataLoader wired per the CUDA pre-flight, with one deliberate departure.

    The pre-flight recommends ``persistent_workers=True`` to avoid paying the
    ~2.2 s Windows spawn cost once per epoch.  **We do not take it.**  Persistent
    workers hold a *pickled copy* of the dataset, so ``set_epoch`` in the parent
    never reaches them and the augmentation freezes on whatever epoch was current
    when the workers spawned -- every epoch after the first would then re-draw
    the identical window offsets and gains, which looks exactly like a working
    run and quietly deletes the augmentation.  Respawning per epoch re-pickles
    the dataset with the new epoch, so correctness costs 2.2 s an epoch.

    The alternative -- a shared epoch counter read by ``worker_init_fn`` -- was
    rejected as cross-process state on a path this project does not use:
    ``--num-workers 0`` is the default and is genuinely faster at this corpus
    size, so the spawn cost is hypothetical.  Workers exist here to keep the
    path proven, not because the training needs them.
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False, generator=generator,
        worker_init_fn=worker_init_fn if num_workers else None,
        persistent_workers=False,
        prefetch_factor=2 if num_workers else None,
    )


def log_epoch(writer, epoch: int, record: dict, metrics: dict) -> None:
    """Everything the owner watches live, flushed so it is on screen at once."""
    if writer is None:
        return
    writer.add_scalar("val/macro_f1", metrics["macro_f1"], epoch)
    writer.add_scalar("val/boundary_pr_auc", metrics["boundary_pr_auc"], epoch)
    writer.add_scalar("val/ece_mean", metrics["ece_mean"], epoch)
    for name, value in metrics["per_class_f1"].items():
        writer.add_scalar(f"val/f1/{name}", value, epoch)
    for name, value in metrics["ece"].items():
        writer.add_scalar(f"val/ece/{name}", value, epoch)
    for key, value in metrics["loss"].items():
        writer.add_scalar(f"val/loss/{key}", value, epoch)
    for key, value in record["train"].items():
        writer.add_scalar(f"epoch/train_{key}", value, epoch)
    writer.add_image("val/confusion", confusion_image(np.array(metrics["confusion"])), epoch)
    writer.flush()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, *,
             weight: torch.Tensor, gamma: float, pos_weight: float,
             tv_lambda: float) -> dict:
    """One pass over the val split -> the metric block logged each epoch."""
    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    probs_chunks: list = []
    label_chunks: list = []
    boundary_scores: list = []
    boundary_labels: list = []
    losses = {"total": 0.0, "focal": 0.0, "boundary": 0.0, "tv": 0.0}
    batches = 0

    with torch.no_grad():
        for mel, labels, label_mask, boundary, boundary_mask in loader:
            mel = mel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_mask = label_mask.to(device, non_blocking=True)
            boundary = boundary.to(device, non_blocking=True)
            boundary_mask = boundary_mask.to(device, non_blocking=True)

            label_logits, boundary_logits = model(mel)
            focal = focal_loss(label_logits, labels, weight=weight, gamma=gamma)
            bce = boundary_bce(boundary_logits, boundary, boundary_mask,
                               pos_weight=pos_weight)
            smooth = tv_penalty(label_logits, label_mask, boundary, boundary_mask)
            losses["focal"] += float(focal)
            losses["boundary"] += float(bce)
            losses["tv"] += float(smooth)
            losses["total"] += float(focal + bce + tv_lambda * smooth)
            batches += 1

            valid = label_mask.reshape(-1).cpu().numpy()
            probabilities = label_logits.softmax(dim=-1).reshape(-1, NUM_CLASSES)
            probabilities = probabilities.float().cpu().numpy()[valid]
            truth = labels.reshape(-1).cpu().numpy()[valid]
            matrix += confusion_matrix(truth, probabilities.argmax(axis=1), NUM_CLASSES)
            probs_chunks.append(probabilities)
            label_chunks.append(truth)

            live = boundary_mask.reshape(-1).cpu().numpy()
            boundary_scores.append(
                torch.sigmoid(boundary_logits).reshape(-1).float().cpu().numpy()[live])
            boundary_labels.append(
                (boundary.reshape(-1).cpu().numpy()[live] >= BOUNDARY_POSITIVE_THRESHOLD))

    probs = np.concatenate(probs_chunks) if probs_chunks else np.zeros((0, NUM_CLASSES))
    truth = np.concatenate(label_chunks) if label_chunks else np.zeros(0, dtype=np.int64)
    ece = per_class_ece(probs, truth)
    divisor = max(batches, 1)
    return {
        "macro_f1": macro_f1(matrix),
        "per_class_f1": {name: float(value) for name, value
                         in zip(V1_ORDER, per_class_f1(matrix))},
        "boundary_pr_auc": pr_auc(np.concatenate(boundary_scores) if boundary_scores
                                  else np.zeros(0),
                                  np.concatenate(boundary_labels) if boundary_labels
                                  else np.zeros(0, dtype=bool)),
        "ece": {name: float(value) for name, value in zip(V1_ORDER, ece)},
        "ece_mean": float(ece.mean()),
        "loss": {key: value / divisor for key, value in losses.items()},
        "confusion": matrix.tolist(),
        "frames": int(len(truth)),
    }


def train(config: dict) -> dict:
    """Run (or resume) one training job; returns the training report."""
    data_dir = Path(config["data_dir"])
    version = config.get("model_version") or MODEL_VERSION
    run_dir = data_dir / MODELS_DIR / version / config["run_name"]
    tb_dir = data_dir / MODELS_DIR / version / TB_DIR / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["seed"])
    device = torch.device(config["device"])

    splits = make_splits(data_dir, write=False)
    track_ids = {ref.youtube_id: ref.track_id for ref in candidate_tracks(data_dir)[0]}
    sections = {str(track.get("id")): parse_sections(track) for track in load_tracks(data_dir)}
    chosen = {name: select_smoke_subset(splits[name], track_ids, config["smoke_tracks"])
              for name in ("train", "val")}
    config = dict(config)
    config["tracks"] = {name: [[track_ids[i], i] for i in ids]
                        for name, ids in chosen.items()}
    config["lr_note"] = lr_note(config["lr"], config["batch_size"])

    train_set = WindowDataset(data_dir, chosen["train"], augment=True,
                              seed=config["seed"], sections_by_youtube_id=sections)
    val_set = WindowDataset(data_dir, chosen["val"], augment=False,
                            seed=config["seed"], sections_by_youtube_id=sections)

    stats = load_target_stats(data_dir, chosen["train"], sections)
    weights = class_weights(stats.class_counts)
    pos_weight = boundary_pos_weight(stats.boundary_positive, stats.boundary_valid)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)

    model = SectionCRNN(n_classes=NUM_CLASSES, label_pool=LABEL_POOL,
                        dropout=config["dropout"]).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    # Schedule length is derived from the dataset, never hardcoded: the corpus
    # is still growing, and a cosine tail computed for a stale window count
    # decays to zero before the run ends (or never gets there).
    steps_per_epoch = max(1, math.ceil(len(train_set) / config["batch_size"]))
    total_steps = steps_per_epoch * config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    fingerprint = config_fingerprint(config)
    history: dict = {"steps": [], "epochs": []}
    best = {"macro_f1": -math.inf, "epoch": 0}
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
        # Before load_state_dict, not after: a label_pool mismatch changes no
        # tensor shape, so the weights would load *cleanly* into a model that
        # decodes at the wrong frame rate.
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
    config["steps_per_epoch"] = steps_per_epoch
    config["total_steps"] = total_steps
    config["train_windows"] = len(train_set)
    config["val_windows"] = len(val_set)
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

    print(f"{config['run_name']}: {params} params (budget {PARAM_BUDGET}) | "
          f"{len(train_set)} train windows / {len(val_set)} val windows | "
          f"{steps_per_epoch} steps/epoch x {config['epochs']} epochs | device {device}",
          flush=True)
    print(f"class counts {stats.class_counts.tolist()} -> weights "
          f"{np.round(weights, 3).tolist()} | boundary pos_weight {pos_weight:.2f}",
          flush=True)
    print(f"lr: {config['lr_note']}", flush=True)

    generator = torch.Generator()
    loader_workers = config["num_workers"]
    val_loader = build_loader(val_set, batch_size=config["batch_size"], shuffle=False,
                              num_workers=loader_workers,
                              pin_memory=device.type == "cuda", generator=None)
    train_loader = build_loader(train_set, batch_size=config["batch_size"], shuffle=True,
                                num_workers=loader_workers,
                                pin_memory=device.type == "cuda", generator=generator)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    trained_steps = 0
    stopped_early = False

    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        train_set.set_epoch(epoch)
        # Re-seeded from (seed, epoch) rather than left to run on: a resumed run
        # then draws the same batch order as an uninterrupted one without having
        # to restore the sampler's internal state.
        generator.manual_seed(config["seed"] * 1_000_003 + epoch)

        model.train()
        epoch_start = time.perf_counter()
        totals = {"total": 0.0, "focal": 0.0, "boundary": 0.0, "tv": 0.0}
        index = -1
        for index, (mel, labels, label_mask, boundary, boundary_mask) in enumerate(train_loader):
            mel = mel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_mask = label_mask.to(device, non_blocking=True)
            boundary = boundary.to(device, non_blocking=True)
            boundary_mask = boundary_mask.to(device, non_blocking=True)

            label_logits, boundary_logits = model(mel)
            focal = focal_loss(label_logits, labels, weight=weight_tensor,
                               gamma=config["focal_gamma"])
            bce = boundary_bce(boundary_logits, boundary, boundary_mask,
                               pos_weight=pos_weight)
            smooth = tv_penalty(label_logits, label_mask, boundary, boundary_mask)
            loss = focal + bce + config["tv_lambda"] * smooth

            step = (epoch - 1) * steps_per_epoch + index + 1
            values = {"total": float(loss.detach()), "focal": float(focal.detach()),
                      "boundary": float(bce.detach()), "tv": float(smooth.detach())}
            # Before the step, not after: a NaN that has already been applied
            # has destroyed the weights, and the checkpoint written at the end
            # of the epoch would preserve the wreckage.
            if not math.isfinite(values["total"]):
                raise RuntimeError(f"non-finite loss at step {step}: {values}")

            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            for key, value in values.items():
                totals[key] += value
            history["steps"].append({"step": step, "epoch": epoch,
                                     "lr": learning_rate, **values})
            trained_steps += 1
            if writer is not None:
                for key, value in values.items():
                    writer.add_scalar(f"train/{key}", value, step)
                writer.add_scalar("train/lr", learning_rate, step)

        batches = max(1, index + 1)
        metrics = evaluate(model, val_loader, device, weight=weight_tensor,
                           gamma=config["focal_gamma"], pos_weight=pos_weight,
                           tv_lambda=config["tv_lambda"])
        record = {
            "epoch": epoch,
            "train": {key: value / batches for key, value in totals.items()},
            "val": {key: value for key, value in metrics.items() if key != "confusion"},
            "confusion": metrics["confusion"],
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": time.perf_counter() - epoch_start,
        }
        history["epochs"].append(record)

        log_epoch(writer, epoch, record, metrics)

        print(f"epoch {epoch:3d} | train {record['train']['total']:.4f} "
              f"(focal {record['train']['focal']:.4f} bce {record['train']['boundary']:.4f} "
              f"tv {record['train']['tv']:.4f}) | val {metrics['loss']['total']:.4f} "
              f"macroF1 {metrics['macro_f1']:.4f} PR-AUC {metrics['boundary_pr_auc']:.4f} "
              f"ECE {metrics['ece_mean']:.4f} | {record['seconds']:.1f}s", flush=True)

        improved = metrics["macro_f1"] > best["macro_f1"]
        if improved:
            best = {"macro_f1": float(metrics["macro_f1"]), "epoch": epoch,
                    "metrics": {key: value for key, value in metrics.items()
                                if key != "confusion"}}
            torch.save({"model": model.state_dict(), "arch": model.arch(),
                        "config": config, "epoch": epoch, "metrics": best},
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
        }, checkpoint_path)

        if config["crash_after_epoch"] and epoch >= config["crash_after_epoch"]:
            # Resume drill: die the way a killed process dies -- checkpoint on
            # disk, nothing flushed, no cleanup -- so `--resume` is proven
            # against a real interruption rather than a graceful shutdown.
            print(f"crash-after-epoch {epoch}: exiting hard", flush=True)
            sys.stdout.flush()
            os._exit(17)

        if epoch - best["epoch"] >= config["patience"]:
            print(f"early stop: no val macro-F1 improvement in {config['patience']} epochs",
                  flush=True)
            stopped_early = True
            break

    wall = time.perf_counter() - wall_start
    report = {
        "config": config,
        "param_count": params,
        "class_counts": stats.class_counts.tolist(),
        "class_weights": weights.tolist(),
        "boundary_pos_weight": pos_weight,
        "lr_note": config["lr_note"],
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

    print(f"done: best val macro-F1 {best['macro_f1']:.4f} @ epoch {best['epoch']} | "
          f"weights {report['weight_hash'][:16]} | {wall:.1f}s "
          f"({report['steps_per_second']:.1f} steps/s)", flush=True)
    return report


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    """Inverse of ``capture_rng``.

    The ``.cpu()`` calls are load-bearing, not decoration: the checkpoint is
    loaded with ``map_location=device``, which moves *every* tensor in it --
    including these RNG ByteTensors -- onto the GPU, and both setters reject
    anything that is not a CPU ByteTensor.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and len(state["cuda"]):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="corpus root; checkpoints land in "
                             f"<data-dir>/{MODELS_DIR}/{MODEL_VERSION}/<run-name> "
                             "(default: %(default)s)")
    parser.add_argument("--run-name", default="v1", help="run directory name")
    parser.add_argument("--model-version", default=MODEL_VERSION,
                        help="artifact generation; runs land in "
                             f"<data-dir>/{MODELS_DIR}/<model-version>/<run-name>. "
                             "A retrain on a grown corpus takes a new generation so "
                             "the checkpoints backing an already-published verdict "
                             "are never overwritten (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=60)
    # 128, not the plan's 32: the pre-flight measured throughput saturating well
    # before it and ~3.1 GB reserved, which fits the 3070 with a desktop open.
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tv-lambda", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10,
                        help="early stop after N epochs without a val macro-F1 gain")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; 0 (default) is fastest for this "
                             "corpus and the only mode that re-rolls augmentation "
                             "every epoch -- see build_loader")
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
    train(config)
    return 0


if __name__ == "__main__":   # spawn re-imports this module; the guard is required
    raise SystemExit(main())
