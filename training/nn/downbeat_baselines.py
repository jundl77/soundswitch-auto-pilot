"""The numbers a downbeat F1 has to be read against.

    uv run python -m training.nn.downbeat_baselines --data-dir <corpus> \
        --checkpoint <run>/best.pt

An F1 without its baseline is a decoration, and *which* baseline decides whether
a result is impressive or trivial.  Three references are computed here, on the
same split and through the same committed peak-picking path the trainer scores
itself with, so none of them is a separate implementation that has to be trusted:

**The informative null: a perfect but PHASE-BLIND beat detector.**  Gaussian
bumps at every annotated beat, no bar knowledge at all, scored against the
annotated *downbeats*.  This is the number that matters, because it is what a
system that has solved beat tracking and learned nothing about bars already
scores.  Structureless nulls (noise, an untrained net) are much weaker and
flatter the model by a factor of three; they are computed too, precisely so the
difference is visible rather than a matter of which one someone quoted.

**The calibration floor.**  The de-weighted ECE is computed against *binarised*
labels while the target is a soft Gaussian, so even an oracle whose de-weighted
probability IS the training target scores well above zero.  That value is the
floor, and an ECE below it says the model is closer to the binary labels than
the target is -- not that it is well calibrated.  Reporting the number without
the floor invites exactly the wrong reading.

**The phase histogram of predicted peaks.**  Where the model's false positives
land, by bar phase.  This is a diagnosis, not a score: a flat distribution over
phases 2/3/4 would mean noise, while a concentration on phase 3 means a half-bar
ambiguity -- a structured error a cyclic phase decoder can remove, and the
strongest single hint about which decoder knob is worth tuning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import _TRAINING_DIR  # noqa: F401  (puts training/ on sys.path)

from .dataset import FRAME_SEC, candidate_tracks, make_splits
from .downbeat_dataset import (
    BEATS_PER_BAR,
    DOWNBEAT_SIGMA_SEC,
    DownbeatTargets,
    DownbeatWindowDataset,
    load_beat_grids,
)
from .downbeat_train import (
    MIN_PEAK_DISTANCE_SEC,
    POSITIVE_THRESHOLD,
    TOLERANCE_SEC,
    binary_ece,
    deweighted,
    frame_times,
    peak_candidates,
    stitch,
    sweep_peak_f1,
)

from build_training_table import default_data_dir  # noqa: E402
from raveform_fetch_annotations import load_tracks, parse_sections  # noqa: E402


def gaussian_at(instants: np.ndarray, times: np.ndarray,
                sigma: float = DOWNBEAT_SIGMA_SEC) -> np.ndarray:
    """Gaussian of the distance to the nearest instant -- the target's own recipe.

    Two binary searches rather than one exp per instant, and identical in form to
    ``track_downbeat_targets`` so the null is built the way the truth is.
    """
    if not instants.size or not times.size:
        return np.zeros(len(times), dtype=np.float64)
    right = np.searchsorted(instants, times)
    before = instants[np.clip(right - 1, 0, instants.size - 1)]
    after = instants[np.clip(right, 0, instants.size - 1)]
    distance = np.minimum(np.abs(times - before), np.abs(times - after))
    return np.exp(-0.5 * (distance / sigma) ** 2)


def phase_blind_activation(targets: DownbeatTargets) -> np.ndarray:
    """What a perfect beat tracker with no bar knowledge would emit."""
    times = frame_times(np.arange(len(targets.downbeat)))
    return gaussian_at(targets.beat_time[targets.beat_frame >= 0], times)


def reference_downbeats(targets: DownbeatTargets, live: np.ndarray) -> np.ndarray:
    """Annotated downbeat instants inside the supervised region."""
    on_grid = (targets.beat_phase == 1) & (targets.beat_frame >= 0)
    return targets.beat_time[on_grid & live[np.maximum(targets.beat_frame, 0)]]


def pick_live(activation: np.ndarray, live: np.ndarray, min_distance: int) -> np.ndarray:
    """The trainer's own picker, restricted to supervised frames."""
    picked = peak_candidates(np.where(live, activation, -1.0), min_distance)
    return picked[live[picked]] if picked.size else picked


def score_activations(dataset, activations: dict, min_distance: int) -> dict:
    """Whole-track activations -> the same swept peak F1 the trainer reports."""
    per_track = []
    for youtube_id in dataset.track_ids():
        targets = dataset.targets_for(youtube_id)
        live = targets.mask
        activation = activations[youtube_id]
        picked = pick_live(activation, live, min_distance)
        per_track.append((frame_times(picked), activation[picked],
                          reference_downbeats(targets, live)))
    return sweep_peak_f1(per_track)


def ece_floor(dataset, pos_weight: float) -> float:
    """De-weighted ECE of an oracle whose de-weighted probability IS the target.

    The floor exists because the ECE labels binarise a *soft* target: a frame
    whose target is 0.4 is labelled negative, so an oracle emitting 0.4 there is
    scored as 0.4 of over-confidence.  Nothing about the model can go below this.
    """
    soft, labels = [], []
    for youtube_id in dataset.track_ids():
        targets = dataset.targets_for(youtube_id)
        live = targets.mask
        soft.append(targets.downbeat[live].astype(np.float64))
        labels.append(targets.downbeat[live] >= POSITIVE_THRESHOLD)
    soft = np.concatenate(soft)
    labels = np.concatenate(labels)
    # deweighted(inflate(target)) == target exactly, so the floor is just the
    # soft target's own ECE against the binarised labels.
    logit = np.log(np.clip(soft, 1e-12, 1 - 1e-12) / (1 - np.clip(soft, 1e-12, 1 - 1e-12)))
    inflated = 1.0 / (1.0 + np.exp(-(logit + np.log(pos_weight))))
    return binary_ece(deweighted(inflated, pos_weight), labels)


def peak_phase_histogram(dataset, activations: dict, threshold: float,
                         min_distance: int) -> dict:
    """Bar phase of every predicted peak, plus how many land on a beat at all.

    ``phase[0]`` counts peaks further than the tolerance from *any* annotated
    beat.  The rest are the phase the nearest beat carries -- so ``phase[1]`` is
    (essentially) the true positives and ``phase[3]`` is the half-bar error.
    """
    counts = np.zeros(BEATS_PER_BAR + 1, dtype=np.int64)
    for youtube_id in dataset.track_ids():
        targets = dataset.targets_for(youtube_id)
        live = targets.mask
        activation = activations[youtube_id]
        picked = pick_live(activation, live, min_distance)
        picked = picked[activation[picked] >= threshold]
        predicted = frame_times(picked)

        on_grid = targets.beat_frame >= 0
        beats = targets.beat_time[on_grid]
        phases = targets.beat_phase[on_grid]
        if not beats.size or not predicted.size:
            continue
        near = np.searchsorted(beats, predicted)
        left = np.clip(near - 1, 0, beats.size - 1)
        right = np.clip(near, 0, beats.size - 1)
        take_left = np.abs(predicted - beats[left]) <= np.abs(predicted - beats[right])
        nearest = np.where(take_left, left, right)
        locked = np.abs(predicted - beats[nearest]) <= TOLERANCE_SEC
        counts[0] += int((~locked).sum())
        for phase in range(1, BEATS_PER_BAR + 1):
            counts[phase] += int(((phases[nearest] == phase) & locked).sum())
    return {"unlocked": int(counts[0]),
            **{f"phase_{phase}": int(counts[phase])
               for phase in range(1, BEATS_PER_BAR + 1)}}


def model_activations(checkpoint, dataset, device, batch_size: int = 128) -> dict:
    """Stitched whole-track activations from a trained checkpoint.

    Imported lazily -- everything above is numpy, and the nulls are worth being
    able to compute on a machine with no torch and no GPU.
    """
    import torch

    from .downbeat_model import DownbeatCRNN
    from .train import build_loader

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = {key: (tuple(value) if isinstance(value, list) else value)
            for key, value in state["arch"].items()}
    model = DownbeatCRNN(**arch).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    sums = {i: np.zeros(len(dataset.targets_for(i).downbeat)) for i in dataset.track_ids()}
    counts = {i: np.zeros(len(dataset.targets_for(i).downbeat), dtype=np.int64)
              for i in dataset.track_ids()}
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                          pin_memory=device.type == "cuda", generator=None)
    index = 0
    with torch.no_grad():
        for mel, _target, _mask in loader:
            probs = torch.sigmoid(model(mel.to(device))).float().cpu().numpy()
            index = stitch(dataset, probs, index, sums, counts)

    out = {}
    for youtube_id in dataset.track_ids():
        seen = counts[youtube_id] > 0
        activation = np.zeros_like(sums[youtube_id])
        np.divide(sums[youtube_id], counts[youtube_id], out=activation, where=seen)
        out[youtube_id] = activation
    return out, float(state.get("pos_weight", 1.0))


def build_dataset(data_dir: Path, split: str, tracks: int, seed: int):
    splits = make_splits(data_dir, write=False)
    track_ids = {ref.youtube_id: ref.track_id for ref in candidate_tracks(data_dir)[0]}
    ids = sorted(splits[split], key=lambda i: track_ids[i])
    if tracks > 0:
        ids = ids[:tracks]
    sections = {str(t.get("id")): parse_sections(t) for t in load_tracks(data_dir)}
    grids, missing = load_beat_grids(data_dir, ids)
    if missing:
        raise RuntimeError(f"{len(missing)} track(s) have no beat grid: {missing[:5]}")
    return DownbeatWindowDataset(data_dir, ids, augment=False, seed=seed,
                                 sections_by_youtube_id=sections,
                                 grids_by_youtube_id=grids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split", default="val",
                        help="never 'test' before the one-shot verdict")
    parser.add_argument("--tracks", type=int, default=0, help="0 = the whole split")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a best.pt; without it only the nulls are computed")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="operating point for the phase histogram")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = build_dataset(args.data_dir, args.split, args.tracks, args.seed)
    min_distance = max(1, int(round(MIN_PEAK_DISTANCE_SEC / FRAME_SEC)))
    print(f"{args.split}: {len(dataset.track_ids())} tracks, {len(dataset)} windows")

    rng = np.random.default_rng(args.seed)
    nulls = {
        "phase-blind beat detector": {i: phase_blind_activation(dataset.targets_for(i))
                                      for i in dataset.track_ids()},
        "iid uniform noise": {i: rng.random(len(dataset.targets_for(i).downbeat))
                              for i in dataset.track_ids()},
        "constant": {i: np.full(len(dataset.targets_for(i).downbeat), 0.5)
                     for i in dataset.track_ids()},
    }
    print("\n-- baselines (same picker, same matcher, same split) --")
    for name, activations in nulls.items():
        swept = score_activations(dataset, activations, min_distance)
        print(f"  {name:<28} F1 {swept['f1']:.4f}  P {swept['precision']:.4f} "
              f"R {swept['recall']:.4f}  @{swept['best_threshold']:.2f}")

    if args.checkpoint is None:
        return 0

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    activations, pos_weight = model_activations(args.checkpoint, dataset, device)
    swept = score_activations(dataset, activations, min_distance)
    print(f"  {'trained model':<28} F1 {swept['f1']:.4f}  P {swept['precision']:.4f} "
          f"R {swept['recall']:.4f}  @{swept['best_threshold']:.2f}")

    floor = ece_floor(dataset, pos_weight)
    print(f"\n-- calibration (pos_weight {pos_weight:.4f}) --")
    print(f"  de-weighted ECE floor (soft target vs binarised labels): {floor:.4f}")

    histogram = peak_phase_histogram(dataset, activations, args.threshold, min_distance)
    total = sum(histogram.values())
    print(f"\n-- phase of predicted peaks at threshold {args.threshold} "
          f"({total} peaks) --")
    for key, value in histogram.items():
        print(f"  {key:<10} {value:7d}  {100.0 * value / max(total, 1):5.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
