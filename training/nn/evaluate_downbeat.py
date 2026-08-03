"""The downbeat verdict: grid reachability, decode quality, and the show ablation."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .downbeat_decoder import (
    BEATS_PER_BAR,
    BarPhaseHMM,
    PhaseParams,
    bar_phase,
    candidate_grid,
    downbeat_times,
)
from .downbeat_infer import SIDECAR_DIR
from .downbeat_train import MODEL_VERSION, TOLERANCE_SEC, match_events, prf

MODELS_DIR = "models"
CONFIG_FILE = "downbeat_decoder_config.json"
ALIGNMENT_FILE = "downbeat_alignment_{split}.json"
EVAL_FILE = "downbeat_eval_{split}.json"
SPLITS_FILE = "splits.json"

LOOK_AHEAD_BUDGET_BEATS = 4

# Owner decisions #81/#133; the plan's original 0.85 sat above published offline SOTA.
GATE_F1 = 0.55

INTERVAL_DEVIATION = 0.15
INTERVAL_WINDOW = 9

LOCK_IQR_BEATS = 0.06

TEMPO_TOLERANCE = 0.02
TEMPO_MULTIPLES = (0.5, 1.0, 2.0)

CONFIDENCE_THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.9)

REACH_LABELS = ("beat", "midpoint", "coast", "no_coverage", "dropout",
                "tempo_mismatch", "fraction_lock", "jitter")
REACHED = ("beat", "midpoint", "coast")

RESIDUAL_BINS = tuple(round(0.05 * step, 2) for step in range(11))

CONDITIONS = ("live", "expert")


def lag_for(look_ahead_beats: int, subdivision: int) -> int:
    return int(look_ahead_beats) * int(subdivision)


def rolling_median(values, window: int = INTERVAL_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    window = max(1, int(window)) | 1
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=1)


def nearest_index(query, reference) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        return np.full(query.size, -1, dtype=np.int64)
    right = np.searchsorted(reference, query)
    left = np.clip(right - 1, 0, reference.size - 1)
    right = np.clip(right, 0, reference.size - 1)
    take_left = np.abs(query - reference[left]) <= np.abs(reference[right] - query)
    return np.where(take_left, left, right)


def nearest_offset(query, reference) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        return np.full(query.size, np.nan, dtype=np.float64)
    return query - reference[nearest_index(query, reference)]


def fold_to_beats(offset_sec, period_sec) -> np.ndarray:
    offset = np.asarray(offset_sec, dtype=np.float64).reshape(-1)
    period = np.asarray(period_sec, dtype=np.float64).reshape(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beats = np.where(period > 0, offset / np.where(period > 0, period, 1.0), np.nan)
    folded = beats - np.floor(beats + 0.5)
    return np.where(folded == -0.5, 0.5, folded)


def local_periods(times, window: int = INTERVAL_WINDOW) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size < 2:
        return np.full(times.size, np.nan, dtype=np.float64)
    intervals = rolling_median(np.diff(times), window)
    return np.concatenate([intervals[:1], intervals])


def pulse_period(times, window: int = 2 * INTERVAL_WINDOW - 1,
                 gap_factor: float = 1.5) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size < 3:
        return np.full(times.size, np.nan, dtype=np.float64)
    intervals = np.diff(times)
    scale = rolling_median(intervals, INTERVAL_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        steps = np.where(scale > 0, np.rint(intervals / scale), 1.0)
    steps = np.maximum(np.nan_to_num(steps, nan=1.0), 1.0)
    continuous = intervals <= gap_factor * scale * steps
    window = max(1, int(window)) | 1
    half = window // 2
    padded_time = np.pad(np.where(continuous, intervals, 0.0), half, mode="edge")
    padded_steps = np.pad(np.where(continuous, steps, 0.0), half, mode="edge")
    kernel = np.ones(window)
    elapsed = np.convolve(padded_time, kernel, mode="valid")
    beats = np.convolve(padded_steps, kernel, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        period = np.where(beats > 0, elapsed / np.where(beats > 0, beats, 1.0), np.nan)
    return np.concatenate([period[:1], period])


def _iqr(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def alignment_row(live, expert, downbeats, *,
                  tolerance: float = TOLERANCE_SEC) -> dict:
    live = np.asarray(live, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64).reshape(-1)
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)

    offsets = nearest_offset(live, expert)
    periods = local_periods(expert)
    matched = nearest_index(live, expert)
    period_at = periods[matched] if expert.size and live.size else np.zeros(0)
    phases = fold_to_beats(offsets, period_at) if live.size else np.zeros(0)

    reverse = nearest_offset(expert, live)
    to_downbeats = nearest_offset(downbeats, live)

    def share(values) -> float:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0
        return float(np.mean(np.abs(finite) <= tolerance))

    return {
        "n_live": int(live.size),
        "n_expert": int(expert.size),
        "n_downbeats": int(downbeats.size),
        "median_offset_sec": float(np.nanmedian(offsets)) if live.size else float("nan"),
        "median_abs_phase": float(np.nanmedian(np.abs(phases))) if live.size else float("nan"),
        "phase_iqr": _iqr(np.abs(phases)) if live.size else float("nan"),
        "live_on_grid": share(offsets),
        "expert_covered": share(reverse),
        "downbeat_on_beats": share(to_downbeats),
        "median_ibi_live": float(np.median(np.diff(live))) if live.size > 1 else float("nan"),
        "median_ibi_expert": float(np.median(np.diff(expert))) if expert.size > 1 else float("nan"),
        "ibi_ratio": (float(np.median(np.diff(live)) / np.median(np.diff(expert)))
                      if live.size > 1 and expert.size > 1 else float("nan")),
        "extra_beats": int(live.size) - int(expert.size),
    }


def _tempo_residual(live_period: float, expert_period: float) -> float:
    if not (np.isfinite(live_period) and np.isfinite(expert_period)) or expert_period <= 0:
        return float("nan")
    ratio = live_period / expert_period
    return float(min(abs(ratio / multiple - 1.0) for multiple in TEMPO_MULTIPLES))


def decoder_instants(beat_times, params: PhaseParams | None = None) -> np.ndarray:
    params = params or PhaseParams(subdivision=2, lag_beats=lag_for(1, 2))
    dense = candidate_grid(beat_times, params.subdivision)
    if dense.size == 0:
        return dense
    decisions = BarPhaseHMM(params).decode(dense, np.full(dense.size, np.nan))
    return np.asarray([d.time for d in decisions], dtype=np.float64)


def bar_rate_ratio(beat_times, downbeats, params: PhaseParams | None = None) -> float:
    params = params or PhaseParams(subdivision=2, lag_beats=lag_for(1, 2))
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    if downbeats.size == 0:
        return float("nan")
    cycle = BEATS_PER_BAR * int(params.subdivision)
    return float(decoder_instants(beat_times, params).size / (cycle * downbeats.size))


def reach_labels(live, downbeats, expert, *, tolerance: float = TOLERANCE_SEC,
                 params: PhaseParams | None = None) -> list:
    params = params or PhaseParams(subdivision=2, lag_beats=lag_for(1, 2))
    live = np.asarray(live, dtype=np.float64).reshape(-1)
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64).reshape(-1)
    if downbeats.size == 0:
        return []
    if live.size == 0:
        return ["no_coverage"] * downbeats.size

    dense = candidate_grid(live, params.subdivision)
    coasted = decoder_instants(live, params)

    to_beat = np.abs(nearest_offset(downbeats, live))
    to_dense = np.abs(nearest_offset(downbeats, dense))
    to_coast = np.abs(nearest_offset(downbeats, coasted))

    live_periods = local_periods(live)
    live_pulse = pulse_period(live)
    expert_pulse = pulse_period(expert) if expert.size > 2 else None
    near = nearest_index(downbeats, live)
    if expert.size:
        beat_phase_residual = fold_to_beats(
            nearest_offset(live, expert),
            local_periods(expert)[nearest_index(live, expert)])
    else:
        beat_phase_residual = np.full(live.size, np.nan)

    labels: list = []
    for index, moment in enumerate(downbeats):
        if to_beat[index] <= tolerance:
            labels.append("beat")
            continue
        if to_dense[index] <= tolerance:
            labels.append("midpoint")
            continue
        if to_coast[index] <= tolerance:
            labels.append("coast")
            continue

        anchor = int(near[index])
        period = float(live_periods[anchor])
        if (moment < live[0] - 0.5 * period) or (moment > live[-1] + 0.5 * period):
            labels.append("no_coverage")
            continue

        after = int(np.searchsorted(live, moment))
        gap = (live[after] - live[after - 1]
               if 0 < after < live.size else float("nan"))
        if np.isfinite(gap) and np.isfinite(period) and period > 0 \
                and gap >= params.coast_ratio * period:
            labels.append("dropout")
            continue

        expert_period = (float(expert_pulse[int(nearest_index([moment], expert)[0])])
                         if expert_pulse is not None else float("nan"))
        residual = _tempo_residual(float(live_pulse[anchor]), expert_period)
        if np.isfinite(residual) and residual > TEMPO_TOLERANCE:
            labels.append("tempo_mismatch")
            continue

        lo = max(0, anchor - BEATS_PER_BAR * 2)
        hi = min(live.size, anchor + BEATS_PER_BAR * 2 + 1)
        labels.append("fraction_lock"
                      if _iqr(beat_phase_residual[lo:hi]) <= LOCK_IQR_BEATS
                      else "jitter")
    return labels


def subdivided_grid(beat_times, subdivision: int) -> np.ndarray:
    """Ceiling analysis only; pinned equal to ``candidate_grid`` at subdivision 2 by test."""
    times = np.asarray(beat_times, dtype=np.float64).reshape(-1)
    subdivision = int(subdivision)
    if subdivision <= 1 or times.size < 2:
        return times
    steps = np.arange(subdivision, dtype=np.float64) / subdivision
    dense = times[:-1, None] + steps[None, :] * np.diff(times)[:, None]
    return np.append(dense.reshape(-1), times[-1])


def grid_ceiling(live, downbeats, subdivision: int,
                 tolerance: float = TOLERANCE_SEC) -> float:
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    if downbeats.size == 0:
        return 0.0
    grid = subdivided_grid(live, subdivision)
    if grid.size == 0:
        return 0.0
    return float(np.mean(np.abs(nearest_offset(downbeats, grid)) <= tolerance))


def downbeat_residuals(live, downbeats, expert) -> np.ndarray:
    live = np.asarray(live, dtype=np.float64).reshape(-1)
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64).reshape(-1)
    if downbeats.size == 0:
        return np.zeros(0, dtype=np.float64)
    if live.size == 0 or expert.size < 2:
        return np.full(downbeats.size, np.nan, dtype=np.float64)
    periods = local_periods(expert)[nearest_index(downbeats, expert)]
    return np.abs(fold_to_beats(nearest_offset(downbeats, live), periods))


def score_downbeats(predicted, truth, tolerance: float = TOLERANCE_SEC) -> dict:
    return prf(*match_events(np.asarray(predicted, dtype=np.float64),
                             np.asarray(truth, dtype=np.float64), tolerance))


def confidence_sweep(decisions, truth, thresholds=CONFIDENCE_THRESHOLDS,
                     tolerance: float = TOLERANCE_SEC) -> dict:
    times = np.asarray([d.time for d in decisions if d.phase == 1], dtype=np.float64)
    confidence = np.asarray([d.confidence for d in decisions if d.phase == 1],
                            dtype=np.float64)
    rows: dict = {}
    for threshold in thresholds:
        kept = times[confidence >= threshold]
        score = score_downbeats(kept, truth, tolerance)
        rows[float(threshold)] = {**{key: score[key] for key in ("tp", "fp", "fn")},
                                  "kept": int(kept.size), "total": int(times.size)}
    return rows


def phase_scores(decisions, subdivision: int, expert_times, expert_phases,
                 tolerance: float = TOLERANCE_SEC) -> dict:
    expert_times = np.asarray(expert_times, dtype=np.float64).reshape(-1)
    expert_phases = np.asarray(expert_phases, dtype=np.int64).reshape(-1)
    times = np.asarray([d.time for d in decisions], dtype=np.float64)
    phases = np.asarray([d.phase for d in decisions], dtype=np.int64)

    correct = covered = interstitial = 0
    if times.size and expert_times.size:
        index = nearest_index(expert_times, times)
        near = np.abs(times[index] - expert_times) <= tolerance
        mapped = np.asarray([bar_phase(int(p), subdivision) for p in phases[index]],
                            dtype=np.int64)
        covered = int(np.count_nonzero(near))
        interstitial = int(np.count_nonzero(near & (mapped == 0)))
        correct = int(np.count_nonzero(near & (mapped == expert_phases)))
    total = int(expert_times.size)
    return {
        "correct": correct,
        "covered": covered,
        "total": total,
        "interstitial": interstitial,
        "accuracy": correct / covered if covered else 0.0,
        "coverage": covered / total if total else 0.0,
    }


def interval_deviation(downbeats, *, deviation: float = INTERVAL_DEVIATION,
                       window: int = INTERVAL_WINDOW) -> dict:
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    if downbeats.size < 3:
        return {"events": 0, "intervals": max(int(downbeats.size) - 1, 0),
                "minutes": 0.0, "per_minute": 0.0}
    intervals = np.diff(downbeats)
    running = rolling_median(intervals, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(running > 0, np.abs(intervals - running) / running, 0.0)
    minutes = float(downbeats[-1] - downbeats[0]) / 60.0
    events = int(np.count_nonzero(relative > deviation))
    return {
        "events": events,
        "intervals": int(intervals.size),
        "minutes": minutes,
        "per_minute": events / minutes if minutes > 0 else 0.0,
    }


def beat_anchored_flips(decisions, subdivision: int) -> dict:
    cycle = BEATS_PER_BAR * int(subdivision)
    records: list = []
    grid_index = 0
    coasted_since = False
    for decision in decisions:
        if decision.virtual:
            coasted_since = True
            continue
        if grid_index % int(subdivision) == 0:
            records.append((int(decision.phase), coasted_since))
            coasted_since = False
        grid_index += 1

    flips = pairs = breaks = 0
    for (previous, _), (current, broken) in zip(records, records[1:]):
        if broken:
            breaks += 1
            continue
        pairs += 1
        if (current - previous) % cycle != int(subdivision) % cycle:
            flips += 1
    return {"flips": flips, "pairs": pairs, "breaks": breaks,
            "beats": len(records)}


def edges_from_downbeats(downbeats) -> np.ndarray:
    edges = np.unique(np.asarray(downbeats, dtype=np.float64).reshape(-1))
    if edges.size < 2:
        raise RuntimeError(
            f"{edges.size} predicted downbeats -- there is no bar grid to decode "
            f"on (the section decoder runs at bar rate by design)")
    return np.append(edges, edges[-1] + float(np.median(np.diff(edges))))


def config_fingerprint(params: PhaseParams, condition: str, *,
                       refine: bool) -> dict:
    payload = {**{key: (float(value) if isinstance(value, float) else value)
                  for key, value in asdict(params).items()},
               "condition": str(condition), "refine": bool(refine),
               "tolerance_sec": float(TOLERANCE_SEC)}
    document = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {"config": document,
            "sha256": hashlib.sha256(document.encode("utf-8")).hexdigest()}


def split_guard(data_dir, ids, split: str = "val", *, reason: str | None = None) -> list:
    reason = reason or ("this mode reads annotated truth to choose a value, "
                        "which is a tuning read")
    path = Path(data_dir) / SPLITS_FILE
    if not path.exists():
        raise RuntimeError(f"no splits at {path} -- it is never regenerated implicitly")
    document = json.loads(path.read_text(encoding="utf-8"))
    if split not in document:
        raise RuntimeError(f"{path} has no {split!r} split (has {sorted(document)})")
    allowed = {str(i) for i in document[split]}
    ids = [str(i) for i in ids]
    outside = [i for i in ids if i not in allowed]
    if outside:
        raise RuntimeError(
            f"{len(outside)} of the {len(ids)} requested ids are not in {split} "
            f"({outside[:5]}{' ...' if len(outside) > 5 else ''}) -- {reason}")
    return ids


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_dir(data_dir) -> Path:
    return Path(data_dir) / MODELS_DIR / MODEL_VERSION


def sidecar_path(data_dir, youtube_id: str) -> Path:
    return Path(data_dir) / SIDECAR_DIR / f"{youtube_id}.npz"


def load_truth(data_dir, ids) -> dict:
    # Lazy: downbeat_dataset pulls torch, which the helpers above must import without.
    from .downbeat_dataset import load_beat_grids

    grids, missing = load_beat_grids(Path(data_dir), ids)
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(list(ids))} tracks have no beat grid "
            f"({missing[:5]}) -- truth is not optional")
    return {youtube_id: (np.asarray(grid.times, dtype=np.float64),
                         np.asarray(grid.phases, dtype=np.int64),
                         np.asarray(grid.downbeat_times, dtype=np.float64))
            for youtube_id, grid in grids.items()}


def read_sidecar(path) -> dict:
    with np.load(Path(path)) as archive:
        data = {"activation": np.asarray(archive["activation"], dtype=np.float64),
                "frame_sec": float(archive["frame_sec"]),
                "t0": float(archive["t0"]),
                "model_sha": str(archive["model_sha"])}
        for condition in CONDITIONS:
            data[f"{condition}_beat_time"] = np.asarray(
                archive[f"{condition}_beat_time"], dtype=np.float64)
            data[f"{condition}_beat_score"] = np.asarray(
                archive[f"{condition}_beat_score"], dtype=np.float64)
    return data


def refine_decisions(decisions, sidecar: dict) -> list:
    from .downbeat_decoder import refine_instants

    return refine_instants(decisions, sidecar["activation"], sidecar["frame_sec"],
                           sidecar["t0"])


def decode_evidence(sidecar: dict, condition: str, params: PhaseParams, *,
                    refine: bool = False) -> list:
    """``decode_track`` off arrays already in memory; pinned equal to it by test."""
    from .downbeat_decoder import aggregate_at_beats

    beats = sidecar[f"{condition}_beat_time"]
    if int(params.subdivision) == 1:
        times, scores = beats, sidecar[f"{condition}_beat_score"]
    else:
        times = candidate_grid(beats, params.subdivision)
        scores, _counts = aggregate_at_beats(
            sidecar["activation"], times, sidecar["frame_sec"], sidecar["t0"])
    decisions = BarPhaseHMM(params).decode(times, scores)
    return refine_decisions(decisions, sidecar) if refine else decisions


def score_decisions(decisions, truth: tuple, subdivision: int) -> dict:
    beat_times, beat_phases, downbeats = truth
    predicted = downbeat_times(decisions)
    score = score_downbeats(predicted, downbeats)
    return {
        **{key: score[key] for key in ("precision", "recall", "f1", "tp", "fp", "fn")},
        "phase": phase_scores(decisions, subdivision, beat_times, beat_phases),
        "stability": beat_anchored_flips(decisions, subdivision),
        "interval": interval_deviation(predicted),
        "confidence": confidence_sweep(decisions, downbeats),
        "n_predicted": int(predicted.size),
        "n_truth": int(downbeats.size),
    }


def aggregate_rows(rows: dict) -> dict:
    values = list(rows.values())
    if not values:
        return {}
    tp = sum(row["tp"] for row in values)
    fp = sum(row["fp"] for row in values)
    fn = sum(row["fn"] for row in values)
    per_track_f1 = np.asarray([row["f1"] for row in values], dtype=np.float64)
    flips = np.asarray([row["stability"]["flips"] for row in values], dtype=np.float64)
    deviation = np.asarray([row["interval"]["per_minute"] for row in values],
                           dtype=np.float64)
    phase_correct = sum(row["phase"]["correct"] for row in values)
    phase_covered = sum(row["phase"]["covered"] for row in values)
    phase_total = sum(row["phase"]["total"] for row in values)
    interstitial = sum(row["phase"]["interstitial"] for row in values)
    events = sum(row["interval"]["events"] for row in values)
    minutes = sum(row["interval"]["minutes"] for row in values)
    return {
        "tracks": len(values),
        **prf(tp, fp, fn),
        "f1_median": float(np.median(per_track_f1)),
        "f1_mean": float(np.mean(per_track_f1)),
        "phase_accuracy": phase_correct / phase_covered if phase_covered else 0.0,
        "phase_coverage": phase_covered / phase_total if phase_total else 0.0,
        "phase_interstitial_share": interstitial / phase_covered if phase_covered else 0.0,
        "predicted_per_truth": ((tp + fp) / (tp + fn)) if (tp + fn) else 0.0,
        "flips_median": float(np.median(flips)),
        "flips_mean": float(np.mean(flips)),
        "flips_max": float(np.max(flips)),
        "flips_le_1_share": float(np.mean(flips <= 1.0)),
        "interval_dev_per_min_median": float(np.median(deviation)),
        "interval_dev_per_min_micro": events / minutes if minutes else 0.0,
        "confidence": {
            str(threshold): {
                **prf(sum(row["confidence"][threshold]["tp"] for row in values),
                      sum(row["confidence"][threshold]["fp"] for row in values),
                      sum(row["confidence"][threshold]["fn"] for row in values)),
                "kept_share": (sum(row["confidence"][threshold]["kept"] for row in values)
                               / max(sum(row["confidence"][threshold]["total"]
                                         for row in values), 1)),
            }
            for threshold in values[0]["confidence"]
        },
    }


def _track_job(args) -> tuple:
    youtube_id, path, truth, specs = args
    sidecar = read_sidecar(path)
    rows: list = []
    for condition, params in specs:
        decisions = decode_evidence(sidecar, condition, params)
        rows.append(score_decisions(decisions, truth, params.subdivision))
        rows.append(score_decisions(refine_decisions(decisions, sidecar), truth,
                                    params.subdivision))
    return youtube_id, rows


def evaluate_configs(data_dir, ids, truth: dict, specs, *,
                     workers: int = 1) -> list:
    data_dir = Path(data_dir)
    specs = list(specs)
    jobs = [(youtube_id, sidecar_path(data_dir, youtube_id), truth[youtube_id],
             specs) for youtube_id in ids]
    collected: dict = {}
    if workers <= 1:
        for job in jobs:
            youtube_id, rows = _track_job(job)
            collected[youtube_id] = rows
    else:
        with futures.ProcessPoolExecutor(max_workers=int(workers)) as pool:
            for youtube_id, rows in pool.map(_track_job, jobs, chunksize=1):
                collected[youtube_id] = rows
    return [{youtube_id: collected[youtube_id][index] for youtube_id in ids}
            for index in range(2 * len(specs))]


def evaluate_ids(data_dir, ids, truth: dict, condition: str, params: PhaseParams,
                 *, refine: bool = False, workers: int = 1) -> dict:
    both = evaluate_configs(data_dir, ids, truth, [(condition, params)],
                            workers=workers)
    return both[1 if refine else 0]


def run_alignment(data_dir, ids, truth: dict) -> dict:
    data_dir = Path(data_dir)
    rows: dict = {}
    reach: dict = {label: 0 for label in REACH_LABELS}
    per_track_reach: dict = {}
    per_track_ceiling: dict = {}
    per_track_rate: dict = {}
    residuals: dict = {label: [] for label in REACH_LABELS}
    for youtube_id in ids:
        sidecar = read_sidecar(sidecar_path(data_dir, youtube_id))
        beat_times, _phases, downbeats = truth[youtube_id]
        live = sidecar["live_beat_time"]
        rows[youtube_id] = alignment_row(live, beat_times, downbeats)
        labels = reach_labels(live, downbeats, beat_times)
        counts = {label: labels.count(label) for label in REACH_LABELS}
        per_track_reach[youtube_id] = counts
        for label, count in counts.items():
            reach[label] += count
        found = sum(counts[label] for label in REACHED)
        per_track_ceiling[youtube_id] = {
            "beats": counts["beat"] / len(labels) if labels else 0.0,
            "grid": (counts["beat"] + counts["midpoint"]) / len(labels) if labels else 0.0,
            "decoder": found / len(labels) if labels else 0.0,
            "quarter_bound": grid_ceiling(live, downbeats, 4),
            "eighth_bound": grid_ceiling(live, downbeats, 8),
        }
        per_track_rate[youtube_id] = {
            "half_beat_grid": bar_rate_ratio(
                live, downbeats, PhaseParams(subdivision=2, lag_beats=lag_for(1, 2))),
            "beat_grid": bar_rate_ratio(
                live, downbeats, PhaseParams(subdivision=1, lag_beats=lag_for(1, 1))),
        }
        for label, residual in zip(labels, downbeat_residuals(live, downbeats, beat_times)):
            if np.isfinite(residual):
                residuals[label].append(float(residual))
    total = sum(reach.values())
    summary = {
        "tracks": len(rows),
        "downbeats": total,
        "reach_counts": reach,
        "reach_share": {label: (count / total if total else 0.0)
                        for label, count in reach.items()},
        "ceiling_beats": sum(reach[label] for label in ("beat",)) / total if total else 0.0,
        "ceiling_grid": sum(reach[label] for label in ("beat", "midpoint")) / total
                        if total else 0.0,
        "ceiling_decoder": sum(reach[label] for label in REACHED) / total if total else 0.0,
    }
    for key in ("median_abs_phase", "phase_iqr", "live_on_grid", "expert_covered",
                "downbeat_on_beats", "ibi_ratio", "median_offset_sec"):
        values = np.asarray([row[key] for row in rows.values()], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"{key}_median"] = float(np.median(finite)) if finite.size else float("nan")
    for key in ("beats", "grid", "decoder", "quarter_bound", "eighth_bound"):
        values = np.asarray([row[key] for row in per_track_ceiling.values()],
                            dtype=np.float64)
        summary[f"ceiling_{key}_micro"] = float(np.average(
            values, weights=[row["n_downbeats"] for row in rows.values()]))
        summary[f"ceiling_{key}_deciles"] = [
            float(np.percentile(values, share)) for share in range(0, 101, 10)]
        summary[f"ceiling_{key}_tracks_at_gate"] = int(
            np.count_nonzero(values >= GATE_F1))
    downbeat_total = sum(row["n_downbeats"] for row in rows.values())
    for key in ("half_beat_grid", "beat_grid"):
        values = np.asarray([row[key] for row in per_track_rate.values()],
                            dtype=np.float64)
        weights = np.asarray([row["n_downbeats"] for row in rows.values()],
                             dtype=np.float64)
        bar_rate_micro = float(np.sum(values * weights) / downbeat_total) if downbeat_total else float("nan")
        summary[f"bar_rate_{key}_micro"] = bar_rate_micro
        summary[f"bar_rate_{key}_median"] = float(np.median(values))
        coverage = (summary["ceiling_decoder"] if key == "half_beat_grid"
                    else summary["ceiling_beats"])
        summary[f"f1_ceiling_{key}"] = 2.0 * coverage / (1.0 + bar_rate_micro)
    summary["residual_histogram"] = {
        label: np.histogram(values, bins=RESIDUAL_BINS)[0].tolist() if values else []
        for label, values in residuals.items()}
    summary["residual_median"] = {
        label: (float(np.median(values)) if values else float("nan"))
        for label, values in residuals.items()}
    summary["residual_bins"] = list(RESIDUAL_BINS)
    return {"summary": summary, "per_track": rows, "per_track_reach": per_track_reach,
            "per_track_ceiling": per_track_ceiling, "per_track_rate": per_track_rate}


def sweep_rows(data_dir, ids, truth: dict, specs, *, workers: int = 1) -> list:
    specs = list(specs)
    per_config = evaluate_configs(data_dir, ids, truth, specs, workers=workers)
    rows: list = []
    for index, (condition, params) in enumerate(specs):
        for offset, refine in ((0, False), (1, True)):
            rows.append({
                "condition": condition,
                "params": {key: (float(v) if isinstance(v, float) else v)
                           for key, v in asdict(params).items()},
                "look_ahead_beats": params.lag_beats / params.subdivision,
                "refine": refine,
                **config_fingerprint(params, condition, refine=refine),
                "aggregate": aggregate_rows(per_config[2 * index + offset]),
            })
    return rows


def naive_grids(data_dir, ids) -> dict:
    data_dir = Path(data_dir)
    return {youtube_id: read_sidecar(sidecar_path(data_dir, youtube_id))
                        ["live_beat_time"][::BEATS_PER_BAR]
            for youtube_id in ids}


def ablation_rows(data_dir, ids, predicted: dict, *, section_dir: str,
                  models_subdir: str, naive: dict | None = None) -> dict:
    from .decoder import DecodeParams, bar_grid
    from .evaluate_v1 import (
        DEFAULT_SPACE,
        POSTERIORS_DIR,
        TrackInputs,
        bar_observations,
        evaluate_config,
        load_decoder_config,
    )
    from .priors import Priors
    from build_training_table import TABLE_FILE
    from evaluate_against_labels import TOLERANCES_SEC as SECTION_TOLERANCES_SEC
    from evaluate_against_labels import load_tracks
    from raveform_fetch_annotations import BEATS_DIR, annotations_dir

    data_dir = Path(data_dir)
    models = data_dir / MODELS_DIR / models_subdir
    params = load_decoder_config(models / "decoder_config.json")
    priors = Priors.load(models / "priors.json")
    by_youtube_id = {t.track_id.split(".", 1)[-1]: t
                     for t in load_tracks(data_dir / TABLE_FILE)}
    beats_dir = annotations_dir(data_dir) / BEATS_DIR
    posteriors_dir = data_dir / (section_dir or POSTERIORS_DIR)

    expert_inputs: list = []
    predicted_inputs: list = []
    naive_inputs: list = []
    skipped: list = []
    for youtube_id in ids:
        track = by_youtube_id.get(youtube_id)
        sidecar = posteriors_dir / f"{youtube_id}.npz"
        if track is None or not sidecar.exists():
            skipped.append({"youtube_id": youtube_id,
                            "reason": "no table rows" if track is None
                                      else "no posterior sidecar"})
            continue
        beat_csv = beats_dir / f"{track.track_id}.beat.csv"
        if not beat_csv.exists():
            skipped.append({"youtube_id": youtube_id, "reason": "no beat grid"})
            continue
        try:
            columns = [(bar_grid(beat_csv), expert_inputs),
                       (edges_from_downbeats(predicted[youtube_id]), predicted_inputs)]
            if naive is not None:
                columns.append((edges_from_downbeats(naive[youtube_id]), naive_inputs))
        except RuntimeError as error:
            skipped.append({"youtube_id": youtube_id, "reason": str(error)})
            continue
        for edges, bucket in columns:
            posteriors, boundary = bar_observations(
                sidecar, edges, min_coverage=params.min_coverage,
                boundary_tolerance_sec=params.boundary_tolerance_sec)
            bucket.append(TrackInputs(
                track_id=track.track_id, youtube_id=youtube_id, edges=edges,
                posteriors=posteriors, boundary=boundary, times=track.times,
                labels=track.labels, intents=track.intents))

    def column(inputs) -> dict:
        result = evaluate_config(inputs, priors, params, space=DEFAULT_SPACE)
        score = result["score"]
        bars = int(sum(item.edges.size - 1 for item in inputs))
        boundary = {}
        for tolerance in SECTION_TOLERANCES_SEC:
            precision, recall, f1 = score.boundary_prf("class", tolerance)
            boundary[f"{tolerance}"] = {"precision": precision, "recall": recall,
                                        "f1": f1}
        return {
            "tracks": score.tracks,
            "bars": bars,
            "macro_f1": float(score.macro_f1),
            "boundary": boundary,
            "flicker_per_audience_minute": {
                f"{tolerance}": float(rate)
                for tolerance, rate in score.flicker_per_minute["class"].items()},
            "undecoded_share": float(score.no_intent_sec / score.exposure_sec
                                     if score.exposure_sec else 0.0),
        }

    expert_column = column(expert_inputs)
    predicted_column = column(predicted_inputs)
    return {
        "section_chain": models_subdir,
        **({"naive_grid": column(naive_inputs)} if naive is not None else {}),
        "posteriors_dir": str(posteriors_dir.name),
        "decoder_config_sha256": file_sha256(models / "decoder_config.json"),
        "priors_sha256": file_sha256(models / "priors.json"),
        "skipped": skipped,
        "expert_grid": expert_column,
        "predicted_grid": predicted_column,
        "delta": {
            "bars_ratio": (predicted_column["bars"] / expert_column["bars"]
                           if expert_column["bars"] else float("nan")),
            "macro_f1": predicted_column["macro_f1"] - expert_column["macro_f1"],
            "boundary_f1": {
                key: predicted_column["boundary"][key]["f1"]
                     - expert_column["boundary"][key]["f1"]
                for key in expert_column["boundary"]},
            "flicker_per_audience_minute": {
                key: predicted_column["flicker_per_audience_minute"][key]
                     - expert_column["flicker_per_audience_minute"][key]
                for key in expert_column["flicker_per_audience_minute"]},
        },
    }


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "training" / "data" / "raveform"


def read_split(data_dir, split: str) -> list:
    path = Path(data_dir) / SPLITS_FILE
    document = json.loads(path.read_text(encoding="utf-8"))
    return [str(i) for i in document[split]]


def sweep_grid(look_ahead, penalties, subdivisions) -> list:
    return [PhaseParams(lag_beats=lag_for(beats_ahead, subdivision),
                        subdivision=subdivision, flip_penalty=float(penalty))
            for subdivision in subdivisions
            for beats_ahead in look_ahead
            for penalty in penalties]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Downbeat tracking verdict: alignment, val sweep, one test read")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--align", action="store_true",
                      help="corpus live-vs-expert alignment + residual decomposition")
    mode.add_argument("--sweep", action="store_true",
                      help="the val lever sweep (flip penalty x look-ahead x refine)")
    mode.add_argument("--freeze", action="store_true",
                      help="write the chosen config, hashed, BEFORE any test read")
    mode.add_argument("--verdict", action="store_true",
                      help="score the frozen config; --split test is the one read")
    parser.add_argument("--flip-penalty", type=float, default=None)
    parser.add_argument("--look-ahead", type=int, default=LOOK_AHEAD_BUDGET_BEATS)
    parser.add_argument("--subdivision", type=int, default=2)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--penalties", type=float, nargs="*",
                        default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0])
    parser.add_argument("--look-aheads", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--subdivisions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--role", default="headline",
                        help="what this frozen config is FOR, recorded in the file")
    parser.add_argument("--config", default=None,
                        help="where the frozen config lives (--freeze writes it, "
                             "--verdict reads it)")
    parser.add_argument("--ablate", action="store_true",
                        help="with --verdict: the show ablation on the same tracks")
    parser.add_argument("--section-dir", default="posteriors")
    parser.add_argument("--section-models", default="v1")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    ids = args.ids if args.ids is not None else read_split(data_dir, args.split)
    if args.limit:
        ids = ids[:args.limit]

    if args.align or args.sweep:
        ids = split_guard(data_dir, ids, "val")
    elif args.verdict:
        ids = split_guard(data_dir, ids, args.split,
                          reason=f"a --split {args.split} verdict is labelled "
                                 f"{args.split} and must score only {args.split}")
    truth = load_truth(data_dir, ids)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()

    if args.align:
        payload = {"generated_at": stamp, "split": "val", **run_alignment(data_dir, ids, truth)}
        out = Path(args.out) if args.out else model_dir(data_dir) / ALIGNMENT_FILE.format(split="val")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["summary"], indent=2))
        return 0

    if args.sweep:
        specs = [(condition, params) for condition in CONDITIONS
                 for params in sweep_grid(args.look_aheads, args.penalties,
                                          args.subdivisions)]
        rows = sweep_rows(data_dir, ids, truth, specs, workers=args.workers)
        payload = {"generated_at": stamp, "split": "val", "tracks": len(ids),
                   "rows": rows}
        out = Path(args.out) if args.out else model_dir(data_dir) / "downbeat_sweep_val.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for row in sorted(rows, key=lambda r: -r["aggregate"]["f1"])[:12]:
            print(f"{row['condition']:6s} sub {row['params']['subdivision']} "
                  f"look-ahead {row['look_ahead_beats']:.0f} beats "
                  f"flip {row['params']['flip_penalty']:5.1f} "
                  f"refine {int(row['refine'])} -> F1 {row['aggregate']['f1']:.4f} "
                  f"P {row['aggregate']['precision']:.3f} R {row['aggregate']['recall']:.3f} "
                  f"flips med {row['aggregate']['flips_median']:.0f}")
        return 0

    if args.freeze:
        if args.flip_penalty is None:
            raise SystemExit("--freeze needs --flip-penalty: the config is a choice")
        params = PhaseParams(lag_beats=lag_for(args.look_ahead, args.subdivision),
                             subdivision=args.subdivision,
                             flip_penalty=float(args.flip_penalty))
        payload = {"frozen_at": stamp,
                   "role": args.role,
                   "chosen_on": "val", "condition": "live",
                   "refine": bool(args.refine),
                   "look_ahead_beats": args.look_ahead,
                   **config_fingerprint(params, "live", refine=args.refine)}
        out = Path(args.config) if args.config else model_dir(data_dir) / CONFIG_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    config_path = Path(args.config) if args.config else model_dir(data_dir) / CONFIG_FILE
    if not config_path.exists():
        raise SystemExit(
            f"no frozen config at {config_path} -- run --freeze first, so the "
            f"choice provably predates the read")
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    chosen = json.loads(frozen["config"])
    params = PhaseParams(**{key: value for key, value in chosen.items()
                            if key not in ("condition", "refine", "tolerance_sec")})
    refine = bool(chosen["refine"])

    rows = {condition: evaluate_ids(data_dir, ids, truth, condition, params,
                                    refine=refine, workers=args.workers)
            for condition in CONDITIONS}
    payload = {
        "generated_at": stamp,
        "split": args.split,
        "tracks": len(ids),
        "frozen_config": frozen,
        "frozen_config_sha256": file_sha256(config_path),
        "model_sha": read_sidecar(sidecar_path(data_dir, ids[0]))["model_sha"],
        "conditions": {condition: aggregate_rows(row) for condition, row in rows.items()},
        "per_track": {condition: row for condition, row in rows.items()},
    }
    if args.ablate:
        predicted = {}
        for youtube_id in ids:
            sidecar = read_sidecar(sidecar_path(data_dir, youtube_id))
            predicted[youtube_id] = downbeat_times(
                decode_evidence(sidecar, "live", params, refine=refine))
        chains = list(zip(args.section_dir.split(","),
                          args.section_models.split(",")))
        naive = naive_grids(data_dir, ids)
        payload["ablation"] = {
            models_subdir: ablation_rows(data_dir, ids, predicted,
                                         section_dir=section_dir,
                                         models_subdir=models_subdir, naive=naive)
            for section_dir, models_subdir in chains}
    out = Path(args.out) if args.out else model_dir(data_dir) / EVAL_FILE.format(split=args.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"split": args.split, "tracks": len(ids),
                      "conditions": payload["conditions"],
                      **({"ablation": {chain: {"tracks": row["expert_grid"]["tracks"],
                                               **row["delta"]}
                                       for chain, row in payload["ablation"].items()}}
                         if args.ablate else {})}, indent=2))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
