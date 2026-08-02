"""Bar-grid supervision: per-frame downbeat targets and per-beat phase labels."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import _TRAINING_DIR  # noqa: F401

from raveform_fetch_annotations import (  # noqa: E402
    beat_csv_path,
    load_tracks,
    parse_beat_csv,
    parse_sections,
)

from .dataset import FRAME_SEC, WindowDataset, _take  # noqa: E402

DOWNBEAT_SIGMA_SEC = 0.070
BEATS_PER_BAR = 4
BEAT_PAST_END = -1

# Above any grid jitter and below a whole missing 4/4 bar (5x); no corpus grid trips it.
GAP_FACTOR = 1.75

# Catches a stuck phase column that parses clean; the corpus scores 0.0 and 1.00.
MAX_PHASE_BREAK_FRACTION = 0.20
BAR_LENGTH_TOLERANCE = 0.25


class BeatGrid(NamedTuple):
    times: np.ndarray
    phases: np.ndarray
    phase_breaks: np.ndarray
    source: str

    @property
    def downbeat_times(self) -> np.ndarray:
        return self.times[self.phases == 1]

    @property
    def bars(self) -> int:
        return int(np.count_nonzero(self.phases == 1))

    @property
    def median_beat_sec(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return float(np.median(np.diff(self.times)))


def parse_beat_grid(rows, source: str = "<memory>") -> BeatGrid:
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"{source}: no beats in the grid -- nothing to supervise")

    try:
        times = np.array([row[0] for row in rows], dtype=np.float64)
        raw_phases = np.array([row[1] for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{source}: a beat row is not a number ({exc})") from exc

    if not np.isfinite(times).all():
        raise RuntimeError(f"{source}: beat time is not a number (nan or inf)")
    if not np.isfinite(raw_phases).all() or (raw_phases != np.rint(raw_phases)).any():
        raise RuntimeError(f"{source}: downbeat phase is not a whole number")
    phases = raw_phases.astype(np.int64)

    if (times < 0.0).any():
        index = int(np.argmax(times < 0.0))
        raise RuntimeError(
            f"{source}: negative beat time {times[index]!r} at beat {index}")
    if len(times) > 1 and (np.diff(times) <= 0.0).any():
        index = int(np.argmax(np.diff(times) <= 0.0))
        raise RuntimeError(
            f"{source}: beat times must be strictly increasing -- beat {index} at "
            f"{times[index]!r} is followed by {times[index + 1]!r}")
    outside = (phases < 1) | (phases > BEATS_PER_BAR)
    if outside.any():
        index = int(np.argmax(outside))
        raise RuntimeError(
            f"{source}: beat {index} has downbeat phase {phases[index]}, expected "
            f"1..{BEATS_PER_BAR}")

    expected = phases[:-1] % BEATS_PER_BAR + 1
    breaks = np.flatnonzero(phases[1:] != expected).astype(np.int64)
    return BeatGrid(times, phases.astype(np.int8), breaks, str(source))


def grid_anomalies(grid: BeatGrid) -> list:
    reasons: list = []
    beats = len(grid.times)
    if beats > 1:
        fraction = len(grid.phase_breaks) / (beats - 1)
        if fraction > MAX_PHASE_BREAK_FRACTION:
            reasons.append(
                f"{fraction:.0%} of beats break the 1..{BEATS_PER_BAR} cycle "
                f"({len(grid.phase_breaks)} of {beats - 1}) -- the phase column "
                f"does not describe bars")

    downbeats = grid.downbeat_times
    beat_sec = grid.median_beat_sec
    if downbeats.size > 2 and beat_sec > 0.0:
        bar_sec = float(np.median(np.diff(downbeats)))
        ratio = bar_sec / (BEATS_PER_BAR * beat_sec)
        if abs(ratio - 1.0) > BAR_LENGTH_TOLERANCE:
            reasons.append(
                f"median bar is {bar_sec:.3f}s against {BEATS_PER_BAR} beats at "
                f"{beat_sec:.3f}s (ratio {ratio:.2f}) -- the downbeats are not "
                f"one per bar")
    return reasons


def load_beat_grid(path) -> BeatGrid:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"missing beat grid: {path}")
    try:
        rows = parse_beat_csv(path)
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{path}: cannot read the beat grid -- expected the 'time', 'downbeat' "
            f"and 'section' column headers ({exc!r})") from exc
    except ValueError as exc:
        raise RuntimeError(
            f"{path}: a beat row is not a number ({exc})") from exc
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(f"{path}: cannot read the beat grid ({exc})") from exc
    return parse_beat_grid(rows, source=str(path))


def load_beat_grids(data_dir, youtube_ids) -> tuple:
    data_dir = Path(data_dir)
    wanted = [str(i) for i in youtube_ids]
    records = {str(track.get("id")): track for track in load_tracks(data_dir)}

    grids: dict = {}
    missing: list = []
    for youtube_id in wanted:
        track = records.get(youtube_id)
        path = beat_csv_path(data_dir, track) if track is not None else None
        if path is None or not path.exists():
            missing.append(youtube_id)
            continue
        grids[youtube_id] = load_beat_grid(path)
    return grids, missing


class DownbeatTargets(NamedTuple):
    downbeat: np.ndarray
    mask: np.ndarray
    beat_time: np.ndarray
    beat_phase: np.ndarray
    beat_frame: np.ndarray


def track_downbeat_targets(grid: BeatGrid, n_frames: int,
                           frame_sec: float = FRAME_SEC,
                           t0: float | None = None,
                           sigma_sec: float = DOWNBEAT_SIGMA_SEC) -> DownbeatTargets:
    t0 = frame_sec if t0 is None else t0
    times = t0 + np.arange(n_frames, dtype=np.float64) * frame_sec

    downbeats = grid.downbeat_times
    target = np.zeros(n_frames, dtype=np.float32)
    if downbeats.size and n_frames:
        distance = _distance_to_nearest_downbeat(times, downbeats)
        target = np.exp(-0.5 * (distance / sigma_sec) ** 2).astype(np.float32)

    mask = np.zeros(n_frames, dtype=bool)
    if n_frames:
        mask = (times >= grid.times[0]) & (times <= grid.times[-1])
        for start, end in _grid_holes(grid):
            mask &= ~((times > start) & (times < end))
    target[~mask] = 0.0

    frames = np.rint((grid.times - t0) / frame_sec).astype(np.int64)
    np.maximum(frames, 0, out=frames)
    frames[frames >= n_frames] = BEAT_PAST_END

    return DownbeatTargets(target, mask, grid.times, grid.phases, frames)


def _distance_to_nearest_downbeat(times: np.ndarray,
                                  downbeats: np.ndarray) -> np.ndarray:
    right = np.searchsorted(downbeats, times)
    before = downbeats[np.clip(right - 1, 0, downbeats.size - 1)]
    after = downbeats[np.clip(right, 0, downbeats.size - 1)]
    return np.minimum(np.abs(times - before), np.abs(times - after))


def _grid_holes(grid: BeatGrid) -> list:
    if len(grid.times) < 3:
        return []
    gaps = np.diff(grid.times)
    limit = GAP_FACTOR * float(np.median(gaps))
    return [(float(grid.times[index]), float(grid.times[index + 1]))
            for index in np.flatnonzero(gaps > limit)]


class DownbeatWindowDataset(WindowDataset):
    def __init__(self, data_dir, youtube_ids, *, grids_by_youtube_id: dict | None = None,
                 sections_by_youtube_id: dict | None = None, **kwargs) -> None:
        data_dir = Path(data_dir)
        records: dict = {}
        if sections_by_youtube_id is None or grids_by_youtube_id is None:
            records = {str(track.get("id")): track for track in load_tracks(data_dir)}
        if sections_by_youtube_id is None:
            sections_by_youtube_id = {youtube_id: parse_sections(track)
                                      for youtube_id, track in records.items()}

        super().__init__(data_dir, youtube_ids,
                         sections_by_youtube_id=sections_by_youtube_id, **kwargs)

        self._grids: dict = dict(grids_by_youtube_id or {})
        self._by_id: dict = {track.youtube_id: track for track in self._tracks}
        self._downbeat_cache: dict = {}
        for youtube_id in self.track_ids():
            grid = self._grids.get(youtube_id)
            if grid is None:
                track = records.get(youtube_id)
                if track is None and not records:
                    raise RuntimeError(
                        f"no beat grid supplied for {youtube_id}: grids were "
                        f"injected, so the corpus annotations were never read")
                path = beat_csv_path(data_dir, track) if track is not None else None
                if path is None:
                    raise RuntimeError(
                        f"no beat grid for {youtube_id}: it has no record in "
                        f"{data_dir}'s annotations, so its grid cannot be located")
                if not path.exists():
                    raise RuntimeError(
                        f"missing beat grid for {youtube_id}: {path} -- it would "
                        f"train on nothing but masked frames")
                grid = load_beat_grid(path)
                self._grids[youtube_id] = grid
            if grid.bars < 2:
                raise RuntimeError(
                    f"{youtube_id}: beat grid has {grid.bars} downbeat(s) -- "
                    f"nothing to supervise")
            anomalies = grid_anomalies(grid)
            if anomalies:
                raise RuntimeError(
                    f"{youtube_id}: beat grid is structurally valid but not "
                    f"believable -- " + "; ".join(anomalies))

    def window(self, index: int, offset: int, gain_db: float = 0.0) -> tuple:
        targets = self.targets_for(self.track_id_of(index))
        return (
            self.mel_window(index, offset, gain_db),
            _take(targets.downbeat, offset, self.window_frames, 0.0),
            _take(targets.mask, offset, self.window_frames, False),
        )

    def grid(self, youtube_id: str) -> BeatGrid:
        return self._grids[youtube_id]

    def targets_for(self, youtube_id: str) -> DownbeatTargets:
        targets = self._downbeat_cache.get(youtube_id)
        if targets is None:
            track = self._by_id[youtube_id]
            targets = track_downbeat_targets(self._grids[youtube_id], track.usable,
                                             FRAME_SEC, FRAME_SEC)
            self._downbeat_cache[youtube_id] = targets
        return targets


def alignment_profile(mel: np.ndarray, targets: DownbeatTargets, *,
                      bands: int = 4, shifts=range(-4, 5)) -> dict:
    energy = mel[:, :bands].mean(axis=1)
    flux = np.maximum(np.diff(energy, prepend=energy[:1]), 0.0)
    base = float(flux.mean()) or 1.0

    profile: dict = {}
    for name, wanted in (("downbeat", targets.beat_phase == 1),
                         ("beat", targets.beat_phase != 1)):
        frames = targets.beat_frame[wanted]
        frames = frames[frames >= 0]
        lifts = {}
        for shift in shifts:
            shifted = frames + shift
            shifted = shifted[(shifted >= 0) & (shifted < len(flux))]
            lifts[int(shift)] = float(flux[shifted].mean()) / base if shifted.size else 0.0
        profile[name] = lifts
    return profile


def validate_tracks(data_dir, youtube_ids) -> list:
    from .dataset import LABEL_POOL, load_sidecar, sidecar_shape
    from build_training_table import FEATURES_DIR

    data_dir = Path(data_dir)
    grids, missing = load_beat_grids(data_dir, youtube_ids)
    rows: list = []
    for youtube_id in [str(i) for i in youtube_ids]:
        if youtube_id in missing:
            rows.append({"youtube_id": youtube_id, "error": "no beat grid"})
            continue
        grid = grids[youtube_id]
        path = data_dir / FEATURES_DIR / f"{youtube_id}.npz"
        if not path.exists():
            rows.append({"youtube_id": youtube_id, "error": "no mel sidecar"})
            continue
        n_frames = (sidecar_shape(path)[0] // LABEL_POOL) * LABEL_POOL
        targets = track_downbeat_targets(grid, n_frames, FRAME_SEC, FRAME_SEC)
        beat_sec = grid.median_beat_sec
        profile = alignment_profile(load_sidecar(path)[:n_frames], targets)
        stamps = FRAME_SEC * (targets.beat_frame + 1)
        clamped = targets.beat_time < FRAME_SEC / 2.0
        on_grid = (targets.beat_frame >= 0) & ~clamped
        offsets = stamps[on_grid] - targets.beat_time[on_grid]
        rows.append({
            "youtube_id": youtube_id,
            "beats": len(grid.times),
            "bars": grid.bars,
            "expected_bars": round(len(grid.times) / BEATS_PER_BAR, 2),
            "phase_breaks": int(len(grid.phase_breaks)),
            "bpm": round(60.0 / beat_sec, 1) if beat_sec else 0.0,
            "first_beat": round(float(grid.times[0]), 3),
            "last_beat": round(float(grid.times[-1]), 2),
            "frames": int(n_frames),
            "masked_pct": round(100.0 * float((~targets.mask).mean()), 2),
            "holes": len(_grid_holes(grid)),
            "peak_frames": int(np.count_nonzero(targets.downbeat > 0.85)),
            "target_mean": round(float(targets.downbeat.mean()), 5),
            "beats_past_end": int(np.count_nonzero(targets.beat_frame < 0)),
            "beats_clamped_leading": int(np.count_nonzero(clamped)),
            "anomalies": grid_anomalies(grid),
            "stamp_offset_ms": round(float(offsets.mean()) * 1000.0, 2),
            "stamp_offset_max_ms": round(float(np.abs(offsets).max()) * 1000.0, 2),
            "align_peak_shift": max(profile["downbeat"], key=profile["downbeat"].get),
            "align_control_shift": max(profile["beat"], key=profile["beat"].get),
            "align_profile": {name: {k: round(v, 2) for k, v in lifts.items()}
                              for name, lifts in profile.items()},
        })
    return rows


def format_report(rows: list) -> str:
    lines = ["youtube_id    beats  bars  bars/4  brk    bpm  mask%  holes  peaks "
             " align  ctrl  stamp_ms",
             "-" * 88]
    for row in rows:
        if "error" in row:
            lines.append(f"{row['youtube_id']:<12}  {row['error']}")
            continue
        lines.append(
            f"{row['youtube_id']:<12} {row['beats']:>6} {row['bars']:>5} "
            f"{row['expected_bars']:>7} {row['phase_breaks']:>4} {row['bpm']:>6} "
            f"{row['masked_pct']:>6} {row['holes']:>6} {row['peak_frames']:>6} "
            f"{row['align_peak_shift']:>+6} {row['align_control_shift']:>+5} "
            f"{row['stamp_offset_ms']:>+9.2f}")
        for reason in row["anomalies"]:
            lines.append(f"{'':<12}  ANOMALY: {reason}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    from build_training_table import default_data_dir
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split", default="train")
    parser.add_argument("--tracks", type=int, default=3,
                        help="how many tracks of the split to validate")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="validate these youtube ids instead of the split's first")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ids:
        ids = [str(i) for i in args.ids]
    else:
        from .priors import split_ids
        ids = split_ids(args.data_dir, args.split)[:args.tracks]
    rows = validate_tracks(args.data_dir, ids)
    print(format_report(rows))
    for row in rows:
        if "error" not in row:
            print(f"\n{row['youtube_id']}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
