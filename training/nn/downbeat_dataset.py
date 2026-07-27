"""Bar-grid supervision: per-frame downbeat targets and per-beat phase labels.

    uv run python -m training.nn.downbeat_dataset --data-dir <corpus> --tracks 3

The section head learns *what* is playing; this head learns *where the bar
starts*, because that is the one thing the live engine cannot get from aubio.
Aubio emits beat instants with no bar phase, so a decoder that commits at bar
rate and a blackout that must land on the bar before a drop both have nothing to
quantise to.  The supervision is the corpus's own expert beat grids
(``annotations/beats/<key>.beat.csv``: a time and a ``downbeat`` phase in 1..4
for every beat).

**Why a Gaussian at 70 ms.**  Same recipe as the section boundary head, at the
tolerance this component is scored at: a downbeat is an *instant*, and a target
that is one at one frame and zero at its neighbour asks the model to resolve
46 ms of mel to a hard edge and punishes a 47 ms answer as hard as a 4 s one.
The Gaussian makes near-misses cheap and keeps the argmax where the annotator
put it.  Sigma is small relative to a beat (70 ms against 350-700 ms), which is
what makes the off-beats genuine negatives rather than smeared positives -- and
also why the mask needs no guard band at the grid's edges: an unlabelled
downbeat one beat outside the grid contributes ``exp(-12)`` to the edge frame.

**Why the phase labels stay per beat.**  The model predicts one thing, a
downbeat activation at frame rate.  Phase 1..4 is the *decoder's* state, resolved
per beat instant by the bar-phase HMM, so the phase label belongs to a beat and
not to a frame.  ``DownbeatTargets`` therefore carries the beat table
(time, phase, and the mel frame each beat lands on) beside the frame target: the
same arrays the decoder is scored against, read once from the same parse.

**Masks, and what is unknowable.**  Ground truth ends where the grid does, so
audio before the first annotated beat and after the last is loss-masked exactly
as the section dataset masks its unannotated lead-in and tail.  An interior hole
-- consecutive beats much further apart than the track's own median -- is masked
too: the annotator dropped beats there, so "no downbeat here" is not something
the corpus actually said.  No grid in the 1,423-track corpus has such a hole
today; the handling exists because a silent all-negative stretch is invisible.

**A malformed grid stops the run.**  Out-of-range phases, times that do not
increase, a renamed column, a grid with no beats: all raise, naming the file.  A
grid that parses to nothing produces a legal-looking all-zero target and trains a
model to predict no downbeats anywhere, which is the failure this refuses to
have.  A *phase discontinuity* is the one anomaly that is recorded rather than
refused -- v1 is 4/4 only, and a 3/4 bar or an annotator's edit does not
invalidate the downbeat instants themselves.

Windows come from ``WindowDataset``: same sidecars, same geometry checks, same
offsets, augmentation and padding.  This module extends it with one head's
targets rather than restating any of that.  Note that it therefore inherits the
section dataset's requirement that a track have ``label_v1`` sections -- true by
construction for everything in ``splits.json``, and one definition of a
trainable track is better than two.

Import this from a decode path *lazily*: it pulls ``nn.dataset``, and with it
torch, which the showtime-shaped objects must never pay for.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import _TRAINING_DIR  # noqa: F401  (puts training/ on sys.path)

from raveform_fetch_annotations import (  # noqa: E402
    beat_csv_path,
    load_tracks,
    parse_beat_csv,
    parse_sections,
)

# ``_take`` is the section dataset's own pad-and-slice helper: the downbeat
# window must pad its tail exactly as the mel does, and two implementations of
# that would drift apart one frame at a time.
from .dataset import FRAME_SEC, WindowDataset, _take  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# sigma = the tolerance the downbeat verdict is scored at (+-70 ms).
DOWNBEAT_SIGMA_SEC = 0.070

# The corpus is 4/4 throughout; the phase alphabet is a constant, not a guess.
BEATS_PER_BAR = 4

# An inter-beat interval this many times the track's own median is a hole in the
# annotation, not a musical event: at 4/4 a whole missing bar is 5x, and the
# widest legitimate spacing (a half-time feel annotated on the slow grid) is 2x.
# Chosen below that and above any jitter -- 0 of the corpus's 1,423 grids trip
# it, so the cost of the tolerance today is exactly zero.
GAP_FACTOR = 1.75


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #


class BeatGrid(NamedTuple):
    """One track's expert beat grid: an instant and a bar phase per beat."""

    times: np.ndarray            # float64 [n], strictly increasing
    phases: np.ndarray           # int8 [n], 1..BEATS_PER_BAR
    phase_breaks: np.ndarray     # int64 [k], indices where the cycle does not hold
    source: str                  # where it came from, for error messages

    @property
    def downbeat_times(self) -> np.ndarray:
        """The phase-1 instants -- the bar grid the decoder quantises to."""
        return self.times[self.phases == 1]

    @property
    def bars(self) -> int:
        return int(np.count_nonzero(self.phases == 1))

    @property
    def median_beat_sec(self) -> float:
        """Median inter-beat interval (0.0 for a grid too short to have one)."""
        if len(self.times) < 2:
            return 0.0
        return float(np.median(np.diff(self.times)))


def parse_beat_grid(rows, source: str = "<memory>") -> BeatGrid:
    """``[(time, phase, section)]`` -> a validated ``BeatGrid``.

    Pure and path-free so the validation is testable against hand-written grids;
    ``load_beat_grid`` is the thin adapter that reads the corpus's CSV.
    """
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
    # Read as float and checked back, because an int cast would turn a phase of
    # 1.5 into a confident 1 -- the exact shape of silent corruption this refuses.
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

    # A break is a statement about the annotation, not about the audio: recorded
    # and carried, never repaired -- a repaired phase would be this module's
    # opinion masquerading as the annotator's.
    expected = phases[:-1] % BEATS_PER_BAR + 1
    breaks = np.flatnonzero(phases[1:] != expected).astype(np.int64)
    return BeatGrid(times, phases.astype(np.int8), breaks, str(source))


def load_beat_grid(path) -> BeatGrid:
    """Read and validate one ``*.beat.csv``, reusing the corpus's own parser."""
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
    except OSError as exc:  # pragma: no cover - unreadable file, not a bad one
        raise RuntimeError(f"{path}: cannot read the beat grid ({exc})") from exc
    return parse_beat_grid(rows, source=str(path))


def load_beat_grids(data_dir, youtube_ids) -> tuple:
    """``({youtube_id: BeatGrid}, missing)`` for the given ids.

    Missing grids are *returned*, not raised: the corpus report wants the count,
    and the dataset -- where a missing grid means a silently unsupervised track
    -- refuses on its own terms.
    """
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


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


class DownbeatTargets(NamedTuple):
    """Supervision for one whole track, on its own mel frame grid."""

    downbeat: np.ndarray     # [n] float32, Gaussian-smeared, 0 where masked
    mask: np.ndarray         # [n] bool
    beat_time: np.ndarray    # [b] float64, every beat in the grid
    beat_phase: np.ndarray   # [b] int8, 1..4
    beat_frame: np.ndarray   # [b] int64, mel frame index or -1 if off the grid


def track_downbeat_targets(grid: BeatGrid, n_frames: int,
                           frame_sec: float = FRAME_SEC,
                           t0: float | None = None,
                           sigma_sec: float = DOWNBEAT_SIGMA_SEC) -> DownbeatTargets:
    """All downbeat supervision arrays for one track.

    Frame ``k`` carries song time ``t0 + k * frame_sec`` -- the sidecars' own
    time base, unchanged, so targets are read at the frame's own timestamp with
    no correction factor.
    """
    t0 = frame_sec if t0 is None else t0
    times = t0 + np.arange(n_frames, dtype=np.float64) * frame_sec

    downbeats = grid.downbeat_times
    target = np.zeros(n_frames, dtype=np.float32)
    if downbeats.size and n_frames:
        # The Gaussian is monotone in |distance|, so the max over all downbeats
        # is the Gaussian of the distance to the *nearest* one: two binary
        # searches instead of one full-length exp per downbeat.
        right = np.searchsorted(downbeats, times)
        before = downbeats[np.clip(right - 1, 0, downbeats.size - 1)]
        after = downbeats[np.clip(right, 0, downbeats.size - 1)]
        distance = np.minimum(np.abs(times - before), np.abs(times - after))
        target = np.exp(-0.5 * (distance / sigma_sec) ** 2).astype(np.float32)

    mask = np.zeros(n_frames, dtype=bool)
    if n_frames:
        mask = (times >= grid.times[0]) & (times <= grid.times[-1])
        for start, end in _grid_holes(grid):
            mask &= ~((times > start) & (times < end))
    target[~mask] = 0.0

    frames = np.rint((grid.times - t0) / frame_sec).astype(np.int64)
    frames[(frames < 0) | (frames >= n_frames)] = -1

    return DownbeatTargets(target, mask, grid.times, grid.phases, frames)


def _grid_holes(grid: BeatGrid) -> list:
    """``[(start, end)]`` intervals where the annotator's beats stop and resume."""
    if len(grid.times) < 3:
        return []
    gaps = np.diff(grid.times)
    limit = GAP_FACTOR * float(np.median(gaps))
    return [(float(grid.times[index]), float(grid.times[index + 1]))
            for index in np.flatnonzero(gaps > limit)]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class DownbeatWindowDataset(WindowDataset):
    """Fixed-length mel windows with masked per-frame downbeat targets.

    One item is ``(mel [W, 40] float32, downbeat [W] float32, mask [W] bool)``.
    Everything about the window -- offsets, gain jitter, eval-mode tiling,
    tail padding, sidecar geometry checks -- is ``WindowDataset``'s and is not
    restated; only the targets differ.

    Grids are parsed once at construction so that a track with no grid, or a
    grid with nothing to supervise, fails here rather than after an epoch of
    all-negative windows.
    """

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
                path = beat_csv_path(data_dir, track) if track is not None else None
                if path is None:
                    raise RuntimeError(
                        f"no beat grid for {youtube_id}: it is neither in the "
                        f"supplied grids nor in {data_dir}'s annotations")
                if not path.exists():
                    raise RuntimeError(
                        f"missing beat grid for {youtube_id}: {path} -- it would "
                        f"train on nothing but masked frames")
                grid = load_beat_grid(path)
                self._grids[youtube_id] = grid
            # Two downbeats is the least that describes a bar length; below that
            # the track is a fully-negative window generator that looks fine.
            if grid.bars < 2:
                raise RuntimeError(
                    f"{youtube_id}: beat grid has {grid.bars} downbeat(s) -- "
                    f"nothing to supervise")

    # -- the window itself -------------------------------------------------- #

    def window(self, index: int, offset: int, gain_db: float = 0.0) -> tuple:
        """The three arrays for item ``index`` at an explicit frame offset."""
        targets = self.targets_for(self.track_id_of(index))
        return (
            self.mel_window(index, offset, gain_db),
            _take(targets.downbeat, offset, self.window_frames, 0.0),
            _take(targets.mask, offset, self.window_frames, False),
        )

    # -- per-track access (the decoder and the evaluator read these) --------- #

    def grid(self, youtube_id: str) -> BeatGrid:
        """The expert beat grid backing one track."""
        return self._grids[youtube_id]

    def targets_for(self, youtube_id: str) -> DownbeatTargets:
        """Whole-track targets on the same frame grid the windows are cut from."""
        targets = self._downbeat_cache.get(youtube_id)
        if targets is None:
            track = self._by_id[youtube_id]
            targets = track_downbeat_targets(self._grids[youtube_id], track.usable,
                                             FRAME_SEC, FRAME_SEC)
            self._downbeat_cache[youtube_id] = targets
        return targets


# --------------------------------------------------------------------------- #
# Corpus validation (the CLI)
# --------------------------------------------------------------------------- #


def alignment_profile(mel: np.ndarray, targets: DownbeatTargets, *,
                      bands: int = 4, shifts=range(-4, 5)) -> dict:
    """Low-band onset flux at the beat frames, per frame shift, as a lift.

    ``{"downbeat": {shift: lift}, "beat": {shift: lift}}``, each lift being the
    mean positive first difference of the low mel bands at the shifted frames
    over that track's own mean.  It answers one question and only one: is the
    grid on the same clock as the mel?  A time base that has slipped moves the
    peak away from where the audio actually is, and no training curve shows it.

    Two things to read it with.  *Energy* is the wrong statistic -- it measures
    loudness, not events, and its profile is noise; the positive flux is the
    onset.  And the off-beats are the control, not a contrast: this corpus is
    four-on-the-floor, so every beat carries a kick and the downbeat profile is
    expected to look like the off-beat profile.  A downbeat-specific *acoustic*
    signature is what the model is for; it is not what this measures.

    Expect the peak at shift +1 rather than 0.  A pooled frame is stamped at the
    END of its window, so the frame that *contains* an onset is the one at or
    after its instant, and aubio's analysis window smears a transient a further
    hop or two.  That is a property of the mel time base, shared with the
    section head, and it is why the targets stay defined against the frame's own
    stamp: the Gaussian peak then sits on the annotated instant itself, so a
    decoded activation peak is an unbiased estimate of the downbeat time.
    """
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
    """Per-track facts a human can check the grids against, in id order."""
    from .dataset import load_sidecar, sidecar_shape
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
        n_frames = sidecar_shape(path)[0]
        targets = track_downbeat_targets(grid, n_frames, FRAME_SEC, FRAME_SEC)
        beat_sec = grid.median_beat_sec
        profile = alignment_profile(load_sidecar(path), targets)
        stamps = FRAME_SEC * (targets.beat_frame + 1)
        on_grid = targets.beat_frame >= 0
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
            "beats_off_grid": int(np.count_nonzero(targets.beat_frame < 0)),
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
