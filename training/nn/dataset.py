"""Corpus splits and the windowed, loss-masked training set for the CRNN."""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import _TRAINING_DIR  # noqa: F401  (imported for its sys.path side effect)

from build_clean_manifest import CLEAN_MANIFEST_FILE, STATUS_OK  # noqa: E402
from build_training_table import (  # noqa: E402
    FEATURES_DIR,
    MEL_BANDS,
    POOL_BUFFERS,
    label_coverage,
)
from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE  # noqa: E402
from lib.label_space import LABEL_INDEX, NUM_SECTION_CLASSES  # noqa: E402
from raveform_fetch_annotations import load_all_tracks, parse_sections  # noqa: E402
from select_eval_set import EVAL_SET_FILE, artist_of, load_eval_set  # noqa: E402

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    class _TorchDataset:  # type: ignore[no-redef]
        pass


FRAME_SEC = POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE

WINDOW_SEC = 16.0
LABEL_POOL = 2

# 4, not LABEL_POOL: a 4x-pooled label head is a live variant of this geometry.
_FRAME_ALIGN = 4
WINDOW_FRAMES = -(-math.ceil(WINDOW_SEC / FRAME_SEC) // _FRAME_ALIGN) * _FRAME_ALIGN
LABEL_FRAMES = WINDOW_FRAMES // LABEL_POOL

BOUNDARY_SIGMA_SEC = 0.5
BOUNDARY_MASK_RADIUS_SEC = 2.0 * BOUNDARY_SIGMA_SEC

GAIN_JITTER_DB = 3.0
LOG_MEL_PER_DB = math.log(10.0) / 20.0

# torch's CrossEntropyLoss default, so a masked label passes straight in.
IGNORE_INDEX = -100

SPLITS_FILE = "splits.json"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED = 1337
SPLIT_RATIOS = (0.70, 0.15, 0.15)

_PARTICIPANT_SPLIT = re.compile(r"\s*(?:&|\bfeat\b\.?|\bft\b\.?|\bvs\b\.?|\bversus\b)\s*")


class TrackRef(NamedTuple):
    track_id: str
    youtube_id: str
    title: str


def artist_participants(title: str) -> frozenset:
    credit = artist_of(title)
    parts = (part.strip(" .-") for part in _PARTICIPANT_SPLIT.split(credit))
    return frozenset(part for part in parts if part)


def excluded_artist_names(eval_tracks) -> frozenset:
    names: set = set()
    for track in eval_tracks:
        names |= artist_participants(str(track.get("title", "")))
    return frozenset(names)


def assign_split(youtube_id: str, seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{youtube_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < SPLIT_RATIOS[0]:
        return "train"
    if value < SPLIT_RATIOS[0] + SPLIT_RATIOS[1]:
        return "val"
    return "test"


def partition(candidates, *, eval_ids, artist_names, seed: int = SPLIT_SEED,
              existing: dict | None = None) -> dict:
    frozen = {}
    for split in SPLIT_NAMES:
        for youtube_id in (existing or {}).get(split) or []:
            frozen[str(youtube_id)] = split

    result: dict = {split: [] for split in SPLIT_NAMES}
    excluded_eval: list = []
    excluded_artist: list = []
    seen: set = set()

    for ref in sorted(candidates, key=lambda candidate: candidate.youtube_id):
        seen.add(ref.youtube_id)
        if ref.youtube_id in eval_ids:
            excluded_eval.append(ref.youtube_id)
            continue
        if artist_names and (artist_participants(ref.title) & artist_names):
            excluded_artist.append(ref.youtube_id)
            continue
        split = frozen.get(ref.youtube_id) or assign_split(ref.youtube_id, seed)
        result[split].append(ref.youtube_id)

    result["excluded_eval_set"] = excluded_eval
    result["excluded_artist"] = excluded_artist
    result["retired"] = sorted(set(frozen) - seen)
    return result


def _clean_manifest_rows(data_dir: Path) -> list:
    path = Path(data_dir) / CLEAN_MANIFEST_FILE
    if not path.exists():
        raise RuntimeError(
            f"missing {path} -- run training/raveform/build_clean_manifest.py first"
        )
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == STATUS_OK]


def candidate_tracks(data_dir: Path) -> tuple:
    data_dir = Path(data_dir)
    by_track_id = {str(track.get("key")): track for track in load_all_tracks(data_dir)}

    candidates: list = []
    no_annotation: list = []
    unlabeled: list = []

    for row in _clean_manifest_rows(data_dir):
        youtube_id = row["youtube_id"]
        track = by_track_id.get(row["track_id"])
        if track is None:
            no_annotation.append(youtube_id)
            continue
        if not label_spans(parse_sections(track)):
            unlabeled.append(youtube_id)
            continue
        candidates.append(TrackRef(row["track_id"], youtube_id, str(track.get("title", ""))))

    return candidates, sorted(no_annotation), sorted(unlabeled)


def make_splits(data_dir, seed: int = SPLIT_SEED, *,
                eval_set_path=EVAL_SET_FILE, write: bool = True) -> dict:
    data_dir = Path(data_dir)
    path = data_dir / SPLITS_FILE

    candidates, no_annotation, unlabeled = candidate_tracks(data_dir)

    document = load_eval_set(Path(eval_set_path))
    eval_ids = frozenset(str(i) for i in document["youtube_ids"])

    titles = {str(track.get("id")): str(track.get("title", ""))
              for track in load_all_tracks(data_dir)}
    recorded = {str(track.get("youtube_id")): str(track.get("title", ""))
                for track in document.get("tracks") or []}
    eval_titles = {}
    unresolved = []
    for youtube_id in sorted(eval_ids):
        title = titles.get(youtube_id) or recorded.get(youtube_id) or ""
        if not title.strip():
            unresolved.append(youtube_id)
        eval_titles[youtube_id] = title
    if unresolved:
        raise RuntimeError(
            f"no title for eval-set track(s) {unresolved} in the corpus "
            f"annotations or in {eval_set_path} -- the artist exclusion would be "
            f"silently weakened, so the split is refused"
        )
    artist_names = excluded_artist_names(
        [{"title": title} for title in eval_titles.values()])

    existing = None
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        recorded_seed = existing.get("seed")
        if recorded_seed is not None and int(recorded_seed) != int(seed):
            raise RuntimeError(
                f"{path} was frozen with seed {recorded_seed}, called with {seed} "
                f"-- delete the file to regenerate deliberately, or pass the "
                f"recorded seed; new ids must not be placed by a different rule"
            )
        recorded_ratios = existing.get("ratios")
        if recorded_ratios and dict(recorded_ratios) != dict(zip(SPLIT_NAMES, SPLIT_RATIOS)):
            raise RuntimeError(
                f"{path} was frozen with ratios {recorded_ratios}, this build uses "
                f"{dict(zip(SPLIT_NAMES, SPLIT_RATIOS))} -- same problem as a seed "
                f"change; delete the file to regenerate deliberately"
            )

    splits = partition(candidates, eval_ids=eval_ids, artist_names=artist_names,
                       seed=seed, existing=existing)
    splits.update({
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
        "eval_set": sorted(eval_ids),
        "excluded_artist_names": sorted(artist_names),
        "skipped_no_annotation": no_annotation,
        "skipped_unlabeled": unlabeled,
        "candidates": len(candidates),
        "written_at": datetime.datetime.now(datetime.timezone.utc)
                              .replace(microsecond=0).isoformat(),
    })

    if write:
        tmp = path.with_suffix(".json.part")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(splits, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp.replace(path)
    return splits


class TrackTargets(NamedTuple):
    label_frame: np.ndarray
    label_mask: np.ndarray
    boundary: np.ndarray
    boundary_mask: np.ndarray
    label_pooled: np.ndarray
    label_pooled_mask: np.ndarray


def label_spans(sections: list) -> list:
    return sorted(label_coverage(sections), key=lambda span: span[0])


def _label_frames(spans: list, times: np.ndarray) -> np.ndarray:
    labels = np.full(len(times), -1, dtype=np.int16)
    if not spans:
        return labels
    starts = np.array([span[0] for span in spans], dtype=np.float64)
    ends = np.array([span[1] for span in spans], dtype=np.float64)
    values = np.array([LABEL_INDEX[span[2]] for span in spans], dtype=np.int16)

    index = np.searchsorted(starts, times, side="right") - 1
    covered = index >= 0
    clipped = np.where(covered, index, 0)
    covered &= times < ends[clipped]
    labels[covered] = values[clipped[covered]]
    return labels


def _boundaries_and_gaps(spans: list) -> tuple:
    """Every contiguous section change is a boundary; a hole is neither.

    Two adjacent sections carrying the same label are still two sections the
    annotator published, so their join is a boundary event even though the label
    head sees nothing change across it.
    """
    boundaries: list = []
    gaps: list = []
    for before, after in zip(spans, spans[1:]):
        if after[0] > before[1] + 1e-9:
            gaps.append((before[1], after[0]))
        else:
            boundaries.append(after[0])
    return boundaries, gaps


def _pool_labels(label_frame: np.ndarray, pool: int = LABEL_POOL) -> tuple:
    usable = (len(label_frame) // pool) * pool
    grouped = label_frame[:usable].reshape(-1, pool)
    groups = grouped.shape[0]

    voting = grouped >= 0
    rows = np.repeat(np.arange(groups), pool).reshape(groups, pool)[voting]
    columns = grouped[voting].astype(np.int64)
    positions = np.tile(np.arange(pool), (groups, 1))[voting]

    counts = np.zeros((groups, NUM_SECTION_CLASSES), dtype=np.int64)
    np.add.at(counts, (rows, columns), 1)
    recency = np.full((groups, NUM_SECTION_CLASSES), -1, dtype=np.int64)
    np.maximum.at(recency, (rows, columns), positions)

    outranks_recency = pool + 1
    choice = (counts * outranks_recency + recency).argmax(axis=1)
    mask = voting.any(axis=1)
    pooled = np.where(mask, choice, IGNORE_INDEX).astype(np.int64)
    return pooled, mask


def track_targets(sections: list, n_frames: int, frame_sec: float = FRAME_SEC,
                  t0: float | None = None) -> TrackTargets:
    t0 = frame_sec if t0 is None else t0
    times = t0 + np.arange(n_frames, dtype=np.float64) * frame_sec
    spans = label_spans(sections)

    label_frame = _label_frames(spans, times)
    label_mask = label_frame >= 0

    boundary = np.zeros(n_frames, dtype=np.float32)
    boundary_mask = np.zeros(n_frames, dtype=bool)

    if spans:
        first_start = spans[0][0]
        last_end = max(span[1] for span in spans)
        boundaries, gaps = _boundaries_and_gaps(spans)

        for instant in boundaries:
            boundary = np.maximum(
                boundary,
                np.exp(-0.5 * ((times - instant) / BOUNDARY_SIGMA_SEC) ** 2),
            ).astype(np.float32)

        radius = BOUNDARY_MASK_RADIUS_SEC
        deleted = np.zeros(n_frames, dtype=bool)
        for start, end in gaps:
            deleted |= (times >= start - radius) & (times <= end + radius)
        if deleted.any() and boundaries:
            known_boundary = np.zeros(n_frames, dtype=bool)
            for instant in boundaries:
                known_boundary |= np.abs(times - instant) <= radius
            deleted &= ~known_boundary

        boundary_mask = (times >= first_start) & (times < last_end) & ~deleted
        boundary[~boundary_mask] = 0.0

    label_pooled, label_pooled_mask = _pool_labels(label_frame)
    return TrackTargets(label_frame, label_mask, boundary, boundary_mask,
                        label_pooled, label_pooled_mask)


def sidecar_shape(path) -> tuple:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("mel.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(handle)
                else:  # pragma: no cover
                    raise ValueError(f"unsupported .npy version {version}")
                return tuple(shape)
    except (KeyError, ValueError, zipfile.BadZipFile):
        with np.load(path) as archive:
            return tuple(archive["mel"].shape)


def load_sidecar(path) -> np.ndarray:
    with np.load(path) as archive:
        mel = np.asarray(archive["mel"], dtype=np.float32)
        frame_sec = float(archive["frame_sec"])
        t0 = float(archive["t0"])
        pool = int(archive["pool_buffers"]) if "pool_buffers" in archive else POOL_BUFFERS
    if not math.isclose(frame_sec, FRAME_SEC, rel_tol=1e-9):
        raise RuntimeError(
            f"{path}: frame_sec {frame_sec!r} does not match this build's "
            f"{FRAME_SEC!r} -- the sidecar was written on a different mel grid"
        )
    if not math.isclose(t0, FRAME_SEC, rel_tol=1e-9):
        raise RuntimeError(
            f"{path}: t0 {t0!r} does not match this build's frame origin "
            f"{FRAME_SEC!r} -- every target would be offset against the audio"
        )
    if pool != POOL_BUFFERS or mel.shape[1] != MEL_BANDS:
        raise RuntimeError(
            f"{path}: mel geometry {mel.shape[1]} bands / pool {pool} does not "
            f"match {MEL_BANDS} / {POOL_BUFFERS}"
        )
    return mel


class _Track(NamedTuple):
    youtube_id: str
    path: Path
    sections: list
    usable: int
    slots: int


class WindowDataset(_TorchDataset):
    def __init__(self, data_dir, youtube_ids, *, augment: bool = False,
                 seed: int = SPLIT_SEED, window_frames: int = WINDOW_FRAMES,
                 gain_jitter_db: float = GAIN_JITTER_DB,
                 sections_by_youtube_id: dict | None = None) -> None:
        if window_frames % LABEL_POOL:
            raise ValueError(
                f"window_frames {window_frames} must be a multiple of {LABEL_POOL}"
            )
        self.data_dir = Path(data_dir)
        self.augment = bool(augment)
        self.window_frames = int(window_frames)
        self.label_frames = self.window_frames // LABEL_POOL
        self.gain_jitter_db = float(gain_jitter_db)
        self._seed = int(seed)
        self._epoch = 0
        self._mel_cache: dict = {}
        self._target_cache: dict = {}

        if sections_by_youtube_id is None:
            sections_by_youtube_id = {
                str(track.get("id")): parse_sections(track)
                for track in load_all_tracks(self.data_dir)
            }

        features = self.data_dir / FEATURES_DIR
        self._tracks: list = []
        self._slots: list = []
        for youtube_id in youtube_ids:
            path = features / f"{youtube_id}.npz"
            if not path.exists():
                raise RuntimeError(f"missing mel sidecar for {youtube_id}: {path}")
            sections = sections_by_youtube_id.get(youtube_id) or []
            if not label_spans(sections):
                raise RuntimeError(
                    f"no labelled sections for {youtube_id} -- it would train on "
                    f"nothing but masked frames"
                )
            n_frames = sidecar_shape(path)[0]
            usable = (n_frames // LABEL_POOL) * LABEL_POOL
            slots = max(1, -(-usable // self.window_frames))
            index = len(self._tracks)
            self._tracks.append(_Track(youtube_id, path, sections, usable, slots))
            self._slots.extend((index, slot) for slot in range(slots))

    def __len__(self) -> int:
        return len(self._slots)

    def __getitem__(self, index: int) -> tuple:
        return self.window(index, self.window_offset(index), self.gain_db(index))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng(self, index: int, salt: int) -> np.random.Generator:
        return np.random.default_rng([self._seed, self._epoch, index, salt])

    def window_offset(self, index: int) -> int:
        track_index, slot = self._slots[index]
        track = self._tracks[track_index]
        limit = max(0, track.usable - self.window_frames)
        if not self.augment:
            return int(min(slot * self.window_frames, limit) // LABEL_POOL * LABEL_POOL)
        draw = self._rng(index, 0).integers(0, limit // LABEL_POOL + 1)
        return int(draw) * LABEL_POOL

    def gain_db(self, index: int) -> float:
        if not self.augment or self.gain_jitter_db <= 0.0:
            return 0.0
        return float(self._rng(index, 1).uniform(-self.gain_jitter_db,
                                                 self.gain_jitter_db))

    def mel_window(self, index: int, offset: int, gain_db: float = 0.0) -> np.ndarray:
        track = self._tracks[self._slots[index][0]]
        mel = _take(self._mel(track), offset, self.window_frames, 0.0)
        if gain_db:
            np.maximum(mel + gain_db * LOG_MEL_PER_DB, 0.0, out=mel)
        return mel

    def track_id_of(self, index: int) -> str:
        return self._tracks[self._slots[index][0]].youtube_id

    def window(self, index: int, offset: int, gain_db: float = 0.0) -> tuple:
        track = self._tracks[self._slots[index][0]]
        targets = self._targets(track)
        mel = self.mel_window(index, offset, gain_db)

        pooled_offset = offset // LABEL_POOL
        return (
            mel,
            _take(targets.label_pooled, pooled_offset, self.label_frames, IGNORE_INDEX),
            _take(targets.label_pooled_mask, pooled_offset, self.label_frames, False),
            _take(targets.boundary, offset, self.window_frames, 0.0),
            _take(targets.boundary_mask, offset, self.window_frames, False),
        )

    def track_ids(self) -> list:
        return [track.youtube_id for track in self._tracks]

    def _mel(self, track: _Track) -> np.ndarray:
        mel = self._mel_cache.get(track.youtube_id)
        if mel is None:
            mel = load_sidecar(track.path)[:track.usable]
            self._mel_cache[track.youtube_id] = mel
        return mel

    def _targets(self, track: _Track) -> TrackTargets:
        targets = self._target_cache.get(track.youtube_id)
        if targets is None:
            targets = track_targets(track.sections, track.usable, FRAME_SEC, FRAME_SEC)
            self._target_cache[track.youtube_id] = targets
        return targets


def _take(source: np.ndarray, offset: int, length: int, fill) -> np.ndarray:
    shape = (length,) + source.shape[1:]
    out = np.full(shape, fill, dtype=source.dtype)
    end = min(offset + length, len(source))
    if end > offset:
        out[:end - offset] = source[offset:end]
    return out
