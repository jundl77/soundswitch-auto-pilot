"""Windowed training set for the CRNN section classifier.

Two independent jobs live here, and both are about *not lying to the model*.

**Splits** (``make_splits``).  The corpus is still downloading, so the split
assignment must be a pure function of the track id: adding tracks may only ever
add, never move a track that is already placed -- otherwise tonight's model and
next week's 1,423-track retrain are not comparable, and a track can migrate from
``test`` into ``train`` between two runs without anyone noticing.  Two families
of track are removed before assignment: the ten ids of the frozen eval set
(``training/eval_set.json``), which are the benchmark the whole plan is judged
on, and every track sharing an *artist* with one of them.  The second is the
subtler leak: producers have a sound, and a net that has heard six other Andy C
rollers has partly memorised the benchmark.  Artist matching is
collaboration-aware -- ``Greg Downey Feat. Bo Bruce`` and a solo ``Greg Downey``
release are the same producer, so the credit is split into participants and any
shared participant excludes.  This over-excludes rather than under-excludes (a
band whose name contains ``&`` splits into two names); losing a few training
tracks is cheap, a contaminated benchmark is not.

**Targets** (``track_targets``, ``WindowDataset``).  The corpus does not label
every second of every track: audio before the first published section (up to
~36 s on this corpus) and audio past the last section end have no ground truth,
and the ``end`` sentinel marks time that must not be re-attributed to a
neighbour.  All three are **loss-masked**, never labelled -- a masked frame
teaches nothing, a mislabelled one teaches the wrong thing.  The boundary head
gets one more mask: where two published sections fold to the same ``label_v1``
class (``breakdown`` + ``cooldown``, ``outro`` + ``altoutro``) the join is a
statement about section identity, not necessarily an audible event, so the
target there is *deleted* rather than taught as a negative.

Time base: mel frame ``k`` carries song time ``t0 + k * frame_sec`` with
``t0 == frame_sec`` (the frame is stamped at the END of its last buffer -- see
``build_training_table.pooled_log_mel``).  Beats, intents and mel frames all
share that base, so targets are read at the frame's own timestamp with no
correction factor.

Nothing here imports torch at module level: the dataset is plain numpy plus an
optional ``torch.utils.data.Dataset`` base, so the target/mask logic stays
testable on a machine that has only synced the default extras.
"""
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

from . import _TRAINING_DIR  # noqa: F401  (puts training/ on sys.path)

from build_clean_manifest import CLEAN_MANIFEST_FILE, STATUS_OK  # noqa: E402
from build_training_table import (  # noqa: E402
    FEATURES_DIR,
    MEL_BANDS,
    POOL_BUFFERS,
    V1_ORDER,
    canonical_coverage,
    label_v1,
)
from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE  # noqa: E402
from raveform_fetch_annotations import load_tracks, parse_sections  # noqa: E402
from select_eval_set import EVAL_SET_FILE, artist_of, load_eval_set  # noqa: E402

try:  # torch lives in the `training` extra; the logic below does not need it
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - exercised only on a torch-less venv
    class _TorchDataset:  # type: ignore[no-redef]
        """Stand-in so this module imports without the training extra."""


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

# One mel frame = POOL_BUFFERS analysis buffers, exactly as the sidecars were
# written.  Derived rather than hardcoded so a change to the pooling factor
# cannot leave the dataset reading the sidecars on the wrong grid; every sidecar
# load re-checks its own recorded `frame_sec` against this.
FRAME_SEC = POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE      # ~46.44 ms

WINDOW_SEC = 16.0        # design spec: trailing ~16 s, decision frame ~8 s in
LABEL_POOL = 2           # frame rate 21.5 Hz / 2 = 10.8 Hz -- the "~10 Hz" head

# The conv front-end pools over frequency only, but a 4x-pooled label head is a
# live variant, so the window is aligned up to a multiple of 4 rather than 2.
# 16 s / 46.44 ms = 344.5 -> 348 frames (16.16 s), the shape the CUDA preflight
# benchmarked.
_FRAME_ALIGN = 4
WINDOW_FRAMES = -(-math.ceil(WINDOW_SEC / FRAME_SEC) // _FRAME_ALIGN) * _FRAME_ALIGN
LABEL_FRAMES = WINDOW_FRAMES // LABEL_POOL

# sigma = the annotation tolerance the evaluator scores at (+-0.5 s).
BOUNDARY_SIGMA_SEC = 0.5
# How far either side of a merged-run join the boundary target is deleted.  2
# sigma keeps the bulk of a Gaussian that *might* belong there out of the loss
# while giving up only ~2 s of supervision per join; 3 sigma would delete half
# again as much for the last 13% of the bump.
BOUNDARY_MASK_RADIUS_SEC = 2.0 * BOUNDARY_SIGMA_SEC

# Amplitude gain in dB is an additive shift in the log domain: log(e * 10^(g/20))
# = log(e) + g * ln(10)/20.  Exact for log(); an approximation under the
# sidecars' log1p(), which is why the result is clamped at zero -- log1p output
# is non-negative and the model must never see an input the encoder cannot make.
GAIN_JITTER_DB = 3.0
LOG_MEL_PER_DB = math.log(10.0) / 20.0

# torch's CrossEntropyLoss default: a masked label is safe to pass straight in.
IGNORE_INDEX = -100

CLASS_INDEX = {label: index for index, label in enumerate(V1_ORDER)}
NUM_CLASSES = len(V1_ORDER)

# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

SPLITS_FILE = "splits.json"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED = 1337
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Decorations that join two credits into one.  Split on them and match on any
# participant: the corpus lists the same producer both solo and behind a Feat.
_PARTICIPANT_SPLIT = re.compile(r"\s*(?:&|\bfeat\b\.?|\bft\b\.?|\bvs\b\.?|\bversus\b)\s*")


class TrackRef(NamedTuple):
    """One split candidate: the three fields the assignment is a function of."""

    track_id: str
    youtube_id: str
    title: str


def artist_participants(title: str) -> frozenset:
    """Every producer credited on a Raveform title, normalised.

    ``artist_of`` already strips chart tags, editorial prefixes and marker
    glyphs and casefolds; this splits what is left on collaboration markers, so
    ``"Greg Downey Feat. Bo Bruce - Come To Me"`` yields both names and a solo
    ``Greg Downey`` release is recognised as the same producer.
    """
    credit = artist_of(title)
    parts = (part.strip(" .-") for part in _PARTICIPANT_SPLIT.split(credit))
    return frozenset(part for part in parts if part)


def excluded_artist_names(eval_tracks) -> frozenset:
    """Union of every participant credited on any eval-set track."""
    names: set = set()
    for track in eval_tracks:
        names |= artist_participants(str(track.get("title", "")))
    return frozenset(names)


def assign_split(youtube_id: str, seed: int = SPLIT_SEED) -> str:
    """Deterministic 70/15/15 bucket for one id.

    A pure function of ``(seed, id)`` -- no corpus-wide shuffle, no dependence
    on how many tracks exist yet.  That is what makes the assignment *extend*
    when the download finishes instead of reshuffling under the frozen file.
    """
    digest = hashlib.sha256(f"{seed}:{youtube_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < SPLIT_RATIOS[0]:
        return "train"
    if value < SPLIT_RATIOS[0] + SPLIT_RATIOS[1]:
        return "val"
    return "test"


def partition(candidates, *, eval_ids, artist_names, seed: int = SPLIT_SEED,
              existing: dict | None = None) -> dict:
    """Candidates -> ``{train, val, test, excluded_*}``, honouring a frozen file.

    ``existing`` is a previously written splits document.  An id it already
    places keeps that placement even if the hash would say otherwise -- the file
    is the record, the hash only decides where *new* ids land.  Exclusions are
    the one thing that overrides it: a track that becomes an eval-set or artist
    match is pulled out of the split it was in rather than left to contaminate.
    """
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
    # Placed once, gone from the corpus now: recorded rather than silently
    # dropped, so a shrinking corpus is visible instead of looking like churn.
    result["retired"] = sorted(set(frozen) - seen)
    return result


def _clean_manifest_rows(data_dir: Path) -> list:
    """``status == ok`` rows of the cleanliness gate, in file order."""
    path = Path(data_dir) / CLEAN_MANIFEST_FILE
    if not path.exists():
        raise RuntimeError(
            f"missing {path} -- run training/build_clean_manifest.py first"
        )
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == STATUS_OK]


def candidate_tracks(data_dir: Path) -> tuple:
    """Trainable tracks, plus why each rejected one was rejected.

    Trainable means all three inputs exist: the audio passed the cleanliness
    gate, the batch sim wrote a mel sidecar, and the annotation carries at least
    one section that survives into ``label_v1`` (a track whose only section is
    the dropped ``end`` sentinel has nothing to supervise).
    """
    data_dir = Path(data_dir)
    features = data_dir / FEATURES_DIR
    by_track_id = {str(track.get("key")): track for track in load_tracks(data_dir)}

    candidates: list = []
    no_sidecar: list = []
    no_annotation: list = []
    unlabeled: list = []

    for row in _clean_manifest_rows(data_dir):
        youtube_id = row["youtube_id"]
        track = by_track_id.get(row["track_id"])
        if track is None:
            no_annotation.append(youtube_id)
            continue
        if not (features / f"{youtube_id}.npz").exists():
            no_sidecar.append(youtube_id)
            continue
        if not v1_spans(parse_sections(track)):
            unlabeled.append(youtube_id)
            continue
        candidates.append(TrackRef(row["track_id"], youtube_id, str(track.get("title", ""))))

    return candidates, sorted(no_sidecar), sorted(no_annotation), sorted(unlabeled)


def make_splits(data_dir, seed: int = SPLIT_SEED, *,
                eval_set_path=EVAL_SET_FILE, write: bool = True) -> dict:
    """Read (or create) ``<data-dir>/splits.json`` and return it.

    Never regenerates implicitly: an existing file's assignments are carried
    through untouched and only new candidates are placed.  The eval set and its
    artists are re-applied on every call, so a widened exclusion rule takes
    effect on the next run without disturbing anything else.
    """
    data_dir = Path(data_dir)
    path = data_dir / SPLITS_FILE

    candidates, no_sidecar, no_annotation, unlabeled = candidate_tracks(data_dir)

    document = load_eval_set(Path(eval_set_path))
    eval_ids = frozenset(str(i) for i in document["youtube_ids"])
    # Titles from the corpus where possible -- the eval-set record is a copy and
    # the corpus is the source the artist parser was written against.
    titles = {ref.youtube_id: ref.title for ref in candidates}
    eval_tracks = [
        {"title": titles.get(str(track.get("youtube_id")), str(track.get("title", "")))}
        for track in document.get("tracks") or []
    ] or [{"title": titles.get(youtube_id, "")} for youtube_id in sorted(eval_ids)]
    artist_names = excluded_artist_names(eval_tracks)

    existing = None
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)

    splits = partition(candidates, eval_ids=eval_ids, artist_names=artist_names,
                       seed=seed, existing=existing)
    splits.update({
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
        "eval_set": sorted(eval_ids),
        "excluded_artist_names": sorted(artist_names),
        "skipped_no_sidecar": no_sidecar,
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


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


class TrackTargets(NamedTuple):
    """Every supervision signal for one whole track, on the mel frame grid."""

    label_frame: np.ndarray         # [n] int16, class index or -1
    label_mask: np.ndarray          # [n] bool
    boundary: np.ndarray            # [n] float32, Gaussian-smeared
    boundary_mask: np.ndarray       # [n] bool
    label_pooled: np.ndarray        # [n // LABEL_POOL] int64, IGNORE_INDEX where masked
    label_pooled_mask: np.ndarray   # [n // LABEL_POOL] bool


def v1_spans(sections: list) -> list:
    """Published sections -> ``[(start, end, v1_label)]``, sorted by start.

    Per published section, not per merged run: merging is a statement about
    section identity and a merged run's span can swallow a dropped ``end``
    sentinel whose time the corpus says must not be re-attributed.
    """
    spans = [
        (start, end, label_v1(label))
        for start, end, label in canonical_coverage(sections)
    ]
    return sorted(spans, key=lambda span: span[0])


def _label_frames(spans: list, times: np.ndarray) -> np.ndarray:
    """Class index per frame, ``-1`` where nothing covers it.

    Vectorised form of ``build_training_table.Timeline.at``: the span with the
    greatest start at or before ``t`` wins, and only if it actually reaches
    ``t`` -- so a zero-width (clamped) section claims nothing and a gap left by
    a dropped sentinel stays uncovered.
    """
    labels = np.full(len(times), -1, dtype=np.int16)
    if not spans:
        return labels
    starts = np.array([span[0] for span in spans], dtype=np.float64)
    ends = np.array([span[1] for span in spans], dtype=np.float64)
    values = np.array([CLASS_INDEX[span[2]] for span in spans], dtype=np.int16)

    index = np.searchsorted(starts, times, side="right") - 1
    covered = index >= 0
    clipped = np.where(covered, index, 0)
    covered &= times < ends[clipped]
    labels[covered] = values[clipped[covered]]
    return labels


def _boundaries_and_joins(spans: list) -> tuple:
    """``(real_boundaries, merged_joins, gaps)`` from consecutive v1 spans.

    A join where the ``label_v1`` class changes is a boundary the model must
    learn.  A join where it does not is a *merged-run join*: the annotator split
    the section, the v1 fold put both halves in one class, and whether anything
    audible happens there is unknowable -- so it is deleted from the boundary
    loss rather than taught as a negative.  A gap (the time of a dropped
    sentinel) is unknowable in both directions and is deleted whole.
    """
    boundaries: list = []
    joins: list = []
    gaps: list = []
    for before, after in zip(spans, spans[1:]):
        if after[0] > before[1] + 1e-9:          # unlabeled time between them
            gaps.append((before[1], after[0]))
        elif before[2] == after[2]:
            joins.append(after[0])
        else:
            boundaries.append(after[0])
    return boundaries, joins, gaps


def _pool_labels(label_frame: np.ndarray, pool: int = LABEL_POOL) -> tuple:
    """Frame-rate labels -> majority label per pooled group.

    Masked frames do not vote; a group with no vote at all is masked.  Ties are
    broken by the latest voting frame in the group, because a pooled frame
    carries the song time of its END -- so the label that was current at the
    frame's own timestamp wins rather than the alphabetically first class.
    """
    usable = (len(label_frame) // pool) * pool
    grouped = label_frame[:usable].reshape(-1, pool)
    groups = grouped.shape[0]

    voting = grouped >= 0
    rows = np.repeat(np.arange(groups), pool).reshape(groups, pool)[voting]
    columns = grouped[voting].astype(np.int64)
    positions = np.tile(np.arange(pool), (groups, 1))[voting]

    counts = np.zeros((groups, NUM_CLASSES), dtype=np.int64)
    np.add.at(counts, (rows, columns), 1)
    recency = np.full((groups, NUM_CLASSES), -1, dtype=np.int64)
    np.maximum.at(recency, (rows, columns), positions)

    # count dominates (recency < pool + 1), recency only breaks ties.
    choice = (counts * (pool + 1) + recency).argmax(axis=1)
    mask = voting.any(axis=1)
    pooled = np.where(mask, choice, IGNORE_INDEX).astype(np.int64)
    return pooled, mask


def track_targets(sections: list, n_frames: int, frame_sec: float = FRAME_SEC,
                  t0: float | None = None) -> TrackTargets:
    """All supervision arrays for one track, on its own mel frame grid."""
    t0 = frame_sec if t0 is None else t0
    times = t0 + np.arange(n_frames, dtype=np.float64) * frame_sec
    spans = v1_spans(sections)

    label_frame = _label_frames(spans, times)
    label_mask = label_frame >= 0

    boundary = np.zeros(n_frames, dtype=np.float32)
    boundary_mask = np.zeros(n_frames, dtype=bool)

    if spans:
        first_start = spans[0][0]
        last_end = max(span[1] for span in spans)
        boundaries, joins, gaps = _boundaries_and_joins(spans)

        for instant in boundaries:
            boundary = np.maximum(
                boundary,
                np.exp(-0.5 * ((times - instant) / BOUNDARY_SIGMA_SEC) ** 2),
            ).astype(np.float32)

        radius = BOUNDARY_MASK_RADIUS_SEC
        deleted = np.zeros(n_frames, dtype=bool)
        for instant in joins:
            deleted |= np.abs(times - instant) <= radius
        for start, end in gaps:
            deleted |= (times >= start - radius) & (times <= end + radius)
        # A genuine transition sitting inside a deleted neighbourhood keeps its
        # supervision -- deletion is for ambiguity, not for erasing known events.
        if deleted.any() and boundaries:
            keep = np.zeros(n_frames, dtype=bool)
            for instant in boundaries:
                keep |= np.abs(times - instant) <= radius
            deleted &= ~keep

        boundary_mask = (times >= first_start) & (times < last_end) & ~deleted
        boundary[~boundary_mask] = 0.0

    label_pooled, label_pooled_mask = _pool_labels(label_frame)
    return TrackTargets(label_frame, label_mask, boundary, boundary_mask,
                        label_pooled, label_pooled_mask)


# --------------------------------------------------------------------------- #
# Sidecars
# --------------------------------------------------------------------------- #


def sidecar_shape(path) -> tuple:
    """``(n_frames, n_bands)`` read from the npz header alone.

    The dataset needs every track's length at construction time to lay out its
    windows; decompressing 600 MB of mel to learn it would cost a minute and
    ~1.5 GB.  The .npy header inside the zip is a few dozen bytes.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("mel.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(handle)
                else:  # pragma: no cover - numpy has shipped 1.0/2.0 only
                    raise ValueError(f"unsupported .npy version {version}")
                return tuple(shape)
    except (KeyError, ValueError, zipfile.BadZipFile):
        # Any surprise in the container: pay the decompression rather than guess.
        with np.load(path) as archive:
            return tuple(archive["mel"].shape)


def load_sidecar(path) -> np.ndarray:
    """The pooled log-mel array, refusing a sidecar built on another grid."""
    with np.load(path) as archive:
        mel = np.asarray(archive["mel"], dtype=np.float32)
        frame_sec = float(archive["frame_sec"])
        pool = int(archive["pool_buffers"]) if "pool_buffers" in archive else POOL_BUFFERS
    if not math.isclose(frame_sec, FRAME_SEC, rel_tol=1e-9):
        raise RuntimeError(
            f"{path}: frame_sec {frame_sec!r} does not match this build's "
            f"{FRAME_SEC!r} -- the sidecar was written on a different mel grid"
        )
    if pool != POOL_BUFFERS or mel.shape[1] != MEL_BANDS:
        raise RuntimeError(
            f"{path}: mel geometry {mel.shape[1]} bands / pool {pool} does not "
            f"match {MEL_BANDS} / {POOL_BUFFERS}"
        )
    return mel


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class _Track(NamedTuple):
    youtube_id: str
    path: Path
    sections: list
    usable: int          # frames, truncated to a whole number of pooled groups
    slots: int           # windows this track contributes per epoch


class WindowDataset(_TorchDataset):
    """Fixed-length mel windows with masked ``label_v1`` targets.

    One item is ``(mel [W, 40] float32, labels [W/2] int64, label_mask [W/2]
    bool, boundary [W] float32, boundary_mask [W] bool)``.  Masked label
    positions carry ``IGNORE_INDEX`` so a forgotten mask degrades to torch's
    default ignore behaviour rather than to a confident wrong class.

    ``augment=True`` (training) draws a fresh window offset and a gain shift per
    item per epoch, seeded from ``(seed, epoch, index)`` so a run is reproducible
    and DataLoader workers cannot collide.  ``augment=False`` (val/test) tiles
    the track from frame 0 and touches nothing -- the same item forever.

    Mel arrays are cached in memory on first touch (1-2 MB per track, so a
    ~540-track train split settles around 1 GB once every window has been seen).
    With ``num_workers > 0`` on Windows each spawned worker builds its own copy
    of that cache, so ``num_workers=0`` is both faster and cheaper for a corpus
    this size (see the CUDA preflight report).
    """

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
                for track in load_tracks(self.data_dir)
            }

        features = self.data_dir / FEATURES_DIR
        self._tracks: list = []
        self._slots: list = []
        for youtube_id in youtube_ids:
            path = features / f"{youtube_id}.npz"
            if not path.exists():
                raise RuntimeError(f"missing mel sidecar for {youtube_id}: {path}")
            sections = sections_by_youtube_id.get(youtube_id) or []
            # An id with no annotation would yield a fully masked track: hours of
            # zero-gradient windows that look exactly like a working dataset.
            if not v1_spans(sections):
                raise RuntimeError(
                    f"no label_v1 sections for {youtube_id} -- it would train on "
                    f"nothing but masked frames"
                )
            n_frames = sidecar_shape(path)[0]
            usable = (n_frames // LABEL_POOL) * LABEL_POOL
            slots = max(1, usable // self.window_frames)
            index = len(self._tracks)
            self._tracks.append(_Track(youtube_id, path, sections, usable, slots))
            self._slots.extend((index, slot) for slot in range(slots))

    # -- torch Dataset ----------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._slots)

    def __getitem__(self, index: int) -> tuple:
        return self.window(index, self.window_offset(index), self.gain_db(index))

    # -- sampling policy --------------------------------------------------- #

    def set_epoch(self, epoch: int) -> None:
        """Re-roll the augmentation for a new epoch (no-op when ``augment`` is off)."""
        self._epoch = int(epoch)

    def _rng(self, index: int, salt: int) -> np.random.Generator:
        return np.random.default_rng([self._seed, self._epoch, index, salt])

    def window_offset(self, index: int) -> int:
        """First mel frame of item ``index`` -- always a multiple of ``LABEL_POOL``.

        Pool alignment is what lets the pooled label targets be sliced straight
        out of the whole-track arrays instead of re-pooled per window.
        """
        track_index, slot = self._slots[index]
        track = self._tracks[track_index]
        limit = max(0, track.usable - self.window_frames)
        if not self.augment:
            return int(min(slot * self.window_frames, limit) // LABEL_POOL * LABEL_POOL)
        draw = self._rng(index, 0).integers(0, limit // LABEL_POOL + 1)
        return int(draw) * LABEL_POOL

    def gain_db(self, index: int) -> float:
        """Gain jitter for item ``index``, in dB (0.0 outside augmentation)."""
        if not self.augment or self.gain_jitter_db <= 0.0:
            return 0.0
        return float(self._rng(index, 1).uniform(-self.gain_jitter_db,
                                                 self.gain_jitter_db))

    # -- the window itself -------------------------------------------------- #

    def window(self, index: int, offset: int, gain_db: float = 0.0) -> tuple:
        """The five arrays for item ``index`` at an explicit frame offset."""
        track_index, _slot = self._slots[index]
        track = self._tracks[track_index]
        mel_full = self._mel(track)
        targets = self._targets(track)

        mel = _take(mel_full, offset, self.window_frames, 0.0)
        if gain_db:
            np.maximum(mel + gain_db * LOG_MEL_PER_DB, 0.0, out=mel)

        pooled_offset = offset // LABEL_POOL
        return (
            mel,
            _take(targets.label_pooled, pooled_offset, self.label_frames, IGNORE_INDEX),
            _take(targets.label_pooled_mask, pooled_offset, self.label_frames, False),
            _take(targets.boundary, offset, self.window_frames, 0.0),
            _take(targets.boundary_mask, offset, self.window_frames, False),
        )

    def track_ids(self) -> list:
        """The youtube ids backing this dataset, in construction order."""
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
    """``source[offset:offset+length]``, zero-padded at the tail.

    A track shorter than one window (or a window running off the end) is padded
    rather than skipped, and the padding is masked by the caller's ``fill`` --
    silence the model is told nothing about beats dropping the track entirely.
    """
    shape = (length,) + source.shape[1:]
    out = np.full(shape, fill, dtype=source.dtype)
    end = min(offset + length, len(source))
    if end > offset:
        out[:end - offset] = source[offset:end]
    return out
