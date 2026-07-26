#!/usr/bin/env python
"""Batch fast-sim over the clean corpus -> label-aligned per-beat training table.

Reads ``<data-dir>/clean_manifest.csv`` (status ``ok`` rows only -- see
``build_clean_manifest.py``), runs every one of those tracks through the
UNMODIFIED fast simulation (``simulate.runner.run_fast_simulation``, the exact
production pipeline on a virtual clock), joins each detected beat to the
expert Raveform label covering it, and writes::

    <data-dir>/training_table.csv.gz        one row per labeled beat
    <data-dir>/training_table.meta.json     counts, histograms, git SHA, timestamp
    <data-dir>/reports/<youtube_id>.json.gz the cached sim report, one per track
    <data-dir>/features/<youtube_id>.npz    pooled log-mel sidecar for the NN

Two stages, deliberately separated:

**Stage A (expensive, parallel, once per track).**  Simulate, cache the report,
export the mel sidecar, delete the decode cache.

**Stage B (cheap, serial, pure).**  Rebuild the whole table from the cached
reports.  Nothing about the join needs audio, so the join can be fixed and the
table regenerated in seconds without re-simulating 460 tracks -- and the join
logic stays unit-testable in isolation (``tests/test_training_table_labels.py``).
``--table-only`` runs Stage B alone.

The report cache
----------------

Each cached report is stamped with the SHA of the pipeline that produced it
(``lib/`` + ``simulate/``, not repo HEAD -- see ``pipeline_sha``) and with the
mp3's size and mtime.  A track whose stamp still matches, and whose mel sidecar
is on disk, skips BOTH the simulation and the multi-second decode, so a rebuild
over an unchanged corpus costs seconds and a rebuild after 20 new downloads
costs 20 tracks.  Any mismatch -- new pipeline, re-encoded audio, missing
sidecar, unreadable cache -- is a miss and is re-simulated; ``--force`` misses
everything.  Hit and miss counts (with reasons) are printed and recorded in the
meta file, so a run that unexpectedly re-simulates the corpus says why.

Label semantics (binding, from the validated corpus)
----------------------------------------------------

* Canonical vocabulary comes from ``raveform_manifest``: ``end`` dropped,
  ``altintro``->``intro``, ``bridge``->``breakdown``.  ``label_v1`` merges two
  more per the NN design spec (``cooldown``->``breakdown``,
  ``altoutro``->``outro``), giving the 5-class space the model trains on.
* **Coverage is per published section, never per merged run.**  ``canonical_runs``
  merges adjacent same-label sections, and a merged run's *span* can swallow a
  dropped ``end`` sentinel sitting between two members -- time the corpus
  explicitly says must not be re-attributed.  So the label lookup uses the
  individual (clamped, folded) sections and ``canonical_runs`` supplies only the
  labeled *bounds* of the track.
* Audio before the first section start is UNANNOTATED (up to 35.9 s on this
  corpus) and audio past the last section end has no ground truth.  Beats in
  either region are dropped and counted, never absorbed into a neighbour.
* Sections with ``end < start`` (one track: ``1020.c1VBubZ2w3M``) are clamped to
  zero width -- they claim no beat and crash nothing.

Time bases
----------

Beat timestamps are song-position seconds.  Intent blocks are audience time --
one look-ahead delay later -- so they are shifted back by the report's own
``metrics.look_ahead_sec`` before being read at a beat.  Mel frames carry the
same stamp convention as beats (see ``pooled_log_mel``).

Decode-cache discipline
-----------------------

The simulation writes ``<mp3>.<samplerate>.npy`` beside the audio (~7.7x the mp3
size; the full corpus would be ~95 GiB).  Each worker deletes its track's cache
as soon as the sidecar is written.  Caches that already existed before the batch
started are left alone -- the run must not clean up after someone else.

A downloader may be writing into ``audio/`` concurrently, so only ``*.mp3``
files older than ``--min-age-sec`` are touched.

Usage::

    uv run python training/build_training_table.py \\
        --data-dir C:\\Users\\Julian\\Projects\\soundswitch-auto-pilot\\training\\data\\raveform
    uv run python training/build_training_table.py --limit 5    # smoke run
    uv run python training/build_training_table.py --table-only # re-join only
    uv run python training/build_training_table.py --force      # ignore the cache
"""

from __future__ import annotations

import argparse
import bisect
import collections
import concurrent.futures
import csv
import datetime
import gzip
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from build_clean_manifest import (  # noqa: E402  (needs the path inserts above)
    CLEAN_MANIFEST_FILE,
    MIN_AGE_SEC,
    STATUS_OK,
    is_settled,
)
from raveform_fetch_annotations import load_tracks, parse_sections  # noqa: E402
from raveform_manifest import CANONICAL_DROP, CANONICAL_MAP, canonical_runs  # noqa: E402

from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE  # noqa: E402

TABLE_FILE = "training_table.csv.gz"
META_FILE = "training_table.meta.json"
REPORTS_DIR = "reports"
FEATURES_DIR = "features"
AUDIO_DIR = "audio"

# Bump when the cached-report envelope changes shape: every existing cache then
# misses and is rebuilt, rather than being read under the wrong assumptions.
CACHE_VERSION = 1

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

# Continuous features get a per-track z-scored twin (``<name>_z``).  Mixes differ
# in loudness, brightness and bass weight by more than sections do within one
# mix, so a fitter reading absolute values learns the mastering as much as the
# music; the z-scored copy is the mix-invariant view.  Both are kept -- the
# absolute values are what the live classifier actually thresholds on.
CONTINUOUS_COLUMNS = (
    "bpm",
    "onset_density",
    "kick_strength",
    "centroid_trend",
    "sub_bass_ratio",
    "rms",
)

TABLE_HEADER = (
    "track_id",
    "youtube_id",
    "t_song",                # song-position seconds of the beat
    "bpm",
    "onset_density",
    "kick_strength",
    "kick_known",            # 0/1: was the kick measurable at this beat
    "centroid_trend",
    "sub_bass_ratio",
    "rms",
    "intent_at_beat",        # committed show state, de-shifted to song time
    "label_canonical",       # 7-class Raveform vocabulary
    "label_raw",             # exactly as published
    "label_v1",              # 5-class space the NN trains on
    "bar_position_unknown",
) + tuple(f"{column}_z" for column in CONTINUOUS_COLUMNS)

# No intent block covers this beat: the show had not committed a state yet (or
# the run ended before the block covering this song time was stamped).  Empty
# rather than a fake state -- a consumer must be able to tell "unknown" from
# "the lights were in ATMOSPHERIC".
NO_INTENT = ""

# The pipeline has no downbeat tracker, so no row can say where in the bar its
# beat falls.  The column is constant today and exists so the schema does not
# change when Stage-2 downbeat tracking lands.
BAR_POSITION_UNKNOWN = 1

# Kick presence is read off the row's own RMS against the analyser's silence
# gate, NOT by testing kick_strength against its sentinel: the sentinel is a
# number in the range of real ratios, so a genuine measurement can land on it
# (lib/analyser/CLAUDE.md, "Kick strength").  This constant MUST equal
# `_KICK_MIN_RMS` in lib/analyser/music_analyser.py; the coupling is pinned by
# tests/test_training_table_features.py rather than imported, so the table's
# schema does not depend on a private name in the pipeline under evaluation.
KICK_MIN_RMS = 0.005

# label_v1: the canonical vocabulary merged down to the 5 classes the neural
# section classifier trains on (docs/superpowers/specs/2026-07-26-nn-section-
# classifier-design.md).  `cooldown` is positionally defined and undecidable
# from a single window; `altoutro` is a variant marker for the same structural
# role as `outro`.
V1_MAP = {"cooldown": "breakdown", "altoutro": "outro"}
V1_ORDER = ("intro", "buildup", "breakdown", "drop", "outro")
CANONICAL_ORDER = ("intro", "buildup", "drop", "breakdown", "cooldown", "outro", "altoutro")

# --------------------------------------------------------------------------- #
# Mel sidecar (NN features)
# --------------------------------------------------------------------------- #

MEL_BANDS = 40          # must equal MusicAnalyser.mel_filters
POOL_BUFFERS = 8        # 8 x 256 samples @ 44.1 kHz ~= 46 ms per frame


# --------------------------------------------------------------------------- #
# Label geometry
# --------------------------------------------------------------------------- #


class Timeline:
    """Half-open ``[start, end)`` spans with an O(log n) point lookup.

    Spans are sorted by start; on overlap the later span wins, which is the only
    tie-break that keeps a lookup single-valued without inventing a rule the
    annotation does not have.
    """

    def __init__(self, spans: list) -> None:
        ordered = sorted(spans, key=lambda span: span[0])
        self._starts = [span[0] for span in ordered]
        self._ends = [span[1] for span in ordered]
        self._values = [span[2] for span in ordered]

    def __len__(self) -> int:
        return len(self._starts)

    def at(self, t: float):
        """Value of the span covering ``t``, or ``None``."""
        index = bisect.bisect_right(self._starts, t) - 1
        if index < 0:
            return None
        return self._values[index] if t < self._ends[index] else None


def _clamped_spans(sections: list) -> list:
    """``[(start, end, label)]`` with every negative-length section clamped."""
    return [
        (float(start), max(float(start), float(end)), str(label))
        for start, end, label in sections
    ]


def canonical_coverage(sections: list) -> list:
    """Labeled spans in the canonical vocabulary -- sentinels removed.

    Per published section, NOT per merged run: merging is a statement about
    section identity, and a merged run's span can cover a dropped sentinel whose
    time must not be re-attributed (see ``canonical_runs``).
    """
    return [
        (start, end, CANONICAL_MAP.get(label, label))
        for start, end, label in _clamped_spans(sections)
        if label not in CANONICAL_DROP
    ]


def raw_coverage(sections: list) -> list:
    """Every published section, labels untouched."""
    return _clamped_spans(sections)


def dropped_coverage(sections: list) -> list:
    """Only the sections the canonical mapping throws away (the ``end`` tail)."""
    return [span for span in _clamped_spans(sections) if span[2] in CANONICAL_DROP]


def labeled_bounds(sections: list) -> tuple:
    """``(first_start, last_end)`` of the canonically labeled region, or ``(None, None)``.

    Uses ``canonical_runs`` -- the corpus's own definition of what counts as a
    labeled stretch -- rather than the raw section list, so a track that both
    starts and ends on a sentinel reports the bounds of real sections.
    """
    runs = canonical_runs(list(sections))
    if not runs:
        return None, None
    first_start = float(runs[0][0])
    last_end = max(max(float(start), float(end)) for start, end, _label, _dur in runs)
    return first_start, last_end


def label_v1(label: str) -> str:
    """Canonical label -> the 5-class space the NN trains on."""
    return V1_MAP.get(label, label)


def song_time_intents(blocks: list, look_ahead_sec: float,
                      default_end: float | None = None) -> list:
    """Intent blocks (audience time) -> ``[(start, end, intent)]`` in song time.

    The engine runs one look-ahead ahead of what the audience hears, so an
    intent block stamped at audience time T was caused by the beats at song time
    T - look_ahead.  Subtracting puts both on the beat's clock.
    """
    spans = []
    for block in blocks:
        start = float(block["t"]) - look_ahead_sec
        raw_end = block.get("end", default_end)
        end = float("inf") if raw_end is None else float(raw_end) - look_ahead_sec
        spans.append((start, max(start, end), str(block["intent"])))
    return spans


def zscores(values: list) -> list:
    """Population z-scores; all-zero for a feature that never moves.

    A constant feature carries no information, and dividing by its zero spread
    would turn a well-defined "no variation" into NaN halfway down a CSV.
    """
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    spread = float(array.std())
    if spread <= 0.0:
        return [0.0] * len(values)
    return ((array - array.mean()) / spread).tolist()


# --------------------------------------------------------------------------- #
# Join
# --------------------------------------------------------------------------- #


class JoinStats(NamedTuple):
    """Where every beat of one track went.

    ``beats_kept + dropped_leading + dropped_gap + dropped_trailing`` always
    equals ``beats_total``; ``dropped_in_dropped_section`` is a diagnostic that
    overlaps the others (it names *why* those beats are unlabeled).
    """

    beats_total: int
    beats_kept: int
    dropped_leading: int
    dropped_gap: int
    dropped_trailing: int
    dropped_in_dropped_section: int
    beats_without_intent: int


def join_track(track_id: str, youtube_id_: str, report: dict, sections: list) -> tuple:
    """One track's sim report + annotation -> ``(rows, JoinStats)``.

    Rows are dicts keyed by ``TABLE_HEADER`` in beat order.  Pure: no I/O, no
    audio, no clock.
    """
    beats = sorted(report.get("beats", []), key=lambda record: float(record["t"]))
    coverage = Timeline(canonical_coverage(sections))
    raw = Timeline(raw_coverage(sections))
    sentinels = Timeline(dropped_coverage(sections))
    first_start, last_end = labeled_bounds(sections)

    look_ahead_sec = float(report.get("metrics", {}).get("look_ahead_sec", 0.0))
    intents = Timeline(song_time_intents(
        report.get("intents", []), look_ahead_sec, report.get("duration_sec")
    ))

    rows: list = []
    leading = gap = trailing = in_dropped = without_intent = 0

    for record in beats:
        t = float(record["t"])
        label = coverage.at(t)
        if label is None:
            if first_start is not None and t < first_start:
                leading += 1
            elif last_end is not None and t >= last_end:
                trailing += 1
            else:
                gap += 1
            if sentinels.at(t) is not None:
                in_dropped += 1
            continue

        intent = intents.at(t)
        if intent is None:
            intent = NO_INTENT
            without_intent += 1

        rms = float(record.get("rms", 0.0))
        rows.append({
            "track_id": track_id,
            "youtube_id": youtube_id_,
            "t_song": t,
            "bpm": float(record.get("bpm", 0.0)),
            "onset_density": float(record.get("onset_density", 0.0)),
            "kick_strength": float(record.get("kick_strength", 0.0)),
            "kick_known": 1 if rms >= KICK_MIN_RMS else 0,
            "centroid_trend": float(record.get("centroid_trend", 0.0)),
            "sub_bass_ratio": float(record.get("sub_bass_ratio", 0.0)),
            "rms": rms,
            "intent_at_beat": intent,
            "label_canonical": label,
            "label_raw": raw.at(t) or "",
            "label_v1": label_v1(label),
            "bar_position_unknown": BAR_POSITION_UNKNOWN,
        })

    _add_zscores(rows)
    stats = JoinStats(
        beats_total=len(beats),
        beats_kept=len(rows),
        dropped_leading=leading,
        dropped_gap=gap,
        dropped_trailing=trailing,
        dropped_in_dropped_section=in_dropped,
        beats_without_intent=without_intent,
    )
    return rows, stats


def _add_zscores(rows: list) -> None:
    """Attach ``<feature>_z`` to every row, standardised over this track only."""
    for column in CONTINUOUS_COLUMNS:
        for row, value in zip(rows, zscores([row[column] for row in rows])):
            row[f"{column}_z"] = value


def _field(value) -> str:
    """One CSV cell.  Fixed-width floats so the file is byte-stable."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int)):
        return str(int(value))
    return f"{float(value):.6f}"


def format_row(row: dict) -> list:
    """Row dict -> string fields in ``TABLE_HEADER`` order."""
    return [_field(row[column]) for column in TABLE_HEADER]


# --------------------------------------------------------------------------- #
# Mel feature sidecar
# --------------------------------------------------------------------------- #


class MelEnergyStream:
    """The pipeline's per-buffer mel filterbank, rebuilt outside the pipeline.

    ``lib/`` is read-only in this plan, so the exporter cannot borrow a live
    ``MusicAnalyser``; it constructs the same aubio objects with the same
    parameters as ``MusicAnalyser._reset_state`` instead.  The duplication is
    pinned by a parity test that feeds both sides the same buffers and demands
    bit-identical energies -- without it, the model could silently train on
    features the runtime never produces.

    Stateful: aubio's phase vocoder keeps an overlap window, so buffers must be
    fed in order from the start of the track, exactly as the pipeline does.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, buffer_size: int = BUFFER_SIZE):
        import aubio  # local: keeps the label-join path free of the DSP import

        self.sample_rate = sample_rate
        self.win_s = buffer_size * 4
        self.hop_s = buffer_size
        self.mel_bands = MEL_BANDS
        self._pvoc = aubio.pvoc(self.win_s, self.hop_s)
        self._filterbank = aubio.filterbank(self.mel_bands, self.win_s)
        self._filterbank.set_mel_coeffs_slaney(sample_rate)

    def process(self, buffer: np.ndarray) -> np.ndarray:
        """Mel band energies for one buffer (same call chain as the analyser)."""
        return self._filterbank(self._pvoc(buffer))


def pooled_log_mel(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                   buffer_size: int = BUFFER_SIZE,
                   pool: int = POOL_BUFFERS) -> tuple:
    """Decoded track -> ``(mel[n_frames, 40] float32, frame_sec, t0)``.

    ``log1p`` compresses the energies (the spec's input transform) and pooling
    ``pool`` consecutive buffers takes the frame rate from ~5.8 ms to ~46 ms,
    which is the rate the CRNN reads.

    Time base: the simulation advances its clock *before* analysing a buffer, so
    an event in buffer ``i`` is stamped at ``(i+1) * buffer_sec``.  A pooled
    frame therefore carries the song time of the END of its last buffer, putting
    frame ``k`` at ``t0 + k * frame_sec`` with ``t0 == frame_sec`` -- mel frames
    and beat rows land on one time base with no correction factor.

    The trailing partial frame is dropped (it would be pooled over fewer buffers
    and weighted unlike every other frame); the trailing partial *buffer* is
    zero-padded, exactly as ``FileAudioClient`` pads it for the simulation.
    """
    stream = MelEnergyStream(sample_rate, buffer_size)
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_buffers = -(-len(samples) // buffer_size)  # ceil: the last one is padded
    n_frames = n_buffers // pool

    mel = np.zeros((n_frames, MEL_BANDS), dtype=np.float32)
    accumulator = np.zeros(MEL_BANDS, dtype=np.float64)
    frame = 0
    for index in range(n_frames * pool):
        start = index * buffer_size
        chunk = samples[start:start + buffer_size]
        if len(chunk) < buffer_size:
            padded = np.zeros(buffer_size, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
        # maximum(): a mel filterbank over a magnitude spectrum is non-negative,
        # but a negative would become NaN under log1p and poison training in
        # silence rather than failing loudly.
        accumulator += np.log1p(np.maximum(stream.process(chunk), 0.0))
        if (index + 1) % pool == 0:
            mel[frame] = accumulator / pool
            accumulator[:] = 0.0
            frame += 1

    frame_sec = pool * buffer_size / sample_rate
    return mel, frame_sec, frame_sec


def write_feature_sidecar(path, mel: np.ndarray, frame_sec: float, t0: float) -> None:
    """Write one ``<youtube_id>.npz`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "wb") as handle:
            np.savez_compressed(
                handle,
                mel=mel.astype(np.float32, copy=False),
                frame_sec=np.float64(frame_sec),
                t0=np.float64(t0),
                sample_rate=np.int32(SAMPLE_RATE),
                pool_buffers=np.int32(POOL_BUFFERS),
            )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- #
# Stage A: simulate one track (pool worker)
# --------------------------------------------------------------------------- #


class SimJob(NamedTuple):
    """One track to simulate.  Picklable: crosses the process pool."""

    track_id: str
    youtube_id: str
    mp3_path: str
    report_path: str
    sidecar_path: str
    keep_cache: bool     # the cache predates this batch -- leave it behind
    pipeline_sha: str    # stamped into the cached report
    mp3_size: int
    mp3_mtime: float


class SimResult(NamedTuple):
    track_id: str
    ok: bool
    detail: str
    beats: int
    frames: int
    sidecar_bytes: int
    wall_sec: float


def decode_cache_path(mp3_path: str) -> str:
    """Where ``FileAudioClient`` parks the decoded samples."""
    return f"{mp3_path}.{SAMPLE_RATE}.npy"


def _write_json_gz(path: Path, payload: dict) -> None:
    """Write a gzipped JSON document atomically and reproducibly.

    ``mtime=0`` keeps the gzip header out of the content: re-simulating an
    unchanged track then yields a byte-identical file, so a diff over the report
    cache shows pipeline changes and nothing else.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6,
                               filename="", mtime=0) as compressed:
                compressed.write(
                    json.dumps(payload, separators=(",", ":"), sort_keys=True)
                    .encode("utf-8")
                )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def report_path(data_dir: Path, youtube_id: str) -> Path:
    """Where one track's cached simulation report lives."""
    return data_dir / REPORTS_DIR / f"{youtube_id}.json.gz"


def report_envelope(job: SimJob, report: dict) -> dict:
    """The cached report plus everything needed to decide it is still valid."""
    return {
        "cache_version": CACHE_VERSION,
        "track_id": job.track_id,
        "youtube_id": job.youtube_id,
        "pipeline_sha": job.pipeline_sha,
        "mp3_size": job.mp3_size,
        "mp3_mtime": job.mp3_mtime,
        "report": report,
    }


def cache_is_fresh(envelope: dict, pipeline_sha_: str,
                   mp3_size: int, mp3_mtime: float) -> bool:
    """Can this cached report stand in for a re-simulation?

    Three things can invalidate it and nothing else does: a different pipeline
    (the report IS the pipeline's output), a different audio file, or a cache
    written by an older layout of this envelope.  Size *and* mtime are compared
    because either alone misses a plausible change -- a re-encode at the same
    length, or a restored file with an old timestamp.
    """
    if not isinstance(envelope, dict):
        return False
    return (
        envelope.get("cache_version") == CACHE_VERSION
        and envelope.get("pipeline_sha") == pipeline_sha_
        and envelope.get("mp3_size") == mp3_size
        and envelope.get("mp3_mtime") == mp3_mtime
        and isinstance(envelope.get("report"), dict)
    )


def simulate_track(job: SimJob) -> SimResult:
    """Run one track through the fast sim, export its sidecar, drop its cache.

    Never raises: a bad track must not take the batch down with it.  The decode
    cache is removed in a ``finally`` so a failure cannot leak ~7.7x the mp3's
    size onto the disk.
    """
    import asyncio

    started = time.monotonic()
    cache_path = decode_cache_path(job.mp3_path)
    try:
        from simulate.fake_audio_client import FileAudioClient
        from simulate.runner import run_fast_simulation

        client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, job.mp3_path)
        _client, event_buffer, command_queue = asyncio.run(run_fast_simulation(client))
        report = event_buffer.to_report(command_queue.get_timing_log())
        _write_json_gz(Path(job.report_path), report_envelope(job, report))

        # Same decoded samples the simulation just consumed -- parity by
        # construction, and the cache is still warm.
        audio = np.load(cache_path)
        mel, frame_sec, t0 = pooled_log_mel(audio, SAMPLE_RATE, BUFFER_SIZE)
        write_feature_sidecar(Path(job.sidecar_path), mel, frame_sec, t0)

        return SimResult(job.track_id, True, "", len(report.get("beats", [])),
                         len(mel), os.path.getsize(job.sidecar_path),
                         time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 -- one bad track, not a dead batch
        return SimResult(job.track_id, False, f"{type(exc).__name__}: {exc}"[:300],
                         0, 0, 0, time.monotonic() - started)
    finally:
        if not job.keep_cache:
            try:
                os.unlink(cache_path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_ok_rows(data_dir: Path) -> list:
    """The ``status == ok`` rows of ``clean_manifest.csv``, sorted by track_id."""
    path = data_dir / CLEAN_MANIFEST_FILE
    if not path.exists():
        raise RuntimeError(
            f"missing {path} -- run training/build_clean_manifest.py first"
        )
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == STATUS_OK]
    if not rows:
        raise RuntimeError(f"no ok rows in {path} -- nothing to build from")
    rows.sort(key=lambda row: row["track_id"])
    return rows


def load_sections_by_track(data_dir: Path) -> dict:
    """``track_id -> [(start, end, label)]`` from ``annotations/segments.json``."""
    return {str(track["key"]): parse_sections(track) for track in load_tracks(data_dir)}


def select_jobs(rows: list, data_dir: Path, force: bool = False,
                min_age_sec: float = MIN_AGE_SEC, now: float | None = None,
                preexisting_caches: set | None = None,
                sha: str | None = None) -> tuple:
    """``(jobs, counts)`` -- which tracks still need simulating, and why.

    A track is a CACHE HIT, and neither simulated nor decoded, when its cached
    report was produced by this pipeline from this exact audio file *and* its
    mel sidecar is on disk.  Everything else is a miss, and ``counts`` records
    which kind so a rebuild that unexpectedly re-runs the corpus says why.

    ``force`` misses everything.  A track whose mp3 was written in the last
    ``min_age_sec`` is left for a later run -- a downloader may still be writing
    it -- and is never a cache hit either, since its bytes are still moving.
    """
    now = time.time() if now is None else now
    preexisting_caches = preexisting_caches or set()
    sha = pipeline_sha() if sha is None else sha
    jobs: list = []
    counts: collections.Counter = collections.Counter()

    for row in rows:
        mp3 = Path(row["mp3_path"])
        cached = report_path(data_dir, row["youtube_id"])
        sidecar = data_dir / FEATURES_DIR / f"{row['youtube_id']}.npz"
        try:
            stat = mp3.stat()
        except OSError:
            counts["missing_audio"] += 1
            continue

        if force:
            reason = "miss_forced"
        else:
            reason = _cache_miss_reason(cached, sidecar, sha, stat.st_size, stat.st_mtime)
            if reason is None:
                counts["hit"] += 1
                continue

        # Counted as too_recent and NOT as a miss: the counters partition the
        # manifest, so sum(miss_*) is exactly the number of jobs dispatched.
        if not is_settled(mp3, now, min_age_sec):
            counts["too_recent"] += 1
            continue

        counts[reason] += 1
        jobs.append(SimJob(
            row["track_id"], row["youtube_id"], str(mp3),
            str(cached), str(sidecar),
            keep_cache=decode_cache_path(str(mp3)) in preexisting_caches,
            pipeline_sha=sha, mp3_size=stat.st_size, mp3_mtime=stat.st_mtime,
        ))
    jobs.sort(key=lambda job: job.track_id)
    return jobs, counts


def _cache_miss_reason(cached: Path, sidecar: Path, sha: str,
                       mp3_size: int, mp3_mtime: float) -> str | None:
    """``None`` when the cache may be used, else the counter name for the miss."""
    if not cached.exists():
        return "miss_new"
    if not sidecar.exists():
        return "miss_no_sidecar"
    try:
        envelope = _read_json_gz(cached)
    except (OSError, ValueError, EOFError):
        return "miss_unreadable"
    if cache_is_fresh(envelope, sha, mp3_size, mp3_mtime):
        return None
    if envelope.get("pipeline_sha") != sha:
        return "miss_pipeline_changed"
    if (envelope.get("mp3_size") != mp3_size
            or envelope.get("mp3_mtime") != mp3_mtime):
        return "miss_audio_changed"
    return "miss_stale_format"


def find_caches(data_dir: Path) -> set:
    """Every decode cache currently sitting in ``audio/``."""
    audio_dir = data_dir / AUDIO_DIR
    if not audio_dir.exists():
        return set()
    return {str(path) for path in audio_dir.glob("*.npy")}


# --------------------------------------------------------------------------- #
# Stage A: batch
# --------------------------------------------------------------------------- #


def default_workers() -> int:
    """Leave two cores for the OS (and for a downloader that may still run)."""
    return max(1, (os.cpu_count() or 4) - 2)


def run_simulations(jobs: list, workers: int, progress_every: int = 10) -> list:
    """Simulate every job; results come back in job order."""
    if not jobs:
        return []
    results = []
    started = time.time()
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            results.append(simulate_track(job))
            _print_progress(index, len(jobs), started, progress_every)
        return results

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(pool.map(simulate_track, jobs, chunksize=1),
                                           start=1):
                results.append(result)
                _print_progress(index, len(jobs), started, progress_every)
    except concurrent.futures.process.BrokenProcessPool as exc:
        # A worker died outright (an OOM kill is the realistic cause on a large
        # corpus).  Everything already written to the report cache stands, so
        # report the damage and let the run finish -- the next invocation picks
        # up exactly where this one stopped.
        print(f"  WARNING: worker pool broke after {len(results)}/{len(jobs)} "
              f"track(s): {exc}.  Re-run to continue -- cached reports are kept.",
              flush=True)
    return results


def _print_progress(done: int, total: int, started: float, every: int) -> None:
    if not every or (done % every and done != total):
        return
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    print(f"  simulated {done}/{total}  {elapsed / 60:.1f} min elapsed  "
          f"{rate * 60:.1f} tracks/min  ~{remaining / 60:.1f} min left", flush=True)


# --------------------------------------------------------------------------- #
# Stage B: table
# --------------------------------------------------------------------------- #


class TableStats(NamedTuple):
    tracks: int
    rows: int
    canonical: collections.Counter
    v1: collections.Counter
    raw: collections.Counter
    intents: collections.Counter
    dropped: collections.Counter
    look_ahead_sec: set
    skipped: list


def build_table(data_dir: Path, rows: list, sections_by_track: dict) -> TableStats:
    """Join every track that has a report on disk and stream the table out.

    Rows are written in (track_id, beat time) order, so the file is identical
    whatever order Stage A happened to finish in.
    """
    path = data_dir / TABLE_FILE
    tmp = path.with_suffix(path.suffix + ".part")
    tracks = row_count = 0
    canonical = collections.Counter()
    v1 = collections.Counter()
    raw = collections.Counter()
    intents = collections.Counter()
    dropped = collections.Counter()
    look_ahead = set()
    skipped = []

    try:
        # mtime=0: the table is a build artefact that must diff cleanly against
        # the previous build, and a gzip header timestamp would make every
        # rebuild look like a change.
        with open(tmp, "wb") as raw_file, \
                gzip.GzipFile(fileobj=raw_file, mode="wb", compresslevel=6,
                              filename="", mtime=0) as compressed, \
                io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(TABLE_HEADER)
            for row in rows:
                track_id = row["track_id"]
                cached = report_path(data_dir, row["youtube_id"])
                if not cached.exists():
                    continue
                sections = sections_by_track.get(track_id)
                if sections is None:
                    skipped.append((track_id, "no annotation record"))
                    continue
                try:
                    report = _read_json_gz(cached)["report"]
                except (OSError, ValueError, EOFError, KeyError, TypeError) as exc:
                    skipped.append((track_id, f"unreadable report: {exc}"))
                    continue

                joined, stats = join_track(track_id, row["youtube_id"], report, sections)
                look_ahead.add(float(report.get("metrics", {}).get("look_ahead_sec", 0.0)))
                tracks += 1
                row_count += len(joined)
                dropped["beats_total"] += stats.beats_total
                dropped["kept"] += stats.beats_kept
                dropped["leading"] += stats.dropped_leading
                dropped["gap"] += stats.dropped_gap
                dropped["trailing"] += stats.dropped_trailing
                dropped["in_dropped_section"] += stats.dropped_in_dropped_section
                dropped["without_intent"] += stats.beats_without_intent
                for joined_row in joined:
                    canonical[joined_row["label_canonical"]] += 1
                    v1[joined_row["label_v1"]] += 1
                    raw[joined_row["label_raw"]] += 1
                    intents[joined_row["intent_at_beat"] or "(none)"] += 1
                    writer.writerow(format_row(joined_row))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return TableStats(tracks, row_count, canonical, v1, raw, intents, dropped,
                      look_ahead, skipped)


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    """HEAD of the repository, or ``unknown``."""
    return _git(repo_root, "rev-parse", "HEAD") or "unknown"


def pipeline_sha(repo_root: Path = REPO_ROOT) -> str:
    """Identity of the code whose output the cached reports are.

    Deliberately NOT repo HEAD: a report is invalidated by a change to the
    pipeline under evaluation (``lib/``, ``simulate/``), not by a commit to this
    script or to a document.  Keying on HEAD would throw away the whole corpus
    cache on every commit, which is exactly what the cache exists to prevent.

    Uncommitted changes under those paths append ``+dirty``, so an edit that has
    not been committed yet still invalidates the cache instead of silently
    reusing reports the current code would no longer produce.
    """
    sha = _git(repo_root, "log", "-1", "--format=%H", "--", "lib", "simulate")
    if not sha:
        return "unknown"
    dirty = _git(repo_root, "status", "--porcelain", "--", "lib", "simulate")
    return f"{sha}+dirty" if dirty else sha


def sidecar_stats(data_dir: Path) -> tuple:
    """``(count, total_bytes)`` of the mel sidecars on disk."""
    features = data_dir / FEATURES_DIR
    if not features.exists():
        return 0, 0
    paths = list(features.glob("*.npz"))
    return len(paths), sum(path.stat().st_size for path in paths)


def write_meta(data_dir: Path, stats: TableStats, failures: list,
               elapsed_sec: float, cache_counts: collections.Counter | None = None,
               sha: str | None = None) -> Path:
    """Everything needed to reproduce and audit the table, beside the table."""
    count, total_bytes = sidecar_stats(data_dir)
    meta = {
        "built_at": datetime.datetime.now(datetime.timezone.utc)
                            .replace(microsecond=0).isoformat(),
        "git_sha": git_sha(),
        "pipeline_sha": pipeline_sha() if sha is None else sha,
        "report_cache": dict(sorted((cache_counts or {}).items())),
        "build_wall_sec": round(elapsed_sec, 1),
        "schema": list(TABLE_HEADER),
        "tracks": stats.tracks,
        "rows": stats.rows,
        "look_ahead_sec": sorted(stats.look_ahead_sec),
        "class_histogram": {
            "canonical": _ordered_counts(stats.canonical, CANONICAL_ORDER),
            "v1": _ordered_counts(stats.v1, V1_ORDER),
            "raw": _ordered_counts(stats.raw, ()),
        },
        "intent_histogram": _ordered_counts(stats.intents, ()),
        "dropped_beats": dict(sorted(stats.dropped.items())),
        "features": {
            "dir": FEATURES_DIR,
            "tracks": count,
            "total_bytes": total_bytes,
            "mel_bands": MEL_BANDS,
            "pool_buffers": POOL_BUFFERS,
            "frame_sec": POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE,
        },
        "failed_tracks": [{"track_id": t, "detail": d} for t, d in failures],
        "skipped_tracks": [{"track_id": t, "detail": d} for t, d in stats.skipped],
    }
    path = data_dir / META_FILE
    _write_json_pretty(path, meta)
    return path


def _write_json_pretty(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _ordered_counts(counter: collections.Counter, order: tuple) -> dict:
    """Counts in musical order first, then anything unexpected by frequency."""
    known = [label for label in order if label in counter]
    extra = sorted(set(counter) - set(order), key=lambda label: (-counter[label], label))
    return {label: counter[label] for label in known + extra}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_report(stats: TableStats, results: list, table_path: Path,
                 meta_path: Path, data_dir: Path, elapsed: float,
                 new_caches: set, cache_counts: collections.Counter | None = None) -> None:
    failures = [result for result in results if not result.ok]
    count, total_bytes = sidecar_stats(data_dir)
    dropped = stats.dropped

    if cache_counts:
        print()
        print("report cache")
        print(f"  hits (no sim, no decode): {cache_counts['hit']}")
        misses = {key: value for key, value in sorted(cache_counts.items())
                  if key.startswith("miss_")}
        print(f"  misses                  : {sum(misses.values())}"
              + (f"  ({', '.join(f'{k[5:]} {v}' for k, v in misses.items())})"
                 if misses else ""))
        for key, caption in (("too_recent", "too recent to touch"),
                             ("missing_audio", "audio file missing")):
            if cache_counts[key]:
                print(f"  {caption:<24}: {cache_counts[key]}")

    print()
    print("training table")
    print(f"  tracks joined         : {stats.tracks}")
    print(f"  rows (labeled beats)  : {stats.rows}")
    print(f"  beats seen            : {dropped['beats_total']}")
    print(f"  dropped leading       : {dropped['leading']}  (before the first section)")
    print(f"  dropped gap           : {dropped['gap']}  (unlabeled interior)")
    print(f"  dropped trailing      : {dropped['trailing']}  (past the last section)")
    print(f"     of which in a dropped 'end' sentinel: {dropped['in_dropped_section']}")
    print(f"  rows without an intent: {dropped['without_intent']}")
    print(f"  look_ahead_sec        : {sorted(stats.look_ahead_sec)}")

    print()
    print("class histogram (canonical)")
    _print_histogram(stats.canonical, stats.rows)
    print()
    print("class histogram (v1)")
    _print_histogram(stats.v1, stats.rows)
    print()
    print("committed intent at beat")
    _print_histogram(stats.intents, stats.rows)

    print()
    print("mel feature sidecars")
    print(f"  tracks    : {count}")
    print(f"  total size: {total_bytes / 2**20:.1f} MiB")
    print(f"  frame     : {MEL_BANDS} bands, "
          f"{POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE * 1000:.2f} ms hop")

    if failures:
        print()
        print(f"failed tracks ({len(failures)}):")
        for result in failures[:40]:
            print(f"  {result.track_id:<20} {result.detail}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more (see {META_FILE})")
    if stats.skipped:
        print()
        print(f"skipped at join time ({len(stats.skipped)}):")
        for track_id, detail in stats.skipped[:40]:
            print(f"  {track_id:<20} {detail}")

    print()
    print(f"decode caches left in {data_dir / AUDIO_DIR}: "
          f"{len(new_caches)} new" + (f" -- {sorted(new_caches)}" if new_caches else ""))
    print(f"wall time: {elapsed / 60:.1f} min")
    print()
    print(f"table: {table_path}  ({table_path.stat().st_size / 2**20:.1f} MiB)")
    print(f"meta : {meta_path}")


def _print_histogram(counter: collections.Counter, total: int) -> None:
    for label, count in counter.most_common():
        share = 100.0 * count / total if total else 0.0
        print(f"  {label or '(none)':<14}{count:>10}{share:>8.1f}%")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_data_dir() -> Path:
    return REPO_ROOT / "training" / "data" / "raveform"


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir(),
        help="corpus root; reads clean_manifest.csv + annotations/, writes "
             "training_table.csv.gz, reports/ and features/ (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=default_workers(),
        help="parallel simulation workers (default: %(default)s = cpu_count - 2)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="simulate at most N tracks (smoke run; 0 = no limit)",
    )
    parser.add_argument(
        "--min-age-sec", type=float, default=MIN_AGE_SEC,
        help="skip audio written more recently than this -- the corpus "
             "downloader may still be running (default: %(default)s)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="ignore the report cache and re-simulate every track",
    )
    parser.add_argument(
        "--table-only", action="store_true",
        help="skip the simulations and re-join the cached reports on disk",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    started = time.time()
    sha = pipeline_sha()
    print("label-aligned training table")
    print(f"data dir : {data_dir}")
    print(f"repo     : {git_sha()}")
    print(f"pipeline : {sha}  (lib/ + simulate/ -- the report cache key)")

    rows = load_ok_rows(data_dir)
    sections_by_track = load_sections_by_track(data_dir)
    print(f"clean manifest: {len(rows)} ok track(s); "
          f"{len(sections_by_track)} annotated track(s)")

    preexisting_caches = find_caches(data_dir)
    if preexisting_caches:
        print(f"NOTE: {len(preexisting_caches)} decode cache(s) predate this run "
              f"and will be left in place")

    results: list = []
    cache_counts: collections.Counter = collections.Counter()
    if args.table_only:
        print("NOTE: --table-only -- no simulations, re-joining cached reports")
    else:
        jobs, cache_counts = select_jobs(
            rows, data_dir, force=args.force, min_age_sec=args.min_age_sec,
            preexisting_caches=preexisting_caches, sha=sha,
        )
        if args.limit:
            jobs = jobs[: args.limit]
            print(f"NOTE: --limit {args.limit} -- simulating a subset only")
        print(f"stage A: {len(jobs)} to simulate, {cache_counts['hit']} cache hit(s)",
              flush=True)
        if jobs:
            print(f"  {args.workers} worker(s)", flush=True)
            results = run_simulations(jobs, workers=args.workers)

    print("stage B: joining beats to labels ...", flush=True)
    stats = build_table(data_dir, rows, sections_by_track)
    elapsed = time.time() - started
    failures = [(result.track_id, result.detail) for result in results if not result.ok]
    meta_path = write_meta(data_dir, stats, failures, elapsed, cache_counts, sha)

    new_caches = find_caches(data_dir) - preexisting_caches
    print_report(stats, results, data_dir / TABLE_FILE, meta_path, data_dir,
                 elapsed, new_caches, cache_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
