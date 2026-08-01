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
sidecar, a sidecar from an older mel exporter, unreadable cache -- is a miss and
is re-simulated; ``--force`` misses everything.  Hit and miss counts (with
reasons) are printed and recorded in the meta file, so a run that unexpectedly
re-simulates the corpus says why.

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

Beat timestamps are song-position seconds.  An intent block is stamped in
AUDIENCE time and carries ``song_t``, the instant of audio it describes, so the
join reads that and infers nothing.  It has to: under the NN engine a block's
delay is the playback delay minus that decision's own measured age, so it
differs per command, and no constant de-shift and no beat-matching rule can
recover song time from the stamp alone -- a bar line is not a beat the report
carries.  Mel frames carry the same stamp convention as beats.

Blocks WITHOUT ``song_t`` -- every report cut before the engine recorded it, of
which the corpus holds thousands -- keep the older inference, and it is what the
rest of this section describes.  Every intent commit rides the delayed command
queue, so every beat-driven block is audience time and shifts back cleanly.  The exception is
beat-absence ATMOSPHERIC: it rides the queue too, but a timer fired it rather
than a beat, so ``t - look_ahead`` lands nowhere near one and the detection can
only read it as song-stamped -- leaving it one look-ahead late.  Measured on the
eval set, that block falls past the last label and scores nothing; a mid-song
silence would misplace it.  Reports cut before the engine unified its commit
paths do carry genuinely song-stamped blocks, and this reading is still right
for them.  ``realign_intents`` records the counts in ``meta.json`` -- a growing
``song_stamped`` means a commit path moved again, and a report mixing recorded
and inferred blocks means it was cut across the change.

Decode-cache discipline
-----------------------

The simulation leaves two derived files beside the audio: the decoded samples
(``<mp3>.<samplerate>.npy``, ~7.7x the mp3, ~95 GiB over the corpus) and D12's
extractor cell sidecar (``<mp3>.<decoder>.mertcells.npz``, ~25 MB a track,
~35 GiB over the corpus).  Each worker deletes both as soon as the features are
out, in a ``finally`` -- and neither buys this batch anything, because it
simulates a track at most once.  Files that already existed before the batch
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
import hashlib
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
# training/raveform/ holds the corpus-acquisition scripts (gate, manifest,
# annotations); they are scripts rather than a package, so their directory has
# to be on the path to import them.
for _path in (
    str(REPO_ROOT),
    str(REPO_ROOT / "training"),
    str(REPO_ROOT / "training" / "raveform"),
):
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

# Re-exported: `corpus_root` is the stdlib-only module that owns where the
# corpus lives, so a show can ask without importing this file.
from corpus_root import default_data_dir  # noqa: E402,F401

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
# in loudness by more than sections do within one mix, so a fitter reading
# absolute values learns the mastering as much as the music; the z-scored copy is
# the mix-invariant view.  Both are kept -- absolute RMS is what the silence gate
# reads, and the NN's own features are the mel sidecar, not these columns.
#
# The rule engine's four features (onset density, kick strength, centroid trend,
# sub-bass ratio) were columns here until the demolition deleted the chains that
# produced them.  A report carries none of those keys now, so emitting them would
# write 0.0 on every beat of every track with nothing to say so.
CONTINUOUS_COLUMNS = (
    "bpm",
    "rms",
)

TABLE_HEADER = (
    "track_id",
    "youtube_id",
    "t_song",                # song-position seconds of the beat
    "bpm",
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

# The geometry every sidecar on disk was written on.  It used to be a coupling
# to the analyser's aubio filterbank; that bank is deleted and no new sidecar can
# be produced, so these two are now a record of the corpus's mel grid and the
# thing `load_sidecar` refuses a mismatch against.
MEL_BANDS = 40
POOL_BUFFERS = 8        # 8 x 256 samples @ 44.1 kHz ~= 46 ms per frame

# Which exporter wrote a sidecar, recorded IN the sidecar.  Bump it whenever the
# exporter produces different numbers for input it already handled -- a different
# compression than log1p, a different pooling reduction, a filterbank change.
# Geometry cannot stand in for this: all three of those changes leave
# the frame rate and the band count exactly where they were, so a corpus rebuilt
# on top of the old sidecars would train one model on two feature generations
# and say nothing.  The stamp lives here and NOT in the cached report because
# the report's bytes are what the eval-set baseline checksums, and a provenance
# field has no business moving a benchmark number.
MEL_EXPORTER_VERSION = 1
MEL_EXPORTER_KEY = "exporter_version"

# `MusicAnalyser` throws its rolling state away every 15 minutes (`lib/main.py`)
# to stop the windows growing without bound.  The simulation runs that code
# unmodified, so a track that reaches the horizon has its beat stream restart
# mid-song while the mel exporter -- which had no such reset -- kept going.  The
# two then describe the same audio from different states, and the training table
# joins them anyway: wrong rows, no error, no counter.  The corpus tops out at
# 899.889 s, i.e. 0.11 s of margin, so this is a live edge and not a hypothetical
# one.  Tracks at or past it are dropped from the build with a line saying why.
ANALYSER_RESET_SEC = 900.0


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
    """Intent blocks -> ``[(start, end, intent)]``, shifting EVERY block back.

    The naive transform: correct only for blocks the engine committed through
    the delayed command queue.  ``realign_intents`` is what the join actually
    uses; this stays as the reference the correction is measured against.
    """
    spans = []
    for block in blocks:
        start = float(block["t"]) - look_ahead_sec
        raw_end = block.get("end", default_end)
        end = float("inf") if raw_end is None else float(raw_end) - look_ahead_sec
        spans.append((start, max(start, end), str(block["intent"])))
    return spans


# A queue commit fires on the first main-loop iteration after it comes due, so a
# queue-stamped block lands within one buffer quantum of (beat + look_ahead).
# 1.5 quanta gives float-accumulation margin while staying two orders of
# magnitude below the ~0.5 s beat spacing that separates the two hypotheses.
_QUEUE_STAMP_TOLERANCE_SEC = 1.5 * BUFFER_SIZE / SAMPLE_RATE
_STAMP_EPS = 1e-9


class IntentAlignment(NamedTuple):
    """How each intent block's timestamp was interpreted."""

    blocks: int
    song_stamped: int      # committed immediately -- already in song time
    clamped_tail: int      # queue commit frozen at the report's end by mark_end
    song_recorded: int = 0  # the block carried the instant it describes
    late: int = 0          # fired more than a look-ahead after the audio it names


def _nearest_gap(sorted_times: list, value: float) -> float:
    """Distance from ``value`` to the closest entry of a sorted list."""
    if not sorted_times:
        return float("inf")
    index = bisect.bisect_left(sorted_times, value)
    best = float("inf")
    for candidate in (index - 1, index):
        if 0 <= candidate < len(sorted_times):
            best = min(best, abs(sorted_times[candidate] - value))
    return best


def realign_intents(blocks: list, look_ahead_sec: float, beat_times: list,
                    duration_sec: float | None = None,
                    tolerance: float = _QUEUE_STAMP_TOLERANCE_SEC) -> tuple:
    """Intent blocks -> song-time spans, respecting BOTH of the engine's clocks.

    **A block that records ``song_t`` needs no inference at all**, and every
    block the NN engine commits does.  Its delay is a per-command quantity now
    (the playback delay minus the decision's measured age, B1), so no constant
    de-shift reaches song time and no beat-matching rule can recover it either:
    a bar line is not a beat the report carries, and the residual wobbles by up
    to a whole feature hop.  The engine knows the instant exactly at commit time
    and says so; this reads it.

    The inference below is what reports cut before that stamping need, and it is
    kept rather than retired because the corpus holds thousands of them.  There,
    the engine commits an intent two different ways and they land in different
    time bases:

    * **Queue commits** are enqueued at a beat and fire one look-ahead later, so
      the block is stamped in AUDIENCE time and must be shifted back.  Every
      beat-driven commit is one of these, the first beat of a run and the beat
      that re-enters after a sound stop included.
    * **Beat-absence ATMOSPHERIC** rides the queue as well, but no beat caused
      it, so nothing explains its stamp and it reads as song time below --
      staying one look-ahead late.  Reports cut before the engine unified its
      commit paths carry genuinely song-stamped blocks, which this same reading
      handles correctly: shifting one of those back would steal up to a
      look-ahead of beats from the intent that preceded it.

    A block is treated as a queue commit when ``t - look_ahead`` lands within
    ``tolerance`` of an actual beat, i.e. when the queue hypothesis *explains*
    it; otherwise it can only have been stamped in song time.  Preferring the
    queue reading is the conservative direction: beat timestamps and block
    timestamps are drawn from the same virtual-clock tick ladder, so ~1.2% of
    ordinary queue commits coincidentally fall on a beat instant, and a rule
    that keyed on that coincidence would mis-shift hundreds of blocks.  The one
    exception is the final block, which ``mark_end`` freezes at the report's
    duration: it is a queue commit whose stamp was clamped, so it is recognised
    explicitly rather than by matching a beat.

    Block boundaries are single instants -- ``set_intent`` closes the previous
    block and opens the next one with the same reading -- so a block's end is
    taken from the following block's corrected start, never de-shifted twice.

    Returns ``(spans, IntentAlignment)``.
    """
    if not blocks:
        return [], IntentAlignment(0, 0, 0, 0)

    starts: list = []
    shifts: list = []
    song_stamped = clamped_tail = song_recorded = 0
    for block in blocks:
        t = float(block["t"])
        # Counted for every block, not only the inferred ones: once the engine
        # started recording `song_t` the inferred branch stopped running, and
        # this tripwire read zero on every report -- which looks exactly like
        # "no block was clamped".
        is_clamped = (duration_sec is not None
                      and abs(t - float(duration_sec)) <= _STAMP_EPS)
        if block.get("song_t") is not None:
            starts.append(float(block["song_t"]))
            song_recorded += 1
            clamped_tail += 1 if is_clamped else 0
            shifts.append(t - starts[-1])
            continue
        explained_by_queue = (
            look_ahead_sec <= 0
            or not beat_times
            or is_clamped
            or _nearest_gap(beat_times, t - look_ahead_sec) <= tolerance
        )
        if explained_by_queue:
            starts.append(t - look_ahead_sec)
            clamped_tail += 1 if is_clamped else 0
        else:
            starts.append(t)
            song_stamped += 1
        shifts.append(t - starts[-1])

    # The first block cannot begin before the first beat: the run's opening
    # commit happens AT that beat.  A no-op whenever the reading above was
    # right, and a floor under it when it was not -- but never applied to a
    # recorded instant, which is not an inference to be corrected.
    if beat_times and not blocks[0].get("song_t"):
        starts[0] = max(starts[0], beat_times[0])

    spans = []
    for index, block in enumerate(blocks):
        if index + 1 < len(blocks):
            end = starts[index + 1]
        else:
            raw_end = block.get("end", duration_sec)
            end = (float("inf") if raw_end is None
                   else float(raw_end) - shifts[index])
        spans.append((starts[index], max(starts[index], end), str(block["intent"])))
    # #154's accepted lateness, counted rather than inferred from a log: a block
    # whose stamp is more than one look-ahead after the audio it names was
    # committed late because the chain was older than the playback delay.  The
    # engine warns once per transition; this is how many blocks it cost.
    #
    # Only a block that RECORDED its instant can be measured this way -- an
    # inferred start is derived from the look-ahead, so its shift is the
    # look-ahead by construction and can never exceed it.  `song_recorded` is
    # therefore the denominator, and it ships beside the count: on the
    # thousands of pre-`song_t` corpus reports "0 late" is structural, and
    # without the denominator it is indistinguishable from a chain that never
    # ran behind.
    late = sum(1 for shift in shifts if shift > look_ahead_sec + tolerance)
    return spans, IntentAlignment(len(blocks), song_stamped, clamped_tail,
                                  song_recorded, late)


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
    intent_blocks_song_stamped: int   # blocks the engine committed immediately
    intent_blocks_song_recorded: int  # blocks carrying the instant they describe
    intent_blocks_late: int           # of those, committed past the budget
    intent_reattributed: int          # rows whose intent the realignment moved


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
    blocks = report.get("intents", [])
    duration_sec = report.get("duration_sec")
    beat_times = [float(record["t"]) for record in beats]
    spans, alignment = realign_intents(blocks, look_ahead_sec, beat_times, duration_sec)
    intents = Timeline(spans)
    # The uniform de-shift, kept only to count what the realignment changed.
    naive_intents = Timeline(song_time_intents(blocks, look_ahead_sec, duration_sec))

    rows: list = []
    leading = gap = trailing = in_dropped = without_intent = reattributed = 0

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
        if (naive_intents.at(t) or NO_INTENT) != (intent or NO_INTENT):
            reattributed += 1
        if intent is None:
            intent = NO_INTENT
            without_intent += 1

        rows.append({
            "track_id": track_id,
            "youtube_id": youtube_id_,
            "t_song": t,
            "bpm": float(record.get("bpm", 0.0)),
            "rms": float(record.get("rms", 0.0)),
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
        intent_blocks_song_stamped=alignment.song_stamped,
        intent_blocks_song_recorded=alignment.song_recorded,
        intent_blocks_late=alignment.late,
        intent_reattributed=reattributed,
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
                **{MEL_EXPORTER_KEY: np.int32(MEL_EXPORTER_VERSION)},
            )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def sidecar_generation(path: Path) -> int:
    """Which mel exporter wrote ``path``.

    Sidecars written before this stamp existed carry no such key.  They are
    reported as generation 1 -- the generation that in fact wrote them -- rather
    than as unknown, so the check grandfathers the corpus instead of ordering a
    1,387-track re-simulation to learn something already known.  A sidecar this
    cannot open reads the same way, deliberately: freshness is not the place to
    diagnose a corrupt file (the dataset builder fails loudly on one), and
    widening this check to catch it would have changed a behaviour nobody asked
    to change.
    """
    try:
        with np.load(path) as archive:
            if MEL_EXPORTER_KEY in archive.files:
                return int(archive[MEL_EXPORTER_KEY])
    except (OSError, ValueError, EOFError, KeyError):
        pass
    return 1


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


def derived_cache_paths(mp3_path: str) -> tuple:
    """Everything one simulation leaves beside the audio.

    Two files now, not one: D12's extractor cell sidecar is ~25 MB a track, so
    over this corpus it is ~35 GiB -- an order of magnitude past the decode
    cache this cleanup was built for, and the batch re-simulates a track at most
    once anyway, so a cache of the pass it just ran buys nothing.
    """
    from simulate.cell_cache import sidecar_path
    from simulate.fake_audio_client import FileAudioClient

    return (decode_cache_path(mp3_path),
            str(sidecar_path(mp3_path, FileAudioClient.decode_path)))


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
    """Run one track through the fast sim and drop its decode cache.

    Never raises: a bad track must not take the batch down with it.  The decode
    cache is removed in a ``finally`` so a failure cannot leak ~7.7x the mp3's
    size onto the disk.
    """
    import asyncio

    started = time.monotonic()
    try:
        from simulate.fake_audio_client import FileAudioClient
        from simulate.runner import run_fast_simulation

        client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, job.mp3_path)
        _client, event_buffer, command_queue = asyncio.run(run_fast_simulation(client))
        report = event_buffer.to_report(command_queue.get_timing_log())
        _write_json_gz(Path(job.report_path), report_envelope(job, report))

        # The mel exporter went with the analyser's filterbank, so a track
        # simulated from here on gets a report and no features.  Existing
        # sidecars are still read; new ones cannot be produced, which is the
        # one-way door the integration plan discloses.
        return SimResult(job.track_id, True, "", len(report.get("beats", [])),
                         0, 0, time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 -- one bad track, not a dead batch
        return SimResult(job.track_id, False, f"{type(exc).__name__}: {exc}"[:300],
                         0, 0, 0, time.monotonic() - started)
    finally:
        if not job.keep_cache:
            for path in derived_cache_paths(job.mp3_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_ok_rows(data_dir: Path) -> list:
    """The ``status == ok``, short-enough rows of ``clean_manifest.csv``.

    Sorted by track_id, and stopping short of the analyser's self-reset -- see
    ``ANALYSER_RESET_SEC``.
    """
    path = data_dir / CLEAN_MANIFEST_FILE
    if not path.exists():
        raise RuntimeError(
            f"missing {path} -- run training/raveform/build_clean_manifest.py first"
        )
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == STATUS_OK]
    rows, rejected = _reject_past_the_analyser_reset(rows)
    for track_id, seconds in rejected:
        print(f"  SKIP {track_id}: {seconds / 60.0:.1f} min reaches MusicAnalyser's "
              f"{ANALYSER_RESET_SEC / 60.0:.0f}-minute self-reset -- its beats and "
              f"its mel sidecar would no longer describe the same audio", flush=True)
    if not rows:
        raise RuntimeError(f"no ok rows in {path} -- nothing to build from")
    rows.sort(key=lambda row: row["track_id"])
    return rows


def _reject_past_the_analyser_reset(rows: list) -> tuple:
    """``(kept, [(track_id, seconds)])`` -- split on ``ANALYSER_RESET_SEC``.

    A blank or unparseable duration is kept: this gate exists to catch one
    specific, measurable condition, and it is not the cleanliness gate.
    """
    kept, rejected = [], []
    for row in rows:
        try:
            seconds = float(row.get("decoded_duration_sec") or 0.0)
        except ValueError:
            seconds = 0.0
        if seconds >= ANALYSER_RESET_SEC:
            rejected.append((row["track_id"], seconds))
        else:
            kept.append(row)
    return kept, rejected


def load_sections_by_track(data_dir: Path) -> dict:
    """``track_id -> [(start, end, label)]`` from ``annotations/segments.json``."""
    return {str(track["key"]): parse_sections(track) for track in load_tracks(data_dir)}


def select_jobs(rows: list, data_dir: Path, force: bool = False,
                min_age_sec: float = MIN_AGE_SEC,
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
    now = time.time()
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
    if sidecar_generation(sidecar) != MEL_EXPORTER_VERSION:
        return "miss_sidecar_generation"
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
    return {str(path) for path in audio_dir.glob("*.npy")} | {
        str(path) for path in audio_dir.glob("*.mertcells.npz")}


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
    missing_reports: list     # ok in the manifest, never simulated
    missing_sidecars: list    # simulated, but no mel features to train on


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
    skipped: list = []
    missing_reports: list = []
    missing_sidecars: list = []

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
                    # Never simulated (a partial batch, or a pool that broke).
                    # Recorded, because meta.json is the audit record: a track
                    # that is silently absent looks the same as one that passed.
                    missing_reports.append(track_id)
                    continue
                if not (data_dir / FEATURES_DIR / f"{row['youtube_id']}.npz").exists():
                    # Skip rather than assert: a half-built corpus must still
                    # produce a usable table.  But emitting rows whose track has
                    # no mel features would hand the NN dataset builder inputs
                    # it cannot featurise, so the track is left out entirely.
                    missing_sidecars.append(track_id)
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
                dropped["intent_blocks_song_stamped"] += stats.intent_blocks_song_stamped
                dropped["intent_blocks_song_recorded"] += \
                    stats.intent_blocks_song_recorded
                dropped["intent_blocks_late"] += stats.intent_blocks_late
                dropped["intent_reattributed"] += stats.intent_reattributed
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
                      look_ahead, skipped, missing_reports, missing_sidecars)


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


# The pipeline under evaluation, as a git pathspec: the Python sources of
# lib/ and simulate/ and nothing else.  Narrower than "those two directories"
# on purpose -- a CLAUDE.md living beside the code cannot change what the
# simulation produces, and treating a doc edit as a pipeline change throws away
# a 10-minute corpus cache for nothing (observed, hence the pathspec).
_PIPELINE_PATHSPEC = (":(glob)lib/**/*.py", ":(glob)simulate/**/*.py")


def pipeline_sha(repo_root: Path = REPO_ROOT) -> str:
    """Identity of the code whose output the cached reports are.

    Deliberately NOT repo HEAD: a report is invalidated by a change to the
    pipeline under evaluation, not by a commit to this script or to a document.
    Keying on HEAD would throw away the whole corpus cache on every commit,
    which is exactly what the cache exists to prevent.

    Uncommitted changes to those sources append ``+dirty.<digest>``, so an edit
    that has not been committed yet still invalidates the cache instead of
    silently reusing reports the current code would no longer produce -- and so
    do TWO different uncommitted edits against each other.  A constant ``+dirty``
    suffix gave every working-tree state the same cache key, which is the one
    state a developer changes the pipeline in most often.

    The digest is over ``git status`` (which names untracked files a diff cannot
    show) plus ``git diff HEAD`` (which carries the content of staged and
    unstaged edits alike).  A CLEAN tree still returns the bare commit sha, so
    every report already cached against a committed pipeline stays valid.
    """
    sha = _git(repo_root, "log", "-1", "--format=%H", "--", *_PIPELINE_PATHSPEC)
    if not sha:
        return "unknown"
    dirty = _git(repo_root, "status", "--porcelain", "--", *_PIPELINE_PATHSPEC)
    if not dirty:
        return sha
    diff = _git(repo_root, "diff", "HEAD", "--", *_PIPELINE_PATHSPEC) or ""
    digest = hashlib.sha256(f"{dirty}\n{diff}".encode("utf-8")).hexdigest()
    return f"{sha}+dirty.{digest[:12]}"


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
        "missing_reports": stats.missing_reports,
        "missing_sidecars": stats.missing_sidecars,
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
    print(f"  song-stamped intent blocks realigned: "
          f"{dropped['intent_blocks_song_stamped']}  "
          f"(rows re-attributed: {dropped['intent_reattributed']})")
    print(f"  intent blocks committed late: {dropped['intent_blocks_late']} of "
          f"{dropped['intent_blocks_song_recorded']} eligible  "
          f"(#154's accepted lateness -- the chain was older than the "
          f"playback delay; only a block that RECORDED its instant can be "
          f"measured, so a zero denominator means the reports predate it)")
    print(f"  tracks with no cached report : {len(stats.missing_reports)}"
          + (f"  {stats.missing_reports[:10]}" if stats.missing_reports else ""))
    print(f"  tracks skipped, no sidecar   : {len(stats.missing_sidecars)}"
          + (f"  {stats.missing_sidecars[:10]}" if stats.missing_sidecars else ""))

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
