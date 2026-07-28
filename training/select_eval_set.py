#!/usr/bin/env python
"""Freeze a small, diverse, expert-labeled Raveform eval set.

Writes ``training/eval_set.json`` -- the COMMITTED identity of the simulation
benchmark.  Ten tracks, chosen from the cleanliness-gated corpus, that the fast
simulation is scored against and that the neural section classifier must never
see: ``dataset.make_splits`` (not written yet -- NN plan Task 1) will exclude
every id listed here from train, val and test alike.  This file is ids + the
evidence behind them; the ten tracks' audio and labels are committed beside it
under derived, opaque names (owner-authorised -- see ``training/eval_assets.py``,
whose scheme this document records in its ``artifacts`` block) so that the
benchmark runs from a fresh clone.  The rest of the corpus stays gitignored.

Once frozen it STAYS frozen: re-running this script over a grown corpus refuses
to overwrite the file without ``--force``, because Task B's baseline records
per-track checksums and scores against exactly these tracks.

Why these criteria, in this order
---------------------------------

1. **All five v1 classes in one track.**  The evaluator scores in the v1 space
   (``intro/buildup/breakdown/drop/outro``).  A track missing a class cannot
   exercise the transitions into or out of it, so a benchmark built from such
   tracks would be blind to a whole failure mode.  Applied as a *preference*
   rather than a filter -- "where possible" -- so a sparsely populated BPM band
   still contributes a track instead of silently dropping out of the set.
2. **BPM diversity.**  Intent classification thresholds onset density and beat
   spacing, both of which move with tempo; a benchmark clustered at 126 BPM
   would certify the classifier only for four-to-the-floor techno.  The corpus
   is heavily massed at 124-130, so equal-*count* bins would reproduce exactly
   that cluster.  Equal-*width* bands across the eligible range are used
   instead, and the bands the corpus leaves empty are filled by farthest-point
   spread, so the set spans the tempo range rather than the corpus histogram.
3. **Duration 3-8 minutes.**  Long enough to contain a real arrangement, short
   enough that the whole set simulates inside the integration-test budget.
4. **At least 8 section boundaries.**  Counted in the v1 space, i.e. the
   transitions the evaluator can actually score.  A track with two boundaries
   is a duration test, not a structure test.
5. **No two tracks from the same family.**  Two readings, at two strengths.
   The *artist* is the substantive guard and is hard: two tracks by the same
   producer are one measurement wearing two hats.  The ``track_id`` index block
   (``0834.xxx`` -> ``08``) spreads the set across the corpus rather than over
   one contiguous stretch of it, but the corpus is not ordered by artist or
   label, so a shared block carries no musical meaning -- and enforcing it hard
   was measured to cost the entire 157-163 BPM band, whose only three
   candidates all sat in blocks already taken.  Criteria are ranked, and BPM
   diversity outranks family spread, so the block rule yields wherever holding
   it would narrow the tempo spread -- in the band pass AND in the tempo-gap
   fill, which is easy to get wrong in opposite directions (see ``select``).

Determinism
-----------

There is no RNG.  The seed enters only as a stable BLAKE2b tiebreak over the
YouTube id, so the same inputs always produce the same ten tracks, on any
platform, in any process.  The inputs themselves are recorded in the output
(row counts, seed, and the SHA-256 of both source files) so a future selection
that differs can be traced to the input that moved.

Stdlib only.

Usage::

    uv run python training/select_eval_set.py \\
        --data-dir C:\\Users\\Julian\\Projects\\soundswitch-auto-pilot\\training\\data\\raveform
    uv run python training/select_eval_set.py --dry-run   # table only, no write
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
# training/ for the eval-pipeline scripts, training/raveform/ for the
# corpus-acquisition ones -- both are scripts, not packages.
for _path in (
    str(Path(__file__).resolve().parent),
    str(REPO_ROOT / "training" / "raveform"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from build_clean_manifest import CLEAN_MANIFEST_FILE, STATUS_OK  # noqa: E402
from build_training_table import V1_ORDER, label_v1  # noqa: E402
from eval_assets import artifacts_block  # noqa: E402
from raveform_fetch_annotations import (  # noqa: E402
    SEGMENTS_FILE,
    annotations_dir,
    load_tracks,
    parse_sections,
)
from raveform_manifest import canonical_runs  # noqa: E402

EVAL_SET_FILE = REPO_ROOT / "training" / "eval_set.json"

DEFAULT_SEED = 20260726
DEFAULT_SIZE = 10

MIN_DURATION_SEC = 180.0
MAX_DURATION_SEC = 480.0
MIN_BOUNDARIES = 8

V1_CLASSES = frozenset(V1_ORDER)

# A beat grid this short cannot give a trustworthy median inter-beat interval,
# and a track that thin is not eval material anyway.
MIN_BEATS = 8


class Candidate(NamedTuple):
    """One gated, annotated track and everything the selection reasons about."""

    track_id: str
    youtube_id: str
    duration_sec: float
    bpm: float
    boundaries: int          # v1-space transitions: len(runs) - 1
    classes: frozenset       # v1 labels present in the track
    genre: str
    title: str
    artist: str
    family: str              # track_id index block


# --------------------------------------------------------------------------- #
# Track facts
# --------------------------------------------------------------------------- #


def v1_runs(sections: list) -> list:
    """RAW sections -> contiguous runs in the 5-class v1 space.

    ``canonical_runs`` already drops the ``end`` sentinel and folds
    ``altintro``/``bridge``; the v1 fold (``cooldown``->``breakdown``,
    ``altoutro``->``outro``) can make two of its runs adjacent-and-equal, so
    they are merged again here.  Without the second merge a
    ``breakdown|cooldown`` pair would be counted as a boundary the evaluator
    can never see.
    """
    runs: list = []
    for start, end, label, _duration in canonical_runs(list(sections)):
        mapped = label_v1(label)
        if runs and runs[-1][2] == mapped:
            runs[-1][1] = end
        else:
            runs.append([start, end, mapped])
    return [tuple(run) for run in runs]


def beat_grid_bpm(times: list) -> float | None:
    """Tempo from a Raveform beat grid: the median inter-beat interval.

    The median, not the mean: a grid that spans a tempo change or carries a
    single mis-placed beat would drag a mean off the tempo the track actually
    sits at for most of its length.
    """
    intervals = [later - earlier for earlier, later in zip(times, times[1:]) if later > earlier]
    if len(times) < MIN_BEATS or not intervals:
        return None
    median = statistics.median(intervals)
    return None if median <= 0.0 else 60.0 / median


def read_beat_times(path: Path) -> list:
    """Beat instants from one ``<track_id>.beat.csv``, or ``[]`` if absent."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [float(row["time"]) for row in csv.DictReader(handle)]


_LEADING_TAG = re.compile(r"^\s*(\[[^\]]*\]|[^:\-]{1,40}:)\s*")


def artist_of(title: str) -> str:
    """Normalised producer name from a Raveform title.

    Titles are ``Artist - Track [Label]``, sometimes behind a chart-position
    tag (``[069] Artist - ...``), an editorial prefix
    (``Record Of The Week: Artist - ...``) or a bare marker glyph
    (``+ Konflict - ...``, which the corpus also lists plainly as
    ``Konflict - ...``).  All three are stripped, so the same producer is
    recognised as the same family however the row was captured -- otherwise the
    guard passes two tracks by one artist purely because one row carries a
    decoration.
    """
    cleaned = _LEADING_TAG.sub("", title, count=1).lstrip("+*~- \t")
    return cleaned.split(" - ", 1)[0].strip().casefold()


def family_of(track_id: str) -> str:
    """The ``track_id`` index block -- ``0834.NyEKXA7_6z0`` -> ``08``."""
    return track_id.split(".", 1)[0][:2]


def build_candidates(data_dir: Path, ok_rows: list, tracks: list) -> list:
    """Gated rows joined to their annotation, as ``Candidate``s.

    A row without an annotation record or without a usable beat grid cannot be
    reasoned about and is left out; the counts are reported by the caller.
    """
    by_track_id = {str(track["key"]): track for track in tracks}
    beats_dir = annotations_dir(data_dir) / "beats"

    candidates = []
    for row in ok_rows:
        track = by_track_id.get(row["track_id"])
        if track is None:
            continue
        bpm = beat_grid_bpm(read_beat_times(beats_dir / f"{row['track_id']}.beat.csv"))
        if bpm is None:
            continue
        runs = v1_runs(parse_sections(track))
        title = str(track.get("title", ""))
        candidates.append(Candidate(
            track_id=row["track_id"],
            youtube_id=row["youtube_id"],
            duration_sec=float(row["decoded_duration_sec"]),
            bpm=bpm,
            boundaries=max(0, len(runs) - 1),
            classes=frozenset(label for _s, _e, label in runs) & V1_CLASSES,
            genre=str(track.get("genre", "")),
            title=title,
            artist=artist_of(title),
            family=family_of(row["track_id"]),
        ))
    candidates.sort(key=lambda candidate: candidate.track_id)
    return candidates


def is_eligible(candidate: Candidate) -> bool:
    """Hard gates: length and structural richness.  Class coverage is a rank."""
    return (
        MIN_DURATION_SEC <= candidate.duration_sec <= MAX_DURATION_SEC
        and candidate.boundaries >= MIN_BOUNDARIES
    )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def tiebreak(seed: int, youtube_id_: str) -> str:
    """Stable pseudo-random ordering key -- the ONLY place the seed acts.

    A hash rather than an RNG so the choice depends on nothing but the seed and
    the id: no draw order, no set iteration order, no platform hash salt.
    """
    digest = hashlib.blake2b(f"{seed}:{youtube_id_}".encode("utf-8"), digest_size=8)
    return digest.hexdigest()


def rank_key(candidate: Candidate, seed: int, target_bpm: float | None = None):
    """Preference order within a group of otherwise-interchangeable candidates.

    Class coverage first (criterion 1), then structural richness, then -- when
    a band is being represented -- closeness to that band's centre, so the
    chosen track speaks for the middle of its tempo range rather than its edge.
    """
    return (
        -len(candidate.classes),
        -candidate.boundaries,
        abs(candidate.bpm - target_bpm) if target_bpm is not None else 0.0,
        tiebreak(seed, candidate.youtube_id),
    )


def equal_width_bins(values: list, count: int) -> list:
    """``count`` equal-width ``(low, high)`` bands spanning ``values``.

    Equal width, not equal count: the point is to cover the tempo *range*, and
    equal-count bins over a corpus massed at 124-130 BPM would put eight of ten
    bands inside six BPM of each other.  The last band's upper edge is
    inclusive so the fastest track is never orphaned.
    """
    if count <= 0 or not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [(low, high)]
    width = (high - low) / count
    return [(low + index * width, low + (index + 1) * width) for index in range(count)]


def _in_bin(candidate: Candidate, band: tuple, is_last: bool) -> bool:
    low, high = band
    return low <= candidate.bpm <= high if is_last else low <= candidate.bpm < high


# Two tempos this close are the same tempo for a benchmark's purposes, so the
# fill pass treats candidates within one bucket of each other as tied on tempo
# and lets the *stated* secondary criteria decide between them.  Without the
# bucket the raw float gap is almost always unique, every later key in the sort
# is dead, and class coverage -- criterion 1 -- never gets a vote.
GAP_BUCKET_BPM = 0.5


def tempo_gap(candidate: Candidate, taken_bpms: list) -> float:
    """Distance from this candidate's tempo to the nearest already-chosen one."""
    if not taken_bpms:
        return 0.0
    return min(abs(candidate.bpm - bpm) for bpm in taken_bpms)


def gap_bucket(gap: float) -> int:
    """``tempo_gap`` quantised to ``GAP_BUCKET_BPM``, so near-ties compare equal."""
    return int(gap / GAP_BUCKET_BPM)


class Pick(NamedTuple):
    """One selected track and the reason it was selected."""

    candidate: Candidate
    reason: str


def select(candidates: list, size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> list:
    """Choose ``size`` eligible tracks, spread across the tempo range.

    Two passes.  The first walks the equal-width BPM bands low to high and
    takes the best-ranked eligible candidate in each, which is what "spread
    across the corpus BPM range" means operationally.  Bands the corpus leaves
    empty -- the 140-160 BPM gap is real -- would otherwise shrink the set, so
    the second pass fills the remaining slots by farthest-point sampling: the
    candidate whose tempo is furthest from every tempo already chosen.  That
    keeps the set at exactly ``size`` without collapsing back onto the mass at
    126 BPM.

    The two passes handle the family rules IDENTICALLY, which takes some care to
    state because they group candidates differently.  The artist rule is
    absolute everywhere.  The index-block rule is a preference that yields
    whenever holding it would cost tempo coverage -- and "would cost tempo
    coverage" has to be evaluated against the group that is competing for the
    slot, never against the pool at large:

    * **Band pass** -- the group is the band.  Prefer its unused-block members;
      if the band has none, take any of its members rather than surrender the
      band.
    * **Fill pass** -- the group is the set of candidates tied at the widest
      available tempo gap (bucketed, see ``GAP_BUCKET_BPM``).  Prefer its
      unused-block members; if that group has none, take any of them rather
      than fall back to a candidate with a narrower gap.

    Applying the preference to the whole remaining pool in the fill pass -- the
    obvious-looking simplification -- silently promotes it to a hard rule,
    because the pool practically always contains *some* unused-block track.
    That inverts the documented priority (2 outranks 5) and was measured to
    trade a 3.6 BPM gap for a 2.0 BPM one.  Hence the bucketing: it defines the
    group the preference is allowed to choose within.
    """
    eligible = [candidate for candidate in candidates if is_eligible(candidate)]
    if not eligible:
        return []

    picks: list = []
    families: set = set()
    artists: set = set()

    def take(candidate: Candidate, reason: str) -> None:
        picks.append(Pick(candidate, reason))
        families.add(candidate.family)
        artists.add(candidate.artist)

    def allowed(candidate: Candidate) -> bool:
        """Hard rule: never two tracks by the same producer."""
        return candidate.artist not in artists

    def preferred(candidate: Candidate) -> bool:
        """Soft rule: also a corpus block no other pick occupies."""
        return allowed(candidate) and candidate.family not in families

    def viable(pool: list) -> tuple:
        """``(members, relaxed)`` -- the soft rule if it leaves anything, else not."""
        strict = [candidate for candidate in pool if preferred(candidate)]
        if strict:
            return strict, False
        return [candidate for candidate in pool if allowed(candidate)], True

    bands = equal_width_bins([candidate.bpm for candidate in eligible], size)
    for index, band in enumerate(bands):
        if len(picks) >= size:
            break
        centre = (band[0] + band[1]) / 2.0
        members, relaxed = viable([
            candidate for candidate in eligible
            if _in_bin(candidate, band, index == len(bands) - 1)
        ])
        if members:
            reason = f"BPM band {band[0]:.1f}-{band[1]:.1f}"
            take(min(members, key=lambda c: rank_key(c, seed, centre)),
                 reason + " (block reused)" if relaxed else reason)

    chosen = {pick.candidate.youtube_id for pick in picks}
    while len(picks) < size:
        pool = [
            candidate for candidate in eligible
            if candidate.youtube_id not in chosen and allowed(candidate)
        ]
        if not pool:
            break
        taken_bpms = [pick.candidate.bpm for pick in picks]
        widest = max(gap_bucket(tempo_gap(candidate, taken_bpms)) for candidate in pool)
        contenders = [
            candidate for candidate in pool
            if gap_bucket(tempo_gap(candidate, taken_bpms)) == widest
        ]
        members, relaxed = viable(contenders)
        best = min(members, key=lambda c: rank_key(c, seed))
        reason = "tempo-gap fill"
        take(best, reason + " (block reused)" if relaxed else reason)
        chosen.add(best.youtube_id)

    picks.sort(key=lambda pick: pick.candidate.track_id)
    return picks


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def rationale_line(pick: Pick) -> str:
    """The one-liner recorded per track in ``eval_set.json``."""
    candidate = pick.candidate
    classes = "/".join(label for label in V1_ORDER if label in candidate.classes)
    return (
        f"{candidate.track_id} {candidate.genre} {candidate.bpm:.1f} BPM, "
        f"{candidate.duration_sec / 60.0:.1f} min, {candidate.boundaries} v1 boundaries, "
        f"classes {classes} -- {pick.reason}"
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(data_dir: Path) -> dict:
    """The files a selection is a function of, keyed as they are recorded.

    One definition, used both to stamp the provenance and to check it later --
    otherwise the writer and the verifier drift and the check quietly passes on
    a file nobody read.
    """
    return {
        CLEAN_MANIFEST_FILE: data_dir / CLEAN_MANIFEST_FILE,
        SEGMENTS_FILE: annotations_dir(data_dir) / SEGMENTS_FILE,
    }


def load_eval_set(path: Path) -> dict:
    """Read a frozen eval set, refusing anything that is not one.

    Task B's benchmark runner and the NN split builder both read this file; a
    truncated or half-written one must announce itself here rather than at the
    point where a missing id looks like a missing track.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise RuntimeError(
            f"missing {path} -- run training/select_eval_set.py to freeze the eval set"
        ) from None
    except ValueError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from None
    ids = document.get("youtube_ids") if isinstance(document, dict) else None
    if not isinstance(ids, list) or not ids:
        raise RuntimeError(f"{path} has no 'youtube_ids' -- it is not an eval set")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{path} lists a duplicate youtube_id")
    return document


def verify_inputs(document: dict, data_dir: Path, only: tuple | None = None) -> list:
    """Which recorded selection inputs no longer match the corpus on disk.

    Empty list means the frozen set was selected from exactly the files that
    are there now.  A non-empty one is NOT an error by itself -- the corpus
    grows, and the whole point of freezing is that the benchmark does not
    follow it -- but it is the difference between "frozen deliberately" and
    "frozen by accident", so every caller gets to see it.

    ``only`` narrows the check to named inputs.  The two recorded files carry
    very different weight: ``clean_manifest.csv`` decided WHICH tracks are in
    the set and legitimately moves with every download batch, while
    ``segments.json`` is the ground truth every score is computed against and
    must not move at all under a committed baseline.  The benchmark runner
    checks the second alone and treats it as fatal (``run_eval_set.run``).
    """
    recorded = (document.get("selected_from") or {}).get("inputs") or {}
    drift = []
    for name, path in input_paths(data_dir).items():
        if only is not None and name not in only:
            continue
        expected = recorded.get(name)
        if expected is None:
            drift.append(f"{name}: no checksum recorded in the eval set")
        elif not path.exists():
            drift.append(f"{name}: recorded but missing from {path.parent}")
        elif sha256_of(path) != expected:
            drift.append(f"{name}: changed since the freeze ({expected[:12]}... on record)")
    return drift


def build_document(picks: list, data_dir: Path, clean_rows: int, candidates: int,
                   eligible: int, seed: int) -> dict:
    """The committed eval-set record: ids, provenance, and the reasoning."""
    return {
        "youtube_ids": [pick.candidate.youtube_id for pick in picks],
        "selected_from": {
            "clean_rows": clean_rows,
            "seed": seed,
            "annotated_candidates": candidates,
            "eligible": eligible,
            "criteria": {
                "duration_sec": [MIN_DURATION_SEC, MAX_DURATION_SEC],
                "min_v1_boundaries": MIN_BOUNDARIES,
                "v1_classes": list(V1_ORDER),
                "prefer_unique_track_id_block": True,   # yields to BPM coverage
                "unique_artist": True,                  # absolute
            },
            "inputs": {
                name: sha256_of(path)
                for name, path in input_paths(data_dir).items()
            },
        },
        # What is committed alongside this file, and how its filenames are
        # derived -- the scheme, never a mapping (see eval_assets).  A re-freeze
        # rewrites this document, so it has to be emitted, not hand-added, or
        # the first --force would drop it.
        "artifacts": artifacts_block(),
        "rationale": {
            pick.candidate.youtube_id: rationale_line(pick) for pick in picks
        },
        "tracks": [
            {
                "track_id": pick.candidate.track_id,
                "youtube_id": pick.candidate.youtube_id,
                "title": pick.candidate.title,
                "genre": pick.candidate.genre,
                "bpm": round(pick.candidate.bpm, 2),
                "duration_sec": round(pick.candidate.duration_sec, 3),
                "v1_boundaries": pick.candidate.boundaries,
                "v1_classes": [label for label in V1_ORDER if label in pick.candidate.classes],
            }
            for pick in picks
        ],
    }


def write_eval_set(path: Path, document: dict) -> Path:
    """Write ``eval_set.json`` atomically, pretty and newline-terminated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def print_table(picks: list) -> None:
    """The per-track rationale table the plan asks the selector to print."""
    print()
    print("selected eval set")
    print(f"  {'track_id':<20}{'youtube_id':<14}{'bpm':>7}{'min':>6}{'bnd':>5}"
          f"  {'v1 classes':<38}{'genre':<20}{'why':<34}title")
    for pick in picks:
        candidate = pick.candidate
        classes = "/".join(label for label in V1_ORDER if label in candidate.classes)
        print(
            f"  {candidate.track_id:<20}{candidate.youtube_id:<14}"
            f"{candidate.bpm:>7.1f}{candidate.duration_sec / 60.0:>6.1f}"
            f"{candidate.boundaries:>5}  {classes:<38}{candidate.genre:<20}"
            f"{pick.reason:<34}{candidate.title[:40]}"
        )


def print_summary(picks: list, clean_rows: int, candidates: int, eligible: int) -> None:
    bpms = sorted(pick.candidate.bpm for pick in picks)
    durations = sorted(pick.candidate.duration_sec for pick in picks)
    covered: collections.Counter = collections.Counter()
    for pick in picks:
        covered.update(pick.candidate.classes)
    print()
    print("summary")
    print(f"  clean-manifest ok rows   : {clean_rows}")
    print(f"  with annotation + beats  : {candidates}")
    print(f"  eligible (all hard gates): {eligible}")
    print(f"  selected                 : {len(picks)}")
    print(f"  BPM spread               : {bpms[0]:.1f} - {bpms[-1]:.1f}  "
          f"({', '.join(f'{value:.0f}' for value in bpms)})")
    print(f"  duration spread          : {durations[0] / 60.0:.1f} - "
          f"{durations[-1] / 60.0:.1f} min  "
          f"(total {sum(durations) / 60.0:.1f} min)")
    print("  v1 class coverage        : "
          + "  ".join(f"{label} {covered.get(label, 0)}/{len(picks)}" for label in V1_ORDER))
    print(f"  distinct genres          : "
          f"{len({pick.candidate.genre for pick in picks})}  "
          f"({', '.join(sorted({pick.candidate.genre for pick in picks}))})")
    print(f"  distinct track_id blocks : {len({pick.candidate.family for pick in picks})}")
    print(f"  distinct artists         : {len({pick.candidate.artist for pick in picks})}")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_ok_rows(data_dir: Path) -> list:
    """``status == ok`` rows of ``clean_manifest.csv``, sorted by track_id."""
    path = data_dir / CLEAN_MANIFEST_FILE
    if not path.exists():
        raise RuntimeError(
            f"missing {path} -- run training/raveform/build_clean_manifest.py first"
        )
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == STATUS_OK]
    if not rows:
        raise RuntimeError(f"no ok rows in {path} -- nothing to select from")
    rows.sort(key=lambda row: row["track_id"])
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_data_dir() -> Path:
    return REPO_ROOT / "training" / "data" / "raveform"


def _describe_frozen(path: Path, data_dir: Path) -> list:
    """Human-readable lines about the eval set already on disk."""
    lines = []
    try:
        document = load_eval_set(path)
    except RuntimeError as exc:
        return [f"  (unreadable: {exc})"]
    lines.append(f"  {len(document['youtube_ids'])} track(s): "
                 f"{', '.join(document['youtube_ids'])}")
    drift = verify_inputs(document, data_dir)
    if drift:
        lines.append("  selection inputs have MOVED since the freeze:")
        lines.extend(f"    - {entry}" for entry in drift)
        lines.append("  (expected as the corpus grows -- the benchmark does not "
                     "follow it, by design)")
    else:
        lines.append("  selection inputs still match the corpus on disk")
    return lines


def refuse_to_overwrite(path: Path, data_dir: Path) -> int:
    """Stop rather than silently re-cut the benchmark under a committed baseline."""
    print()
    print(f"REFUSING to overwrite the frozen eval set at {path}")
    for line in _describe_frozen(path, data_dir):
        print(line)
    print()
    print("  The eval set is FROZEN: Task B's baseline file records per-track")
    print("  checksums and scores against exactly these ten tracks, and the NN")
    print("  splits will exclude exactly these ids.  Re-cutting it silently after")
    print("  a download batch would invalidate both without changing a test.")
    print()
    print("  --dry-run  to see what a fresh selection would pick, changing nothing")
    print("  --force    to re-freeze deliberately (then re-cut the baseline too)")
    return 1


def warn_refreeze(path: Path, data_dir: Path) -> None:
    """Loudly announce a deliberate re-freeze before it happens."""
    print()
    print("!" * 78)
    print("RE-FREEZING the eval set (--force).  The benchmark is being redefined.")
    print(f"replacing {path}")
    for line in _describe_frozen(path, data_dir):
        print(line)
    print()
    print("Any committed eval_set_baseline.json is now DESYNCHRONIZED: its per-track")
    print("checksums and scores describe tracks that may no longer be in the set.")
    print("Re-cut the baseline in the same change, or the regression gate is lying.")
    print("!" * 78)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir(),
        help="corpus root; reads clean_manifest.csv + annotations/ (default: %(default)s)",
    )
    # No --out and no --size.  The output path IS the benchmark's identity (a
    # committed baseline, a test that reads EVAL_SET_FILE, and the NN splits all
    # name it), and the size is what the baseline was cut over; a flag for
    # either is a way to produce a file that looks frozen and gates nothing.
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="tiebreak seed, recorded in the output (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the table and summary without writing the file")
    parser.add_argument("--force", action="store_true",
                        help="RE-FREEZE: overwrite an existing eval set. Changes the "
                             "benchmark and desynchronizes any committed baseline")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    print("raveform eval-set selection")
    print(f"data dir: {data_dir}")

    if EVAL_SET_FILE.exists() and not (args.force or args.dry_run):
        return refuse_to_overwrite(EVAL_SET_FILE, data_dir)

    ok_rows = load_ok_rows(data_dir)
    candidates = build_candidates(data_dir, ok_rows, load_tracks(data_dir))
    eligible = [candidate for candidate in candidates if is_eligible(candidate)]
    picks = select(candidates, size=DEFAULT_SIZE, seed=args.seed)
    if not picks:
        # Loud and specific: an empty selection means the gate or the fetch has
        # not run, not that the corpus is merely thin.  Crashing in the summary
        # printer would hide which of the two it was.
        print(f"ERROR: no track in {len(candidates)} candidate(s) satisfies the "
              f"criteria (duration {MIN_DURATION_SEC:.0f}-{MAX_DURATION_SEC:.0f} s, "
              f">= {MIN_BOUNDARIES} v1 boundaries) -- nothing written")
        return 1
    if len(picks) < DEFAULT_SIZE:
        print(f"WARNING: only {len(picks)}/{DEFAULT_SIZE} tracks satisfy the criteria "
              f"-- the corpus is too small or too uniform for this size")

    print_table(picks)
    print_summary(picks, len(ok_rows), len(candidates), len(eligible))

    if args.dry_run:
        print()
        print("--dry-run: nothing written")
        return 0

    if EVAL_SET_FILE.exists():
        warn_refreeze(EVAL_SET_FILE, data_dir)

    document = build_document(picks, data_dir, len(ok_rows), len(candidates),
                              len(eligible), args.seed)
    path = write_eval_set(EVAL_SET_FILE, document)
    print()
    print(f"eval set: {path}")
    print(f"  ids: {', '.join(document['youtube_ids'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
