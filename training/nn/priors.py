"""Structural priors for the section decoder, fitted from Raveform.

    uv run python -m training.nn.priors --data-dir <corpus>

Everything the decoder knows that is not in the audio lives here: which section
can follow which, how long a section lasts, and how much of a track each class
occupies.  Three principles shape the fit, and none of them is "maximise
likelihood".

**Structure is hard, preference is soft.**  Over 538 train tracks, intro is
*only ever* the first run and outro is *never* followed by anything.  Those are
not tendencies to be smoothed, they are -inf entries -- the invalid-transition
graph the engine used to enforce with a veto, moved into the matrix where the
decoder can route around it instead of silently holding state.  The claim is
checked against the data on every fit (``strict``): if a later corpus revision
contradicts it, the fit fails loudly rather than discarding the evidence.

**An uninformative prior must be made uninformative.**  Buildup resolves to
breakdown 235 times and to drop 182 -- about 0.15 nats, which is nothing, and
the fork is exactly the decision the 16 s look-ahead window exists to make.  A
56/44 prior would be a thumb on the scale disguised as a measurement, so the
fork is forced uniform while keeping its combined mass.

**Duration is our strongest prior and the easiest to overfit.**  The corpus is
studio masters; the show is a DJ set.  So the per-class duration model is not
the empirical distribution but a *widened* two-part one: a hard min-duration
floor at the corpus p05 bar count, then a **memoryless** geometric tail whose
survival halves at the corpus median.  Above the floor the hazard is constant,
so the prior never *pushes* a switch at a particular bar -- it only charges a
small, uniform toll per bar of continuing.  A DJ cutting a 32-bar drop at 20
bars pays that toll once instead of fighting a peak.

Bars, not seconds.  Duration priors are musically meaningful in bars and the
decoder runs on the bar grid, so the counts come from the corpus's own beat
CSVs (``downbeat == 1``).  A bar is ~1.9 s here, so seconds would tie the model
to tempo for nothing.

Fitted from the **train split only**.  The priors are model parameters; fitting
them on val or test would leak the answer into every number Tasks 5 and 6
produce, and nothing downstream could detect it.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import _TRAINING_DIR  # noqa: F401  (puts training/ on sys.path for the
                             # lazy imports below)

# **Nothing heavy at module scope.**  ``decoder`` imports this module, and the
# decoder is the object that runs live at showtime -- so the import that a show
# pays for must be numpy and nothing else.  ``nn.dataset`` alone pulls torch
# (1.9 s, 1,127 modules) and ``build_training_table`` pulls the whole eval
# pipeline; both are needed only to *fit* priors from the corpus, never to read
# a fitted one.  They are therefore imported inside the functions that use them.
# ``lib.label_space`` is the exception, and is why it lives outside the training
# tree at all: the vocabulary is stdlib-only, so the one definition of it costs
# the show nothing and never has to be copied into a second place.
from lib.label_space import SECTION_LABELS, check_class_space  # noqa: E402

PRIORS_FILE = "priors.json"
MODELS_DIR = "models"
MODEL_VERSION = "v1"

# Jeffreys' prior on the transition counts.  Half a count is enough to keep a
# legal-but-unobserved edge (intro->outro: a track that opens into its outro)
# representable without moving an edge that has hundreds of observations.
SMOOTHING = 0.5

# The min-duration floor.  p05, not p01: the bottom of the corpus is where
# annotation noise lives, and a floor set by the single shortest drop in 538
# tracks would be no floor at all.
FLOOR_PERCENTILE = 5.0

# Forks the corpus cannot inform.  The spec measures buildup->{breakdown, drop}
# at ~0.15 nats and rules that the look-ahead evidence decides it, not the
# prior.  Written as data rather than as a branch so a second uninformative
# fork (should one be measured) is one line.
UNIFORM_FORKS = {"buildup": ("breakdown", "drop")}

PRIORS_VERSION = 1


def section_classes() -> tuple:
    """The class space to fit in: the whole published vocabulary.

    Callers that already hold a ``Priors`` must read ``priors.classes`` instead
    -- every table in a fitted object is indexed by the space it was fitted on,
    which may be a subset of this one.
    """
    return SECTION_LABELS


# --------------------------------------------------------------------------- #
# The structural graph
# --------------------------------------------------------------------------- #


INTRO_FAMILY = frozenset({"intro", "altintro"})
OUTRO_FAMILY = frozenset({"outro", "altoutro"})


def transition_allowed(src: str, dst: str) -> bool:
    """Can a decoded section of class ``src`` be followed by one of ``dst``?

    Three rules, each a measured fact about the corpus rather than taste:

    * **nothing enters the intro family** -- intro is pure-initial (0 of 3,747
      train transitions land on it); a track does not go back to its
      introduction.
    * **nothing leaves the outro family** -- outro is terminal (0 of 525 train
      outros have a successor); once the decoder commits an outro the track is
      over.
    * **nothing self-transitions** -- the states are *merged runs*, so a
      self-loop is not a section change at all.  Persistence within a run is the
      duration model's job; putting it in the matrix too would double-count it.

    The first two are stated over *families* because that is the granularity
    they were measured at: the fold collapsed each pair into one class, so the
    counts never distinguished ``altintro`` from ``intro``.  A within-family
    move is therefore unmeasured rather than absent, and it is exactly the
    beat-in the owner asked the vocabulary to be able to express.
    """
    if src in INTRO_FAMILY and dst in INTRO_FAMILY:
        return src != dst
    if src in OUTRO_FAMILY and dst in OUTRO_FAMILY:
        return src != dst
    if dst in INTRO_FAMILY or src in OUTRO_FAMILY:
        return False
    return src != dst


def legal_mask(classes=None) -> np.ndarray:
    """``[C, C]`` boolean: ``True`` where a transition is structurally legal."""
    classes = section_classes() if classes is None else tuple(classes)
    return np.array(
        [[transition_allowed(src, dst) for dst in classes] for src in classes],
        dtype=bool,
    )


# --------------------------------------------------------------------------- #
# The fitted object
# --------------------------------------------------------------------------- #


class Priors(NamedTuple):
    """Everything the decoder knows before it hears anything.

    Probabilities, not log-probabilities, are what gets stored: an illegal
    transition is a plain ``0.0`` on disk and becomes ``-inf`` on load.  JSON has
    no infinity (``json.dumps`` emits a Python-only ``-Infinity`` literal), and a
    priors file that only Python can read is a priors file that cannot be
    checked by eye or by another tool.
    """

    classes: tuple
    initial: np.ndarray         # [C] P(first run is this class)
    transition: np.ndarray      # [C, C] P(next run | this run), 0 where illegal
    floor_bars: np.ndarray      # [C] int, min bars before a switch is allowed
    hazard: np.ndarray          # [C] per-bar P(switch) once past the floor
    class_prior: np.ndarray     # [C] share of corpus BARS, for scaled likelihoods
    corpus: dict                # provenance and the raw statistics behind the fit

    # -- derived views ------------------------------------------------------ #

    @property
    def log_initial(self) -> np.ndarray:
        return _log(self.initial)

    @property
    def log_transition(self) -> np.ndarray:
        return _log(self.transition)

    def index(self, label: str) -> int:
        return self.classes.index(label)

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "version": PRIORS_VERSION,
            "classes": list(self.classes),
            "initial": [float(v) for v in self.initial],
            "transition": [[float(v) for v in row] for row in self.transition],
            "duration": {
                "floor_bars": [int(v) for v in self.floor_bars],
                "hazard": [float(v) for v in self.hazard],
            },
            "class_prior": [float(v) for v in self.class_prior],
            "corpus": self.corpus,
        }

    @classmethod
    def from_dict(cls, document: dict, source: str = "priors document") -> "Priors":
        version = int(document.get("version", 0))
        if version != PRIORS_VERSION:
            raise RuntimeError(
                f"priors document is version {version}, this build reads "
                f"{PRIORS_VERSION} -- refit rather than reinterpret"
            )
        classes = tuple(document["classes"])
        # Every table below is indexed by this list, so a space that is merely
        # the right length loads cleanly and decodes the wrong classes.
        check_class_space(classes, source)
        duration = document["duration"]
        return cls(
            classes=classes,
            initial=np.asarray(document["initial"], dtype=np.float64),
            transition=np.asarray(document["transition"], dtype=np.float64),
            floor_bars=np.asarray(duration["floor_bars"], dtype=np.int64),
            hazard=np.asarray(duration["hazard"], dtype=np.float64),
            class_prior=np.asarray(document["class_prior"], dtype=np.float64),
            corpus=dict(document.get("corpus") or {}),
        )

    def save(self, path) -> None:
        """Write the priors as sorted, indented JSON -- byte-stable by content.

        No timestamp: this file is an input to every decode in Tasks 5 and 6, so
        two fits of the same split must produce the same bytes.  Provenance that
        does not change the numbers belongs in ``corpus``, and the split file
        already records when it was frozen.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "Priors":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), str(path))


def _log(values: np.ndarray) -> np.ndarray:
    """``log`` with ``log(0) == -inf`` and no divide-by-zero warning."""
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(values, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Corpus -> run sequences
# --------------------------------------------------------------------------- #


def label_runs(sections: list) -> list:
    """Published sections -> merged runs ``[(start, end, label)]``.

    The ``end`` sentinel is dropped and a merge across it joins the runs on
    either side -- consistent with ``section_runs`` -- but the sentinel's own
    time is never re-attributed, which is why bar counts are summed per member
    rather than taken from the merged span.
    """
    from build_training_table import label_coverage
    spans = sorted(
        ((float(start), float(end), label)
         for start, end, label in label_coverage(sections)),
        key=lambda span: span[0],
    )
    runs: list = []
    for start, end, label in spans:
        if runs and runs[-1][2] == label:
            runs[-1][1] = max(runs[-1][1], end)
        else:
            runs.append([start, end, label])
    return [tuple(run) for run in runs]


def bar_runs(sections: list, downbeats: np.ndarray) -> list:
    """Merged runs as ``[(label, n_bars)]`` on a track's own bar grid.

    A run's length is the number of downbeats inside its *member* spans, summed
    -- not the downbeats between the merged run's start and end.  The two differ
    exactly when a dropped ``end`` sentinel sits between two members, and that
    sentinel's bars belong to nobody.
    """
    from build_training_table import label_coverage
    downbeats = np.asarray(downbeats, dtype=np.float64)
    runs: list = []
    for start, end, label in sorted(
            ((float(s), float(e), l) for s, e, l in label_coverage(sections)),
            key=lambda span: span[0]):
        bars = int(np.count_nonzero((downbeats >= start) & (downbeats < end)))
        if runs and runs[-1][0] == label:
            runs[-1][1] += bars
        else:
            runs.append([label, bars])
    return [(label, bars) for label, bars in runs]


def corpus_bar_runs(data_dir, youtube_ids) -> tuple:
    """``([[ (label, bars) ]], skipped)`` for the given ids, in id order.

    A track with no beat grid is skipped rather than fitted in seconds: mixing
    two duration units in one prior would be invisible in the output and wrong
    everywhere.
    """
    from raveform_fetch_annotations import (
        beat_csv_path, load_all_tracks, parse_beat_csv, parse_sections)
    data_dir = Path(data_dir)
    wanted = set(str(i) for i in youtube_ids)
    by_id = {str(track.get("id")): track for track in load_all_tracks(data_dir)}

    sequences: list = []
    skipped: list = []
    for youtube_id in sorted(wanted):
        track = by_id.get(youtube_id)
        if track is None:
            skipped.append(youtube_id)
            continue
        path = beat_csv_path(data_dir, track)
        if not path.exists():
            skipped.append(youtube_id)
            continue
        downbeats = np.array(
            [time for time, position, _section in parse_beat_csv(path) if position == 1],
            dtype=np.float64)
        if downbeats.size < 2:
            skipped.append(youtube_id)
            continue
        runs = [(label, bars) for label, bars in
                bar_runs(parse_sections(track), downbeats) if bars > 0]
        if runs:
            sequences.append(runs)
        else:
            skipped.append(youtube_id)
    return sequences, skipped


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #


def fit_runs(sequences, *, classes=None, smoothing: float = SMOOTHING,
             floor_percentile: float = FLOOR_PERCENTILE,
             uniform_forks: dict = UNIFORM_FORKS, strict: bool = True,
             provenance: dict | None = None) -> Priors:
    """Fit priors from ``[[(label, n_bars), ...], ...]`` -- one list per track.

    Pure: no paths, no I/O, no corpus.  ``fit`` is the thin adapter that reads
    the split and the beat grids and calls this, which is what lets the whole
    prior be tested against hand-written sequences on a machine that has never
    seen the corpus.
    """
    classes = section_classes() if classes is None else tuple(classes)
    index = {label: i for i, label in enumerate(classes)}
    n = len(classes)
    legal = legal_mask(classes)

    counts = np.zeros((n, n), dtype=np.float64)
    initial_counts = np.zeros(n, dtype=np.float64)
    durations: dict = {label: [] for label in classes}
    illegal = collections.Counter()
    tracks = 0
    runs_seen = 0

    for sequence in sequences:
        pairs = [(str(label), int(bars)) for label, bars in sequence]
        if not pairs:
            continue
        tracks += 1
        runs_seen += len(pairs)
        initial_counts[index[pairs[0][0]]] += 1.0
        for label, bars in pairs:
            durations[label].append(bars)
        for (src, _bars), (dst, _next) in zip(pairs, pairs[1:]):
            i, j = index[src], index[dst]
            if legal[i, j]:
                counts[i, j] += 1.0
            else:
                illegal[f"{src}->{dst}"] += 1

    if illegal and strict:
        detail = ", ".join(f"{pair} x{count}" for pair, count in sorted(illegal.items()))
        raise RuntimeError(
            f"the corpus contains transitions the structural graph calls "
            f"impossible: {detail}.  The -inf entries in the transition matrix "
            f"are a claim about this data -- either the claim is wrong (update "
            f"transition_allowed) or the annotation is; fitting past it would "
            f"silently throw the evidence away.  Pass strict=False to record "
            f"the violation and fit anyway."
        )

    transition = _transition_matrix(counts, legal, classes, smoothing, uniform_forks)
    initial = _smoothed(initial_counts, np.ones(n, dtype=bool), smoothing)
    floor_bars, hazard, stats = _duration_model(durations, classes, floor_percentile)

    total_bars = float(sum(sum(v) for v in durations.values()))
    if total_bars <= 0.0:
        raise RuntimeError("no bars in the fitting corpus -- every run is empty")
    class_prior = np.array([sum(durations[label]) for label in classes],
                           dtype=np.float64) / total_bars

    corpus = {
        "tracks": tracks,
        "runs": runs_seen,
        "bars": int(total_bars),
        "transition_counts": {
            f"{src}->{dst}": int(counts[i, j])
            for i, src in enumerate(classes) for j, dst in enumerate(classes)
            if legal[i, j]
        },
        "initial_counts": {label: int(initial_counts[i])
                           for i, label in enumerate(classes)},
        "illegal_observed": {pair: int(count) for pair, count in sorted(illegal.items())},
        "duration_bars": stats,
        "uniform_forks": {src: list(dsts) for src, dsts in (uniform_forks or {}).items()},
        "smoothing": float(smoothing),
        "floor_percentile": float(floor_percentile),
    }
    corpus.update(provenance or {})

    return Priors(classes, initial, transition, floor_bars, hazard, class_prior,
                  corpus)


def _smoothed(counts: np.ndarray, allowed: np.ndarray, smoothing: float) -> np.ndarray:
    """Additive smoothing over the allowed entries, renormalised; 0 elsewhere."""
    values = np.where(allowed, counts + smoothing, 0.0)
    total = values.sum()
    return values / total if total > 0 else values


def _transition_matrix(counts, legal, classes, smoothing, uniform_forks) -> np.ndarray:
    """Row-stochastic over legal successors, with the uninformative forks levelled.

    Levelling happens *after* normalisation and preserves the fork's combined
    mass: the corpus is trusted about how often buildup resolves at all, and
    disbelieved only about which way.
    """
    index = {label: i for i, label in enumerate(classes)}
    matrix = np.zeros_like(counts)
    for i in range(len(classes)):
        matrix[i] = _smoothed(counts[i], legal[i], smoothing)

    for src, members in (uniform_forks or {}).items():
        if src not in index:
            continue
        i = index[src]
        columns = [index[label] for label in members if label in index and legal[i, index[label]]]
        if len(columns) < 2:
            continue
        matrix[i, columns] = matrix[i, columns].sum() / len(columns)
    return matrix


def _duration_model(durations: dict, classes, floor_percentile: float) -> tuple:
    """Per-class ``(floor_bars, hazard, stats)`` -- min-floor + geometric tail.

    The floor is the corpus p05 in bars.  Above it the tail is *memoryless*:
    survival ``(1 - hazard) ** k`` with the hazard chosen so that survival hits
    one half exactly at the corpus median residual length.  That is the widening
    the spec asks for -- the empirical distribution's peak is replaced by a flat
    per-bar toll, so evidence for an early switch pays a constant price instead
    of climbing out of a trough.
    """
    floors = np.ones(len(classes), dtype=np.int64)
    hazards = np.ones(len(classes), dtype=np.float64)
    stats: dict = {}

    for i, label in enumerate(classes):
        values = np.asarray(durations.get(label) or [], dtype=np.float64)
        if values.size == 0:
            # No observations: a floor of one bar and a hazard of one half is the
            # least committed thing that is still a distribution.
            floors[i], hazards[i] = 1, 0.5
            stats[label] = {"n": 0}
            continue
        floor = max(1, int(round(float(np.percentile(values, floor_percentile)))))
        median = float(np.median(values))
        # At least one bar of tail: a class whose median equals its floor still
        # has to be able to end.
        residual = max(1.0, median - floor)
        floors[i] = floor
        hazards[i] = 1.0 - 0.5 ** (1.0 / residual)
        stats[label] = {
            "n": int(values.size),
            "p05": float(np.percentile(values, 5.0)),
            "p25": float(np.percentile(values, 25.0)),
            "median": median,
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95.0)),
            "floor": int(floor),
            "residual_median": residual,
            "hazard": float(hazards[i]),
            "expected_bars": float(floor + (1.0 - hazards[i]) / hazards[i]),
        }
    return floors, hazards, stats


# --------------------------------------------------------------------------- #
# Corpus entry point
# --------------------------------------------------------------------------- #


def split_ids(data_dir, split: str = "train") -> list:
    """The frozen split's youtube ids.  Never regenerates the split file."""
    from .dataset import SPLITS_FILE
    path = Path(data_dir) / SPLITS_FILE
    if not path.exists():
        raise RuntimeError(
            f"missing {path} -- run training/nn/dataset.make_splits first; the "
            f"priors must be fitted on the frozen assignment, not a fresh one"
        )
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if split not in document:
        raise RuntimeError(f"{path} has no '{split}' split")
    return [str(i) for i in document[split]]


def fit(data_dir, *, split: str = "train", strict: bool = True, **kwargs) -> Priors:
    """Fit priors from one split of the corpus.

    ``split`` defaults to ``train`` and should stay there.  Val is the decoder
    sweep set and test is the benchmark; fitting the priors on either makes
    every number produced afterwards an in-sample number, and nothing
    downstream can tell.
    """
    data_dir = Path(data_dir)
    ids = split_ids(data_dir, split)
    sequences, skipped = corpus_bar_runs(data_dir, ids)
    if not sequences:
        raise RuntimeError(
            f"no usable tracks in split '{split}' of {data_dir} -- every one is "
            f"missing its beat grid or its annotation"
        )
    provenance = {
        "split": split,
        "split_size": len(ids),
        "skipped_no_beat_grid": skipped,
    }
    return fit_runs(sequences, strict=strict, provenance=provenance, **kwargs)


def priors_path(data_dir, version: str = MODEL_VERSION) -> Path:
    return Path(data_dir) / MODELS_DIR / version / PRIORS_FILE


def format_report(priors: Priors) -> str:
    """The fitted tables, as a human reads them.  ``.`` marks a -inf entry."""
    classes = priors.classes
    width = max(len(label) for label in classes)
    lines = []

    corpus = priors.corpus
    lines.append(f"fitted on {corpus.get('tracks', '?')} tracks / "
                 f"{corpus.get('runs', '?')} runs / {corpus.get('bars', '?')} bars "
                 f"(split={corpus.get('split', '?')})")
    if corpus.get("illegal_observed"):
        lines.append(f"  WARNING structurally illegal transitions observed: "
                     f"{corpus['illegal_observed']}")

    lines.append("")
    lines.append("initial P(first run):")
    lines.append("  " + "  ".join(f"{label}={priors.initial[i]:.4f}"
                                  for i, label in enumerate(classes)))

    lines.append("")
    lines.append("transition P(next run | this run)   ('.' = structurally impossible)")
    lines.append(" " * (width + 2) + "".join(f"{label:>11}" for label in classes))
    for i, src in enumerate(classes):
        cells = "".join("          ." if priors.transition[i, j] == 0.0
                        else f"{priors.transition[i, j]:>11.4f}"
                        for j in range(len(classes)))
        lines.append(f"  {src:<{width}}{cells}")

    lines.append("")
    lines.append("duration (bars): floor = corpus p05, then a memoryless tail "
                 "halving at the corpus median")
    lines.append(f"  {'class':<{width}}  {'n':>5} {'p05':>6} {'median':>7} "
                 f"{'p95':>6} {'floor':>6} {'hazard':>8} {'E[bars]':>8}")
    for i, label in enumerate(classes):
        stats = priors.corpus.get("duration_bars", {}).get(label, {})
        if not stats.get("n"):
            lines.append(f"  {label:<{width}}  {0:>5}")
            continue
        lines.append(
            f"  {label:<{width}}  {stats['n']:>5} {stats['p05']:>6.1f} "
            f"{stats['median']:>7.1f} {stats['p95']:>6.1f} "
            f"{int(priors.floor_bars[i]):>6} {priors.hazard[i]:>8.4f} "
            f"{stats['expected_bars']:>8.1f}")

    lines.append("")
    lines.append("class prior (share of corpus bars, for scaled likelihoods):")
    lines.append("  " + "  ".join(f"{label}={priors.class_prior[i]:.4f}"
                                  for i, label in enumerate(classes)))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    from build_training_table import default_data_dir
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split", default="train",
                        help="fit on this split (default: %(default)s -- moving "
                             "it off train leaks into every later metric)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"default: <data-dir>/{MODELS_DIR}/{MODEL_VERSION}/{PRIORS_FILE}")
    parser.add_argument("--allow-illegal", action="store_true",
                        help="record rather than refuse transitions the "
                             "structural graph calls impossible")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    priors = fit(args.data_dir, split=args.split, strict=not args.allow_illegal)
    out = Path(args.out) if args.out else priors_path(args.data_dir)
    priors.save(out)
    print(format_report(priors))
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
