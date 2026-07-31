"""Fixed-lag Viterbi decoder: posteriors in, immutable bar decisions out.

The network is the acoustic model; this is the *committer*.  It owns stability,
latency policy and every show-tunable knob, and it is the only component
permitted to say what the lights are doing.  Pure numpy over a cached posterior
array, so a whole-corpus parameter sweep costs seconds.

**Why an HSMM and not a smoother.**  The engine's vote buffer, min-dwell,
PEAK-promotion counter and invalid-transition veto were a hand-rolled degenerate
HMM with hand-chosen constants.  They are replaced here by one model whose
parameters are fitted from the corpus: transitions carry the structural graph
(illegal pairs are -inf, so the decoder routes around them instead of holding
state), and an explicit duration model carries persistence.  A single anomalous
bar cannot flip the show because leaving a state costs the duration prior, not
because a counter says so.

**The state space is (class, bars-so-far), capped at the class floor.**  Below
the floor a switch is impossible; above it the fitted tail is memoryless, so the
counter has nothing left to remember and saturates.  That turns an
explicit-duration HSMM into an ordinary Viterbi over ~48 states -- a few hundred
max-plus operations per bar, microseconds -- with no truncation error, because
the geometric tail genuinely is the model rather than an approximation of one.

**Immutability is enforced, not hoped for.**  A light show cannot un-fire a
strobe, so the freeze rule is structural: when the decision for bar B is read
off the backtrace at lag, every trellis state whose own history disagrees with
it is set to -inf.  Afterwards *every* surviving path passes through that
decision, so no amount of future audio can revise it -- and the emitted sequence
is still exactly one legal HSMM path, which is what keeps the min-duration and
-inf guarantees true globally rather than locally.

**The boundary head is a ranking score, not a probability.**  Task 2b measured
normalisation destroying most of its PR-AUC and the head was trained with
``pos_weight`` 44.8, so nothing may read it as P(boundary).  It enters as a
bounded *relative* hazard on switch transitions -- ``weight * (score - ref)``
added to the log-probability of changing state at that bar -- which uses its
ordering (its only trustworthy property) and never its calibration.  Boundaries
decide *where*; the label posteriors decide *what*.

**Bars, not frames.**  Duration priors are meaningful in bars, Raveform's
boundaries are downbeat-aligned by annotation policy, and DJ cue points cluster
on 8/16-bar multiples.  Grid-snapping -- not model localisation -- is what meets
the spec's 200 ms commit requirement.

**Thin evidence is refused, not guessed.**  The last ~1 s of a track is covered
only by the deliberately-unread outer margin of a single window (``coverage``
records it).  Those frames are dropped from the bar average; a bar with nothing
else left gets a flat emission, so the duration prior -- which prefers staying
-- carries it.  Hold-last-state, arrived at by the model rather than bolted on.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .priors import Priors

# ~5.6 s at the corpus median bar of 1.875 s: the spec's "5-6 s into the 8 s
# look-ahead budget", leaving margin for quantisation, inference cadence and
# actuation.  Sweepable in Task 5; this is the value the budget was designed to.
DEFAULT_LAG_BARS = 3

# Scaled likelihoods: the posterior is divided by ``class_prior ** strength``.
#
# 0.0, not 1.0, and the reason is the training objective rather than a number
# picked off the val split.  ``train.class_weights`` is inverse-frequency, so the
# focal loss already rebalanced the label head to a *uniform* effective prior --
# its softmax is close to a scaled likelihood before the decoder touches it.
# Dividing by the corpus occupancy on top of that applies the same correction
# twice, and a 12-track val peek shows exactly that: strength 1.0 drops per-bar
# accuracy from 71.5 % to 39.3 % by inflating the two rarest classes.  So the
# *residual* correction the decoder owes is zero, and the scalar stays as the
# sweep axis Task 5 calibrates -- positive divides further, negative puts corpus
# prior mass back, which is the direction this net would actually want.
DEFAULT_PRIOR_STRENGTH = 0.0

# Asymmetric commit cost: the odds a decoder will accept to avoid missing a
# drop, charged ONCE per drop run at the entry edge (see ``_commit_bonus`` -- a
# per-bar charge would compound with run length and make the number meaningless).
# 1.0 is neutral, so any drop-recall gain in a sweep is attributable to the knob
# rather than baked into the baseline.
DEFAULT_DROP_MISS_COST = 1.0

# The boundary hazard's neutral point and gain.  ref 0.5 is the sigmoid's own
# midpoint, so a score above it pays for a switch and below it charges for one;
# the bonus is bounded to +-weight/2, which keeps a miscalibrated head from ever
# overruling the label evidence outright.
DEFAULT_BOUNDARY_REF = 0.5
DEFAULT_BOUNDARY_WEIGHT = 2.0
# Half the annotation tolerance the boundary head was trained at (sigma 0.5 s).
DEFAULT_BOUNDARY_TOLERANCE_SEC = 0.5

# A frame with fewer than this many windows behind it is an unread window edge
# rather than evidence.  2 == "more than the one window that donated it".
DEFAULT_MIN_COVERAGE = 2

# Outro is TERMINAL in the fitted graph, and that is a true statement about the
# annotation: 0 of 525 train outros have a successor, so `transition_allowed`
# refuses `outro -> X` and the corpus never contradicts it.  It is nonetheless
# the single most expensive decision the decoder makes, because it is
# irrevocable and the decoder is not always right about when a track has ended.
# Measured on val-215: 11 100.6 s of exposure follows the first outro commit, of
# which 6 359.3 s (57.3 %) is NOT outro -- 21.3 % of all decoded error, across
# 73 of 215 tracks, from one absorbing state.
#
# So the escape is deliberately NOT fitted and NOT a refit: the data has nothing
# to fit it from, and making `outro -> X` merely *legal* would hand it the
# Jeffreys smoothing mass -- with zero observed outgoing counts that is 0.5 to
# each legal target, three orders of magnitude too much.  It is a decoder
# POLICY: an explicit per-bar probability of discovering the track was not over,
# swept like any other knob.  0.0 reproduces the terminal behaviour exactly.
DEFAULT_OUTRO_ESCAPE = 0.0

# Emission temperature on the frame posteriors before bar-averaging.  Distinct
# from `prior_strength`, which adds a class-constant offset AFTER the log: this
# scales the data term itself, so it changes how strongly a confident frame
# outvotes an unconfident one within the same bar.
DEFAULT_TEMPERATURE = 1.0

# Guards log(0) on a one-hot posterior without perturbing a real one.
EPS = 1e-12


class Decision(NamedTuple):
    """One immutable commit: bar index, class index, class name."""

    bar: int
    class_index: int
    label: str


@dataclass(frozen=True)
class DecodeParams:
    """Every knob, in one sweepable record.

    Frozen and comparable so a Task 5 sweep can use it as a dict key and record
    it verbatim next to the metric it produced.
    """

    lag_bars: int = DEFAULT_LAG_BARS
    class_prior_division: bool = True
    prior_strength: float = DEFAULT_PRIOR_STRENGTH
    drop_miss_cost: float = DEFAULT_DROP_MISS_COST
    boundary_weight: float = DEFAULT_BOUNDARY_WEIGHT
    boundary_ref: float = DEFAULT_BOUNDARY_REF
    boundary_tolerance_sec: float = DEFAULT_BOUNDARY_TOLERANCE_SEC
    min_coverage: int = DEFAULT_MIN_COVERAGE
    floor_scale: float = 1.0
    # Absolute per-class floors, overriding ``floor_scale`` when given.  A tuple
    # so the record stays hashable and a sweep can key on it.  The scalar cannot
    # express what the corpus asks for: breakdown's truth p25 run is 8 bars while
    # drop's is 16, so one multiplier cannot be right for both.
    floor_bars: tuple | None = None
    # Per-bar probability of leaving a committed ``outro`` for breakdown or drop
    # (each), 0.0 keeping outro terminal.  See ``_apply_outro_escape``.
    outro_escape: float = DEFAULT_OUTRO_ESCAPE
    # Softmax temperature applied to the frame posteriors before they are
    # averaged onto the bar grid.  >1 flattens, <1 sharpens; 1.0 is the
    # posterior as written.
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        """Normalise ``floor_bars`` to a tuple.

        JSON has no tuples, so a config read back off disk arrives holding a
        list -- which makes the record unhashable, and the sweep keys ``seen``
        on it.  The failure would land hours into a search rather than at the
        read, and only for the configs that carry a floor vector.  Coerced here
        rather than in the loader so every construction path shares the
        invariant.
        """
        if self.floor_bars is not None and not isinstance(self.floor_bars, tuple):
            object.__setattr__(self, "floor_bars", tuple(self.floor_bars))


# The config the show is meant to run on, committed rather than synthesised: a
# runtime that rebuilt `floors x 0.75` from a multiplier would be reconstructing
# the swept vector from a name, and the two have already diverged once (the
# scalar is inert once the vector is present). A sweep's own output lands beside
# its generation in the gitignored data directory; this is the one that ships.
SHIPPING_DECODER_CONFIG = Path(__file__).with_name("decoder_config.json")


def load_decoder_config(path) -> DecodeParams:
    """Read a sweep's ``decoder_config.json`` into the record it describes.

    An unknown key RAISES.  It used to be filtered out against
    ``dataclasses.fields``, which meant a config naming a knob this generation
    of the decoder does not have decoded silently with that knob absent -- and
    the shipped pick carries three that master did not have, so the whole file
    would have loaded, run, and reported a decoder nobody chose.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    chosen = document.get("chosen", document)
    known = {field.name for field in dataclasses.fields(DecodeParams)}
    unknown = sorted(set(chosen) - known)
    if unknown:
        raise ValueError(
            f"{path}: decoder config names {', '.join(unknown)}, which "
            f"DecodeParams does not have -- dropping them would decode with a "
            f"different decoder than the one this config was measured on. "
            f"Known knobs: {', '.join(sorted(known))}")
    return DecodeParams(**chosen)


# --------------------------------------------------------------------------- #
# The decoder
# --------------------------------------------------------------------------- #


class FixedLagViterbi:
    """Bar-rate HSMM decoder that commits once, at a fixed lag, and never revises.

    ``push`` one bar's posterior at a time and it returns the decisions that
    became final because of it -- the runtime shape.  ``decode`` runs a whole
    cached track through the same code path, which is what makes an offline
    sweep and the live engine provably the same decoder.
    """

    def __init__(self, priors: Priors, lag_bars: int = DEFAULT_LAG_BARS,
                 class_prior_division: bool = True,
                 drop_miss_cost: float = DEFAULT_DROP_MISS_COST, *,
                 prior_strength: float = DEFAULT_PRIOR_STRENGTH,
                 boundary_weight: float = DEFAULT_BOUNDARY_WEIGHT,
                 boundary_ref: float = DEFAULT_BOUNDARY_REF,
                 floor_scale: float = 1.0,
                 floor_bars=None,
                 outro_escape: float = DEFAULT_OUTRO_ESCAPE) -> None:
        if int(lag_bars) < 0:
            raise ValueError(f"lag_bars must be >= 0, got {lag_bars}")
        if float(floor_scale) <= 0.0:
            raise ValueError(f"floor_scale must be > 0, got {floor_scale}")
        if float(drop_miss_cost) <= 0.0:
            raise ValueError(f"drop_miss_cost must be > 0, got {drop_miss_cost}")
        # Two escapes at 0.5 would leave no probability to stay, and anything
        # above that is a negative stay probability -- nonsense rather than an
        # aggressive setting, so it is refused rather than clipped.
        if not 0.0 <= float(outro_escape) < 0.5:
            raise ValueError(
                f"outro_escape must lie in [0, 0.5) -- got {outro_escape}; it is "
                f"a per-bar probability and TWO targets are opened, so the stay "
                f"probability is 1 - 2 * escape")

        self.priors = priors
        self.classes = tuple(priors.classes)
        self.lag_bars = int(lag_bars)
        self.class_prior_division = bool(class_prior_division)
        self.prior_strength = float(prior_strength)
        self.drop_miss_cost = float(drop_miss_cost)
        self.boundary_weight = float(boundary_weight)
        self.boundary_ref = float(boundary_ref)
        self.floor_scale = float(floor_scale)
        self.outro_escape = float(outro_escape)
        self.floor_bars = tuple(floor_bars) if floor_bars is not None else None

        hazard = np.asarray(priors.hazard, dtype=np.float64)
        if not np.all((hazard > 0.0) & (hazard < 1.0)):
            raise ValueError(
                f"every class hazard must lie in (0, 1) -- got {hazard.tolist()}; "
                f"0 makes a class unleavable and 1 caps every run at its floor")

        if self.floor_bars is not None:
            if len(self.floor_bars) != len(self.classes):
                raise ValueError(
                    f"floor_bars has {len(self.floor_bars)} entries for "
                    f"{len(self.classes)} classes {self.classes} -- a floor "
                    f"vector silently misaligned by one would apply drop's floor "
                    f"to breakdown and never raise")
            self._floors = np.maximum(
                1, np.asarray(self.floor_bars, dtype=np.int64))
        else:
            self._floors = np.maximum(
                1, np.rint(np.asarray(priors.floor_bars, dtype=np.float64)
                           * self.floor_scale).astype(np.int64))
        self._emission_bonus = self._class_bonus()
        self._entry_bonus = self._commit_bonus()
        self._build_states()
        self.reset()

    # -- construction ------------------------------------------------------- #

    def _build_states(self) -> None:
        """Enumerate ``(class, bars-so-far)`` and precompute the log-transitions.

        ``bars-so-far`` saturates at the class floor: past it the fitted tail is
        memoryless, so every longer run is the same state.  The switch entries
        are kept as a mask because the boundary hazard is added to exactly those
        and to nothing else.

        The asymmetric commit costs are folded into the *entry* edges here --
        every arc that starts a run of class c, plus c's initial probability --
        so a run pays its cost exactly once no matter how long it turns out to
        be.  See ``_commit_bonus``.
        """
        floors = self._floors
        # States are laid out contiguously per class in dwell order, which is
        # what lets "one more bar in the same class" be ``state + 1``.
        self._state_class = np.concatenate(
            [np.full(int(f), c, dtype=np.int64) for c, f in enumerate(floors)])
        offsets = np.concatenate([[0], np.cumsum(floors)])
        self._entry_state = offsets[:-1].copy()          # (c, 0), the run's first bar
        self._final_state = offsets[1:] - 1              # (c, floor - 1), saturated
        n_states = int(offsets[-1])

        log_transition = self.priors.log_transition
        hazard = np.clip(np.asarray(self.priors.hazard, dtype=np.float64), EPS, 1.0)
        with np.errstate(divide="ignore"):
            log_stay = np.log(1.0 - hazard)
            log_leave = np.log(hazard)

        transition = np.full((n_states, n_states), -np.inf, dtype=np.float64)
        switch = np.zeros((n_states, n_states), dtype=bool)
        for state in range(n_states):
            c = int(self._state_class[state])
            saturated = state == self._final_state[c]
            if not saturated:
                # Below the floor a switch is impossible, so continuing is free.
                transition[state, state + 1] = 0.0
            else:
                transition[state, state] = log_stay[c]
                for other in range(len(self.classes)):
                    if other == c:
                        continue
                    target = int(self._entry_state[other])
                    transition[state, target] = (log_leave[c] + log_transition[c, other]
                                                 + self._entry_bonus[other])
                    switch[state, target] = True

        self._apply_outro_escape(transition, switch)

        self._transition = transition
        self._switch = switch
        self._log_initial = np.full(n_states, -np.inf, dtype=np.float64)
        self._log_initial[self._entry_state] = (self.priors.log_initial
                                                + self._entry_bonus)

    # Where a track that was declared over can turn out not to be.  Only the two
    # classes a track can plausibly resume into: an outro back into intro or
    # buildup is not a mis-commit being repaired, it is a different track.
    ESCAPE_TARGETS = ("breakdown", "drop")

    def _apply_outro_escape(self, transition: np.ndarray, switch: np.ndarray) -> None:
        """Open a bounded way out of a committed ``outro``.

        The fitted graph makes outro absorbing and the corpus agrees with it, so
        this does not touch ``priors`` at all -- it rewrites the trellis arc that
        the absorbing state is made of.  The escape probability is stated
        DIRECTLY as a per-bar rate rather than routed through the fitted hazard,
        because ``hazard[outro]`` is fitted from runs that never end and means
        nothing here; going through it would make the knob's real value depend on
        a number nobody calibrated.

        Deliberately not symmetric with the other classes: leaving outro costs
        ``log(escape)`` flat, with no ``log_transition`` term, because there is no
        fitted distribution over outro's successors to consult.  The entry bonus
        still applies, so a drop-biased config biases the escape too.
        """
        if self.outro_escape <= 0.0 or "outro" not in self.classes:
            return
        source_class = self.classes.index("outro")
        source = int(self._final_state[source_class])
        targets = [self.classes.index(name) for name in self.ESCAPE_TARGETS
                   if name in self.classes]
        if not targets:
            return

        stay = 1.0 - self.outro_escape * len(targets)
        transition[source, source] = math.log(stay) if stay > 0.0 else -np.inf
        for other in targets:
            entry = int(self._entry_state[other])
            transition[source, entry] = (math.log(self.outro_escape)
                                         + self._entry_bonus[other])
            switch[source, entry] = True

    def _class_bonus(self) -> np.ndarray:
        """Per-class additive log-score applied to every bar's **emission**.

        Only the scaled-likelihood correction belongs here, because it is a
        statement about each individual observation.  ``prior_strength`` is
        deliberately signed: this net was trained with inverse-frequency class
        weights, so a *positive* strength corrects an imbalance that is already
        corrected, and a negative one hands corpus prior mass back -- which is
        the direction a doubly-rebalanced head actually wants.  Clamping it at
        zero would hide the useful half of the axis from a sweep.
        """
        bonus = np.zeros(len(self.classes), dtype=np.float64)
        if self.class_prior_division and self.prior_strength:
            prior = np.clip(np.asarray(self.priors.class_prior, dtype=np.float64),
                            EPS, None)
            bonus -= self.prior_strength * np.log(prior)
        return bonus

    def _commit_bonus(self) -> np.ndarray:
        """Per-class additive log-score applied **once, when a run is entered**.

        The spec puts the asymmetric error costs "at the commit step", and a
        commit is a *run*, not a bar.  Putting ``log(drop_miss_cost)`` on the
        emission instead compounds it with run length: at 1.5 the effective bias
        over drop's 16-bar floor is x657 and at 3.0 it is x4.3e7, so a nominal
        [1, 3] sweep is really a sweep from "neutral" to "everything is a drop"
        -- measured as drop occupancy 41 % -> 64 % while accuracy *fell*.  Folded
        into the entry edges the number means what it says: a decoder that pays
        3x for a missed drop will take a 3:1 odds bet to avoid one, once per
        drop, whether that drop lasts 16 bars or 60.
        """
        bonus = np.zeros(len(self.classes), dtype=np.float64)
        if "drop" in self.classes and self.drop_miss_cost != 1.0:
            bonus[self.classes.index("drop")] += np.log(self.drop_miss_cost)
        return bonus

    # -- streaming ---------------------------------------------------------- #

    def reset(self) -> None:
        """Forget the track.  A reset decoder decodes exactly like a fresh one."""
        self._delta = None
        self._psi: list = []
        self._bars = 0
        self._next_commit = 0

    def push(self, posterior, boundary=None) -> list:
        """Advance one bar; return the decisions that became final.

        ``posterior`` is the bar's aggregated class distribution (it is
        renormalised, so unnormalised weights are fine).  ``None``, all-NaN, or
        a zero row means *no usable evidence at this bar* -- the emission goes
        flat and the duration prior carries the state through.  ``boundary`` is
        the raw boundary score at this bar's downbeat, or ``None`` for no
        modulation.
        """
        emission = self._emission(posterior)
        bar = self._bars

        if bar == 0:
            self._delta = self._log_initial + emission
            self._psi.append(np.full(len(emission), -1, dtype=np.int64))
        else:
            step = self._transition
            bonus = self._switch_bonus(boundary)
            if bonus:
                step = step + bonus * self._switch
            scored = self._delta[:, None] + step
            back = scored.argmax(axis=0)
            self._delta = scored[back, np.arange(scored.shape[1])] + emission
            self._psi.append(back)

        self._bars += 1
        return self._commit_due()

    def flush(self) -> list:
        """Force out every bar still inside the lag window.  Idempotent."""
        if self._delta is None or self._next_commit >= self._bars:
            return []
        return self._commit_through(self._bars - 1)

    def decode(self, posteriors, boundary=None) -> list:
        """Decode a whole cached track through the streaming path.

        Not a separate batch implementation: an offline sweep that disagreed
        with the runtime by one line of code would be measuring the wrong
        decoder.
        """
        self.reset()
        posteriors = np.asarray(posteriors, dtype=np.float64)
        if posteriors.ndim != 2 or posteriors.shape[1] != len(self.classes):
            raise ValueError(
                f"posteriors must be [bars, {len(self.classes)}], got "
                f"{posteriors.shape}")
        scores = None if boundary is None else np.asarray(boundary, dtype=np.float64)
        if scores is not None and len(scores) != len(posteriors):
            raise ValueError(
                f"boundary has {len(scores)} entries for {len(posteriors)} bars")

        decisions: list = []
        for bar, row in enumerate(posteriors):
            decisions.extend(self.push(row, None if scores is None else scores[bar]))
        decisions.extend(self.flush())
        return decisions

    # -- internals ---------------------------------------------------------- #

    def _emission(self, posterior) -> np.ndarray:
        """Per-state log-emission for one bar (flat when there is no evidence)."""
        n_states = len(self._state_class)
        if posterior is None:
            return np.zeros(n_states, dtype=np.float64)
        row = np.asarray(posterior, dtype=np.float64).reshape(-1)
        if row.size != len(self.classes):
            raise ValueError(
                f"posterior must have {len(self.classes)} entries, got {row.size}")
        if np.any(row < 0.0):
            # log() of a negative turns the whole trellis into NaN, and NaN
            # compares false against every -inf, so the decode would come back
            # confidently wrong instead of failing.
            raise ValueError(f"posterior has negative entries: {row.tolist()}")
        total = row.sum()
        if not np.isfinite(total) or total <= 0.0:
            # Thin evidence: a flat emission leaves the duration prior in charge,
            # and the duration prior prefers staying -- hold-last-state, derived
            # from the model rather than special-cased around it.
            return np.zeros(n_states, dtype=np.float64)
        scores = np.log(row / total + EPS) + self._emission_bonus
        return scores[self._state_class]

    def _switch_bonus(self, boundary) -> float:
        """Log-multiplier on every switch transition into this bar.

        Deliberately *not* renormalised against the stay probability: this is a
        hazard, not a re-weighted distribution.  A score above the reference
        makes changing state cheaper, one below makes it dearer, and either way
        the effect is bounded by ``boundary_weight / 2`` -- so a miscalibrated
        head (and this one is a ranking score, not a probability) can move where
        a switch lands but can never manufacture one the label evidence does not
        support.  A NaN score, which ``bar_observations`` produces for a bar with
        no reliable frames, is no evidence rather than evidence of no boundary.
        """
        if boundary is None or not self.boundary_weight:
            return 0.0
        score = float(boundary)
        if not np.isfinite(score):
            return 0.0
        return self.boundary_weight * (score - self.boundary_ref)

    def _commit_due(self) -> list:
        target = self._bars - 1 - self.lag_bars
        if target < self._next_commit:
            return []
        return self._commit_through(target)

    def _commit_through(self, target: int) -> list:
        """Emit bars ``_next_commit .. target`` off one backtrace, then prune.

        Pruning is the whole point.  Reading the backtrace alone is a *guess*
        that the current best path will survive; setting every disagreeing state
        to -inf makes it a fact, because from here on no path exists that says
        anything else about those bars.
        """
        ancestors = self._ancestors(self._bars - 1, self._next_commit)
        best = int(np.argmax(self._delta))
        decisions: list = []
        for bar in range(self._next_commit, target + 1):
            state = int(ancestors[bar - self._next_commit][best])
            index = int(self._state_class[state])
            decisions.append(Decision(bar, index, self.classes[index]))

        committed = int(self._state_class[ancestors[target - self._next_commit][best]])
        self._delta = np.where(
            self._state_class[ancestors[target - self._next_commit]] == committed,
            self._delta, -np.inf)
        self._next_commit = target + 1
        return decisions

    def _ancestors(self, frm: int, to: int) -> list:
        """``[bar_to .. bar_frm]`` ancestor state of every current state.

        One vectorised hop per bar: ``psi`` maps a state at bar k to its
        predecessor at k-1, so walking it backwards for all states at once costs
        an array index per bar rather than a loop per state.  Only the last
        ``lag_bars + 1`` entries are ever read (nothing before a commit can be
        revisited), so a live decoder can hold ``_psi`` in a ring buffer of that
        size; offline the whole history is a few hundred kilobytes per track and
        keeping it makes the indexing bar-absolute and obvious.
        """
        index = np.arange(len(self._state_class), dtype=np.int64)
        chain = [index]
        for bar in range(frm, to, -1):
            index = self._psi[bar][index]
            chain.append(index)
        return chain[::-1]


def segments(decisions) -> list:
    """Run-length encode a decision stream into ``[(first_bar, end_bar, label)]``."""
    spans: list = []
    for decision in decisions:
        if spans and spans[-1][2] == decision.label and spans[-1][1] == decision.bar:
            spans[-1][1] = decision.bar + 1
        else:
            spans.append([decision.bar, decision.bar + 1, decision.label])
    return [(start, end, label) for start, end, label in spans]


# --------------------------------------------------------------------------- #
# Track adapters
# --------------------------------------------------------------------------- #


def bar_grid(beat_csv_path) -> np.ndarray:
    """Downbeat times from a Raveform beat CSV, plus a closing edge.

    ``B`` bars need ``B + 1`` edges, and the corpus grid ends on the last
    downbeat rather than on a bar line, so the final bar is closed at the median
    bar length.  Using the median rather than the last observed interval keeps
    one mis-tracked beat at the very end of a track from producing a bar that
    swallows (or drops) several seconds of posterior.
    """
    path = Path(beat_csv_path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        downbeats = [float(row["time"]) for row in csv.DictReader(handle)
                     if int(row["downbeat"]) == 1]
    if len(downbeats) < 2:
        raise RuntimeError(
            f"{path}: fewer than two downbeat rows -- there is no bar grid to "
            f"decode on (the decoder runs at bar rate by design)")
    edges = np.asarray(sorted(downbeats), dtype=np.float64)
    return np.append(edges, edges[-1] + float(np.median(np.diff(edges))))


def temper(label_post: np.ndarray, temperature: float) -> np.ndarray:
    """Re-sharpen or flatten frame posteriors: ``p ** (1/T)``, renormalised.

    Applied per FRAME, before the bar average, and the order is the whole point.
    Tempering after the mean would only rescale a number the averaging has
    already decided; tempering before it changes how much a confident frame
    outvotes an unconfident one *within* the bar, which is where a 16-bar drop
    with two ambiguous bars at its edge is won or lost.

    Distinct from ``prior_strength``, which is a class-constant offset added
    after the log and therefore cannot change the relative weight of two frames
    of the same class.
    """
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if temperature == 1.0:
        return label_post
    powered = np.power(np.clip(label_post, EPS, None), 1.0 / temperature)
    return powered / powered.sum(axis=-1, keepdims=True)


def bar_observations(posterior_npz, edges, *, min_coverage: int = DEFAULT_MIN_COVERAGE,
                     boundary_tolerance_sec: float = DEFAULT_BOUNDARY_TOLERANCE_SEC,
                     temperature: float = DEFAULT_TEMPERATURE) -> tuple:
    """Aggregate a posterior sidecar onto a bar grid.

    Returns ``(posteriors [B, C], boundary [B])``.  A bar with no frame that
    more than one window voted on is returned as all-NaN -- the sidecar's
    ``coverage`` array is the only thing that distinguishes the confident middle
    of a track from its unread edges, and a decoder given a confident-looking
    average of edge frames would commit on the thinnest evidence in the file.

    The two grids have different origins by design (``label_post`` is stamped at
    the END of its pooled group), so each is read at its own timestamps rather
    than at a shared one.
    """
    with np.load(posterior_npz) as archive:
        label_post = np.asarray(archive["label_post"], dtype=np.float64)
        boundary = np.asarray(archive["boundary"], dtype=np.float64)
        coverage = np.asarray(archive["coverage"], dtype=np.int64)
        frame_sec = float(archive["frame_sec"])
        label_frame_sec = float(archive["label_frame_sec"])
        label_t0 = float(archive["label_t0"])
        label_pool = int(archive["label_pool"])
        t0 = float(archive["t0"])

    label_post = temper(label_post, temperature)

    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("need at least two bar edges to define one bar")
    n_bars = edges.size - 1

    label_times = label_t0 + np.arange(len(label_post)) * label_frame_sec
    label_ok = coverage[::label_pool][:len(label_post)] >= int(min_coverage)
    frame_times = t0 + np.arange(len(boundary)) * frame_sec
    frame_ok = coverage[:len(boundary)] >= int(min_coverage)

    # A track where NOTHING clears the threshold is not a thin-evidence track,
    # it is a mispaired config -- and it decodes *silently*, because every bar
    # falls back to the duration prior and the result still looks like a show.
    # The case that motivated this: a whole-track model writes coverage 1
    # everywhere (it has no windows to average), and the shipped configs ask for
    # 2, so 100% of the label evidence is discarded and the reported macro-F1
    # measures the priors.  Nothing downstream can tell that apart from a bad
    # model.
    if not label_ok.any():
        raise RuntimeError(
            f"{posterior_npz}: no frame has coverage >= {min_coverage} "
            f"(max is {int(coverage.max()) if coverage.size else 0}) -- every "
            f"bar would decode from the duration prior alone. Either these "
            f"sidecars were written by a single-pass model (coverage 1) and the "
            f"config wants overlap-averaged ones, or the two were mispaired")

    posteriors = np.full((n_bars, label_post.shape[1]), np.nan, dtype=np.float64)
    scores = np.full(n_bars, np.nan, dtype=np.float64)

    lo = np.searchsorted(label_times, edges[:-1], side="left")
    hi = np.searchsorted(label_times, edges[1:], side="left")
    for bar in range(n_bars):
        window = label_ok[lo[bar]:hi[bar]]
        if window.any():
            posteriors[bar] = label_post[lo[bar]:hi[bar]][window].mean(axis=0)

    # The boundary is read at the bar LINE, not over the bar: it answers "does a
    # section change here", and the head was trained with a 0.5 s Gaussian, so a
    # tolerance window either side of the downbeat is the matching read.
    starts = np.searchsorted(frame_times, edges[:-1] - boundary_tolerance_sec, "left")
    ends = np.searchsorted(frame_times, edges[:-1] + boundary_tolerance_sec, "right")
    for bar in range(n_bars):
        window = frame_ok[starts[bar]:ends[bar]]
        if window.any():
            scores[bar] = boundary[starts[bar]:ends[bar]][window].max()
    return posteriors, scores


def decode_track(posterior_npz, beat_csv, params: DecodeParams | None = None, *,
                 priors: Priors) -> list:
    """One track, end to end: ``[(bar_start_seconds, label), ...]``.

    The convenience Task 5 sweeps and Task 6 evaluates through.  Bar-stamped
    rather than run-length encoded on purpose -- the evaluator wants a decision
    per grid position, and ``segments`` collapses it when a timeline is what is
    wanted instead.

    ``priors`` is required.  Guessing them from the sidecar's directory layout
    put a second, silent notion of "which generation is this" beside the
    ``--model-version`` every caller already passes -- and the guess was pinned
    to v1, so a v2 sweep would have decoded against v1 priors without a word.
    """
    params = params or DecodeParams()
    edges = bar_grid(beat_csv)
    posteriors, boundary = bar_observations(
        posterior_npz, edges, min_coverage=params.min_coverage,
        boundary_tolerance_sec=params.boundary_tolerance_sec,
        temperature=params.temperature)

    decoder = FixedLagViterbi(
        priors, params.lag_bars,
        class_prior_division=params.class_prior_division,
        drop_miss_cost=params.drop_miss_cost,
        prior_strength=params.prior_strength,
        boundary_weight=params.boundary_weight,
        boundary_ref=params.boundary_ref,
        floor_scale=params.floor_scale,
        floor_bars=params.floor_bars,
        outro_escape=params.outro_escape)
    decisions = decoder.decode(posteriors, boundary)
    return [(float(edges[d.bar]), d.label) for d in decisions]
