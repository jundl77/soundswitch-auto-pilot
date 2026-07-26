"""Tests for the raveform manifest's canonical-label fold (training/raveform_manifest.py).

The canonical fold is the definition of the training vocabulary: it decides
which labels exist, how long each section is, and therefore what the duration
and transition priors are trained on.  A silent regression here would not crash
anything -- it would quietly train a different model, so it is worth pinning.
"""
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from raveform_manifest import (  # noqa: E402  (needs the path insert above)
    CANONICAL_DROP,
    CANONICAL_MAP,
    canonical_runs,
    raw_runs,
    section_length,
)


# --------------------------------------------------------------------------- #
# section_length
# --------------------------------------------------------------------------- #


def test_section_length_is_the_plain_difference():
    assert section_length(1.0, 3.5) == 2.5


def test_section_length_of_an_empty_section_is_zero():
    assert section_length(4.0, 4.0) == 0.0


def test_section_length_clamps_a_negative_section_to_zero():
    # The real anomaly (1020.c1VBubZ2w3M) is a final section whose start exceeds
    # its end by 0.6 ms.  It must never subtract from an aggregate.
    assert section_length(5.0006, 5.0) == 0.0


def test_section_length_clamps_a_grossly_negative_section_to_zero():
    assert section_length(90.0, 10.0) == 0.0


# --------------------------------------------------------------------------- #
# canonical_runs -- drop
# --------------------------------------------------------------------------- #


def test_canonical_runs_drops_the_end_sentinel():
    sections = [(0.0, 100.0, "intro"), (100.0, 104.8, "end")]
    assert canonical_runs(sections) == [(0.0, 100.0, "intro", 100.0)]


def test_canonical_runs_of_only_a_sentinel_is_empty():
    assert canonical_runs([(0.0, 4.8, "end")]) == []


def test_canonical_runs_of_no_sections_is_empty():
    assert canonical_runs([]) == []


def test_end_is_the_only_dropped_label():
    # A second dropped label would remove time from the corpus without anyone
    # deciding to; the ruling drops exactly one.
    assert CANONICAL_DROP == frozenset({"end"})


# --------------------------------------------------------------------------- #
# canonical_runs -- fold
# --------------------------------------------------------------------------- #


def test_canonical_runs_folds_altintro_into_intro():
    assert canonical_runs([(0.0, 8.0, "altintro")]) == [(0.0, 8.0, "intro", 8.0)]


def test_canonical_runs_folds_bridge_into_breakdown():
    assert canonical_runs([(0.0, 8.0, "bridge")]) == [(0.0, 8.0, "breakdown", 8.0)]


def test_canonical_runs_keeps_altoutro_unfolded():
    # altoutro is part of the documented seven-label vocabulary; folding it away
    # would shrink the vocabulary the model is trained on.
    assert canonical_runs([(0.0, 8.0, "altoutro")]) == [(0.0, 8.0, "altoutro", 8.0)]
    assert "altoutro" not in CANONICAL_MAP


def test_fold_map_is_exactly_the_two_ruled_variants():
    assert CANONICAL_MAP == {"altintro": "intro", "bridge": "breakdown"}


# --------------------------------------------------------------------------- #
# canonical_runs -- merge
# --------------------------------------------------------------------------- #


def test_canonical_runs_merges_adjacent_same_label_sections():
    sections = [(0.0, 10.0, "drop"), (10.0, 25.0, "drop")]
    assert canonical_runs(sections) == [(0.0, 25.0, "drop", 25.0)]


def test_canonical_runs_merges_across_a_fold():
    # altintro then intro is one intro: the fold happens before the merge.
    sections = [(0.0, 5.0, "altintro"), (5.0, 9.0, "intro")]
    assert canonical_runs(sections) == [(0.0, 9.0, "intro", 9.0)]


def test_canonical_runs_keeps_different_labels_apart():
    sections = [(0.0, 10.0, "intro"), (10.0, 20.0, "buildup"), (20.0, 30.0, "drop")]
    assert canonical_runs(sections) == [
        (0.0, 10.0, "intro", 10.0),
        (10.0, 20.0, "buildup", 10.0),
        (20.0, 30.0, "drop", 10.0),
    ]


def test_canonical_runs_folds_then_merges_a_whole_track():
    sections = [
        (0.0, 5.0, "altintro"),
        (5.0, 9.0, "intro"),
        (9.0, 15.0, "bridge"),
        (15.0, 20.0, "breakdown"),
        (20.0, 30.0, "drop"),
        (30.0, 34.0, "end"),
    ]
    assert canonical_runs(sections) == [
        (0.0, 9.0, "intro", 9.0),
        (9.0, 20.0, "breakdown", 11.0),
        (20.0, 30.0, "drop", 10.0),
    ]


# --------------------------------------------------------------------------- #
# canonical_runs -- summed-duration semantics
# --------------------------------------------------------------------------- #


def test_merged_run_joins_across_a_dropped_sentinel():
    sections = [(0.0, 10.0, "drop"), (10.0, 12.0, "end"), (12.0, 20.0, "drop")]
    assert [run[:3] for run in canonical_runs(sections)] == [(0.0, 20.0, "drop")]


def test_merged_duration_sums_members_and_excludes_the_dropped_span():
    # This is the whole point of carrying a separate duration: the run spans
    # 0..20 but only 18 s of it were labelled `drop`.  Re-deriving the duration
    # as end - start would silently re-attribute the dropped sentinel's 2 s.
    sections = [(0.0, 10.0, "drop"), (10.0, 12.0, "end"), (12.0, 20.0, "drop")]
    (start, end, label, duration), = canonical_runs(sections)
    assert (start, end, label) == (0.0, 20.0, "drop")
    assert duration == 18.0
    assert duration != end - start


def test_merged_duration_ignores_a_negative_member():
    # A negative-length member contributes 0 s, not negative time.
    sections = [(0.0, 10.0, "drop"), (10.0, 9.9994, "drop")]
    (_start, _end, _label, duration), = canonical_runs(sections)
    assert duration == 10.0


def test_unmerged_run_duration_matches_its_span():
    (start, end, _label, duration), = canonical_runs([(3.0, 11.0, "drop")])
    assert duration == end - start == 8.0


def test_canonical_runs_returns_tuples_not_internal_lists():
    # The builder accumulates in lists; leaking them would let a caller mutate
    # a run in place and corrupt the statistics computed from it.
    assert all(isinstance(run, tuple) for run in canonical_runs([(0.0, 1.0, "drop")]))


# --------------------------------------------------------------------------- #
# raw_runs
# --------------------------------------------------------------------------- #


def test_raw_runs_neither_drops_nor_folds_nor_merges():
    sections = [(0.0, 5.0, "altintro"), (5.0, 9.0, "intro"), (9.0, 13.0, "end")]
    assert raw_runs(sections) == [
        (0.0, 5.0, "altintro", 5.0),
        (5.0, 9.0, "intro", 4.0),
        (9.0, 13.0, "end", 4.0),
    ]


def test_raw_runs_clamps_negative_lengths_too():
    assert raw_runs([(5.0006, 5.0, "outro")]) == [(5.0006, 5.0, "outro", 0.0)]
