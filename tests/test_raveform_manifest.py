import sys
from pathlib import Path

RAVEFORM_DIR = Path(__file__).resolve().parents[1] / "training" / "raveform"
if str(RAVEFORM_DIR) not in sys.path:
    sys.path.insert(0, str(RAVEFORM_DIR))

import pytest  # noqa: E402

import raveform_manifest  # noqa: E402
from lib.label_space import DROPPED_LABELS  # noqa: E402
from raveform_manifest import (  # noqa: E402
    raw_runs,
    section_length,
    section_runs,
)


def test_section_length_is_the_plain_difference():
    assert section_length(1.0, 3.5) == 2.5


def test_section_length_of_an_empty_section_is_zero():
    assert section_length(4.0, 4.0) == 0.0


def test_section_length_clamps_a_negative_section_to_zero():
    # Real annotations contain sub-millisecond negative sections (1020.c1VBubZ2w3M).
    assert section_length(5.0006, 5.0) == 0.0


def test_section_length_clamps_a_grossly_negative_section_to_zero():
    assert section_length(90.0, 10.0) == 0.0


def test_section_runs_drops_the_end_sentinel():
    sections = [(0.0, 100.0, "intro"), (100.0, 104.8, "end")]
    assert section_runs(sections) == [(0.0, 100.0, "intro", 100.0)]


def test_section_runs_of_only_a_sentinel_is_empty():
    assert section_runs([(0.0, 4.8, "end")]) == []


def test_section_runs_of_no_sections_is_empty():
    assert section_runs([]) == []


def test_end_is_the_only_dropped_label():
    assert DROPPED_LABELS == frozenset({"end"})


@pytest.mark.parametrize("label", ["altintro", "bridge", "cooldown", "altoutro"])
def test_a_formerly_folded_label_survives_as_itself(label):
    assert section_runs([(0.0, 8.0, label)]) == [(0.0, 8.0, label, 8.0)]


def test_the_fold_machinery_is_gone_not_merely_unused():
    for retired in ("CANONICAL_MAP", "CANONICAL_ORDER", "CANONICAL_DROP",
                    "canonical_runs"):
        assert not hasattr(raveform_manifest, retired), retired


def test_section_runs_merges_adjacent_same_label_sections():
    sections = [(0.0, 10.0, "drop"), (10.0, 25.0, "drop")]
    assert section_runs(sections) == [(0.0, 25.0, "drop", 25.0)]


def test_a_formerly_folded_pair_is_now_two_runs_with_a_boundary_between():
    sections = [(0.0, 5.0, "altintro"), (5.0, 9.0, "intro")]
    assert section_runs(sections) == [
        (0.0, 5.0, "altintro", 5.0),
        (5.0, 9.0, "intro", 4.0),
    ]


def test_section_runs_keeps_different_labels_apart():
    sections = [(0.0, 10.0, "intro"), (10.0, 20.0, "buildup"), (20.0, 30.0, "drop")]
    assert section_runs(sections) == [
        (0.0, 10.0, "intro", 10.0),
        (10.0, 20.0, "buildup", 10.0),
        (20.0, 30.0, "drop", 10.0),
    ]


def test_section_runs_drops_then_merges_a_whole_track():
    sections = [
        (0.0, 5.0, "altintro"),
        (5.0, 9.0, "intro"),
        (9.0, 15.0, "bridge"),
        (15.0, 20.0, "breakdown"),
        (20.0, 30.0, "drop"),
        (30.0, 34.0, "end"),
    ]
    assert section_runs(sections) == [
        (0.0, 5.0, "altintro", 5.0),
        (5.0, 9.0, "intro", 4.0),
        (9.0, 15.0, "bridge", 6.0),
        (15.0, 20.0, "breakdown", 5.0),
        (20.0, 30.0, "drop", 10.0),
    ]


def test_merged_run_joins_across_a_dropped_sentinel():
    sections = [(0.0, 10.0, "drop"), (10.0, 12.0, "end"), (12.0, 20.0, "drop")]
    assert [run[:3] for run in section_runs(sections)] == [(0.0, 20.0, "drop")]


def test_merged_duration_sums_members_and_excludes_the_dropped_span():
    sections = [(0.0, 10.0, "drop"), (10.0, 12.0, "end"), (12.0, 20.0, "drop")]
    (start, end, label, duration), = section_runs(sections)
    assert (start, end, label) == (0.0, 20.0, "drop")
    assert duration == 18.0
    assert duration != end - start


def test_merged_duration_ignores_a_negative_member():
    sections = [(0.0, 10.0, "drop"), (10.0, 9.9994, "drop")]
    (_start, _end, _label, duration), = section_runs(sections)
    assert duration == 10.0


def test_unmerged_run_duration_matches_its_span():
    (start, end, _label, duration), = section_runs([(3.0, 11.0, "drop")])
    assert duration == end - start == 8.0


def test_section_runs_returns_tuples_not_internal_lists():
    assert all(isinstance(run, tuple) for run in section_runs([(0.0, 1.0, "drop")]))


def test_raw_runs_neither_drops_nor_merges():
    sections = [(0.0, 5.0, "altintro"), (5.0, 9.0, "intro"), (9.0, 13.0, "end")]
    assert raw_runs(sections) == [
        (0.0, 5.0, "altintro", 5.0),
        (5.0, 9.0, "intro", 4.0),
        (9.0, 13.0, "end", 4.0),
    ]


def test_raw_runs_clamps_negative_lengths_too():
    assert raw_runs([(5.0006, 5.0, "outro")]) == [(5.0006, 5.0, "outro", 0.0)]
