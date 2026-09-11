"""The #348 arc read asks how long an intent was IN FORCE.

Not where a block started.  Intent blocks are spans: a block stamped
``song_t`` holds until the next block's, and the last block in a report holds
to the end of the track.  The instrument's first reading looked for a block
*starting* inside the labelled window, so a chain that entered DROP before the
window and held it at full power across the whole labelled drop scored as
having LOST that drop.  That is
the false finding on ``DCLNE1S7_opus`` -- blocks at 254.740 breakdown, 267.604
buildup, 292.844 drop, 391.895 breakdown, against a labelled drop of
342.307-386.551, every second of which the rig spent in DROP.

The opposite error is just as easy and was measured on the same night: NMK on
opus and BC/NIWDW15 on inyathi genuinely left a labelled drop dark for tens of
seconds, and the gate closed on them correctly.  A reader that answered "was
the intent in force ANYWHERE in the span" would pass a chain that lit one
second of forty-four.  So the primary quantity is SECONDS in force over the
labelled span, with the boolean derived from a stated threshold, and the
first-entry reading kept under ``first_entry_*`` keys because the campaign has
banked artifacts carrying its digits.

These cases use synthetic block lists rather than the real reports so the
arithmetic is inspectable.
"""
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
for _path in (str(TRAINING_DIR), str(TRAINING_DIR / "l9c_campaign")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import ng_arc_read  # noqa: E402

# The labelled drop from the probe report the false finding was read off.
OPUS_DROP = [("drop", 342.307, 386.551)]
SPAN = 386.551 - 342.307
# ...and an arc, so the buildup cases have a labelled climb in front of a drop.
ARC_SPANS = [("buildup", 300.0, 340.0), ("drop", 340.0, 380.0)]


def block(song_t, intent):
    return {"song_t": song_t, "intent": intent, "trigger": "classifier"}


def test_a_drop_held_from_before_the_window_is_in_force():
    items = [block(254.740, "breakdown"), block(267.604, "buildup"),
             block(292.844, "drop"), block(391.895, "breakdown")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force"] is True
    assert drop["in_force_kept"] is True
    assert drop["in_force_fraction"] == 1.0
    assert drop["in_force_dark_seconds"] == 0.0
    assert drop["in_force_song_t"] == 292.84
    assert drop["in_force_lag"] == -49.46
    assert drop["in_force_held_from_before_window"] is True
    assert drop["in_force_preceded_by"] == "buildup"
    # the banked reading, preserved verbatim: it scored this drop as lost.
    assert drop["first_entry_landed"] is False


def test_a_drop_lit_for_a_fraction_of_the_span_is_not_kept():
    # The NMK shape: DROP committed on the label and abandoned a second later,
    # leaving 43 s of a labelled drop dark.  Both booleans the instrument has
    # ever printed call this a landed, strict drop -- only the seconds do not.
    items = [block(341.0, "breakdown"), block(343.0, "drop"),
             block(344.0, "breakdown")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force"] is True
    assert drop["in_force_seconds"] == 1.0
    assert drop["in_force_dark_seconds"] == 43.24
    assert drop["in_force_kept"] is False
    assert drop["first_entry_landed"] is True
    assert drop["first_entry_strict"] is True


def test_a_brief_gap_inside_the_span_still_counts_as_kept():
    # The rule is stated, not measured: a labelled drop is KEPT when at least
    # KEPT_FRACTION of its seconds are in force.  A 2 s hole in 44 s clears it.
    assert ng_arc_read.KEPT_FRACTION == 0.75
    items = [block(300.0, "drop"), block(360.0, "breakdown"),
             block(362.0, "drop"), block(400.0, "breakdown")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force_dark_seconds"] == 2.0
    assert drop["in_force_fraction"] == 0.955
    assert drop["in_force_kept"] is True


def test_a_gap_past_the_threshold_is_not_kept():
    # The same shape with a 12 s hole falls under KEPT_FRACTION and is refused,
    # so the threshold is pinned from both sides rather than left implicit.
    items = [block(300.0, "drop"), block(350.0, "breakdown"),
             block(362.0, "drop"), block(400.0, "breakdown")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force_dark_seconds"] == 12.0
    assert drop["in_force_fraction"] == 0.729
    assert drop["in_force_kept"] is False


def test_a_drop_entered_inside_the_window_still_lands():
    items = [block(300.0, "breakdown"), block(342.8, "drop"),
             block(400.0, "breakdown")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force"] is True
    assert drop["in_force_kept"] is True
    assert drop["in_force_lag"] == 0.49
    assert drop["in_force_strict"] is True
    assert drop["in_force_held_from_before_window"] is False
    assert drop["in_force_preceded_by"] == "breakdown"
    # unchanged from the banked reading -- this is the case it got right.
    assert drop["first_entry_landed"] is True
    assert drop["first_entry_strict"] is True


def test_a_drop_that_ended_before_the_window_is_not_in_force():
    items = [block(100.0, "drop"), block(200.0, "breakdown")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force"] is False
    assert drop["in_force_seconds"] == 0.0
    assert drop["in_force_song_t"] is None
    assert drop["in_force_lag"] is None
    assert drop["first_entry_landed"] is False


def test_a_drop_that_starts_after_the_window_is_not_in_force():
    items = [block(10.0, "breakdown"), block(400.0, "drop")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force"] is False
    assert drop["in_force_seconds"] == 0.0
    assert drop["first_entry_landed"] is False


def test_the_last_block_in_a_report_stays_in_force_to_the_end():
    items = [block(100.0, "drop")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force"] is True
    assert drop["in_force_seconds"] == round(SPAN, 2)
    assert drop["in_force_kept"] is True
    assert drop["in_force_song_t"] == 100.0
    assert drop["in_force_held_from_before_window"] is True
    assert drop["first_entry_landed"] is False


def test_a_buildup_held_from_before_the_arc_window_holds_into_the_drop():
    items = [block(250.0, "buildup"), block(345.0, "drop"),
             block(420.0, "breakdown")]

    read = ng_arc_read.arc_read(items, ARC_SPANS)
    arc = read["arcs"][0]

    assert arc["in_force_entered"] is True
    assert arc["in_force_entry_lag"] == -50.0
    assert arc["in_force_entered_before_window"] is True
    assert arc["in_force_fraction"] == 1.0
    assert arc["in_force_held_into_drop"] is True
    assert read["arcs_entered_in_force"] == 1
    assert read["arcs_held_in_force"] == 1
    # the banked reading saw neither, which is the finding this test corrects.
    assert arc["first_entry_entered"] is False
    assert arc["first_entry_held_into_drop"] is False
    assert read["arcs_entered_first_entry"] == 0
    assert read["arcs_held_first_entry"] == 0


def test_the_strict_clause_reads_the_block_nearest_the_onset():
    # An early DROP inside the 8 s lead followed by one on the label: the
    # first-entry reading takes the early block and calls the drop lost, while
    # the rig demonstrably landed it.  in_force_strict asks whether ANY drop
    # block covering the window sits inside +-1.5 s, so it can only ever turn a
    # banked False into True -- never the reverse.
    items = [block(336.0, "drop"), block(341.0, "breakdown"),
             block(342.5, "drop"), block(400.0, "breakdown")]

    drop = ng_arc_read.arc_read(items, OPUS_DROP)["drops"][0]

    assert drop["in_force_strict"] is True
    assert drop["first_entry_strict"] is False
