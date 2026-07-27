"""Tests for the beat -> Raveform-label join (training/build_training_table.py).

The join is where a beat produced by the DSP pipeline meets an expert
annotation.  Every downstream number -- the baseline confusion matrix, the NN's
training targets -- inherits its mistakes silently: a beat attributed to the
wrong section is indistinguishable from a classifier error, and an unlabeled
region attributed to its neighbour teaches the model something the annotator
never said.  So the rules that decide *which* beats become rows, and *what*
label each one carries, are pinned here.

Only pure logic is exercised: no audio, no simulation, no corpus on disk.
"""
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from build_training_table import (  # noqa: E402  (needs the path insert above)
    ANALYSER_RESET_SEC,
    BAR_POSITION_UNKNOWN,
    CACHE_VERSION,
    CLEAN_MANIFEST_FILE,
    CONTINUOUS_COLUMNS,
    FEATURES_DIR,
    KICK_MIN_RMS,
    MEL_EXPORTER_KEY,
    MEL_EXPORTER_VERSION,
    NO_INTENT,
    REPORTS_DIR,
    TABLE_FILE,
    TABLE_HEADER,
    Timeline,
    build_table,
    cache_is_fresh,
    canonical_coverage,
    format_row,
    join_track,
    label_v1,
    load_ok_rows,
    realign_intents,
    pipeline_sha,
    report_path,
    select_jobs,
    sidecar_generation,
    song_time_intents,
    write_feature_sidecar,
    zscores,
    _read_json_gz,
    _write_json_gz,
)

LOOK_AHEAD = 2.5


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #


def beat(t: float, **overrides) -> dict:
    """One beat record in the shape EventBuffer.to_report() emits."""
    record = {
        "t": t,
        "bpm": 128.0,
        "onset_density": 4.0,
        "kick_strength": 2.0,
        "centroid_trend": 1.0,
        "sub_bass_ratio": 0.3,
        "rms": 0.1,
    }
    record.update(overrides)
    return record


def report(beats: list, intents: list | None = None,
           look_ahead_sec: float = LOOK_AHEAD, duration_sec: float = 1000.0) -> dict:
    return {
        "duration_sec": duration_sec,
        "beats": beats,
        "intents": intents if intents is not None else [],
        "metrics": {"look_ahead_sec": look_ahead_sec},
    }


def join(sections, beats, intents=None, **kwargs):
    return join_track("0001.abc", "abc", report(beats, intents, **kwargs), sections)


def labels(rows: list, column: str = "label_canonical") -> list:
    return [row[column] for row in rows]


# --------------------------------------------------------------------------- #
# Coverage boundaries: which beats become rows at all
# --------------------------------------------------------------------------- #


def test_beats_before_the_first_section_are_dropped_and_counted():
    """The leading offset is UNANNOTATED audio (up to 35.9 s on this corpus).

    Attributing it to the first section would teach an intro that the annotator
    never marked; it must be excluded, not absorbed.
    """
    sections = [(10.0, 20.0, "intro"), (20.0, 30.0, "drop")]
    rows, stats = join(sections, [beat(5.0), beat(15.0), beat(25.0)])

    assert labels(rows) == ["intro", "drop"]
    assert stats.dropped_leading == 1
    assert stats.dropped_trailing == 0


def test_beats_past_the_last_section_end_are_dropped_and_counted():
    """Audio can outlast the annotation; those beats have no ground truth."""
    sections = [(0.0, 20.0, "drop")]
    rows, stats = join(sections, [beat(10.0), beat(20.0), beat(24.9)])

    assert labels(rows) == ["drop"]
    assert stats.dropped_trailing == 2  # the boundary itself is past the end


def test_beats_inside_the_end_sentinel_are_unlabeled_not_merged_across():
    """`end` is a tail sentinel, not a section: its time is never re-attributed.

    canonical_runs() merges drop+end+drop into ONE run whose span covers the
    sentinel -- which is why the join must not use a merged run's span as
    coverage.  A beat inside the sentinel has no label.
    """
    sections = [(0.0, 10.0, "drop"), (10.0, 12.0, "end"), (12.0, 20.0, "drop")]
    rows, stats = join(sections, [beat(5.0), beat(11.0), beat(15.0)])

    assert [row["t_song"] for row in rows] == [5.0, 15.0]
    assert labels(rows) == ["drop", "drop"]
    assert stats.dropped_gap == 1
    assert stats.dropped_in_dropped_section == 1


def test_trailing_end_sentinel_beats_count_as_trailing():
    sections = [(0.0, 10.0, "outro"), (10.0, 14.0, "end")]
    rows, stats = join(sections, [beat(5.0), beat(12.0)])

    assert labels(rows) == ["outro"]
    assert stats.dropped_trailing == 1
    assert stats.dropped_in_dropped_section == 1


def test_every_beat_is_accounted_for_exactly_once():
    """kept + leading + gap + trailing == total, always -- the join may never
    lose a beat silently."""
    sections = [(5.0, 10.0, "intro"), (10.0, 12.0, "end"), (12.0, 20.0, "drop")]
    beats = [beat(t) for t in (1.0, 6.0, 11.0, 15.0, 25.0)]
    rows, stats = join(sections, beats)

    assert stats.beats_total == len(beats)
    assert stats.beats_kept == len(rows)
    assert (stats.beats_kept + stats.dropped_leading
            + stats.dropped_gap + stats.dropped_trailing) == stats.beats_total


def test_track_without_sections_yields_no_rows():
    rows, stats = join([], [beat(1.0), beat(2.0)])

    assert rows == []
    assert stats.beats_total == 2
    assert stats.dropped_gap == 2


def test_track_without_beats_yields_no_rows():
    rows, stats = join([(0.0, 10.0, "drop")], [])

    assert rows == []
    assert stats.beats_total == 0


# --------------------------------------------------------------------------- #
# Anomalous annotations
# --------------------------------------------------------------------------- #


def test_negative_length_section_is_clamped_not_crashing():
    """One track (1020.c1VBubZ2w3M) ends with start > end by 0.6 ms."""
    sections = [(0.0, 10.0, "intro"), (10.0, 20.0, "drop"), (20.0, 19.9994, "end")]
    rows, stats = join(sections, [beat(15.0), beat(20.5)])

    assert labels(rows) == ["drop"]
    assert stats.dropped_trailing == 1


def test_negative_length_labeled_section_covers_nothing():
    """A clamped section has zero width -- it can never claim a beat."""
    sections = [(0.0, 10.0, "intro"), (10.0, 9.5, "drop")]
    rows, stats = join(sections, [beat(5.0), beat(10.0)])

    assert labels(rows) == ["intro"]
    assert stats.beats_kept == 1
    assert stats.dropped_trailing == 1


def test_boundary_is_half_open_so_a_beat_on_a_boundary_belongs_to_the_later_section():
    sections = [(0.0, 10.0, "intro"), (10.0, 20.0, "drop")]
    rows, _stats = join(sections, [beat(10.0)])

    assert labels(rows) == ["drop"]


# --------------------------------------------------------------------------- #
# Label vocabulary
# --------------------------------------------------------------------------- #


def test_canonical_mapping_folds_altintro_and_bridge():
    sections = [(0.0, 10.0, "altintro"), (10.0, 20.0, "bridge")]
    rows, _stats = join(sections, [beat(5.0), beat(15.0)])

    assert labels(rows) == ["intro", "breakdown"]
    assert labels(rows, "label_raw") == ["altintro", "bridge"]


def test_label_v1_merges_cooldown_into_breakdown_and_altoutro_into_outro():
    sections = [
        (0.0, 10.0, "altintro"),
        (10.0, 20.0, "bridge"),
        (20.0, 30.0, "cooldown"),
        (30.0, 40.0, "altoutro"),
    ]
    rows, _stats = join(sections, [beat(t) for t in (5.0, 15.0, 25.0, 35.0)])

    assert labels(rows, "label_canonical") == ["intro", "breakdown", "cooldown", "altoutro"]
    assert labels(rows, "label_v1") == ["intro", "breakdown", "breakdown", "outro"]


def test_label_v1_is_identity_on_the_five_class_space():
    for label in ("intro", "buildup", "breakdown", "drop", "outro"):
        assert label_v1(label) == label


def test_canonical_coverage_drops_sentinels_and_clamps():
    spans = canonical_coverage([(0.0, 5.0, "altintro"), (5.0, 7.0, "end"), (7.0, 6.0, "drop")])

    assert spans == [(0.0, 5.0, "intro"), (7.0, 7.0, "drop")]


# --------------------------------------------------------------------------- #
# Intent realignment (audience time -> song time)
# --------------------------------------------------------------------------- #


def test_intent_blocks_are_shifted_back_by_the_look_ahead():
    blocks = [{"t": 5.0, "intent": "GROOVE", "end": 10.0},
              {"t": 10.0, "intent": "DROP", "end": 20.0}]

    assert song_time_intents(blocks, 2.5) == [(2.5, 7.5, "GROOVE"), (7.5, 17.5, "DROP")]


def test_intent_at_beat_uses_song_time_not_audience_time():
    sections = [(0.0, 30.0, "drop")]
    # Queue commits: enqueued at the beats at 3 s and 8 s, stamped 2.5 s later.
    intents = [queue_block(3.0, "GROOVE", 10.5), queue_block(8.0, "DROP", 20.0)]
    rows, stats = join(sections, [beat(t) for t in (1.0, 3.0, 8.0, 18.0)], intents)

    assert labels(rows, "intent_at_beat") == [NO_INTENT, "GROOVE", "DROP", NO_INTENT]
    assert stats.beats_without_intent == 2


def test_open_ended_intent_block_is_closed_at_the_report_duration():
    """to_report() closes the final block, but a hand-built report may not."""
    sections = [(0.0, 30.0, "drop")]
    intents = [{"t": 5.0, "intent": "PEAK"}]
    rows, _stats = join(sections, [beat(10.0)], intents, duration_sec=20.0)

    assert labels(rows, "intent_at_beat") == ["PEAK"]


def test_zero_look_ahead_leaves_intent_blocks_where_they_are():
    blocks = [{"t": 5.0, "intent": "GROOVE", "end": 10.0}]

    assert song_time_intents(blocks, 0.0) == [(5.0, 10.0, "GROOVE")]


# --------------------------------------------------------------------------- #
# Intent realignment: the engine's two commit paths stamp in different bases
# --------------------------------------------------------------------------- #


def queue_block(beat_t: float, intent: str, end: float, look_ahead=LOOK_AHEAD) -> dict:
    """A block committed through the delayed queue: stamped one look-ahead late."""
    return {"t": beat_t + look_ahead, "intent": intent, "end": end}


def test_a_queue_stamped_block_is_shifted_back():
    beats = [10.0, 10.5, 11.0]
    blocks = [queue_block(10.0, "GROOVE", 20.0)]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert spans[0][0] == pytest.approx(10.0)
    assert alignment.song_stamped == 0


def test_a_song_stamped_block_is_left_where_it_is():
    """The first beat commits immediately, inside on_beat -- already song time.
    Shifting it back would start the run's first intent 2.5 s before the music."""
    beats = [10.0, 10.5, 11.0]
    blocks = [{"t": 10.0, "intent": "GROOVE", "end": 20.0}]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert spans[0][0] == pytest.approx(10.0)
    assert alignment.song_stamped == 1


def test_a_mid_track_song_stamped_block_does_not_steal_the_previous_intent():
    """Re-entry after a sound stop commits immediately too.  De-shifting it
    would hand it the 2.5 s of beats belonging to the intent before it."""
    beats = [10.0, 11.0, 12.0, 13.0, 20.0, 21.0]
    blocks = [
        queue_block(10.0, "GROOVE", 20.0),      # ends where the next begins
        {"t": 20.0, "intent": "DROP", "end": 30.0},   # song-stamped re-entry
    ]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)
    timeline = Timeline(spans)

    assert alignment.song_stamped == 1
    assert spans[1][0] == pytest.approx(20.0)
    # The beats at 12 s and 13 s stay with GROOVE instead of being annexed.
    assert timeline.at(12.0) == "GROOVE"
    assert timeline.at(13.0) == "GROOVE"
    assert timeline.at(20.0) == "DROP"


def test_a_block_boundary_is_de_shifted_once_not_twice():
    """set_intent closes the old block and opens the new one at ONE instant, so
    a block's end must come from the next block's corrected start."""
    beats = [10.0, 11.0, 20.0]
    blocks = [
        queue_block(10.0, "GROOVE", 20.0),
        {"t": 20.0, "intent": "DROP", "end": 30.0},
    ]

    spans, _alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert spans[0][1] == spans[1][0]
    assert spans[0][1] == pytest.approx(20.0)


def test_the_final_block_frozen_by_mark_end_is_still_a_queue_commit():
    """to_report clamps the last commit to the report duration, so it matches no
    beat -- but it came off the queue and must still be shifted back."""
    beats = [10.0, 11.0, 12.0]
    blocks = [queue_block(10.0, "GROOVE", 30.0),
              {"t": 30.0, "intent": "PEAK", "end": 30.0}]

    _spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert alignment.song_stamped == 0
    assert alignment.clamped_tail == 1


def test_a_queue_commit_landing_on_a_beat_instant_is_still_a_queue_commit():
    """Beats and blocks are drawn from the same virtual-clock tick ladder, so
    ~1% of ordinary commits coincide with a beat.  Keying on that coincidence
    would mis-shift hundreds of blocks, so the queue reading wins when it
    explains the stamp."""
    beats = [10.0, 12.5]                        # 12.5 == 10.0 + look_ahead
    blocks = [queue_block(10.0, "GROOVE", 20.0)]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert alignment.song_stamped == 0
    assert spans[0][0] == pytest.approx(10.0)


def test_realignment_tolerates_the_one_buffer_queue_latency():
    """The queue fires on the next main-loop iteration, up to one buffer late."""
    quantum = 256 / 44100
    beats = [10.0, 10.5]
    blocks = [{"t": 10.0 + LOOK_AHEAD + quantum, "intent": "GROOVE", "end": 20.0}]

    _spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert alignment.song_stamped == 0


def test_the_first_block_never_starts_before_the_first_beat():
    beats = [10.0, 11.0]
    blocks = [{"t": 10.0, "intent": "GROOVE", "end": 20.0}]

    spans, _alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert spans[0][0] >= beats[0]


def test_realignment_of_nothing_is_nothing():
    spans, alignment = realign_intents([], LOOK_AHEAD, [1.0], duration_sec=30.0)

    assert spans == []
    assert alignment == (0, 0, 0)


def test_a_report_without_beats_is_left_uniformly_shifted():
    blocks = [{"t": 10.0, "intent": "GROOVE", "end": 20.0}]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, [], duration_sec=30.0)

    assert spans[0][0] == pytest.approx(7.5)
    assert alignment.song_stamped == 0


def test_join_counts_the_rows_the_realignment_moved():
    sections = [(0.0, 40.0, "drop")]
    beats = [beat(t) for t in (10.0, 11.0, 18.0, 19.0, 20.0)]
    intents = [
        queue_block(10.0, "GROOVE", 20.0),
        {"t": 20.0, "intent": "DROP", "end": 30.0},   # song-stamped re-entry
    ]
    rows, stats = join(sections, beats, intents, duration_sec=30.0)

    assert stats.intent_blocks_song_stamped == 1
    # The naive de-shift starts DROP at 17.5 s and annexes the beats at 18 and
    # 19 s from GROOVE, which really held the lights until 20 s.
    assert stats.intent_reattributed == 2
    assert labels(rows, "intent_at_beat") == ["GROOVE"] * 4 + ["DROP"]


def test_join_reports_no_reattribution_when_every_block_came_off_the_queue():
    sections = [(0.0, 40.0, "drop")]
    beats = [beat(t) for t in (10.0, 11.0, 12.0)]
    intents = [queue_block(10.0, "GROOVE", 30.0)]
    _rows, stats = join(sections, beats, intents, duration_sec=30.0)

    assert stats.intent_blocks_song_stamped == 0
    assert stats.intent_reattributed == 0


# --------------------------------------------------------------------------- #
# Feature columns
# --------------------------------------------------------------------------- #


def test_kick_known_follows_the_rms_silence_gate():
    """The kick sentinel is a number in the range of real ratios, so presence is
    read off the row's own RMS (lib/analyser/CLAUDE.md)."""
    sections = [(0.0, 30.0, "drop")]
    beats = [beat(1.0, rms=KICK_MIN_RMS - 0.001),
             beat(2.0, rms=KICK_MIN_RMS),
             beat(3.0, rms=KICK_MIN_RMS + 0.001)]
    rows, _stats = join(sections, beats)

    assert [row["kick_known"] for row in rows] == [0, 1, 1]


def test_feature_columns_are_copied_through_verbatim():
    sections = [(0.0, 30.0, "drop")]
    rows, _stats = join(sections, [beat(1.0, bpm=124.5, onset_density=6.0,
                                        kick_strength=3.25, centroid_trend=1.4,
                                        sub_bass_ratio=0.42, rms=0.09)])
    row = rows[0]

    assert row["bpm"] == 124.5
    assert row["onset_density"] == 6.0
    assert row["kick_strength"] == 3.25
    assert row["centroid_trend"] == 1.4
    assert row["sub_bass_ratio"] == 0.42
    assert row["rms"] == 0.09


def test_bar_position_is_flagged_unknown_on_every_row():
    """The pipeline has no downbeat tracker; the column exists so the schema
    does not change when Stage-2 downbeat tracking lands."""
    rows, _stats = join([(0.0, 30.0, "drop")], [beat(1.0), beat(2.0)])

    assert {row["bar_position_unknown"] for row in rows} == {BAR_POSITION_UNKNOWN}


def test_track_identity_is_on_every_row():
    rows, _stats = join([(0.0, 30.0, "drop")], [beat(1.0)])

    assert rows[0]["track_id"] == "0001.abc"
    assert rows[0]["youtube_id"] == "abc"


# --------------------------------------------------------------------------- #
# Per-track z-scores
# --------------------------------------------------------------------------- #


def test_zscores_are_zero_for_a_flat_feature():
    assert zscores([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]


def test_zscores_use_the_population_standard_deviation():
    result = zscores([1.0, 2.0, 3.0])

    assert result[1] == pytest.approx(0.0)
    assert result[0] == pytest.approx(-math.sqrt(1.5))
    assert result[2] == pytest.approx(math.sqrt(1.5))


def test_zscores_are_computed_over_the_kept_rows_of_one_track():
    """Dropped beats must not move the track's mean -- the z-scores describe the
    training rows, not the audio."""
    sections = [(10.0, 30.0, "drop")]
    beats = [beat(1.0, rms=100.0), beat(11.0, rms=1.0),
             beat(12.0, rms=2.0), beat(13.0, rms=3.0)]
    rows, _stats = join(sections, beats)

    assert [row["rms_z"] for row in rows] == pytest.approx(
        [-math.sqrt(1.5), 0.0, math.sqrt(1.5)]
    )


def test_every_continuous_feature_gets_a_z_column():
    rows, _stats = join([(0.0, 30.0, "drop")], [beat(1.0), beat(2.0, bpm=130.0)])

    for column in CONTINUOUS_COLUMNS:
        assert f"{column}_z" in rows[0]


# --------------------------------------------------------------------------- #
# Row shape
# --------------------------------------------------------------------------- #


def test_rows_carry_exactly_the_header_columns():
    rows, _stats = join([(0.0, 30.0, "drop")], [beat(1.0)])

    assert set(rows[0]) == set(TABLE_HEADER)


def test_rows_are_ordered_by_song_time():
    rows, _stats = join([(0.0, 30.0, "drop")], [beat(t) for t in (1.0, 2.0, 3.0)])

    assert [row["t_song"] for row in rows] == [1.0, 2.0, 3.0]


def test_format_row_emits_the_header_order_as_strings():
    rows, _stats = join([(0.0, 30.0, "drop")], [beat(1.5)])
    fields = format_row(rows[0])

    assert len(fields) == len(TABLE_HEADER)
    assert all(isinstance(field, str) for field in fields)
    assert fields[TABLE_HEADER.index("track_id")] == "0001.abc"
    assert fields[TABLE_HEADER.index("t_song")] == "1.500000"
    assert fields[TABLE_HEADER.index("label_canonical")] == "drop"


# --------------------------------------------------------------------------- #
# Table file
# --------------------------------------------------------------------------- #


def write_corpus(tmp_path: Path, tracks: dict) -> tuple:
    """``tracks`` = ``{track_id: (youtube_id, sections, report)}`` -> cached reports."""
    rows = []
    sections_by_track = {}
    (tmp_path / FEATURES_DIR).mkdir(parents=True, exist_ok=True)
    for track_id, (youtube, sections, payload) in sorted(tracks.items()):
        (tmp_path / FEATURES_DIR / f"{youtube}.npz").write_bytes(b"npz")
        _write_json_gz(report_path(tmp_path, youtube), {
            "cache_version": CACHE_VERSION,
            "track_id": track_id,
            "youtube_id": youtube,
            "pipeline_sha": "deadbeef",
            "mp3_size": 1,
            "mp3_mtime": 1.0,
            "report": payload,
        })
        rows.append({"track_id": track_id, "youtube_id": youtube})
        sections_by_track[track_id] = sections
    return rows, sections_by_track


def read_table(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_table_writes_the_header_and_one_row_per_labeled_beat(tmp_path):
    rows, sections = write_corpus(tmp_path, {
        "0001.abc": ("abc", [(0.0, 30.0, "drop")], report([beat(1.0), beat(2.0)])),
    })

    stats = build_table(tmp_path, rows, sections)
    table = read_table(tmp_path / TABLE_FILE)

    assert table[0] == list(TABLE_HEADER)
    assert len(table) == 3
    assert stats.tracks == 1
    assert stats.rows == 2
    assert stats.canonical["drop"] == 2
    assert stats.v1["drop"] == 2


def test_table_is_byte_identical_when_rebuilt_from_the_same_reports(tmp_path):
    """A rebuild must diff cleanly against the previous one -- otherwise a
    threshold change and a timestamp are indistinguishable in the output."""
    rows, sections = write_corpus(tmp_path, {
        "0001.abc": ("abc", [(0.0, 30.0, "drop")], report([beat(1.0), beat(2.0)])),
        "0002.def": ("def", [(0.0, 30.0, "intro")], report([beat(3.0)])),
    })

    build_table(tmp_path, rows, sections)
    first = (tmp_path / TABLE_FILE).read_bytes()
    build_table(tmp_path, rows, sections)

    assert (tmp_path / TABLE_FILE).read_bytes() == first


def test_an_unsimulated_track_is_recorded_not_silently_dropped(tmp_path):
    """meta.json is the audit record: a track that vanishes without a line looks
    identical to one that passed."""
    rows, sections = write_corpus(tmp_path, {
        "0001.abc": ("abc", [(0.0, 30.0, "drop")], report([beat(1.0)])),
    })
    rows.append({"track_id": "0009.zzz", "youtube_id": "zzz"})
    sections["0009.zzz"] = [(0.0, 30.0, "drop")]

    stats = build_table(tmp_path, rows, sections)

    assert stats.tracks == 1
    assert stats.missing_reports == ["0009.zzz"]
    assert stats.missing_sidecars == []
    assert stats.skipped == []


def test_a_track_without_mel_features_contributes_no_rows(tmp_path):
    """Rows whose track has no sidecar would reach the NN dataset builder with
    nothing to featurise."""
    rows, sections = write_corpus(tmp_path, {
        "0001.abc": ("abc", [(0.0, 30.0, "drop")], report([beat(1.0), beat(2.0)])),
        "0002.def": ("def", [(0.0, 30.0, "intro")], report([beat(3.0)])),
    })
    (tmp_path / FEATURES_DIR / "def.npz").unlink()

    stats = build_table(tmp_path, rows, sections)
    table = read_table(tmp_path / TABLE_FILE)

    assert stats.missing_sidecars == ["0002.def"]
    assert stats.tracks == 1
    assert stats.rows == 2
    assert {row[0] for row in table[1:]} == {"0001.abc"}


def test_table_records_a_track_whose_report_is_unreadable(tmp_path):
    rows, sections = write_corpus(tmp_path, {
        "0001.abc": ("abc", [(0.0, 30.0, "drop")], report([beat(1.0)])),
    })
    report_path(tmp_path, "abc").write_bytes(b"not gzip at all")

    stats = build_table(tmp_path, rows, sections)

    assert stats.tracks == 0
    assert [track_id for track_id, _detail in stats.skipped] == ["0001.abc"]


# --------------------------------------------------------------------------- #
# Report cache
# --------------------------------------------------------------------------- #

FRESH = {"cache_version": CACHE_VERSION, "pipeline_sha": "sha1",
         "mp3_size": 100, "mp3_mtime": 1.5, "report": {"beats": []}}


def test_cache_is_fresh_when_pipeline_and_audio_match():
    assert cache_is_fresh(FRESH, "sha1", 100, 1.5)


@pytest.mark.parametrize("sha,size,mtime", [
    ("sha2", 100, 1.5),   # pipeline changed -- the report IS its output
    ("sha1", 101, 1.5),   # audio re-encoded at a different size
    ("sha1", 100, 2.0),   # audio replaced, same size
])
def test_cache_is_stale_when_anything_it_depends_on_moved(sha, size, mtime):
    assert not cache_is_fresh(FRESH, sha, size, mtime)


def test_cache_of_an_older_envelope_version_is_stale():
    assert not cache_is_fresh({**FRESH, "cache_version": CACHE_VERSION - 1},
                              "sha1", 100, 1.5)


def test_cache_without_a_report_is_stale():
    assert not cache_is_fresh({k: v for k, v in FRESH.items() if k != "report"},
                              "sha1", 100, 1.5)


def stage_track(tmp_path: Path, sha: str = "sha1", sidecar: bool = True,
                cached: bool = True) -> list:
    """One manifest row with an mp3, and optionally its cache + sidecar."""
    mp3 = tmp_path / "audio" / "abc.mp3"
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"x" * 100)
    stat = mp3.stat()
    if cached:
        _write_json_gz(report_path(tmp_path, "abc"), {
            "cache_version": CACHE_VERSION, "track_id": "0001.abc",
            "youtube_id": "abc", "pipeline_sha": sha,
            "mp3_size": stat.st_size, "mp3_mtime": stat.st_mtime,
            "report": report([beat(1.0)]),
        })
    if sidecar:
        (tmp_path / FEATURES_DIR).mkdir(parents=True, exist_ok=True)
        (tmp_path / FEATURES_DIR / "abc.npz").write_bytes(b"npz")
    return [{"track_id": "0001.abc", "youtube_id": "abc", "mp3_path": str(mp3)}]


def select(tmp_path, rows, **kwargs):
    """select_jobs with the mtime guard disabled (the fixture mp3 is brand new)."""
    kwargs.setdefault("sha", "sha1")
    kwargs.setdefault("min_age_sec", -1.0)
    return select_jobs(rows, tmp_path, **kwargs)


def test_a_fresh_cache_skips_the_simulation_entirely(tmp_path):
    rows = stage_track(tmp_path)

    jobs, counts = select(tmp_path, rows)

    assert jobs == []
    assert counts["hit"] == 1


def test_a_new_track_is_a_miss(tmp_path):
    rows = stage_track(tmp_path, cached=False, sidecar=False)

    jobs, counts = select(tmp_path, rows)

    assert len(jobs) == 1
    assert counts["miss_new"] == 1


def test_a_pipeline_change_invalidates_every_cached_report(tmp_path):
    rows = stage_track(tmp_path, sha="old")

    jobs, counts = select(tmp_path, rows)

    assert len(jobs) == 1
    assert counts["miss_pipeline_changed"] == 1


def test_replacing_the_audio_invalidates_its_report(tmp_path):
    rows = stage_track(tmp_path)
    Path(rows[0]["mp3_path"]).write_bytes(b"y" * 400)

    jobs, counts = select(tmp_path, rows)

    assert len(jobs) == 1
    assert counts["miss_audio_changed"] == 1


def test_a_missing_sidecar_forces_the_track_back_through_the_decode(tmp_path):
    """The sidecar can only be built from decoded audio, so a report alone is
    not enough to call the track done."""
    rows = stage_track(tmp_path, sidecar=False)

    jobs, counts = select(tmp_path, rows)

    assert len(jobs) == 1
    assert counts["miss_no_sidecar"] == 1


def test_an_unreadable_cache_is_a_miss_not_a_crash(tmp_path):
    rows = stage_track(tmp_path)
    report_path(tmp_path, "abc").write_bytes(b"truncated")

    jobs, counts = select(tmp_path, rows)

    assert len(jobs) == 1
    assert counts["miss_unreadable"] == 1


def test_force_ignores_a_perfectly_good_cache(tmp_path):
    rows = stage_track(tmp_path)

    jobs, counts = select(tmp_path, rows, force=True)

    assert len(jobs) == 1
    assert counts["miss_forced"] == 1
    assert counts["hit"] == 0


def test_a_freshly_written_mp3_is_left_for_the_next_run(tmp_path):
    """The downloader may still be writing it -- and a file whose bytes are
    still moving must not be cached against either."""
    rows = stage_track(tmp_path, cached=False, sidecar=False)

    jobs, counts = select_jobs(rows, tmp_path, sha="sha1", min_age_sec=3600.0)

    assert jobs == []
    assert counts["too_recent"] == 1


def test_a_missing_mp3_is_counted_not_dispatched(tmp_path):
    rows = stage_track(tmp_path, cached=False, sidecar=False)
    Path(rows[0]["mp3_path"]).unlink()

    jobs, counts = select(tmp_path, rows)

    assert jobs == []
    assert counts["missing_audio"] == 1


def test_the_job_carries_the_stamp_the_cache_will_be_checked_against(tmp_path):
    rows = stage_track(tmp_path, cached=False, sidecar=False)
    stat = Path(rows[0]["mp3_path"]).stat()

    jobs, _counts = select(tmp_path, rows)

    assert jobs[0].pipeline_sha == "sha1"
    assert jobs[0].mp3_size == stat.st_size
    assert jobs[0].mp3_mtime == stat.st_mtime


# --------------------------------------------------------------------------- #
# Mel-sidecar generation
# --------------------------------------------------------------------------- #
#
# The sidecar's freshness used to be "the file exists".  A change to
# `pooled_log_mel` that keeps the frame rate and the band count -- a different
# compression, a different pooling reduction -- writes a sidecar of exactly the
# right shape and entirely different numbers, and the corpus would then carry
# two feature generations under one training table with nothing to say so.


def stamp_sidecar(path: Path, version=MEL_EXPORTER_VERSION) -> None:
    """A real npz; ``version=None`` writes one from before the stamp existed."""
    payload = {"mel": np.zeros((1, 40), dtype=np.float32)}
    if version is not None:
        payload[MEL_EXPORTER_KEY] = np.int32(version)
    np.savez_compressed(path, **payload)


def test_a_sidecar_from_another_exporter_generation_is_re_simulated(tmp_path):
    rows = stage_track(tmp_path)
    stamp_sidecar(tmp_path / FEATURES_DIR / "abc.npz", MEL_EXPORTER_VERSION + 1)

    jobs, counts = select(tmp_path, rows)

    assert len(jobs) == 1
    assert counts["miss_sidecar_generation"] == 1


def test_a_sidecar_from_this_exporter_generation_is_a_hit(tmp_path):
    rows = stage_track(tmp_path)
    stamp_sidecar(tmp_path / FEATURES_DIR / "abc.npz")

    jobs, counts = select(tmp_path, rows)

    assert jobs == []
    assert counts["hit"] == 1


def test_an_unstamped_sidecar_is_grandfathered_rather_than_rebuilt(tmp_path):
    """Every sidecar on disk predates the stamp.  Invalidating them would order
    a 1,387-track re-simulation to discover what is already known: they were
    written by generation 1."""
    rows = stage_track(tmp_path)
    stamp_sidecar(tmp_path / FEATURES_DIR / "abc.npz", version=None)

    jobs, counts = select(tmp_path, rows)

    assert jobs == []
    assert counts["hit"] == 1


def test_sidecar_generation_reads_the_stamp_and_defaults_to_one(tmp_path):
    stamped, bare, junk = (tmp_path / n for n in ("s.npz", "b.npz", "j.npz"))
    stamp_sidecar(stamped, 7)
    stamp_sidecar(bare, version=None)
    junk.write_bytes(b"not an npz")

    assert sidecar_generation(stamped) == 7
    assert sidecar_generation(bare) == 1
    assert sidecar_generation(junk) == 1


def test_the_cache_counters_partition_the_manifest(tmp_path):
    """Hits + misses + skips must add up to the tracks looked at, and the misses
    must be exactly the jobs dispatched -- otherwise the summary can report a
    full cache while quietly re-simulating the corpus."""
    hit = stage_track(tmp_path)
    miss = tmp_path / "audio" / "def.mp3"
    miss.write_bytes(b"y" * 50)
    gone = tmp_path / "audio" / "ghi.mp3"
    rows = hit + [
        {"track_id": "0002.def", "youtube_id": "def", "mp3_path": str(miss)},
        {"track_id": "0003.ghi", "youtube_id": "ghi", "mp3_path": str(gone)},
    ]

    jobs, counts = select(tmp_path, rows)

    assert sum(counts.values()) == len(rows)
    assert sum(v for k, v in counts.items() if k.startswith("miss_")) == len(jobs)


def test_a_too_recent_track_is_not_also_counted_as_a_miss(tmp_path):
    rows = stage_track(tmp_path, cached=False, sidecar=False)

    jobs, counts = select_jobs(rows, tmp_path, sha="sha1", min_age_sec=3600.0)

    assert jobs == []
    assert counts["too_recent"] == 1
    assert sum(v for k, v in counts.items() if k.startswith("miss_")) == 0


def staged_repo(tmp_path: Path) -> str:
    """A one-commit repo with a pipeline source and a doc; returns its clean sha."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                       capture_output=True, stdin=subprocess.DEVNULL)

    git("init", "-q")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "engine.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "lib" / "CLAUDE.md").write_text("docs\n", encoding="utf-8")
    git("add", "-A"); git("commit", "-qm", "init")
    return pipeline_sha(tmp_path)


def test_pipeline_sha_ignores_documentation_beside_the_code(tmp_path):
    """A CLAUDE.md living next to the pipeline cannot change what the simulation
    produces; treating a doc edit as a pipeline change discards the whole corpus
    cache for nothing."""
    baseline = staged_repo(tmp_path)

    (tmp_path / "lib" / "CLAUDE.md").write_text("docs, revised\n", encoding="utf-8")
    assert pipeline_sha(tmp_path) == baseline

    (tmp_path / "lib" / "engine.py").write_text("x = 2\n", encoding="utf-8")
    assert pipeline_sha(tmp_path).startswith(f"{baseline}+dirty")


def test_a_committed_tree_keys_on_the_bare_commit_sha(tmp_path):
    """The whole corpus cache is keyed on this string.  Decorating a CLEAN tree's
    key -- with anything, for any reason -- re-simulates 1,387 tracks."""
    baseline = staged_repo(tmp_path)

    assert pipeline_sha(tmp_path) == baseline
    assert "+" not in baseline


def test_two_different_uncommitted_edits_get_two_different_keys(tmp_path):
    """A constant `+dirty` gave every working-tree state one cache key, so the
    second edit of an afternoon read back reports the first one produced."""
    staged_repo(tmp_path)
    source = tmp_path / "lib" / "engine.py"

    source.write_text("x = 2\n", encoding="utf-8")
    first = pipeline_sha(tmp_path)
    source.write_text("x = 3\n", encoding="utf-8")
    second = pipeline_sha(tmp_path)

    assert first != second
    assert "+dirty" in first and "+dirty" in second


def test_the_dirty_key_returns_when_the_edit_does(tmp_path):
    """The key is a function of the working tree's CONTENT, not of how many
    times it was touched -- undoing an edit must restore the cache, not orphan
    it."""
    staged_repo(tmp_path)
    source = tmp_path / "lib" / "engine.py"

    source.write_text("x = 2\n", encoding="utf-8")
    edited = pipeline_sha(tmp_path)
    source.write_text("x = 3\n", encoding="utf-8")
    source.write_text("x = 2\n", encoding="utf-8")

    assert pipeline_sha(tmp_path) == edited


def test_an_untracked_pipeline_source_moves_the_key(tmp_path):
    """A brand-new lib/ module is invisible to `git diff` and changes what the
    simulation does; the status output is what carries it into the digest."""
    baseline = staged_repo(tmp_path)

    (tmp_path / "lib" / "extra.py").write_text("y = 1\n", encoding="utf-8")

    assert pipeline_sha(tmp_path) not in (baseline, "unknown")
    assert pipeline_sha(tmp_path).startswith(f"{baseline}+dirty")


def test_cached_reports_are_byte_identical_when_rewritten(tmp_path):
    payload = {"cache_version": CACHE_VERSION, "report": report([beat(1.0)])}
    path = tmp_path / "abc.json.gz"

    _write_json_gz(path, payload)
    first = path.read_bytes()
    _write_json_gz(path, payload)

    assert path.read_bytes() == first
    assert _read_json_gz(path) == payload


# --------------------------------------------------------------------------- #
# The analyser self-reset horizon
# --------------------------------------------------------------------------- #


def write_manifest(tmp_path: Path, tracks: list) -> None:
    """A clean manifest of ``(track_id, decoded_duration_sec)`` ok rows."""
    lines = ["track_id,youtube_id,mp3_path,decoded_duration_sec,status"]
    lines += [f"{track_id},{track_id[-3:]},audio/{track_id}.mp3,{seconds:.3f},ok"
              for track_id, seconds in tracks]
    (tmp_path / CLEAN_MANIFEST_FILE).write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")


def test_a_track_reaching_the_analyser_reset_is_left_out_of_the_build(tmp_path, capsys):
    """MusicAnalyser throws its rolling state away every 15 minutes; the mel
    exporter has no such reset.  Past the horizon the two describe the same
    audio from different states and the join produces wrong rows in silence."""
    write_manifest(tmp_path, [("0001.short", 300.0),
                              ("0002.long", ANALYSER_RESET_SEC)])

    rows = load_ok_rows(tmp_path)

    assert [row["track_id"] for row in rows] == ["0001.short"]
    printed = capsys.readouterr().out
    assert "0002.long" in printed and "self-reset" in printed


def test_a_track_just_under_the_horizon_still_builds(tmp_path):
    """The longest track in the corpus clears it by 0.11 s -- the gate has to be
    on the horizon itself, not near it."""
    write_manifest(tmp_path, [("0001.edge", ANALYSER_RESET_SEC - 0.111)])

    assert [row["track_id"] for row in load_ok_rows(tmp_path)] == ["0001.edge"]


def test_a_manifest_of_nothing_but_over_long_tracks_fails_loudly(tmp_path):
    write_manifest(tmp_path, [("0001.long", ANALYSER_RESET_SEC + 60.0)])

    with pytest.raises(RuntimeError, match="nothing to build from"):
        load_ok_rows(tmp_path)


def test_a_blank_duration_is_not_this_gates_business(tmp_path):
    """The cleanliness gate already ruled on these rows; a missing number here
    is not evidence of a long track."""
    (tmp_path / CLEAN_MANIFEST_FILE).write_text(
        "track_id,youtube_id,mp3_path,decoded_duration_sec,status\n"
        "0001.blank,ank,audio/a.mp3,,ok\n", encoding="utf-8")

    assert [row["track_id"] for row in load_ok_rows(tmp_path)] == ["0001.blank"]


# --------------------------------------------------------------------------- #
# Timeline lookup
# --------------------------------------------------------------------------- #


def test_timeline_lookup_is_half_open():
    timeline = Timeline([(0.0, 1.0, "a"), (1.0, 2.0, "b")])

    assert timeline.at(0.0) == "a"
    assert timeline.at(0.999) == "a"
    assert timeline.at(1.0) == "b"
    assert timeline.at(2.0) is None
    assert timeline.at(-1.0) is None


def test_timeline_on_overlap_prefers_the_later_span():
    timeline = Timeline([(0.0, 5.0, "a"), (2.0, 6.0, "b")])

    assert timeline.at(3.0) == "b"


def test_empty_timeline_returns_none():
    assert Timeline([]).at(1.0) is None
