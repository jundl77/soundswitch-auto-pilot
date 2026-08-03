"""Tests for the beat -> Raveform-label join (training/build_training_table.py)."""
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

from build_training_table import (  # noqa: E402
    ANALYSER_RESET_SEC,
    BAR_POSITION_UNKNOWN,
    CACHE_VERSION,
    CLEAN_MANIFEST_FILE,
    CONTINUOUS_COLUMNS,
    FEATURES_DIR,
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


def beat(t: float, **overrides) -> dict:
    record = {
        "t": t,
        "bpm": 128.0,
        "strength": 0.0,
        "change": False,
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


def test_beats_before_the_first_section_are_dropped_and_counted():
    sections = [(10.0, 20.0, "intro"), (20.0, 30.0, "drop")]
    rows, stats = join(sections, [beat(5.0), beat(15.0), beat(25.0)])

    assert labels(rows) == ["intro", "drop"]
    assert stats.dropped_leading == 1
    assert stats.dropped_trailing == 0


def test_beats_past_the_last_section_end_are_dropped_and_counted():
    sections = [(0.0, 20.0, "drop")]
    rows, stats = join(sections, [beat(10.0), beat(20.0), beat(24.9)])

    assert labels(rows) == ["drop"]
    assert stats.dropped_trailing == 2


def test_beats_inside_the_end_sentinel_are_unlabeled_not_merged_across():
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


def test_negative_length_section_is_clamped_not_crashing():
    sections = [(0.0, 10.0, "intro"), (10.0, 20.0, "drop"), (20.0, 19.9994, "end")]
    rows, stats = join(sections, [beat(15.0), beat(20.5)])

    assert labels(rows) == ["drop"]
    assert stats.dropped_trailing == 1


def test_negative_length_labeled_section_covers_nothing():
    sections = [(0.0, 10.0, "intro"), (10.0, 9.5, "drop")]
    rows, stats = join(sections, [beat(5.0), beat(10.0)])

    assert labels(rows) == ["intro"]
    assert stats.beats_kept == 1
    assert stats.dropped_trailing == 1


def test_boundary_is_half_open_so_a_beat_on_a_boundary_belongs_to_the_later_section():
    sections = [(0.0, 10.0, "intro"), (10.0, 20.0, "drop")]
    rows, _stats = join(sections, [beat(10.0)])

    assert labels(rows) == ["drop"]


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


def test_intent_blocks_are_shifted_back_by_the_look_ahead():
    blocks = [{"t": 5.0, "intent": "GROOVE", "end": 10.0},
              {"t": 10.0, "intent": "DROP", "end": 20.0}]

    assert song_time_intents(blocks, 2.5) == [(2.5, 7.5, "GROOVE"), (7.5, 17.5, "DROP")]


def test_intent_at_beat_uses_song_time_not_audience_time():
    sections = [(0.0, 30.0, "drop")]
    intents = [queue_block(3.0, "GROOVE", 10.5), queue_block(8.0, "DROP", 20.0)]
    rows, stats = join(sections, [beat(t) for t in (1.0, 3.0, 8.0, 18.0)], intents)

    assert labels(rows, "intent_at_beat") == [NO_INTENT, "GROOVE", "DROP", NO_INTENT]
    assert stats.beats_without_intent == 2


def test_open_ended_intent_block_is_closed_at_the_report_duration():
    sections = [(0.0, 30.0, "drop")]
    intents = [{"t": 5.0, "intent": "PEAK"}]
    rows, _stats = join(sections, [beat(10.0)], intents, duration_sec=20.0)

    assert labels(rows, "intent_at_beat") == ["PEAK"]


def test_zero_look_ahead_leaves_intent_blocks_where_they_are():
    blocks = [{"t": 5.0, "intent": "GROOVE", "end": 10.0}]

    assert song_time_intents(blocks, 0.0) == [(5.0, 10.0, "GROOVE")]


def queue_block(beat_t: float, intent: str, end: float, look_ahead=LOOK_AHEAD) -> dict:
    return {"t": beat_t + look_ahead, "intent": intent, "end": end}


def test_a_recorded_song_instant_is_taken_as_it_stands():
    beats = [10.0, 10.5, 11.0]
    blocks = [{"t": 24.0, "song_t": 10.31, "intent": "DROP", "end": 40.0}]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=50.0)

    assert spans[0][0] == pytest.approx(10.31)
    assert spans[0][1] == pytest.approx(40.0 - (24.0 - 10.31))
    assert alignment.song_recorded == 1
    assert alignment.song_stamped == 0


def test_a_recorded_instant_is_not_floored_at_the_first_beat():
    blocks = [{"t": 20.0, "song_t": 6.0, "intent": "ATMOSPHERIC", "end": 30.0}]
    spans, _alignment = realign_intents(blocks, LOOK_AHEAD, [9.0, 9.5],
                                        duration_sec=40.0)
    assert spans[0][0] == pytest.approx(6.0)


def test_recorded_and_inferred_blocks_can_share_one_report():
    beats = [10.0, 10.5, 11.0]
    blocks = [queue_block(10.0, "GROOVE", 20.0),
              {"t": 34.0, "song_t": 20.0, "intent": "DROP", "end": 44.0}]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=50.0)

    assert spans[0][0] == pytest.approx(10.0)
    assert spans[1][0] == pytest.approx(20.0)
    assert (alignment.song_recorded, alignment.song_stamped) == (1, 0)


def test_a_queue_stamped_block_is_shifted_back():
    beats = [10.0, 10.5, 11.0]
    blocks = [queue_block(10.0, "GROOVE", 20.0)]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert spans[0][0] == pytest.approx(10.0)
    assert alignment.song_stamped == 0


def test_a_song_stamped_block_is_left_where_it_is():
    beats = [10.0, 10.5, 11.0]
    blocks = [{"t": 10.0, "intent": "GROOVE", "end": 20.0}]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert spans[0][0] == pytest.approx(10.0)
    assert alignment.song_stamped == 1


def test_a_mid_track_song_stamped_block_does_not_steal_the_previous_intent():
    beats = [10.0, 11.0, 12.0, 13.0, 20.0, 21.0]
    blocks = [
        queue_block(10.0, "GROOVE", 20.0),
        {"t": 20.0, "intent": "DROP", "end": 30.0},
    ]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)
    timeline = Timeline(spans)

    assert alignment.song_stamped == 1
    assert spans[1][0] == pytest.approx(20.0)
    assert timeline.at(12.0) == "GROOVE"
    assert timeline.at(13.0) == "GROOVE"
    assert timeline.at(20.0) == "DROP"


def test_a_block_boundary_is_de_shifted_once_not_twice():
    beats = [10.0, 11.0, 20.0]
    blocks = [
        queue_block(10.0, "GROOVE", 20.0),
        {"t": 20.0, "intent": "DROP", "end": 30.0},
    ]

    spans, _alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert spans[0][1] == spans[1][0]
    assert spans[0][1] == pytest.approx(20.0)


def test_the_final_block_frozen_by_mark_end_is_still_a_queue_commit():
    beats = [10.0, 11.0, 12.0]
    blocks = [queue_block(10.0, "GROOVE", 30.0),
              {"t": 30.0, "intent": "PEAK", "end": 30.0}]

    _spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert alignment.song_stamped == 0
    assert alignment.clamped_tail == 1


def test_a_queue_commit_landing_on_a_beat_instant_is_still_a_queue_commit():
    beats = [10.0, 10.0 + LOOK_AHEAD]
    blocks = [queue_block(10.0, "GROOVE", 20.0)]

    spans, alignment = realign_intents(blocks, LOOK_AHEAD, beats, duration_sec=30.0)

    assert alignment.song_stamped == 0
    assert spans[0][0] == pytest.approx(10.0)


def test_realignment_tolerates_the_one_buffer_queue_latency():
    one_buffer_sec = 256 / 44100
    beats = [10.0, 10.5]
    blocks = [{"t": 10.0 + LOOK_AHEAD + one_buffer_sec, "intent": "GROOVE",
               "end": 20.0}]

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
    assert alignment == (0, 0, 0, 0, 0)


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
        {"t": 20.0, "intent": "DROP", "end": 30.0},
    ]
    rows, stats = join(sections, beats, intents, duration_sec=30.0)

    assert stats.intent_blocks_song_stamped == 1
    assert stats.intent_reattributed == 2
    assert labels(rows, "intent_at_beat") == ["GROOVE"] * 4 + ["DROP"]


def test_join_reports_no_reattribution_when_every_block_came_off_the_queue():
    sections = [(0.0, 40.0, "drop")]
    beats = [beat(t) for t in (10.0, 11.0, 12.0)]
    intents = [queue_block(10.0, "GROOVE", 30.0)]
    _rows, stats = join(sections, beats, intents, duration_sec=30.0)

    assert stats.intent_blocks_song_stamped == 0
    assert stats.intent_reattributed == 0


def test_every_feature_column_reads_a_key_the_report_still_carries():
    live = {"t": 1.0, "bpm": 128.0, "strength": 0.0, "change": False, "rms": 0.1}
    rows, _stats = join([(0.0, 30.0, "drop")], [live])

    derived = {"track_id", "youtube_id", "t_song", "intent_at_beat",
               "label_canonical", "label_raw", "label_v1", "bar_position_unknown"}
    derived |= {f"{column}_z" for column in CONTINUOUS_COLUMNS}

    assert set(rows[0]) - derived <= set(live)
    assert set(TABLE_HEADER) - derived <= set(live)


def test_feature_columns_are_copied_through_verbatim():
    sections = [(0.0, 30.0, "drop")]
    rows, _stats = join(sections, [beat(1.0, bpm=124.5, rms=0.09)])
    row = rows[0]

    assert row["bpm"] == 124.5
    assert row["rms"] == 0.09


def test_bar_position_is_flagged_unknown_on_every_row():
    rows, _stats = join([(0.0, 30.0, "drop")], [beat(1.0), beat(2.0)])

    assert {row["bar_position_unknown"] for row in rows} == {BAR_POSITION_UNKNOWN}


def test_track_identity_is_on_every_row():
    rows, _stats = join([(0.0, 30.0, "drop")], [beat(1.0)])

    assert rows[0]["track_id"] == "0001.abc"
    assert rows[0]["youtube_id"] == "abc"


def test_zscores_are_zero_for_a_flat_feature():
    assert zscores([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]


def test_zscores_use_the_population_standard_deviation():
    result = zscores([1.0, 2.0, 3.0])

    assert result[1] == pytest.approx(0.0)
    assert result[0] == pytest.approx(-math.sqrt(1.5))
    assert result[2] == pytest.approx(math.sqrt(1.5))


def test_zscores_are_computed_over_the_kept_rows_of_one_track():
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


def write_corpus(tmp_path: Path, tracks: dict) -> tuple:
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
    rows, sections = write_corpus(tmp_path, {
        "0001.abc": ("abc", [(0.0, 30.0, "drop")], report([beat(1.0), beat(2.0)])),
        "0002.def": ("def", [(0.0, 30.0, "intro")], report([beat(3.0)])),
    })

    build_table(tmp_path, rows, sections)
    first = (tmp_path / TABLE_FILE).read_bytes()
    build_table(tmp_path, rows, sections)

    assert (tmp_path / TABLE_FILE).read_bytes() == first


def test_an_unsimulated_track_is_recorded_not_silently_dropped(tmp_path):
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


FRESH = {"cache_version": CACHE_VERSION, "pipeline_sha": "sha1",
         "mp3_size": 100, "mp3_mtime": 1.5, "report": {"beats": []}}


def test_cache_is_fresh_when_pipeline_and_audio_match():
    assert cache_is_fresh(FRESH, "sha1", 100, 1.5)


@pytest.mark.parametrize("sha,size,mtime", [
    ("sha2", 100, 1.5),
    ("sha1", 101, 1.5),
    ("sha1", 100, 2.0),
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


def test_a_missing_sidecar_is_not_a_reason_to_re_simulate(tmp_path):
    rows = stage_track(tmp_path, sidecar=False)

    jobs, counts = select(tmp_path, rows)

    assert jobs == []
    assert counts["hit"] == 1


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


def stamp_sidecar(path: Path, version=MEL_EXPORTER_VERSION) -> None:
    payload = {"mel": np.zeros((1, 40), dtype=np.float32)}
    if version is not None:
        payload[MEL_EXPORTER_KEY] = np.int32(version)
    np.savez_compressed(path, **payload)


def test_a_sidecar_from_another_exporter_generation_is_excluded_not_re_simulated(tmp_path):
    rows = stage_track(tmp_path)
    stamp_sidecar(tmp_path / FEATURES_DIR / "abc.npz", MEL_EXPORTER_VERSION + 1)

    jobs, counts = select(tmp_path, rows)

    assert jobs == []
    assert counts["hit"] == 1


def test_a_sidecar_from_this_exporter_generation_is_a_hit(tmp_path):
    rows = stage_track(tmp_path)
    stamp_sidecar(tmp_path / FEATURES_DIR / "abc.npz")

    jobs, counts = select(tmp_path, rows)

    assert jobs == []
    assert counts["hit"] == 1


def test_an_unstamped_sidecar_is_grandfathered_rather_than_rebuilt(tmp_path):
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
    baseline = staged_repo(tmp_path)

    (tmp_path / "lib" / "CLAUDE.md").write_text("docs, revised\n", encoding="utf-8")
    assert pipeline_sha(tmp_path) == baseline

    (tmp_path / "lib" / "engine.py").write_text("x = 2\n", encoding="utf-8")
    assert pipeline_sha(tmp_path).startswith(f"{baseline}+dirty")


def test_a_committed_tree_keys_on_the_bare_commit_sha(tmp_path):
    baseline = staged_repo(tmp_path)

    assert pipeline_sha(tmp_path) == baseline
    assert "+" not in baseline


def test_two_different_uncommitted_edits_get_two_different_keys(tmp_path):
    staged_repo(tmp_path)
    source = tmp_path / "lib" / "engine.py"

    source.write_text("x = 2\n", encoding="utf-8")
    first = pipeline_sha(tmp_path)
    source.write_text("x = 3\n", encoding="utf-8")
    second = pipeline_sha(tmp_path)

    assert first != second
    assert "+dirty" in first and "+dirty" in second


def test_the_dirty_key_returns_when_the_edit_does(tmp_path):
    staged_repo(tmp_path)
    source = tmp_path / "lib" / "engine.py"

    source.write_text("x = 2\n", encoding="utf-8")
    edited = pipeline_sha(tmp_path)
    source.write_text("x = 3\n", encoding="utf-8")
    source.write_text("x = 2\n", encoding="utf-8")

    assert pipeline_sha(tmp_path) == edited


def test_an_untracked_pipeline_source_moves_the_key(tmp_path):
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


def write_manifest(tmp_path: Path, tracks: list) -> None:
    lines = ["track_id,youtube_id,mp3_path,decoded_duration_sec,status"]
    lines += [f"{track_id},{track_id[-3:]},audio/{track_id}.mp3,{seconds:.3f},ok"
              for track_id, seconds in tracks]
    (tmp_path / CLEAN_MANIFEST_FILE).write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")


def test_a_track_reaching_the_analyser_reset_is_left_out_of_the_build(tmp_path, capsys):
    write_manifest(tmp_path, [("0001.short", 300.0),
                              ("0002.long", ANALYSER_RESET_SEC)])

    rows = load_ok_rows(tmp_path)

    assert [row["track_id"] for row in rows] == ["0001.short"]
    printed = capsys.readouterr().out
    assert "0002.long" in printed and "self-reset" in printed


def test_a_track_just_under_the_horizon_still_builds(tmp_path):
    # The longest track in the corpus clears the horizon by 0.11 s.
    write_manifest(tmp_path, [("0001.edge", ANALYSER_RESET_SEC - 0.111)])

    assert [row["track_id"] for row in load_ok_rows(tmp_path)] == ["0001.edge"]


def test_a_manifest_of_nothing_but_over_long_tracks_fails_loudly(tmp_path):
    write_manifest(tmp_path, [("0001.long", ANALYSER_RESET_SEC + 60.0)])

    with pytest.raises(RuntimeError, match="nothing to build from"):
        load_ok_rows(tmp_path)


def test_a_blank_duration_is_not_this_gates_business(tmp_path):
    (tmp_path / CLEAN_MANIFEST_FILE).write_text(
        "track_id,youtube_id,mp3_path,decoded_duration_sec,status\n"
        "0001.blank,ank,audio/a.mp3,,ok\n", encoding="utf-8")

    assert [row["track_id"] for row in load_ok_rows(tmp_path)] == ["0001.blank"]


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


def test_the_clamped_tail_tripwire_still_sees_a_recorded_block():
    blocks = [{"t": 10.0, "song_t": 5.0, "intent": "DROP", "end": 30.0},
              {"t": 30.0, "song_t": 20.0, "intent": "BREAKDOWN", "end": 30.0}]

    _, alignment = realign_intents(blocks, LOOK_AHEAD, [5.0, 20.0],
                                   duration_sec=30.0)

    assert alignment.song_recorded == 2
    assert alignment.clamped_tail == 1


def test_a_block_committed_past_the_budget_is_counted():
    blocks = [{"t": 10.0, "song_t": 6.0, "intent": "DROP", "end": 40.0},
              {"t": 26.5, "song_t": 24.0, "intent": "BREAKDOWN", "end": 60.0}]

    _, alignment = realign_intents(blocks, LOOK_AHEAD, [6.0, 24.0],
                                   duration_sec=60.0)

    assert alignment.late == 1


def test_a_block_inside_the_budget_is_not_late():
    blocks = [{"t": 8.5, "song_t": 6.0, "intent": "DROP", "end": 40.0}]

    _, alignment = realign_intents(blocks, LOOK_AHEAD, [6.0], duration_sec=60.0)

    assert alignment.late == 0


def test_lateness_ships_with_the_denominator_that_makes_it_readable():
    inferred = [queue_block(6.0, "DROP", 40.0)]
    _, alignment = realign_intents(inferred, LOOK_AHEAD, [6.0], duration_sec=60.0)
    assert (alignment.late, alignment.song_recorded) == (0, 0)

    recorded = [{"t": 8.5, "song_t": 6.0, "intent": "DROP", "end": 40.0}]
    _, alignment = realign_intents(recorded, LOOK_AHEAD, [6.0], duration_sec=60.0)
    assert (alignment.late, alignment.song_recorded) == (0, 1)


def test_join_stats_carry_the_lateness_denominator_to_the_batch():
    sections = [(0.0, 40.0, "drop")]
    beats = [beat(t) for t in (6.0, 7.0, 8.0)]
    intents = [{"t": 8.5, "song_t": 6.0, "intent": "DROP", "end": 40.0}]
    _rows, stats = join(sections, beats, intents, duration_sec=40.0)

    assert stats.intent_blocks_song_recorded == 1
    assert stats.intent_blocks_late == 0


def test_the_batch_tidies_every_file_one_simulation_leaves(tmp_path):
    from build_training_table import derived_cache_paths

    mp3 = tmp_path / "0001.abc.mp3"
    derived = derived_cache_paths(str(mp3))

    assert len(derived) == 2
    assert any(path.endswith(".npy") for path in derived)
    assert any(path.endswith(".mertcells.npz") for path in derived)
    assert all(Path(path).parent == tmp_path for path in derived)


def test_pre_existing_caches_of_both_kinds_are_seen(tmp_path):
    from build_training_table import AUDIO_DIR, find_caches

    audio = tmp_path / AUDIO_DIR
    audio.mkdir()
    (audio / "a.mp3.44100.npy").write_bytes(b"")
    (audio / "b.mp3.librosa.mertcells.npz").write_bytes(b"")

    assert len(find_caches(tmp_path)) == 2


def test_the_batch_deletes_the_cell_sidecar_it_wrote_even_beside_an_old_decode_cache(tmp_path):
    from build_training_table import derived_cache_paths

    mp3 = tmp_path / "0001.abc.mp3"
    decode, cells = derived_cache_paths(str(mp3))

    job = _job_with(preexisting=(decode,))
    assert decode in job.preexisting
    assert cells not in job.preexisting

    job = _job_with(preexisting=(cells,))
    assert cells in job.preexisting
    assert decode not in job.preexisting


def _job_with(**over):
    from build_training_table import SimJob

    fields = dict(track_id="0001", youtube_id="abc", mp3_path="a.mp3",
                  report_path="c", sidecar_path="s", preexisting=(),
                  pipeline_sha="sha", mp3_size=1, mp3_mtime=1.0)
    fields.update(over)
    return SimJob(**fields)
