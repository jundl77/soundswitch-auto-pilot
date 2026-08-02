import math
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from build_training_table import write_feature_sidecar  # noqa: E402
from nn.dataset import FRAME_SEC, LABEL_POOL, WINDOW_FRAMES  # noqa: E402
from nn.downbeat_dataset import (  # noqa: E402
    BEATS_PER_BAR,
    DOWNBEAT_SIGMA_SEC,
    GAP_FACTOR,
    DownbeatWindowDataset,
    alignment_profile,
    grid_anomalies,
    load_beat_grid,
    load_beat_grids,
    parse_beat_grid,
    track_downbeat_targets,
)

from tests.test_nn_dataset import corpus_tracks, fake_corpus  # noqa: E402


def grid_rows(bpm=120.0, bars=16, start=0.5, start_phase=1, section="drop"):
    period = 60.0 / bpm
    rows = []
    time, phase = start, start_phase
    for _index in range(bars * BEATS_PER_BAR):
        rows.append((round(time, 4), phase, section))
        time += period
        phase = phase % BEATS_PER_BAR + 1
    return rows


def write_grid(path, rows, header="time,downbeat,section"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header] + [f"{time},{phase},{label}" for time, phase, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def frame_of(song_time):
    return int(round(song_time / FRAME_SEC)) - 1


def targets_for(rows, duration=120.0):
    grid = parse_beat_grid(rows)
    n_frames = int(duration / FRAME_SEC)
    return grid, track_downbeat_targets(grid, n_frames, FRAME_SEC, FRAME_SEC)


def test_sigma_is_the_evaluation_tolerance():
    assert DOWNBEAT_SIGMA_SEC == pytest.approx(0.070)


def test_sigma_is_narrow_enough_that_a_neighbouring_bar_leaks_nothing():
    for bpm in (87.0, 120.0, 174.0, 200.0):
        beat_period = 60.0 / bpm
        assert math.exp(-0.5 * (beat_period / DOWNBEAT_SIGMA_SEC) ** 2) < 1e-3


def test_parse_reads_times_phases_and_derives_the_bar_grid():
    grid = parse_beat_grid(grid_rows(bpm=120.0, bars=8, start=0.5))

    assert len(grid.times) == 8 * BEATS_PER_BAR
    assert grid.times[0] == pytest.approx(0.5)
    assert grid.phases[0] == 1
    assert grid.bars == 8
    assert len(grid.downbeat_times) == 8
    assert np.allclose(np.diff(grid.downbeat_times), 4 * 0.5)
    assert not len(grid.phase_breaks)


def test_parse_keeps_a_grid_that_starts_mid_bar():
    # 254 of the corpus's 1,423 grids start on phase 2, 3 or 4.
    grid = parse_beat_grid(grid_rows(bars=4, start_phase=3))

    assert grid.phases[0] == 3
    assert list(grid.phases[:6]) == [3, 4, 1, 2, 3, 4]
    assert not len(grid.phase_breaks)
    assert len(grid.downbeat_times) == grid.bars


def test_phases_cycle_one_to_four_over_a_whole_grid():
    grid = parse_beat_grid(grid_rows(bars=32))
    expected = np.tile(np.arange(1, BEATS_PER_BAR + 1), 32)
    assert np.array_equal(grid.phases, expected.astype(grid.phases.dtype))


def test_load_beat_grid_reads_a_csv_off_disk(tmp_path):
    path = write_grid(tmp_path / "0001.abc.beat.csv", grid_rows(bars=6))
    grid = load_beat_grid(path)
    assert grid.bars == 6
    assert grid.source == str(path)


@pytest.mark.parametrize("rows, header, match", [
    ([], "time,downbeat,section", "no beats"),
    (grid_rows(bars=2), "time,beat,section", "column"),
    ([(0.5, 1, "drop"), ("later", 2, "drop")], "time,downbeat,section", "not a number"),
    ([(0.5, 1, "drop"), (1.0, 9, "drop")], "time,downbeat,section", "phase"),
    ([(0.5, 1, "drop"), (1.0, 0, "drop")], "time,downbeat,section", "phase"),
    ([(0.5, 1, "drop"), (0.4, 2, "drop")], "time,downbeat,section", "increas"),
    ([(0.5, 1, "drop"), (0.5, 2, "drop")], "time,downbeat,section", "increas"),
    ([(-0.5, 1, "drop"), (0.5, 2, "drop")], "time,downbeat,section", "negative"),
])
def test_a_malformed_grid_raises_and_names_the_file(tmp_path, rows, header, match):
    path = write_grid(tmp_path / "0001.abc.beat.csv", rows, header=header)
    with pytest.raises(RuntimeError, match=match) as raised:
        load_beat_grid(path)
    assert "0001.abc.beat.csv" in str(raised.value)


def test_a_fractional_phase_is_refused_rather_than_truncated():
    with pytest.raises(RuntimeError, match="whole number"):
        parse_beat_grid([(0.5, 1, "drop"), (1.0, 2.5, "drop")])


def test_a_non_numeric_row_is_refused_by_the_pure_parser():
    with pytest.raises(RuntimeError, match="not a number"):
        parse_beat_grid([(0.5, 1, "drop"), ("later", 2, "drop")])


def test_a_missing_grid_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        load_beat_grid(tmp_path / "nothing.beat.csv")


def test_a_bar_missing_a_beat_keeps_its_downbeats_and_records_the_phase_break():
    rows = grid_rows(bars=4)
    beat_carrying_phase_2 = 5
    rows.pop(beat_carrying_phase_2)

    grid = parse_beat_grid(rows)

    assert list(grid.phase_breaks) == [4]
    assert grid.bars == 4


def test_a_healthy_grid_has_no_anomalies_even_when_it_starts_mid_bar():
    assert grid_anomalies(parse_beat_grid(grid_rows(bars=40))) == []
    assert grid_anomalies(parse_beat_grid(grid_rows(bars=40, start_phase=3))) == []


def test_a_stuck_phase_column_is_caught_by_both_invariants():
    rows = [(time, 1, section) for time, _phase, section in grid_rows(bars=40)]

    reasons = grid_anomalies(parse_beat_grid(rows))

    assert len(reasons) == 2
    assert any("cycle" in reason for reason in reasons)
    assert any("one per bar" in reason for reason in reasons)


def test_a_repeated_phase_pair_is_caught_even_though_downbeats_look_sane():
    rows = [(time, (index // 2) % BEATS_PER_BAR + 1, section)
            for index, (time, _phase, section) in enumerate(grid_rows(bars=40))]

    reasons = grid_anomalies(parse_beat_grid(rows))

    assert any("one per bar" in reason for reason in reasons)


def test_one_broken_bar_is_not_an_anomaly():
    rows = grid_rows(bars=40)
    rows.pop(5)
    assert grid_anomalies(parse_beat_grid(rows)) == []


def test_target_peaks_on_the_downbeat_frame_with_a_gaussian_shape():
    rows = grid_rows(bpm=120.0, bars=8, start=1.0)
    grid, targets = targets_for(rows)
    downbeat = grid.downbeat_times[2]

    peak = frame_of(downbeat)
    assert abs(int(np.argmax(targets.downbeat[peak - 4:peak + 5])) - 4) <= 1
    for offset in (-2, -1, 0, 1, 2):
        index = peak + offset
        delta = (FRAME_SEC * (index + 1)) - downbeat
        expected = math.exp(-0.5 * (delta / DOWNBEAT_SIGMA_SEC) ** 2)
        assert targets.downbeat[index] == pytest.approx(expected, abs=1e-5)


def test_offbeats_are_negatives_not_targets():
    grid, targets = targets_for(grid_rows(bpm=120.0, bars=8, start=1.0))

    for index, phase in enumerate(grid.phases):
        if phase == 1:
            continue
        assert targets.downbeat[frame_of(grid.times[index])] < 0.01


def test_every_downbeat_gets_a_peak_and_nothing_else_does():
    grid, targets = targets_for(grid_rows(bpm=128.0, bars=20, start=0.7))

    past_the_masked_leading_edge = grid.downbeat_times[1:]
    for instant in past_the_masked_leading_edge:
        near = frame_of(instant)
        assert targets.downbeat[near - 1:near + 2].max() > 0.85

    peaks = np.flatnonzero(targets.downbeat > 0.85)
    for peak in peaks:
        distance = np.abs(grid.downbeat_times - FRAME_SEC * (peak + 1)).min()
        assert distance <= FRAME_SEC


def test_the_target_is_flat_zero_between_bars():
    grid, targets = targets_for(grid_rows(bpm=120.0, bars=8, start=1.0))
    half_bar_past_a_downbeat = grid.downbeat_times[3] + 1.0
    assert targets.downbeat[frame_of(half_bar_past_a_downbeat)] == pytest.approx(0.0, abs=1e-6)


def test_audio_outside_the_grid_is_masked_and_zeroed():
    rows = grid_rows(bpm=120.0, bars=8, start=20.0)
    grid, targets = targets_for(rows, duration=120.0)

    assert not targets.mask[:frame_of(19.5)].any()
    assert not targets.mask[frame_of(grid.times[-1] + 1.0):].any()
    assert targets.mask[frame_of(30.0)]
    assert (targets.downbeat[~targets.mask] == 0.0).all()


def test_a_hole_in_the_grid_is_masked_rather_than_taught_as_silence():
    rows = grid_rows(bpm=120.0, bars=8, start=1.0)
    two_bars = 2 * BEATS_PER_BAR
    kept = rows[:12] + rows[12 + two_bars:]
    grid, targets = targets_for(kept)

    hole_start = kept[11][0]
    hole_end = kept[12][0]
    assert hole_end - hole_start > GAP_FACTOR * 0.5
    assert not targets.mask[frame_of(hole_start + 0.5):frame_of(hole_end - 0.5)].any()
    assert targets.mask[frame_of(hole_start - 1.0)]
    assert targets.mask[frame_of(hole_end + 1.0)]


def test_beat_labels_carry_phase_time_and_frame_together():
    grid, targets = targets_for(grid_rows(bpm=140.0, bars=10, start=0.9))

    assert np.array_equal(targets.beat_phase, grid.phases)
    assert np.allclose(targets.beat_time, grid.times)
    stamped = FRAME_SEC * (targets.beat_frame + 1)
    assert np.all(np.abs(stamped - grid.times) <= FRAME_SEC / 2 + 1e-9)
    downbeats = targets.beat_frame[targets.beat_phase == 1]
    assert (targets.downbeat[downbeats[1:]] > 0.85).all()


def test_the_mask_stops_at_the_first_beat_even_next_to_a_downbeat():
    four_ms = 0.004
    frame_before_the_first_beat, frame_of_the_first_beat = 4, 5
    start = 5 * FRAME_SEC + four_ms
    _grid, targets = targets_for(grid_rows(bpm=120.0, bars=8, start=start))

    assert not targets.mask[frame_before_the_first_beat]
    assert targets.downbeat[frame_before_the_first_beat] == 0.0
    assert targets.mask[frame_of_the_first_beat]
    assert 0.8 < targets.downbeat[frame_of_the_first_beat] < 0.9


def test_beats_past_the_end_of_the_mel_are_marked_not_clipped():
    grid, targets = targets_for(grid_rows(bpm=120.0, bars=16, start=1.0), duration=10.0)

    inside = targets.beat_frame >= 0
    assert inside.any() and not inside.all()
    assert (targets.beat_frame[~inside] == -1).all()
    assert (targets.beat_time[~inside] > 10.0 - FRAME_SEC).all()


def test_a_beat_before_the_first_frame_stamp_is_frame_zero_not_the_sentinel():
    # 72 corpus grids open before the first frame's stamp; 54 of those beats are downbeats.
    rows = grid_rows(bpm=120.0, bars=8, start=0.0)
    grid, targets = targets_for(rows)

    assert grid.times[0] == 0.0
    assert grid.phases[0] == 1
    assert targets.beat_frame[0] == 0
    assert (targets.beat_frame >= 0).all()
    assert targets.mask[0] and targets.downbeat[0] > 0.5


def test_a_grid_with_no_downbeats_produces_an_empty_but_valid_target():
    grid, targets = targets_for([(1.0, 2, "drop"), (1.5, 3, "drop")])

    assert grid.bars == 0
    assert targets.downbeat.max() == 0.0
    assert targets.mask[frame_of(1.2)]


def test_alignment_profile_finds_the_shift_the_onsets_actually_sit_at():
    grid, targets = targets_for(grid_rows(bpm=120.0, bars=12, start=1.0))
    n_frames = len(targets.downbeat)
    mel = np.zeros((n_frames, 40), dtype=np.float32)
    for frame in targets.beat_frame[targets.beat_phase == 1]:
        one_frame_late = frame + 1
        if 0 <= one_frame_late < n_frames:
            mel[one_frame_late, :4] = 1.0

    profile = alignment_profile(mel, targets)

    assert max(profile["downbeat"], key=profile["downbeat"].get) == 1
    assert profile["downbeat"][1] > 5.0
    assert max(profile["beat"].values()) < 1.0
    assert grid.bars == 12


def with_grids(tmp_path, count=4, frames=3000, bpm=128.0, bars=200):
    tracks = corpus_tracks(count)
    data_dir, _eval_set = fake_corpus(tmp_path, tracks, frames=frames)
    for track_id, _youtube, _title in tracks:
        write_grid(data_dir / "annotations" / "beats" / f"{track_id}.beat.csv",
                   grid_rows(bpm=bpm, bars=bars, start=1.0))
    ids = [youtube for _t, youtube, _title in tracks]
    return data_dir, ids


def test_dataset_yields_mel_target_and_mask_at_the_documented_shapes(tmp_path):
    data_dir, ids = with_grids(tmp_path)
    data = DownbeatWindowDataset(data_dir, ids)

    mel, downbeat, mask = data[0]
    assert mel.shape == (WINDOW_FRAMES, 40) and mel.dtype == np.float32
    assert downbeat.shape == (WINDOW_FRAMES,) and downbeat.dtype == np.float32
    assert mask.shape == (WINDOW_FRAMES,) and mask.dtype == np.bool_
    assert data.track_ids() == ids


def test_window_targets_are_the_track_targets_sliced_at_the_offset(tmp_path):
    data_dir, ids = with_grids(tmp_path)
    data = DownbeatWindowDataset(data_dir, ids)

    index = len(data) // 2
    offset = data.window_offset(index)
    whole = data.targets_for(data.track_id_of(index))
    _mel, downbeat, mask = data[index]

    end = min(offset + WINDOW_FRAMES, len(whole.downbeat))
    assert np.array_equal(downbeat[:end - offset], whole.downbeat[offset:end])
    assert np.array_equal(mask[:end - offset], whole.mask[offset:end])


def test_eval_mode_is_deterministic_and_augmentation_leaves_targets_alone(tmp_path):
    data_dir, ids = with_grids(tmp_path)
    plain = DownbeatWindowDataset(data_dir, ids)
    jittered = DownbeatWindowDataset(data_dir, ids, augment=True)

    first = plain[2][0].copy()
    plain.set_epoch(7)
    assert np.array_equal(plain[2][0], first)

    offset = jittered.window_offset(0)
    assert offset % LABEL_POOL == 0
    assert np.array_equal(jittered[0][1], plain.window(0, offset)[1])


def test_padding_past_the_end_of_a_short_track_is_masked(tmp_path):
    data_dir, ids = with_grids(tmp_path, count=1, frames=WINDOW_FRAMES // 2, bars=8)
    data = DownbeatWindowDataset(data_dir, ids)
    mel, downbeat, mask = data[0]

    assert (mel[WINDOW_FRAMES // 2:] == 0.0).all()
    assert not mask[WINDOW_FRAMES // 2:].any()
    assert (downbeat[WINDOW_FRAMES // 2:] == 0.0).all()


def test_dataset_refuses_a_track_with_no_beat_grid(tmp_path):
    data_dir, ids = with_grids(tmp_path, count=2)
    (data_dir / "annotations" / "beats" / "0001.id00000001.beat.csv").unlink()

    with pytest.raises(RuntimeError, match="beat grid"):
        DownbeatWindowDataset(data_dir, ids)


def test_dataset_refuses_an_id_whose_grid_was_not_supplied(tmp_path):
    data_dir, ids = with_grids(tmp_path, count=1)
    sections = {ids[0]: [(0.0, 30.0, "intro"), (30.0, 60.0, "drop")]}

    with pytest.raises(RuntimeError, match="no beat grid"):
        DownbeatWindowDataset(data_dir, ids, sections_by_youtube_id=sections,
                              grids_by_youtube_id={})


def test_dataset_refuses_a_track_whose_grid_has_no_bars(tmp_path):
    data_dir, ids = with_grids(tmp_path, count=1)
    rows = [(t, 2 if i % 2 else 3, "drop")
            for i, (t, _p, _s) in enumerate(grid_rows(bars=40))]
    write_grid(data_dir / "annotations" / "beats" / "0000.id00000000.beat.csv", rows)

    with pytest.raises(RuntimeError, match="downbeat"):
        DownbeatWindowDataset(data_dir, ids)


def test_dataset_refuses_a_grid_whose_phase_column_says_nothing(tmp_path):
    data_dir, ids = with_grids(tmp_path, count=1)
    rows = [(t, 1, s) for t, _p, s in grid_rows(bars=40)]
    write_grid(data_dir / "annotations" / "beats" / "0000.id00000000.beat.csv", rows)

    with pytest.raises(RuntimeError, match="not believable"):
        DownbeatWindowDataset(data_dir, ids)


def test_load_beat_grids_reports_what_it_could_not_load(tmp_path):
    data_dir, ids = with_grids(tmp_path, count=3)
    (data_dir / "annotations" / "beats" / "0002.id00000002.beat.csv").unlink()

    grids, missing = load_beat_grids(data_dir, ids)

    assert sorted(grids) == ["id00000000", "id00000001"]
    assert missing == ["id00000002"]
    assert all(grid.bars == 200 for grid in grids.values())
