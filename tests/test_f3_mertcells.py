import json
import math
from pathlib import Path

import numpy as np
import pytest

from lib.analyser.mert_stream import (CellAccumulator, StreamingResampler,
                                      encoder_samples)
from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
from simulate.cell_cache import open_replay
from training.f3_mertcells import (bookkeeping, construct,
                                   reset_from_first_trigger)

HOP_SEC = 1.0
MARGIN_SEC = 3.0
LABEL_FRAME_SEC = 4096 / 44100
ENCODER_RATE = 24000


def measured_triggers(total_pushed: int, reset_at: int) -> list:
    resampler = StreamingResampler(SAMPLE_RATE, ENCODER_RATE)
    hop = encoder_samples(HOP_SEC)
    written = 0
    passes = 0
    pushed = 0
    triggers = []
    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    while pushed < total_pushed:
        if reset_at and pushed == reset_at:
            resampler.reset()
            written = 0
            passes = 0
        written += len(resampler.push(buffer))
        pushed += BUFFER_SIZE
        while written >= (passes + 1) * hop:
            passes += 1
            triggers.append(pushed)
    return triggers


@pytest.mark.parametrize("reset_at", [0, 13312, 26624, 39936])
def test_the_trigger_formula_matches_the_real_resampler(reset_at):
    total = reset_at + 8 * SAMPLE_RATE
    total = BUFFER_SIZE * (total // BUFFER_SIZE)
    triggers, _, _, _ = bookkeeping(total, reset_at, margin_sec=MARGIN_SEC,
                                    hop_sec=HOP_SEC,
                                    label_frame_sec=LABEL_FRAME_SEC)
    assert list(triggers) == measured_triggers(total, reset_at)
    assert len(triggers) >= 6


def test_the_cell_schedule_matches_the_real_accumulator():
    total = 12 * SAMPLE_RATE
    triggers, offsets, indices, seen = bookkeeping(
        total, 0, margin_sec=MARGIN_SEC, hop_sec=HOP_SEC,
        label_frame_sec=LABEL_FRAME_SEC)
    hop = encoder_samples(HOP_SEC)
    margin = encoder_samples(MARGIN_SEC)
    cells = CellAccumulator(1, 2, LABEL_FRAME_SEC)
    frame_times = np.arange(0.0, 12.0, 320 / ENCODER_RATE) + 0.5 * 320 / ENCODER_RATE
    rows = np.ones((len(frame_times), 1, 2), dtype=np.float64)
    lo = 0.0
    emitted = []
    for k in range(len(triggers)):
        hi = ((k + 1) * hop - margin) / ENCODER_RATE
        if hi <= lo:
            assert offsets[k + 1] == offsets[k]
            continue
        keep = (frame_times >= lo) & (frame_times < hi)
        cells.add(rows[keep], frame_times[keep], lo, hi)
        drained = cells.drain(hi)
        emitted.extend(index for index, _ in drained)
        assert offsets[k + 1] - offsets[k] == len(drained)
        lo = hi
    assert emitted == list(indices)
    assert [(k + 1) * hop / ENCODER_RATE for k in range(len(triggers))
            for _ in range(offsets[k + 1] - offsets[k])] == list(seen)


def test_reset_from_first_trigger_inverts_the_schedule():
    for reset_at in (0, 13312, 26624, 39936):
        triggers, _, _, _ = bookkeeping(reset_at + 4 * SAMPLE_RATE, reset_at,
                                        margin_sec=MARGIN_SEC, hop_sec=HOP_SEC,
                                        label_frame_sec=LABEL_FRAME_SEC)
        assert reset_from_first_trigger(int(triggers[0]), hop_sec=HOP_SEC,
                                        margin_sec=MARGIN_SEC) == reset_at


def fake_f3(path, n_cells: int) -> None:
    emb = (np.arange(n_cells * 2 * 3, dtype=np.float32)
           .reshape(n_cells, 2, 3).astype(np.float16))
    np.savez(path, emb=emb, label_frame_sec=np.float64(LABEL_FRAME_SEC),
             stream_margin_sec=np.float64(MARGIN_SEC),
             stream_hop_sec=np.float64(HOP_SEC),
             stream_buffer_sec=np.float64(30.0))


def replayed_cells(path, key, total_pushed):
    replay, reason = open_replay(path, key, expected_samples=total_pushed)
    assert reason == "hit"
    cells = []
    pushed = 0
    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    while pushed < total_pushed:
        replay.push_audio(buffer)
        pushed += BUFFER_SIZE
        while replay.due():
            cells.extend(replay.run_pass())
    return cells


def test_construct_round_trips_through_the_replay(tmp_path):
    key = {"schema": "mert-cells/1", "extractor": "x", "decode": "librosa",
           "encoder": {}, "framing": {"label_frame_sec": LABEL_FRAME_SEC},
           "backend": {}, "source_rate": SAMPLE_RATE,
           "audio_size": 1, "audio_mtime": 1.0}
    f3 = tmp_path / "track.npz"
    fake_f3(f3, 200)
    total = BUFFER_SIZE * ((10 * SAMPLE_RATE) // BUFFER_SIZE)
    out = tmp_path / "track.mp3.librosa.mertcells.npz"
    built = construct(f3, out, key=key, total_pushed=total, reset_at=13312,
                      cell_shift=3)
    cells = replayed_cells(out, key, total)
    assert len(cells) == built["cells"] > 0
    emb = np.load(f3)["emb"]
    for cell in cells:
        expected = emb[cell.index + 3].reshape(-1).astype(np.float32)
        assert np.array_equal(cell.features, expected)
    assert [cell.index for cell in cells] == list(range(len(cells)))
    assert cells[0].time_sec == pytest.approx(LABEL_FRAME_SEC)


def test_construct_truncates_when_the_f3_track_runs_short(tmp_path):
    key = {"schema": "mert-cells/1", "extractor": "x", "decode": "librosa",
           "encoder": {}, "framing": {"label_frame_sec": LABEL_FRAME_SEC},
           "backend": {}, "source_rate": SAMPLE_RATE,
           "audio_size": 1, "audio_mtime": 1.0}
    f3 = tmp_path / "short.npz"
    fake_f3(f3, 40)
    total = BUFFER_SIZE * ((10 * SAMPLE_RATE) // BUFFER_SIZE)
    out = tmp_path / "short.mp3.librosa.mertcells.npz"
    built = construct(f3, out, key=key, total_pushed=total, reset_at=0,
                      cell_shift=5)
    assert built["cells"] == 35
    cells = replayed_cells(out, key, total)
    assert [cell.index for cell in cells] == list(range(35))
    with np.load(out) as archive:
        offsets = archive["pass_offset"]
        assert int(offsets[-1]) == 35
        assert len(offsets) == len(archive["pass_trigger"]) + 1
