"""The live MERT extractor's geometry -- the part a show has to obey."""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from lib.analyser import mert_stream as M
from tests.fixtures import offline_stream_extract as OFFLINE

SR = M.ENCODER_SAMPLE_RATE


def _samples(seconds: float) -> int:
    return M.encoder_samples(seconds)


def _schedule(track_sec, *, margin=3.0, hop_sec=1.0, buffer_sec=30.0):
    return list(M.pass_schedule(_samples(track_sec), length=_samples(buffer_sec),
                                hop=_samples(hop_sec), margin=_samples(margin)))


def _noise(n, seed=0):
    return np.random.default_rng(seed).normal(size=n).astype(np.float32)


def test_the_streaming_resampler_reproduces_a_whole_array_resample():
    audio = _noise(44100 * 7 + 913)
    resampler = M.StreamingResampler()
    out = [resampler.push(audio[i:i + 256]) for i in range(0, len(audio), 256)]
    out.append(resampler.flush())
    live = np.concatenate(out)
    offline = resample_poly(audio, resampler.up, resampler.down).astype(np.float32)
    assert len(live) == len(offline)
    assert np.array_equal(live, offline)


@pytest.mark.parametrize("block", [1, 147, 256, 4096, 44100])
def test_the_resampler_is_indifferent_to_how_the_audio_is_chopped(block):
    audio = _noise(44100 * 3 + 77, seed=block)
    resampler = M.StreamingResampler()
    parts = [resampler.push(audio[i:i + block]) for i in range(0, len(audio), block)]
    parts.append(resampler.flush())
    assert np.array_equal(np.concatenate(parts),
                          resample_poly(audio, 80, 147).astype(np.float32))


def test_the_resampler_never_emits_audio_it_has_not_been_given():
    resampler = M.StreamingResampler()
    produced = 0
    fed = 0
    for _ in range(400):
        produced += len(resampler.push(_noise(256, seed=produced)))
        fed += 256
        assert produced <= math.ceil(fed * resampler.up / resampler.down)


def test_reset_restores_the_head_of_stream_padding():
    audio = _noise(44100)
    first = M.StreamingResampler()
    first.push(audio)
    first.flush()
    first.reset()
    second = M.StreamingResampler()
    assert np.array_equal(
        np.concatenate([first.push(audio), first.flush()]),
        np.concatenate([second.push(audio), second.flush()]))


def test_the_resampler_ratio_comes_from_the_two_rates():
    resampler = M.StreamingResampler()
    assert (resampler.up, resampler.down) == (80, 147)


def test_the_ring_reports_a_monotonic_sample_index():
    ring = M.SampleRing(1000)
    assert ring.written == 0
    ring.write(np.arange(300, dtype=np.float32))
    ring.write(np.arange(300, dtype=np.float32))
    assert ring.written == 600


def test_the_ring_returns_the_samples_written_at_an_absolute_span():
    ring = M.SampleRing(1000)
    data = np.arange(2500, dtype=np.float32)
    for start in range(0, len(data), 137):
        ring.write(data[start:start + 137])
    assert np.array_equal(ring.snapshot(1600, 2500), data[1600:2500])


def test_a_span_the_ring_has_already_overwritten_is_refused():
    ring = M.SampleRing(1000)
    ring.write(np.arange(2500, dtype=np.float32))
    with pytest.raises(M.RingOverrun, match="overwritten"):
        ring.snapshot(0, 100)


def test_a_span_the_ring_has_not_reached_is_refused():
    ring = M.SampleRing(1000)
    ring.write(np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError, match="not written"):
        ring.snapshot(50, 200)


def test_a_write_larger_than_the_ring_still_advances_the_index():
    ring = M.SampleRing(100)
    ring.write(np.arange(250, dtype=np.float32))
    assert ring.written == 250
    assert np.array_equal(ring.snapshot(150, 250), np.arange(150, 250))


def test_reset_restarts_the_sample_index():
    ring = M.SampleRing(1000)
    ring.write(np.ones(400, dtype=np.float32))
    ring.reset()
    assert ring.written == 0


class _Probe(np.ndarray):
    def __new__(cls, source, hook, *, on_read: bool):
        array = np.asarray(source).view(cls)
        array.hook = hook
        array.on_read = on_read
        array.armed = True
        return array

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.hook = getattr(obj, "hook", None)
        self.on_read = getattr(obj, "on_read", False)
        self.armed = False

    def _fire(self, kind: bool) -> None:
        if self.armed and self.on_read is kind:
            self.armed = False
            self.hook()

    def __getitem__(self, key):
        self._fire(True)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self._fire(False)
        super().__setitem__(key, value)


def test_a_write_is_published_before_it_touches_the_buffer():
    ring = M.SampleRing(1000)
    ring.write(np.zeros(1000, dtype=np.float32))
    caught = []

    def read_mid_write():
        with pytest.raises(M.RingOverrun):
            ring.snapshot(0, 1000)
        caught.append(True)

    ring._buffer = _Probe(ring._buffer, read_mid_write, on_read=False)
    ring.write(np.ones(600, dtype=np.float32))
    assert caught == [True]


def test_a_write_landing_inside_a_snapshot_copy_is_refused_not_returned_torn():
    ring = M.SampleRing(1000)
    ring.write(np.zeros(1000, dtype=np.float32))
    ring._buffer = _Probe(ring._buffer,
                          lambda: ring.write(np.ones(600, dtype=np.float32)),
                          on_read=True)
    with pytest.raises(M.RingOverrun):
        ring.snapshot(0, 1000)


def test_a_snapshot_the_concurrent_write_did_not_reach_still_returns():
    ring = M.SampleRing(1000)
    ring.write(np.arange(1000, dtype=np.float32))
    ring._buffer = _Probe(ring._buffer,
                          lambda: ring.write(np.ones(100, dtype=np.float32)),
                          on_read=True)
    assert np.array_equal(ring.snapshot(200, 1000), np.arange(200, 1000))


@pytest.mark.parametrize("margin,hop", [(3.0, 1.0), (2.0, 2.5), (5.0, 1.0),
                                        (0.0, 1.0)])
def test_no_emitted_cell_sees_more_future_than_margin_plus_hop(margin, hop):
    schedule = _schedule(300.0, margin=margin, hop_sec=hop)
    n = _samples(300.0)
    limit = _samples(margin + hop)
    for start, end, (lo, hi) in schedule:
        if end >= n:
            continue
        assert end - lo <= limit, (start, end, lo, hi)
        assert end - hi == _samples(margin)


def test_the_final_pass_flushes_the_tail_and_never_waits_for_more():
    n = _samples(300.0)
    schedule = _schedule(300.0)
    _start, end, (_lo, hi) = schedule[-1]
    assert end == n
    assert hi == n
    assert all(end <= n for _s, end, _span in schedule)


def test_startup_uses_the_short_buffer_it_has_rather_than_waiting():
    start, end, span = _schedule(300.0, margin=3.0)[0]
    assert start == 0
    assert end == _samples(4.0)
    assert span == (0, _samples(1.0))


def test_past_is_bounded_by_the_buffer():
    for start, end, _span in _schedule(300.0, buffer_sec=30.0):
        assert end - start <= _samples(30.0)


@pytest.mark.parametrize("track_sec", [0.4, 3.0, 47.3, 300.0, 912.7])
@pytest.mark.parametrize("margin", [0.0, 3.0, 5.0])
def test_emission_tiles_the_stream_exactly_once(track_sec, margin):
    n = _samples(track_sec)
    spans = [span for _s, _e, span in _schedule(track_sec, margin=margin)]
    assert spans
    assert spans[0][0] == 0
    assert spans[-1][1] == n
    for (_, previous_hi), (lo, _) in zip(spans, spans[1:]):
        assert lo == previous_hi


def test_a_track_shorter_than_one_hop_is_a_single_flush_pass():
    schedule = _schedule(0.4)
    assert len(schedule) == 1
    start, end, span = schedule[0]
    assert (start, end) == (0, _samples(0.4))
    assert span == (0, _samples(0.4))


CELL = 0.1


def _frames(values, layers=1, dim=1):
    return np.asarray(values, dtype=np.float32).reshape(-1, layers, dim)


def test_a_cell_is_the_mean_of_the_encoder_frames_that_reached_it():
    accumulator = M.CellAccumulator(1, 1, CELL)
    times = np.array([0.01, 0.05, 0.09, 0.11], dtype=np.float64)
    accumulator.add(_frames([1.0, 2.0, 3.0, 10.0]), times, 0.0, 0.2)
    cells = accumulator.drain(0.2)
    assert [index for index, _row in cells] == [0, 1]
    assert cells[0][1].ravel()[0] == pytest.approx(2.0)
    assert cells[1][1].ravel()[0] == pytest.approx(10.0)


def test_a_cell_split_across_two_passes_is_still_its_own_mean():
    accumulator = M.CellAccumulator(1, 1, CELL)
    accumulator.add(_frames([1.0, 2.0]), np.array([0.01, 0.04]), 0.0, 0.05)
    assert accumulator.drain(0.05) == []
    accumulator.add(_frames([6.0]), np.array([0.07]), 0.05, 0.15)
    cells = accumulator.drain(0.15)
    assert [index for index, _row in cells] == [0]
    assert cells[0][1].ravel()[0] == pytest.approx(3.0)


def test_an_unreached_cell_is_forward_filled_not_zeroed():
    accumulator = M.CellAccumulator(1, 1, CELL)
    accumulator.add(_frames([4.0, 9.0]), np.array([0.01, 0.31]), 0.0, 0.45)
    cells = accumulator.drain(0.45)
    assert [index for index, _row in cells] == [0, 1, 2, 3]
    assert [float(row.ravel()[0]) for _index, row in cells] == [4.0, 4.0, 4.0, 9.0]


def test_a_leading_gap_is_back_filled_from_the_first_cell_reached():
    accumulator = M.CellAccumulator(1, 1, CELL)
    accumulator.add(_frames([7.0]), np.array([0.25]), 0.0, 0.35)
    cells = accumulator.drain(0.35)
    assert [float(row.ravel()[0]) for _index, row in cells] == [7.0, 7.0, 7.0]


def test_cells_are_emitted_in_order_and_exactly_once():
    accumulator = M.CellAccumulator(1, 1, CELL)
    seen = []
    for step in range(20):
        lo, hi = step * 0.07, (step + 1) * 0.07
        times = np.arange(lo, hi, 0.01)
        accumulator.add(_frames(np.full(len(times), float(step))), times, lo, hi)
        seen.extend(index for index, _row in accumulator.drain(hi))
    assert seen == list(range(len(seen)))


def test_draining_a_partial_cell_waits_for_the_rest_of_it():
    accumulator = M.CellAccumulator(1, 1, CELL)
    accumulator.add(_frames([1.0]), np.array([0.05]), 0.0, 0.06)
    assert accumulator.drain(0.06) == []
    assert [index for index, _row in accumulator.drain(0.1)] == [0]


def test_a_final_drain_emits_the_partial_tail_cell():
    accumulator = M.CellAccumulator(1, 1, CELL)
    accumulator.add(_frames([1.0, 5.0]), np.array([0.05, 0.12]), 0.0, 0.15)
    cells = accumulator.drain(0.15, final=True)
    assert [index for index, _row in cells] == [0, 1]


class FakeEncoder:
    sample_rate = SR
    do_normalize = False
    model_id = "fake"
    model_sha = "0" * 16

    def __init__(self, layers=(6, 22), dim=3):
        self.layers = tuple(layers)
        self.dim = int(dim)
        self.passes = []

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def encode(self, segment, *, offset_samples, lo_sec, hi_sec):
        n = M.encoder_frames(len(segment))
        times, keep = M.frame_selection(n, offset_samples=offset_samples,
                                        lo_sec=lo_sec, hi_sec=hi_sec,
                                        sample_rate=self.sample_rate)
        rate = self.sample_rate
        self.passes.append((offset_samples, offset_samples + len(segment),
                            round(lo_sec * rate), round(hi_sec * rate)))
        return self.frames(segment, offset_samples, keep), times

    def frames(self, segment, offset_samples, keep):
        step = M.ENCODER_SAMPLES_PER_FRAME
        segment = np.asarray(segment, dtype=np.float64)
        keep = np.asarray(keep, dtype=np.int64)
        starts = (offset_samples + keep * step) / 1e5
        digest = float(np.abs(segment).mean()) if len(segment) else 0.0
        local = np.array([segment[i * step:i * step + M.ENCODER_RECEPTIVE_FIELD]
                          .mean() for i in keep], dtype=np.float64) \
            if len(keep) else np.zeros(0)
        stacked = (starts[:, None, None] + digest + local[:, None, None]
                   + np.arange(self.n_layers)[None, :, None] * 0.5
                   + np.arange(self.dim)[None, None, :] * 0.25)
        return stacked.astype(np.float32)


def _stream(encoder, *, margin=3.0, hop=1.0, buffer=30.0, cell=CELL,
            rate=M.SOURCE_SAMPLE_RATE):
    return M.MertStream(encoder, geometry=M.StreamGeometry(
        model_id="fake", layers=(6, 22), margin_sec=margin, hop_sec=hop,
        buffer_sec=buffer, label_frame_sec=cell), source_rate=rate)


def _feed(stream, seconds, *, block=256):
    audio = np.zeros(int(seconds * M.SOURCE_SAMPLE_RATE), dtype=np.float32)
    cells = []
    for start in range(0, len(audio), block):
        stream.push_audio(audio[start:start + block])
        while stream.due():
            cells.extend(stream.run_pass())
    return cells


def _live_form(offline, *, n_samples, hop, margin):
    flat = [(start, end, lo, hi) for start, end, (lo, hi) in offline]
    if n_samples % hop:
        return flat
    # A track ending exactly on a hop boundary repeats its final pass, because
    # the driver already ran that buffer before the flush asked for the margin.
    start, end, lo, hi = flat[-1]
    return flat[:-1] + [(start, end, lo, hi - margin),
                        (start, end, hi - margin, hi)]


@pytest.mark.parametrize("track_sec", [40.0, 40.3, 45.0, 9.5])
def test_the_live_driver_reproduces_the_offline_pass_schedule(track_sec):
    encoder = FakeEncoder()
    stream = _stream(encoder, rate=SR)
    _drive(stream, track_sec)
    stream.flush()
    offline = list(M.pass_schedule(stream.samples_seen, length=_samples(30.0),
                                   hop=_samples(1.0), margin=_samples(3.0)))
    assert encoder.passes == _live_form(
        offline, n_samples=stream.samples_seen, hop=_samples(1.0),
        margin=_samples(3.0))


def _cell_horizons(n_samples, *, margin=3.0, hop=1.0, buffer=30.0, cell=CELL):
    # Read off the schedule, never off the declared margin + hop budget: a
    # driver whose spans drifted still meets the budget on different audio.
    horizons: dict = {}
    for _start, end, (lo, hi) in M.pass_schedule(
            n_samples, length=_samples(buffer), hop=_samples(hop),
            margin=_samples(margin)):
        for index in range(int(math.floor(lo / SR / cell)),
                           int(math.ceil(hi / SR / cell))):
            horizons[index] = max(horizons.get(index, 0), end)
    return horizons


def _poisoned_run(audio, poison_from):
    audio = np.array(audio, dtype=np.float32)
    audio[poison_from:] = 7.0
    return _run_cells(audio)


def _run_cells(audio):
    stream = _stream(FakeEncoder(), rate=SR)
    cells = []
    for start in range(0, len(audio), SR // 10):
        stream.push_audio(audio[start:start + SR // 10])
        while stream.due():
            cells.extend(stream.run_pass())
    cells.extend(stream.flush())
    return cells


@pytest.mark.parametrize("poison_sec", [8.0, 14.0, 25.0])
def test_no_emitted_cell_is_moved_by_audio_its_own_passes_never_saw(poison_sec):
    audio = _noise(30 * SR, seed=4)
    horizon = _cell_horizons(len(audio))
    clean = _run_cells(audio)
    dirty = _poisoned_run(audio, int(poison_sec * SR))

    checked = moved = 0
    for before, after in zip(clean, dirty):
        assert before.index == after.index
        if horizon[before.index] <= poison_sec * SR:
            assert np.array_equal(before.features, after.features), before.index
            checked += 1
        elif not np.array_equal(before.features, after.features):
            moved += 1
    assert checked > 20
    assert moved > 20


def test_the_stream_emits_every_cell_once_in_order():
    stream = _stream(FakeEncoder())
    cells = _feed(stream, 25.0)
    cells.extend(stream.flush())
    assert [cell.index for cell in cells] == list(range(len(cells)))
    assert len(cells) == pytest.approx(25.0 / CELL, abs=2)


def test_a_cell_is_stamped_at_the_end_of_its_own_span():
    stream = _stream(FakeEncoder())
    cells = _feed(stream, 12.0)
    for cell in cells:
        assert cell.time_sec == pytest.approx((cell.index + 1) * CELL)


def test_an_encoder_that_does_not_speak_the_geometrys_rate_is_refused():
    encoder = FakeEncoder()
    encoder.sample_rate = 16000
    with pytest.raises(ValueError, match="16000"):
        _stream(encoder)


def test_no_emitted_cell_saw_audio_beyond_the_margin_plus_hop():
    stream = _stream(FakeEncoder(), margin=3.0, hop=1.0)
    cells = _feed(stream, 45.0)
    assert cells
    for cell in cells:
        future = cell.audio_seen_sec - cell.time_sec
        assert future <= 3.0 + 1.0 + 1e-9, (cell.index, future)


def test_the_stream_reads_features_of_the_width_the_student_expects():
    encoder = FakeEncoder(dim=1024)
    stream = _stream(encoder)
    cells = _feed(stream, 10.0)
    assert cells
    assert cells[0].features.shape == (2 * 1024,)
    assert cells[0].features.dtype == np.float32


def test_features_are_quantised_to_the_grid_the_student_trained_on():
    stream = _stream(FakeEncoder())
    cells = _feed(stream, 10.0)
    assert cells
    for cell in cells[:20]:
        assert np.array_equal(cell.features,
                              cell.features.astype(np.float16).astype(np.float32))


def test_reset_clears_the_ring_and_the_accumulator():
    encoder = FakeEncoder()
    stream = _stream(encoder)
    first = _feed(stream, 12.0)
    stream.reset()
    encoder.passes.clear()
    second = _feed(stream, 12.0)
    assert [cell.index for cell in second] == [cell.index for cell in first]
    assert np.array_equal(second[0].features, first[0].features)


def test_a_pass_is_only_due_once_its_audio_has_arrived():
    stream = _stream(FakeEncoder())
    assert not stream.due()
    stream.push_audio(np.zeros(M.SOURCE_SAMPLE_RATE // 2, dtype=np.float32))
    assert not stream.due()
    stream.push_audio(np.zeros(M.SOURCE_SAMPLE_RATE, dtype=np.float32))
    assert stream.due()


def test_the_stream_flushes_the_tail_at_a_song_boundary():
    stream = _stream(FakeEncoder())
    cells = _feed(stream, 9.5)
    tail = stream.flush()
    assert tail
    assert [cell.index for cell in tail] == list(
        range(len(cells), len(cells) + len(tail)))


def test_a_flush_is_terminal_until_the_stream_is_reset():
    stream = _stream(FakeEncoder())
    _feed(stream, 9.5)
    stream.flush()
    with pytest.raises(M.Flushed):
        stream.push_audio(np.zeros(256, dtype=np.float32))
    with pytest.raises(M.Flushed):
        stream.flush()
    with pytest.raises(M.Flushed):
        stream.resync()
    assert stream.run_pass() == []
    stream.reset()
    stream.push_audio(np.zeros(256, dtype=np.float32))


_SWEEP = [(track, margin, hop, buffer)
          for track in (0.4, 3.0, 9.5, 30.0, 40.0, 47.3)
          for margin in (0.0, 2.0, 3.0, 5.0)
          for hop, buffer in ((1.0, 30.0), (2.5, 5.0))]


def test_the_schedule_generator_is_the_offline_one(
        ):
    for n_samples in (9600, 228000, 720000, 967200, 1135040):
        for margin in (0.0, 3.0, 5.0):
            mine = list(M.pass_schedule(n_samples, length=_samples(30.0),
                                        hop=_samples(1.0),
                                        margin=_samples(margin)))
            theirs = [(start, end, spans[float(margin)]) for start, end, spans
                      in OFFLINE.pass_schedule(
                          n_samples, length=OFFLINE.chunk_samples(30.0),
                          hop=OFFLINE.chunk_samples(1.0),
                          margin_samples={float(margin):
                                          OFFLINE.chunk_samples(margin)})]
            assert mine == theirs, (n_samples, margin)


@pytest.mark.parametrize("track_sec,margin,hop,buffer", _SWEEP)
def test_the_live_stage_pools_the_cells_the_offline_extractor_pools(
        track_sec, margin, hop, buffer):
    cell = OFFLINE.LABEL_FRAME_SEC
    audio = _noise(int(track_sec * SR), seed=int(track_sec * 10 + margin))
    stream = _stream(FakeEncoder(), margin=margin, hop=hop, buffer=buffer,
                     cell=cell, rate=SR)
    cells = []
    for start in range(0, len(audio), SR // 10):
        stream.push_audio(audio[start:start + SR // 10])
        while stream.due():
            cells.extend(stream.run_pass())
    cells.extend(stream.flush())

    pooled = OFFLINE.extract(
        FakeEncoder(), audio, margin_sec=margin, hop_sec=hop,
        buffer_sec=buffer,
        n_pooled=int(math.ceil(len(audio) / SR / cell)) + 4)
    assert cells
    for item in cells:
        expected = pooled[item.index].reshape(-1) \
            .astype(np.float16).astype(np.float32)
        assert np.array_equal(item.features, expected), item.index


def _drive(stream, seconds, *, rate=SR, block=None):
    audio = np.zeros(int(seconds * rate), dtype=np.float32)
    block = block or rate // 10
    cells = []
    for start in range(0, len(audio), block):
        stream.push_audio(audio[start:start + block])
        while stream.due():
            cells.extend(stream.run_pass())
    return cells


def _stalled(seconds=40.0):
    stream = _stream(FakeEncoder(), rate=SR)
    stream.push_audio(np.zeros(int(seconds * SR), dtype=np.float32))
    return stream


def _pump(stream):
    cells = []
    while stream.due():
        cells.extend(stream.run_pass())
    return cells


def test_a_pass_whose_audio_was_overwritten_raises_a_typed_overrun():
    with pytest.raises(M.RingOverrun):
        _pump(_stalled())


def test_an_overrun_is_not_the_type_a_programming_error_raises():
    assert not issubclass(M.RingOverrun, ValueError)
    ring = M.SampleRing(1000)
    ring.write(np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError):
        ring.snapshot(50, 200)
    with pytest.raises(ValueError):
        ring.snapshot(-1, 10)


def test_an_overrun_pass_is_not_recorded_as_taken():
    stream = _stalled()
    with pytest.raises(M.RingOverrun):
        _pump(stream)
    taken = stream.passes
    for _ in range(35):
        with pytest.raises(M.RingOverrun):
            stream.run_pass()
    assert stream.passes == taken
    assert stream.due()


def test_resync_skips_to_the_live_edge_and_reports_the_audio_lost():
    stream = _stalled(40.0)
    with pytest.raises(M.RingOverrun):
        _pump(stream)
    report = stream.resync()
    assert report.lost_sec == pytest.approx(36.0)
    assert report.cells_lost == report.first_cell_index > 0
    assert stream.due()
    recovered = stream.run_pass()
    assert recovered
    assert recovered[0].index == report.first_cell_index


def test_the_gap_a_resync_skipped_is_never_filled_from_later_audio():
    stalled = _stalled(40.0)
    with pytest.raises(M.RingOverrun):
        _pump(stalled)
    stalled.resync()
    recovered = stalled.run_pass()

    healthy = {cell.index: cell for cell in _drive(_stream(FakeEncoder(), rate=SR),
                                                   40.0)}
    assert recovered[0].index > 300
    for cell in recovered:
        assert np.array_equal(cell.features, healthy[cell.index].features)


def test_a_resync_leaves_the_stream_running_the_ordinary_schedule():
    stream = _stalled(40.0)
    with pytest.raises(M.RingOverrun):
        _pump(stream)
    stream.resync()
    cells = stream.run_pass()
    cells.extend(_drive(stream, 5.0))
    assert [cell.index for cell in cells] == list(
        range(cells[0].index, cells[0].index + len(cells)))


def _affine(tmp_path, *, geometry, dim=8, layers=(6, 22)):
    path = tmp_path / "affine.npz"
    np.savez(path, mean=np.zeros(dim, np.float32), std=np.ones(dim, np.float32),
             dim=np.int32(dim), layers=np.asarray(layers, np.int32),
             geometry=np.str_(json.dumps(geometry, sort_keys=True)))
    return path


def test_the_stream_geometry_is_read_from_the_shipped_affine(tmp_path):
    path = _affine(tmp_path, geometry={"causal": 1, "margin_sec": 3.0,
                                       "hop_sec": 1.0, "buffer_sec": 30.0})
    geometry = M.load_stream_geometry(path, label_frame_sec=0.25)
    assert (geometry.margin_sec, geometry.hop_sec, geometry.buffer_sec) == (
        3.0, 1.0, 30.0)
    assert geometry.layers == (6, 22)
    assert geometry.label_frame_sec == 0.25


def test_a_non_causal_affine_is_refused(tmp_path):
    path = _affine(tmp_path, geometry={"causal": 0, "margin_sec": 3.0,
                                       "hop_sec": 1.0, "buffer_sec": 30.0})
    with pytest.raises(ValueError, match="causal"):
        M.load_stream_geometry(path, label_frame_sec=0.25)


def test_an_affine_that_names_no_encoder_layers_is_refused(tmp_path):
    path = tmp_path / "nolayers.npz"
    np.savez(path, mean=np.zeros(4, np.float32), std=np.ones(4, np.float32),
             geometry=np.str_(json.dumps({"causal": 1, "margin_sec": 3.0,
                                          "hop_sec": 1.0, "buffer_sec": 30.0})))
    with pytest.raises(ValueError, match="layers"):
        M.load_stream_geometry(path, label_frame_sec=CELL)


def test_an_affine_with_no_geometry_record_is_refused(tmp_path):
    path = tmp_path / "bare.npz"
    np.savez(path, mean=np.zeros(4, np.float32), std=np.ones(4, np.float32))
    with pytest.raises(ValueError, match="geometry"):
        M.load_stream_geometry(path, label_frame_sec=0.25)


def test_an_encoder_whose_weights_hash_differs_is_refused():
    with pytest.raises(RuntimeError, match="encoder weights"):
        M.check_encoder_sha("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb")
    M.check_encoder_sha("aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa")


def test_an_unpinned_encoder_is_refused_rather_than_waved_through():
    with pytest.raises(ValueError, match="unpinned"):
        M.check_encoder_sha("aaaaaaaaaaaaaaaa", None)


class _StubTensor:
    def detach(self):
        return self

    def to(self, _device):
        return self

    def numpy(self):
        return np.arange(4, dtype=np.float32)


class _StubEncoder:
    sampling_rate = SR
    do_normalize = False
    config = type("config", (), {"hidden_size": 4})

    def eval(self):
        return self

    def half(self):
        return self

    def to(self, _device):
        return self

    def state_dict(self):
        return {"weight": _StubTensor()}


def _stub_transformers(monkeypatch, calls):
    class _Loader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs.get("revision"),
                          kwargs.get("trust_remote_code")))
            return _StubEncoder()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModel=_Loader, Wav2Vec2FeatureExtractor=_Loader))


def _pinned_geometry(**kwargs):
    return M.StreamGeometry(model_id="stub/encoder", layers=(0,), margin_sec=3.0,
                            hop_sec=1.0, buffer_sec=30.0, label_frame_sec=CELL,
                            **kwargs)


def test_the_encoder_is_fetched_at_the_pinned_revision(monkeypatch):
    calls: list = []
    _stub_transformers(monkeypatch, calls)
    sha = M.state_dict_sha(_StubEncoder())
    encoder = M.load_encoder(_pinned_geometry(encoder_sha=sha), device="cpu",
                             fp16=False)
    assert calls == [("stub/encoder", M.DEFAULT_MODEL_REVISION, True)] * 2
    assert encoder.model_sha == sha


def test_an_encoder_whose_weights_are_not_the_pinned_ones_never_loads(monkeypatch):
    _stub_transformers(monkeypatch, [])
    with pytest.raises(RuntimeError, match="encoder weights"):
        M.load_encoder(_pinned_geometry(encoder_sha="0" * 16), device="cpu",
                       fp16=False)


def test_the_shipped_geometry_carries_the_encoder_pin(tmp_path):
    path = _affine(tmp_path, geometry={"causal": 1, "margin_sec": 3.0,
                                       "hop_sec": 1.0, "buffer_sec": 30.0})
    geometry = M.load_stream_geometry(path, label_frame_sec=CELL)
    assert geometry.model_id == M.DEFAULT_MODEL_ID
    assert geometry.revision == M.DEFAULT_MODEL_REVISION
    assert geometry.encoder_sha == M.DEFAULT_ENCODER_SHA


def test_encoder_frame_count_follows_the_conv_stack():
    assert M.encoder_frames(720000) == 2249
    assert M.encoder_frames(399) == 0


def test_encoder_frame_times_sit_at_the_centre_of_the_receptive_field():
    times = M.encoder_frame_times(3, offset_samples=0, sample_rate=SR)
    assert times[0] == pytest.approx(200.0 / SR)
    assert times[1] - times[0] == pytest.approx(320.0 / SR)


def _phase_b_dir() -> Path:
    import run_eval_set

    return Path(run_eval_set.corpus_dir()) / "models" / "phase_b"


def _require_artifacts():
    directory = _phase_b_dir()
    affine = directory / "input_affine_F3.npz"
    from lib.section_chain import MODEL_VERSION

    onnx = (directory / MODEL_VERSION / "online_step.onnx")
    if not affine.exists() or not onnx.exists():
        pytest.skip(f"shipping artifacts absent under {directory} -- "
                    f"they live in the gitignored corpus data directory")
    return affine, onnx


@pytest.mark.integration
def test_the_shipped_geometry_fits_the_eight_second_budget():
    from lib.analyser import section_model as S

    affine, onnx = _require_artifacts()
    head = S.load_head_geometry(onnx)
    geometry = M.load_stream_geometry(affine,
                                      label_frame_sec=head.label_frame_sec)
    total = geometry.margin_sec + geometry.hop_sec + head.future_sec
    assert total <= 8.0, total
    assert 8.0 - total < head.label_frame_sec


def _sidecar_track(geometry):
    import run_eval_set

    corpus = Path(run_eval_set.corpus_dir())
    tag = "-".join(str(int(layer)) for layer in geometry.layers)
    directory = (corpus / "features_stream"
                 / f"{geometry.model_id.split('/')[-1]}_L{tag}"
                   f"_F{geometry.margin_sec:g}_hop{geometry.hop_sec:g}")
    for path in sorted(directory.glob("*.npz")):
        audio = corpus / "audio" / f"{path.stem}.mp3"
        if audio.exists():
            return path, audio
    pytest.skip(f"no stream sidecar with its audio under {directory} -- both "
                f"live in the gitignored corpus data directory")


@pytest.mark.integration
def test_the_pinned_encoder_hash_is_the_one_the_sidecars_were_cut_with():
    affine, onnx = _require_artifacts()
    from lib.analyser import section_model as S

    geometry = M.load_stream_geometry(
        affine, label_frame_sec=S.load_head_geometry(onnx).label_frame_sec)
    sidecar, _audio = _sidecar_track(geometry)
    with np.load(sidecar) as archive:
        assert str(archive["model_sha"]) == M.DEFAULT_ENCODER_SHA
        assert str(archive["model_id"]) == geometry.model_id
        assert float(archive["stream_margin_sec"]) == geometry.margin_sec
        assert float(archive["stream_hop_sec"]) == geometry.hop_sec
        assert float(archive["stream_buffer_sec"]) == geometry.buffer_sec
        assert float(archive["label_frame_sec"]) == geometry.label_frame_sec


@pytest.fixture(scope="module")
def real_stack():
    from lib.analyser import section_model as S

    affine, onnx = _require_artifacts()
    head = S.load_head_geometry(onnx)
    geometry = M.load_stream_geometry(affine,
                                      label_frame_sec=head.label_frame_sec)
    return M.load_encoder(geometry, device=M.best_device(), fp16=True), geometry, head


@pytest.mark.integration
def test_the_real_encoder_streams_the_same_cells_twice_in_one_process(
        real_stack, anchor_mp3):
    encoder, geometry, head = real_stack

    audio = _source_audio(anchor_mp3)[:int(20.0 * M.SOURCE_SAMPLE_RATE)]
    runs = [_run_stream(encoder, geometry, audio) for _ in range(2)]

    assert len(runs[0]) == len(runs[1]) > 0
    for first, second in zip(*runs):
        assert first.index == second.index
        assert np.array_equal(first.features, second.features)
    assert runs[0][0].features.shape == (head.input_dim,)


@pytest.mark.integration
def test_the_live_stage_reproduces_a_corpus_sidecar_for_a_track_prefix(
        real_stack, capsys):
    encoder, geometry, _head = real_stack
    sidecar, audio_path = _sidecar_track(geometry)
    with np.load(sidecar) as archive:
        emb = np.asarray(archive["emb"], dtype=np.float32)

    audio = _ffmpeg_audio(audio_path, M.ENCODER_SAMPLE_RATE)[:60 * M.ENCODER_SAMPLE_RATE]
    cells = _run_stream(encoder, geometry, audio,
                        source_rate=M.ENCODER_SAMPLE_RATE)
    limit = 500
    deltas = [float(np.abs(cell.features - emb[cell.index].reshape(-1)).max())
              for cell in cells if cell.index < limit]
    with capsys.disabled():
        print(f"\nsidecar reproduction ({sidecar.stem}, {len(deltas)} cells): "
              f"max abs delta {max(deltas):g}")
    assert len(deltas) >= 400
    assert max(deltas) == 0.0


@pytest.mark.integration
def test_the_live_resampler_reproduces_the_offline_ffmpeg_cells(
        real_stack, anchor_mp3, capsys):
    encoder, geometry, _head = real_stack
    affine, onnx = _require_artifacts()
    mean, std = M.load_input_affine(affine)

    seconds = 60.0
    take24 = int(seconds * M.ENCODER_SAMPLE_RATE)
    take441 = int(seconds * M.SOURCE_SAMPLE_RATE)
    reference = _run_stream(
        encoder, geometry, _ffmpeg_audio(anchor_mp3, M.ENCODER_SAMPLE_RATE)[:take24],
        source_rate=M.ENCODER_SAMPLE_RATE)
    resampled = _run_stream(
        encoder, geometry,
        _ffmpeg_audio(anchor_mp3, M.SOURCE_SAMPLE_RATE)[:take441])
    simulated = _run_stream(encoder, geometry, _source_audio(anchor_mp3)[:take441])

    baseline = np.stack([cell.features for cell in reference])
    adjacent = np.median(np.abs(np.diff(baseline, axis=0)) / std[None, :])
    decided = _decisions(onnx, mean, reference)
    arms = {}
    for name, cells in (("resampler_only_ffmpeg_44k1", resampled),
                        ("simulation_path_librosa_44k1", simulated)):
        arms[name] = _cell_delta(np.stack([cell.features for cell in cells]),
                                 baseline, std, adjacent)
        arms[name].update(_decision_delta(_decisions(onnx, mean, cells), decided))
    with capsys.disabled():
        print(f"\nD4 parity ({len(baseline)} cells, median adjacent-cell "
              f"distance {adjacent:.4f} affine std):\n{json.dumps(arms, indent=2)}")

    resampler = arms["resampler_only_ffmpeg_44k1"]
    assert resampler["class_argmax_disagreements"] <= 0.01 * resampler["class_cells"], \
        resampler
    assert max(resampler["flip_reference_top_two_gap"], default=0.0) < 0.01, resampler
    assert resampler["median_delta_in_affine_std"] < 0.10, resampler
    assert resampler["share_of_adjacent_cell_distance"] < 0.20, resampler


def _decisions(onnx, mean, cells) -> dict:
    from lib.analyser import section_model as S

    model = S.SectionModel(onnx, mean=mean)
    out = [model.push(cell.features) for cell in cells]
    out.extend(model.flush())
    return {item.index: item for item in out if item is not None}


def _decision_delta(arm: dict, reference: dict) -> dict:
    shared = sorted(set(arm) & set(reference))
    assert shared
    flipped = [i for i in shared
               if np.argmax(arm[i].posterior) != np.argmax(reference[i].posterior)]
    return {
        "class_cells": len(shared),
        "class_argmax_disagreements": len(flipped),
        "flip_reference_top_two_gap": [
            round(float(np.diff(np.sort(reference[i].posterior)[-2:])[0]), 5)
            for i in flipped],
        "max_abs_posterior_delta": max(
            float(np.abs(arm[i].posterior - reference[i].posterior).max())
            for i in shared),
        "max_abs_boundary_delta": max(
            abs(arm[i].boundary - reference[i].boundary) for i in shared),
    }


def _cell_delta(a, b, std, adjacent):
    shared = min(len(a), len(b))
    assert shared > 400
    delta = np.abs(a[:shared] - b[:shared])
    normalised = delta / std[None, :]
    median = float(np.median(normalised))
    return {
        "cells": shared,
        "median_delta_in_affine_std": median,
        "p99_delta_in_affine_std": float(np.percentile(normalised, 99)),
        "max_delta_in_affine_std": float(normalised.max()),
        "share_of_adjacent_cell_distance": float(median / adjacent),
        "feature_argmax_disagreements": int(
            (a[:shared].argmax(axis=1) != b[:shared].argmax(axis=1)).sum()),
        "min_cosine": float(np.min(
            (a[:shared] * b[:shared]).sum(1)
            / (np.linalg.norm(a[:shared], axis=1)
               * np.linalg.norm(b[:shared], axis=1) + 1e-12))),
    }


def _run_stream(encoder, geometry, audio, *, source_rate=M.SOURCE_SAMPLE_RATE):
    stream = M.MertStream(encoder, geometry=geometry, source_rate=source_rate)
    cells = []
    for start in range(0, len(audio), source_rate // 10):
        stream.push_audio(audio[start:start + source_rate // 10])
        while stream.due():
            cells.extend(stream.run_pass())
    cells.extend(stream.flush())
    return cells


def _source_audio(path: str) -> np.ndarray:
    import librosa

    audio, _rate = librosa.load(path, sr=M.SOURCE_SAMPLE_RATE, mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _ffmpeg_audio(path: str, rate: int) -> np.ndarray:
    import subprocess

    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)
