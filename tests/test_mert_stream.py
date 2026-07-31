"""The live MERT extractor's geometry -- the part a show has to obey.

Ported from phase-b's `test_ceiling_stream_extract.py`. The offline extractor
knows the track length up front and can run `pass_schedule` as a generator; a
live one is driven by audio arriving, so the same arithmetic is re-derived
incrementally. These tests interrogate the geometry rather than the encoder:
which audio an emitted cell was allowed to see, that emission tiles the stream
exactly once, and that start-up does not quietly wait for audio it would not
have. An off-by-one here is invisible in every downstream number.

Running MERT needs a GPU and 1.3 GB of weights; those cases are marked
integration and read the gitignored data directory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from lib.analyser import mert_stream as M

SR = M.ENCODER_SAMPLE_RATE


def _samples(seconds: float) -> int:
    return M.encoder_samples(seconds)


def _schedule(track_sec, *, margin=3.0, hop_sec=1.0, buffer_sec=30.0):
    return list(M.pass_schedule(_samples(track_sec), length=_samples(buffer_sec),
                                hop=_samples(hop_sec), margin=_samples(margin)))


# --------------------------------------------------------------------------- #
# The resampler (D4)
# --------------------------------------------------------------------------- #


def _noise(n, seed=0):
    return np.random.default_rng(seed).normal(size=n).astype(np.float32)


def test_the_streaming_resampler_reproduces_a_whole_array_resample():
    """Block-by-block filtering must be the same filtering, sample for sample.

    Not a nicety: it is what lets the live cells be compared against an offline
    decode at all. A resampler that drifts at every block edge would put a
    signature into the features that no offline measurement can see.
    """
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


# --------------------------------------------------------------------------- #
# The ring buffer
# --------------------------------------------------------------------------- #


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
    with pytest.raises(ValueError, match="overwritten"):
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


# --------------------------------------------------------------------------- #
# Future dependence -- the number the whole design is budgeted against
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# The cell accumulator
# --------------------------------------------------------------------------- #

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
    """A zero row is not "no information" to a network -- it is a confident,
    out-of-distribution input."""
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


# --------------------------------------------------------------------------- #
# The live driver against a fake encoder
# --------------------------------------------------------------------------- #


class FakeEncoder:
    """A pure function of absolute sample position, so a cell is predictable."""

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
        self.passes.append((offset_samples, offset_samples + len(segment)))
        starts = (offset_samples + keep * M.ENCODER_SAMPLES_PER_FRAME) / 1e5
        stacked = (starts[:, None, None]
                   + np.arange(self.n_layers)[None, :, None] * 0.5
                   + np.arange(self.dim)[None, None, :] * 0.25)
        return stacked.astype(np.float32), times


def _stream(encoder, *, margin=3.0, hop=1.0, buffer=30.0, cell=CELL):
    return M.MertStream(encoder, geometry=M.StreamGeometry(
        model_id="fake", layers=(6, 22), margin_sec=margin, hop_sec=hop,
        buffer_sec=buffer, label_frame_sec=cell))


def _feed(stream, seconds, *, block=256):
    audio = np.zeros(int(seconds * M.SOURCE_SAMPLE_RATE), dtype=np.float32)
    cells = []
    for start in range(0, len(audio), block):
        stream.push_audio(audio[start:start + block])
        while stream.due():
            cells.extend(stream.run_pass())
    return cells


def test_the_live_driver_reproduces_the_offline_pass_schedule():
    """The rule is not re-derived, it is checked against the offline generator."""
    encoder = FakeEncoder()
    stream = _stream(encoder)
    _feed(stream, 40.3)
    stream.flush()
    offline = [(start, end) for start, end, _span in M.pass_schedule(
        stream.samples_seen, length=_samples(30.0), hop=_samples(1.0),
        margin=_samples(3.0))]
    assert encoder.passes == offline


def test_the_stream_emits_every_cell_once_in_order():
    stream = _stream(FakeEncoder())
    cells = _feed(stream, 25.0)
    cells.extend(stream.flush())
    assert [cell.index for cell in cells] == list(range(len(cells)))
    assert len(cells) == pytest.approx(25.0 / CELL, abs=2)


def test_a_cell_is_stamped_at_the_start_of_its_own_span():
    stream = _stream(FakeEncoder())
    cells = _feed(stream, 12.0)
    for cell in cells:
        assert cell.time_sec == pytest.approx(cell.index * CELL)


def test_no_emitted_cell_saw_audio_beyond_the_margin_plus_hop():
    """Measured from the END of the cell: a cell's own span is not its future."""
    stream = _stream(FakeEncoder(), margin=3.0, hop=1.0)
    cells = _feed(stream, 45.0)
    assert cells
    for cell in cells:
        future = cell.audio_seen_sec - (cell.time_sec + CELL)
        assert future <= 3.0 + 1.0 + 1e-9, (cell.index, future)


def test_the_stream_reads_features_of_the_width_the_student_expects():
    encoder = FakeEncoder(dim=1024)
    stream = _stream(encoder)
    cells = _feed(stream, 10.0)
    assert cells
    assert cells[0].features.shape == (2 * 1024,)
    assert cells[0].features.dtype == np.float32


def test_features_are_quantised_to_the_grid_the_student_trained_on():
    """Offline sidecars are float16; a live path handing the model float32
    precision is feeding it inputs it has never seen."""
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
    assert stream.flush() == []


# --------------------------------------------------------------------------- #
# Geometry is read from the shipped artifact, never retyped (D2)
# --------------------------------------------------------------------------- #


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


def test_an_affine_with_no_geometry_record_is_refused(tmp_path):
    path = tmp_path / "bare.npz"
    np.savez(path, mean=np.zeros(4, np.float32), std=np.ones(4, np.float32))
    with pytest.raises(ValueError, match="geometry"):
        M.load_stream_geometry(path, label_frame_sec=0.25)


def test_an_encoder_whose_weights_hash_differs_is_refused():
    with pytest.raises(RuntimeError, match="encoder weights"):
        M.check_encoder_sha("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb")
    M.check_encoder_sha("aaaaaaaaaaaaaaaa", None)
    M.check_encoder_sha("aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa")


def test_encoder_frame_count_follows_the_conv_stack():
    assert M.encoder_frames(720000) == 2249
    assert M.encoder_frames(399) == 0


def test_encoder_frame_times_sit_at_the_centre_of_the_receptive_field():
    times = M.encoder_frame_times(3, offset_samples=0, sample_rate=SR)
    assert times[0] == pytest.approx(200.0 / SR)
    assert times[1] - times[0] == pytest.approx(320.0 / SR)


# --------------------------------------------------------------------------- #
# The real stack (integration)
# --------------------------------------------------------------------------- #


def _phase_b_dir() -> Path:
    import run_eval_set

    return Path(run_eval_set.corpus_dir()) / "models" / "phase_b"


def _require_artifacts():
    directory = _phase_b_dir()
    affine = directory / "input_affine_F3.npz"
    onnx = (directory / "student_kd_t2_w05_s1234" / "online_step.onnx")
    if not affine.exists() or not onnx.exists():
        pytest.skip(f"shipping artifacts absent under {directory} -- "
                    f"they live in the gitignored corpus data directory")
    return affine, onnx


@pytest.mark.integration
def test_the_shipped_geometry_fits_the_eight_second_budget():
    """F + hop + the head's own future, computed from the files (D2)."""
    from lib.analyser import section_model as S

    affine, onnx = _require_artifacts()
    head = S.load_head_geometry(onnx)
    geometry = M.load_stream_geometry(affine,
                                      label_frame_sec=head.label_frame_sec)
    total = geometry.margin_sec + geometry.hop_sec + head.future_sec
    assert total <= 8.0, total
    assert 8.0 - total < head.label_frame_sec


@pytest.mark.integration
def test_the_real_encoder_streams_the_same_cells_twice_in_one_process():
    from lib.analyser import section_model as S

    affine, onnx = _require_artifacts()
    head = S.load_head_geometry(onnx)
    geometry = M.load_stream_geometry(affine,
                                      label_frame_sec=head.label_frame_sec)
    encoder = M.load_encoder(geometry, device=M.best_device(), fp16=True)

    audio = _source_audio(anchor_mp3_path())[:int(20.0 * M.SOURCE_SAMPLE_RATE)]
    runs = [_run_stream(encoder, geometry, audio) for _ in range(2)]

    assert len(runs[0]) == len(runs[1]) > 0
    for first, second in zip(*runs):
        assert first.index == second.index
        assert np.array_equal(first.features, second.features)
    assert runs[0][0].features.shape == (head.input_dim,)


@pytest.mark.integration
def test_the_live_resampler_reproduces_the_offline_ffmpeg_cells(capsys):
    """D4: the offline features came out of ffmpeg's resampler, the live ones
    out of a polyphase one. Nothing fails loudly if these disagree -- the
    posteriors just get quietly worse -- so the delta is measured and printed."""
    from lib.analyser import section_model as S

    affine, onnx = _require_artifacts()
    head = S.load_head_geometry(onnx)
    geometry = M.load_stream_geometry(affine,
                                      label_frame_sec=head.label_frame_sec)
    encoder = M.load_encoder(geometry, device=M.best_device(), fp16=True)

    seconds = 60.0
    path = anchor_mp3_path()
    live = _run_stream(encoder, geometry,
                       _source_audio(path)[:int(seconds * M.SOURCE_SAMPLE_RATE)])
    offline = _run_stream(
        encoder, geometry,
        _ffmpeg_audio(path, M.ENCODER_SAMPLE_RATE)[
            :int(seconds * M.ENCODER_SAMPLE_RATE)],
        source_rate=M.ENCODER_SAMPLE_RATE)

    shared = min(len(live), len(offline))
    assert shared > 400
    a = np.stack([cell.features for cell in live[:shared]])
    b = np.stack([cell.features for cell in offline[:shared]])
    delta = np.abs(a - b)
    report = {
        "cells": shared,
        "max_cell_delta": float(delta.max()),
        "median_cell_delta": float(np.median(delta)),
        "max_abs_offline_feature": float(np.abs(b).max()),
        "argmax_disagreements": int((a.argmax(axis=1) != b.argmax(axis=1)).sum()),
        "relative_max": float(delta.max() / max(np.abs(b).max(), 1e-9)),
    }
    with capsys.disabled():
        print(f"\nD4 resampler parity: {json.dumps(report)}")
    assert report["relative_max"] < 0.05, report


def _run_stream(encoder, geometry, audio, *, source_rate=M.SOURCE_SAMPLE_RATE):
    stream = M.MertStream(encoder, geometry=geometry, source_rate=source_rate)
    cells = []
    for start in range(0, len(audio), source_rate // 10):
        stream.push_audio(audio[start:start + source_rate // 10])
        while stream.due():
            cells.extend(stream.run_pass())
    cells.extend(stream.flush())
    return cells


def anchor_mp3_path() -> str:
    from conftest import anchor_mp3_path as resolve

    return resolve()


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
