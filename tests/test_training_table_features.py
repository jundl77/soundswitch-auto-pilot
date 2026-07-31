"""Tests for the NN mel-feature sidecar exporter.

The exporter's parity pin against the live analyser is gone with the analyser's
aubio filterbank: the runtime no longer computes a mel stream, so there is
nothing left for the exporter to be identical to.  What remains under test is
the pooling arithmetic and the sidecar file format, which the NN dataset reads.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from build_training_table import (  # noqa: E402  (needs the path insert above)
    MEL_BANDS,
    MEL_EXPORTER_KEY,
    MEL_EXPORTER_VERSION,
    POOL_BUFFERS,
    MelEnergyStream,
    pooled_log_mel,
    sidecar_generation,
    write_feature_sidecar,
)


def random_buffers(count: int, seed: int = 20260726) -> list:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(BUFFER_SIZE).astype(np.float32) for _ in range(count)]


# --------------------------------------------------------------------------- #
# Pooling and time base
# --------------------------------------------------------------------------- #


def test_pooled_frames_are_the_mean_of_log1p_energies_over_eight_buffers():
    buffers = random_buffers(2 * POOL_BUFFERS)
    audio = np.concatenate(buffers)

    reference = MelEnergyStream(SAMPLE_RATE, BUFFER_SIZE)
    per_buffer = np.array([np.log1p(np.maximum(reference.process(b), 0.0))
                           for b in buffers])
    expected = per_buffer.reshape(2, POOL_BUFFERS, MEL_BANDS).mean(axis=1)

    mel, _frame_sec, _t0 = pooled_log_mel(audio, SAMPLE_RATE, BUFFER_SIZE)

    assert mel.shape == (2, MEL_BANDS)
    assert mel.dtype == np.float32
    # A few float32 ulps: the exporter accumulates the pool in float64 before
    # dividing, this reference means in float32.  The pooling *semantics* are
    # what is under test; bit-exactness against a different summation order is
    # not a property worth pinning.
    assert np.allclose(mel, expected.astype(np.float32), rtol=1e-6, atol=0)


def test_frame_rate_is_eight_buffers_and_t0_matches_the_pipeline_stamp():
    """The pipeline advances its clock BEFORE analysing a buffer, so an event in
    buffer i is stamped at (i+1)*buffer_sec.  A pooled frame therefore carries
    the song time of the END of its last buffer -- which puts frame 0 at exactly
    one frame hop, keeping mel frames and beat rows on one time base."""
    audio = np.concatenate(random_buffers(POOL_BUFFERS))
    mel, frame_sec, t0 = pooled_log_mel(audio, SAMPLE_RATE, BUFFER_SIZE)

    assert len(mel) == 1
    assert frame_sec == pytest.approx(POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE)
    assert t0 == pytest.approx(frame_sec)


def test_trailing_partial_frame_is_dropped():
    """A frame pooled over fewer than eight buffers would be weighted
    differently from every other frame."""
    audio = np.concatenate(random_buffers(POOL_BUFFERS + 3))
    mel, _frame_sec, _t0 = pooled_log_mel(audio, SAMPLE_RATE, BUFFER_SIZE)

    assert len(mel) == 1


def test_final_partial_buffer_is_zero_padded_like_the_audio_client():
    """FileAudioClient pads its last buffer with silence; the exporter reads the
    same decoded array and must agree on how many buffers that array holds."""
    buffers = random_buffers(POOL_BUFFERS)
    short = np.concatenate(buffers)[: -BUFFER_SIZE // 2]

    mel, _frame_sec, _t0 = pooled_log_mel(short, SAMPLE_RATE, BUFFER_SIZE)

    assert len(mel) == 1


def test_audio_shorter_than_one_frame_yields_no_frames():
    mel, _frame_sec, _t0 = pooled_log_mel(np.zeros(BUFFER_SIZE, dtype=np.float32),
                                          SAMPLE_RATE, BUFFER_SIZE)

    assert mel.shape == (0, MEL_BANDS)
    assert mel.dtype == np.float32


def test_silence_pools_to_zero_not_to_negative_infinity():
    audio = np.zeros(POOL_BUFFERS * BUFFER_SIZE, dtype=np.float32)
    mel, _frame_sec, _t0 = pooled_log_mel(audio, SAMPLE_RATE, BUFFER_SIZE)

    assert np.all(np.isfinite(mel))
    assert np.all(mel >= 0.0)


# --------------------------------------------------------------------------- #
# Sidecar file
# --------------------------------------------------------------------------- #


def test_sidecar_roundtrips_the_arrays_the_spec_requires(tmp_path):
    audio = np.concatenate(random_buffers(3 * POOL_BUFFERS))
    mel, frame_sec, t0 = pooled_log_mel(audio, SAMPLE_RATE, BUFFER_SIZE)
    path = tmp_path / "abc.npz"

    write_feature_sidecar(path, mel, frame_sec, t0)

    with np.load(path) as loaded:
        assert np.array_equal(loaded["mel"], mel)
        assert loaded["mel"].dtype == np.float32
        assert float(loaded["frame_sec"]) == pytest.approx(frame_sec)
        assert float(loaded["t0"]) == pytest.approx(t0)
        assert int(loaded["sample_rate"]) == SAMPLE_RATE
        assert int(loaded["pool_buffers"]) == POOL_BUFFERS


def test_a_new_sidecar_records_which_exporter_wrote_it(tmp_path):
    """Geometry cannot answer "same features?": a different log transform or a
    different pooling reduction keeps the frame rate and the band count and
    changes every number.  The generation says so explicitly, and the cache
    check reads it off the file rather than assuming."""
    path = tmp_path / "abc.npz"
    write_feature_sidecar(path, np.zeros((2, MEL_BANDS), dtype=np.float32), 0.046, 0.046)

    with np.load(path) as loaded:
        assert int(loaded[MEL_EXPORTER_KEY]) == MEL_EXPORTER_VERSION
    assert sidecar_generation(path) == MEL_EXPORTER_VERSION
