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
    write_feature_sidecar,
)


def test_sidecar_roundtrips_the_arrays_the_spec_requires(tmp_path):
    rng = np.random.default_rng(20260726)
    mel = rng.standard_normal((3, MEL_BANDS)).astype(np.float32)
    frame_sec = t0 = POOL_BUFFERS * BUFFER_SIZE / SAMPLE_RATE
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
    path = tmp_path / "abc.npz"
    write_feature_sidecar(path, np.zeros((2, MEL_BANDS), dtype=np.float32), 0.046, 0.046)

    with np.load(path) as loaded:
        assert int(loaded[MEL_EXPORTER_KEY]) == MEL_EXPORTER_VERSION
