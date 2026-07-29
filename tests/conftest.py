import sys
from pathlib import Path

import pytest

from lib.engine.effect_controller import EffectController
from simulate.stub_clients import StubMidiClient

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# The one real track the non-benchmark tests read.  It used to be the bundled
# Generate mp3; the eval set retired that file, and the committed eval-set audio
# is committed for exactly the same reason -- so a fresh clone with no corpus can
# still run everything.  Named by youtube id because the file on disk carries a
# derived, opaque name (see training/eval_assets.py).
#
# This id is also the migration digest's anchor: tests/fixtures/
# pipeline_digest_baseline.json holds a pre-madmom digest of it, cut on master's
# own code, which is how the anchor keeps its provenance across the retirement.
ANCHOR_YOUTUBE_ID = "PNpXKsge4xM"


def anchor_mp3_path() -> str:
    """The committed anchor track, or one clear failure line naming it."""
    import run_eval_set

    path = run_eval_set.audio_path(run_eval_set.corpus_dir(), ANCHOR_YOUTUBE_ID)
    if not Path(path).exists():
        pytest.fail(
            f"the committed anchor track is missing: {path} -- restore it from "
            f"training/eval_audio/ (see the benchmark section of CLAUDE.md)"
        )
    return str(path)


@pytest.fixture
def anchor_mp3():
    return anchor_mp3_path()


@pytest.fixture
def stub_midi():
    return StubMidiClient()


@pytest.fixture
def effect_controller(stub_midi):
    return EffectController(stub_midi)
