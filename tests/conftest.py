import sys
from pathlib import Path

import pytest

from lib.engine.effect_controller import EffectController
from simulate.stub_clients import StubMidiClient

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# A youtube id, because the committed file on disk carries a name derived from
# it rather than a readable one (see training/eval_assets.py).
ANCHOR_YOUTUBE_ID = "PNpXKsge4xM"


def anchor_mp3_path() -> str:
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


@pytest.fixture
def nn_artifacts():
    from lib import section_chain

    if not section_chain.artifacts_present():
        pytest.skip('the shipped NN artifacts are absent -- they live in the '
                    'gitignored corpus directory, not in the repository')
