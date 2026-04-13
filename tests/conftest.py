import pytest
from lib.engine.effect_controller import EffectController
from simulate.stub_clients import StubMidiClient


@pytest.fixture
def stub_midi():
    return StubMidiClient()


@pytest.fixture
def effect_controller(stub_midi):
    return EffectController(stub_midi)
