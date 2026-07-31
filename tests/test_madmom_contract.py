import functools
import inspect

import pytest

pytestmark = pytest.mark.integration


@functools.lru_cache(maxsize=None)
def _beat_stage():
    """The stage object the pipeline actually runs, not a rebuild from kwargs
    copied into this file — a copy cannot fail when the shipped one changes."""
    from lib.analyser.madmom_rhythm import _BeatStage
    return _BeatStage()


def _networks(processor):
    from madmom.ml.nn import NeuralNetwork

    def walk(node):
        yield node
        for child in getattr(node, 'processors', []) or []:
            yield from walk(child)

    return [n for n in walk(processor) if isinstance(n, NeuralNetwork)]


def _layer_kinds(processor) -> set[str]:
    return {type(layer).__name__
            for net in _networks(processor) for layer in net.layers}


def _frame_sizes(processor) -> list[int]:
    def walk(node):
        yield node
        for child in getattr(node, 'processors', []) or []:
            yield from walk(child)

    return [n.frame_size for n in walk(processor) if hasattr(n, 'frame_size')]


def test_the_beat_network_we_build_contains_no_bidirectional_layer():
    kinds = _layer_kinds(_beat_stage()._rnn)
    assert 'BidirectionalLayer' not in kinds
    assert 'LSTMLayer' in kinds, f'expected unidirectional LSTMs, got {kinds}'


def test_the_offline_variant_really_is_bidirectional():
    from madmom.features.beats import RNNBeatProcessor
    assert 'BidirectionalLayer' in _layer_kinds(RNNBeatProcessor())


def test_the_frame_geometry_is_the_online_one():
    from lib.analyser.madmom_rhythm import FRAME_SIZE

    beat = _frame_sizes(_beat_stage()._rnn)
    assert beat == [2048], f'online beat geometry changed: {beat}'
    assert max(beat) == FRAME_SIZE, (
        'the adapter buffers FRAME_SIZE samples before handing over a frame; it '
        'must be at least the largest window the chain reads')


def test_online_beat_decoding_is_forward_only():
    from madmom.features.beats import DBNBeatTrackingProcessor
    src = inspect.getsource(DBNBeatTrackingProcessor.process_online)
    assert 'self.hmm.forward(' in src
    assert 'viterbi' not in src.lower()


def test_online_beat_times_never_run_ahead_of_the_frame_counter():
    from madmom.features.beats import DBNBeatTrackingProcessor
    src = inspect.getsource(DBNBeatTrackingProcessor.process_online)
    assert '(frame + self.counter) / float(self.fps)' in src


def test_one_activation_in_yields_at_most_one_event_out():
    import numpy as np
    from madmom.features.beats import DBNBeatTrackingProcessor

    dbn = DBNBeatTrackingProcessor(fps=100, online=True)
    dbn.reset()
    for value in np.tile([0.9, 0.1, 0.1, 0.1, 0.1], 200):
        assert len(np.atleast_1d(
            dbn.process_online(np.array([value]), reset=False))) <= 1


def test_processors_expose_reset_so_the_analyser_need_not_rebuild_models():
    from madmom.features.beats import DBNBeatTrackingProcessor
    assert callable(DBNBeatTrackingProcessor.reset)


def test_the_live_path_never_imports_the_offline_downbeat_tracker():
    from pathlib import Path
    lib = Path(__file__).parent.parent / 'lib'
    offenders = [p for p in lib.rglob('*.py')
                 if 'DownBeat' in p.read_text(encoding='utf-8')]
    assert not offenders, f'offline downbeat tracker referenced in {offenders}'



def test_the_pinned_version_is_the_one_that_imports_on_this_python():
    # PyPI 0.16.1 dies at import on Python >= 3.10; the pin is git main, which reports 0.17.dev0.
    import madmom
    assert madmom.__version__.startswith('0.17')
