"""The exact madmom surface the live analyser is allowed to depend on.

madmom is pinned to a git SHA rather than a release, so an upgrade is a
deliberate act — and these tests are what make it a *checked* one.

They assert properties of the CONSTRUCTED processors, not of the keyword
arguments passed to build them. An earlier version checked that `online` was an
accepted parameter, which cannot fail for the reason the test exists: an
upgrade that renamed or dropped the flag would hand the live path bidirectional
networks with every assertion still green. What is checked here instead is the
thing that actually makes the pipeline causal — that no network in the live path
contains a layer which needs to see the future.
"""

import inspect

import pytest

pytestmark = pytest.mark.integration  # loading madmom's models is not a unit cost


def _networks(processor):
    """Every neural network reachable from a processor tree."""
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


# ---------------------------------------------------------------------------
# Causality — the property the whole migration rests on
# ---------------------------------------------------------------------------

def test_the_beat_network_we_build_contains_no_bidirectional_layer():
    """A bidirectional layer consumes the sequence in both directions, so it
    cannot emit anything until the audio has ended. Its presence in the live
    path would mean the pipeline is not causal, whatever the flags say."""
    from madmom.features.beats import RNNBeatProcessor
    kinds = _layer_kinds(RNNBeatProcessor(online=True, origin='stream',
                                          num_frames=1, fps=100))
    assert 'BidirectionalLayer' not in kinds
    assert 'LSTMLayer' in kinds, f'expected unidirectional LSTMs, got {kinds}'


def test_the_onset_network_we_build_contains_no_bidirectional_layer():
    from madmom.features.onsets import RNNOnsetProcessor
    kinds = _layer_kinds(RNNOnsetProcessor(online=True, origin='stream',
                                           num_frames=1, fps=100))
    assert 'BidirectionalLayer' not in kinds
    assert 'RecurrentLayer' in kinds, f'expected unidirectional RNNs, got {kinds}'


def test_the_offline_variants_really_are_bidirectional():
    """The control. Without it the two tests above would also pass against a
    madmom that had quietly stopped building bidirectional nets at all, and
    would then be asserting nothing."""
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.onsets import RNNOnsetProcessor
    assert 'BidirectionalLayer' in _layer_kinds(RNNBeatProcessor())
    assert 'BidirectionalLayer' in _layer_kinds(RNNOnsetProcessor())


def test_the_frame_geometry_is_the_online_one():
    """Online and offline chains are built over different window sizes. These
    are the sizes the adapter's 2048-sample buffer is sized against, so a change
    here is a change the adapter must be told about."""
    from lib.analyser.madmom_rhythm import FRAME_SIZE
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.onsets import RNNOnsetProcessor

    beat = _frame_sizes(RNNBeatProcessor(online=True, origin='stream',
                                         num_frames=1, fps=100))
    onset = _frame_sizes(RNNOnsetProcessor(online=True, origin='stream',
                                           num_frames=1, fps=100))
    assert beat == [2048], f'online beat geometry changed: {beat}'
    assert onset == [512, 1024, 2048], f'online onset geometry changed: {onset}'
    assert max(beat + onset) == FRAME_SIZE, (
        'the adapter buffers FRAME_SIZE samples and hands the same frame to '
        'both chains; it must be at least the largest window either one reads')


# ---------------------------------------------------------------------------
# Decoding — forward only, no result depending on unheard audio
# ---------------------------------------------------------------------------

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
    """The adapter stamps each event with the hop it was produced on, which is
    only sound because a single-frame call cannot return two events."""
    import numpy as np
    from madmom.features.beats import DBNBeatTrackingProcessor

    dbn = DBNBeatTrackingProcessor(fps=100, online=True)
    dbn.reset()
    for value in np.tile([0.9, 0.1, 0.1, 0.1, 0.1], 200):
        assert len(np.atleast_1d(
            dbn.process_online(np.array([value]), reset=False))) <= 1


def test_processors_expose_reset_so_the_analyser_need_not_rebuild_models():
    from madmom.features.beats import DBNBeatTrackingProcessor
    from madmom.features.onsets import OnsetPeakPickingProcessor
    assert callable(DBNBeatTrackingProcessor.reset)
    assert callable(OnsetPeakPickingProcessor.reset)


def test_the_live_path_never_imports_the_offline_downbeat_tracker():
    """madmom has no online downbeat mode (`DBNDownBeatTrackingProcessor` is a
    whole-sequence Viterbi over a bidirectional RNN). Naming it in `lib/` would
    be the easiest way to make the pipeline quietly non-causal."""
    from pathlib import Path
    lib = Path(__file__).parent.parent / 'lib'
    offenders = [p for p in lib.rglob('*.py')
                 if 'DownBeat' in p.read_text(encoding='utf-8')]
    assert not offenders, f'offline downbeat tracker referenced in {offenders}'


def test_the_pinned_version_is_the_one_that_imports_on_this_python():
    """PyPI 0.16.1 dies at import on >= 3.10; the pin is git main for that
    reason, and 0.17.dev0 is what that SHA reports."""
    import madmom
    assert madmom.__version__.startswith('0.17')
