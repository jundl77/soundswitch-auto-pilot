"""The exact madmom surface the live analyser is allowed to depend on.

madmom is pinned to a git SHA rather than a release, so an upgrade is a
deliberate act — and these tests are what makes it a *checked* one. They assert
the properties the live path rests on, not that the library is present:

  * the beat and onset chains have an ONLINE mode and we ask for it;
  * decoding is forward-only, so no result depends on future audio;
  * the offline downbeat path — which has no online mode at all and would
    silently make the pipeline non-causal — is never imported.
"""

import inspect

import pytest

pytestmark = pytest.mark.integration  # loading madmom's models is not a unit cost


def test_beat_chain_offers_an_online_mode():
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    assert 'online' in inspect.signature(RNNBeatProcessor.__init__).parameters or \
        'kwargs' in inspect.signature(RNNBeatProcessor.__init__).parameters
    assert 'online' in inspect.signature(DBNBeatTrackingProcessor.__init__).parameters
    assert hasattr(DBNBeatTrackingProcessor, 'process_online')


def test_onset_chain_offers_an_online_mode():
    from madmom.features.onsets import (OnsetPeakPickingProcessor,
                                        RNNOnsetProcessor)
    assert 'online' in inspect.signature(OnsetPeakPickingProcessor.__init__).parameters
    assert hasattr(OnsetPeakPickingProcessor, 'process_online')
    # RNNOnsetProcessor selects unidirectional nets from **kwargs['online'].
    assert 'kwargs' in inspect.signature(RNNOnsetProcessor.__init__).parameters


def test_online_beat_decoding_is_forward_only():
    """`process_online` must use the HMM forward algorithm. A Viterbi decode
    would need the whole sequence and would make every reported beat depend on
    audio the live path has not heard yet."""
    from madmom.features.beats import DBNBeatTrackingProcessor
    src = inspect.getsource(DBNBeatTrackingProcessor.process_online)
    assert 'self.hmm.forward(' in src
    assert 'viterbi' not in src.lower()


def test_online_beat_times_never_run_ahead_of_the_frame_counter():
    """A reported beat is stamped at a frame already consumed — the decoder
    cannot report a beat in the future."""
    from madmom.features.beats import DBNBeatTrackingProcessor
    src = inspect.getsource(DBNBeatTrackingProcessor.process_online)
    assert '(frame + self.counter) / float(self.fps)' in src


def test_processors_expose_reset_so_the_analyser_need_not_rebuild_models():
    """`MusicAnalyser` resets every 15 minutes and on every sound stop.
    Rebuilding would reload eight pickled LSTMs mid-show."""
    from madmom.features.beats import DBNBeatTrackingProcessor
    from madmom.features.onsets import OnsetPeakPickingProcessor
    assert callable(DBNBeatTrackingProcessor.reset)
    assert callable(OnsetPeakPickingProcessor.reset)


def test_the_live_path_never_imports_the_offline_downbeat_tracker():
    """madmom has no online downbeat mode (`DBNDownBeatTrackingProcessor` is a
    whole-sequence Viterbi over a bidirectional RNN). Importing it in `lib/`
    would be the easiest way to make the pipeline quietly non-causal."""
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
