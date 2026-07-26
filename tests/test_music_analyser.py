"""
Unit tests for MusicAnalyser: onset-density trend, beat timing, virtual-clock
behaviour, and the vectorized silence check.
"""
import datetime
import pytest
import numpy as np
from collections import deque
from lib.analyser.music_analyser import MusicAnalyser


class _StubHandler:
    """Minimal handler — satisfies the interface without any real behaviour."""
    def on_sound_start(self): pass
    def on_sound_stop(self): pass
    async def on_cycle(self): pass
    async def on_onset(self): pass
    async def on_beat(self, beat_number, bpm, bpm_changed): pass
    async def on_note(self): pass
    async def on_section_change(self): pass


@pytest.fixture
def analyser():
    return MusicAnalyser(
        sample_rate=44100,
        buffer_size=256,
        handler=_StubHandler(),
    )


# ---------------------------------------------------------------------------
# get_onset_density_trend
# ---------------------------------------------------------------------------

def test_trend_returns_one_when_fewer_than_four_samples(analyser):
    analyser._density_samples = deque([1.0, 2.0, 3.0], maxlen=12)
    assert analyser.get_onset_density_trend() == 1.0


def test_trend_rising(analyser):
    # past half [1, 1], recent half [3, 3] → ratio 3.0
    analyser._density_samples = deque([1.0, 1.0, 3.0, 3.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(3.0)


def test_trend_stable(analyser):
    analyser._density_samples = deque([4.0, 4.0, 4.0, 4.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(1.0)


def test_trend_falling(analyser):
    # past half [4, 4], recent half [2, 2] → ratio 0.5
    analyser._density_samples = deque([4.0, 4.0, 2.0, 2.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(0.5)


def test_trend_returns_one_when_past_mean_is_zero(analyser):
    # Avoid division by zero — past half is all zeros
    analyser._density_samples = deque([0.0, 0.0, 2.0, 2.0], maxlen=12)
    assert analyser.get_onset_density_trend() == 1.0


def test_trend_uses_all_twelve_samples(analyser):
    # Twelve samples: first 6 = 1.0, last 6 = 2.0 → trend ≈ 2.0
    samples = [1.0] * 6 + [2.0] * 6
    analyser._density_samples = deque(samples, maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# get_seconds_since_last_beat
# ---------------------------------------------------------------------------

def test_seconds_since_last_beat_approximately_correct(analyser):
    analyser.last_beat_detected = datetime.datetime.now() - datetime.timedelta(seconds=1.5)
    elapsed = analyser.get_seconds_since_last_beat()
    assert 1.4 < elapsed < 1.7


def test_seconds_since_last_beat_small_when_recent(analyser):
    analyser.last_beat_detected = datetime.datetime.now()
    assert analyser.get_seconds_since_last_beat() < 0.1


# ---------------------------------------------------------------------------
# Virtual clock
# ---------------------------------------------------------------------------

from lib.clock import VirtualClock


def _make_analyser(clock):
    return MusicAnalyser(
        sample_rate=44100,
        buffer_size=256,
        handler=_StubHandler(),
        clock=clock,
    )


def test_seconds_since_last_beat_on_virtual_clock():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    analyser.last_beat_detected = clock.now()
    clock.advance(1.5)
    assert analyser.get_seconds_since_last_beat() == 1.5


def test_onset_density_window_prunes_on_virtual_time():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    # Two onsets now, then advance beyond the 1.5 s rolling window.
    analyser._onset_times.append(clock.now())
    analyser._onset_times.append(clock.now())
    assert analyser.get_onset_density() == 2 / 1.5
    clock.advance(2.0)
    assert analyser.get_onset_density() == 0.0


# ---------------------------------------------------------------------------
# _is_silence — vectorized equivalence with the old list-comp semantics
# ---------------------------------------------------------------------------

def test_is_silence_all_near_zero(analyser):
    assert analyser._is_silence(np.zeros(40, dtype=np.float32)) is True
    assert analyser._is_silence(np.full(40, 0.00009, dtype=np.float64)) is True
    assert analyser._is_silence(np.full(40, -0.00009, dtype=np.float64)) is True


def test_is_silence_boundary_is_exclusive(analyser):
    # old semantics: -0.0001 < n < 0.0001 (strict) → exactly 0.0001 is NOT silence
    energies = np.zeros(40)
    energies[7] = 0.0001
    assert analyser._is_silence(energies) is False


def test_is_silence_single_loud_band(analyser):
    energies = np.zeros(40)
    energies[0] = 0.5
    assert analyser._is_silence(energies) is False


def test_dead_accumulation_arrays_removed(analyser):
    assert not hasattr(analyser, 'mfccs')
    assert not hasattr(analyser, 'energies')


# ---------------------------------------------------------------------------
# _fold_bpm — octave folding into [85, 170)
# ---------------------------------------------------------------------------

def test_fold_bpm_double_tempo_folds_down():
    # aubio warmup double-tempo lock: 257.8 must fold to 128.9
    assert MusicAnalyser._fold_bpm(257.8) == pytest.approx(128.9)


def test_fold_bpm_half_tempo_folds_up():
    assert MusicAnalyser._fold_bpm(64.0) == pytest.approx(128.0)


def test_fold_bpm_in_range_untouched():
    assert MusicAnalyser._fold_bpm(128.0) == pytest.approx(128.0)


def test_fold_bpm_boundary_170_folds_to_85():
    assert MusicAnalyser._fold_bpm(170.0) == pytest.approx(85.0)


def test_fold_bpm_zero_and_negative_return_zero():
    assert MusicAnalyser._fold_bpm(0.0) == 0.0
    assert MusicAnalyser._fold_bpm(-10.0) == 0.0


# ---------------------------------------------------------------------------
# get_kick_strength — transient capture, ratio cap, silence gate
# ---------------------------------------------------------------------------

def test_kick_beat_sample_captures_transient_peak(analyser):
    # Simulate 12 buffers of quiet sub-bass with one kick spike 5 buffers ago
    for v in [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 1.0, 1.0]:
        analyser._all_sub_bass_samples.append(v)
    analyser._rms_window.append(0.2)  # clearly audible
    analyser._capture_beat_sub_bass()
    # The captured value must be the transient peak (8.0), not the last buffer (1.0)
    assert analyser._beat_sub_bass_samples[-1] == pytest.approx(8.0)


def test_kick_ratio_capped(analyser):
    analyser._rms_window.append(0.2)
    analyser._all_sub_bass_samples.extend([0.001] * 50)
    analyser._beat_sub_bass_samples.extend([5.0] * 5)
    assert analyser.get_kick_strength() == pytest.approx(10.0)


def test_kick_near_silence_returns_unknown(analyser):
    analyser._rms_window.append(0.001)  # -60 dBFS: fade-out / silence
    analyser._all_sub_bass_samples.extend([1e-6] * 50)
    analyser._beat_sub_bass_samples.extend([1e-3] * 5)
    assert analyser.get_kick_strength() == 1.0
