"""
Tests for MusicAnalyser methods introduced in this PR:
  - get_onset_density_trend()
  - get_seconds_since_last_beat()
"""
import datetime
import pytest
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
        visualizer_updater=None,
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
