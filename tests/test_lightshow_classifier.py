import pytest
from lib.analyser.lightshow_classifier import classify_track, LightShowType


def test_hip_hop_genre_returns_hip_hop():
    result = classify_track(['hip hop'], bpm=90, energy=0.6, loudness=-7.0, danceability=0.7)
    assert result == LightShowType.HIP_HOP


def test_high_energy_dance_returns_high_intensity():
    result = classify_track(['dance', 'house'], bpm=128, energy=0.95, loudness=-6.0, danceability=0.8)
    assert result == LightShowType.HIGH_INTENSITY


def test_loud_track_triggers_high_intensity_regardless_of_genre():
    # loudness > -4.5 is its own OR-branch in classify_track
    result = classify_track(['pop'], bpm=100, energy=0.5, loudness=-3.0, danceability=0.5)
    assert result == LightShowType.HIGH_INTENSITY


def test_low_bpm_returns_low_intensity():
    # 'mellow' has no high/medium genres; bpm < 90 triggers LOW
    result = classify_track(['mellow'], bpm=70, energy=0.3, loudness=-12.0, danceability=0.2)
    assert result == LightShowType.LOW_INTENSITY


def test_pop_mid_energy_returns_medium_intensity():
    result = classify_track(['pop'], bpm=120, energy=0.6, loudness=-8.0, danceability=0.6)
    assert result == LightShowType.MEDIUM_INTENSITY


def test_low_energy_returns_low_intensity():
    result = classify_track(['pop'], bpm=120, energy=0.2, loudness=-10.0, danceability=0.5)
    assert result == LightShowType.LOW_INTENSITY
