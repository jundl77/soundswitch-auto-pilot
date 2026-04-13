import pytest
from lib.engine.light_engine import _classify_intent
from lib.engine.effect_definitions import LightIntent


def test_drop_on_density_spike_at_dance_bpm():
    assert _classify_intent(128.0, 9.0) == LightIntent.DROP


def test_drop_requires_bpm_floor():
    # High density but BPM below 100 — should not trigger DROP
    result = _classify_intent(80.0, 10.0)
    assert result != LightIntent.DROP


def test_drop_beats_peak_at_high_bpm_high_density():
    # 140 BPM + 9 density: density spike wins, DROP before PEAK
    assert _classify_intent(140.0, 9.0) == LightIntent.DROP


def test_peak_at_high_bpm_moderate_density():
    assert _classify_intent(140.0, 4.0) == LightIntent.PEAK


def test_breakdown_on_sparse_density():
    # density < 3.0 → BREAKDOWN regardless of BPM
    assert _classify_intent(128.0, 1.5) == LightIntent.BREAKDOWN


def test_buildup_on_rising_trend():
    # density >= 3.0 and trend >= 1.3 → BUILDUP
    assert _classify_intent(120.0, 5.0, density_trend=1.5) == LightIntent.BUILDUP


def test_no_buildup_without_rising_trend():
    # density >= 3.0 but trend stable → GROOVE, not BUILDUP
    assert _classify_intent(120.0, 5.0, density_trend=1.0) == LightIntent.GROOVE


def test_groove_is_default_at_moderate_conditions():
    assert _classify_intent(100.0, 4.0, density_trend=1.0) == LightIntent.GROOVE


def test_atmospheric_never_returned_by_classifier():
    # ATMOSPHERIC is set via beat-absence only, never by _classify_intent
    cases = [
        (60.0, 0.0), (80.0, 1.0), (100.0, 5.0), (130.0, 3.5), (145.0, 2.0),
    ]
    for bpm, density in cases:
        assert _classify_intent(bpm, density) != LightIntent.ATMOSPHERIC


def test_buildup_trend_threshold_boundary():
    # trend exactly at threshold fires BUILDUP
    assert _classify_intent(120.0, 5.0, density_trend=1.3) == LightIntent.BUILDUP
    # trend just below threshold falls to GROOVE
    assert _classify_intent(120.0, 5.0, density_trend=1.29) == LightIntent.GROOVE
