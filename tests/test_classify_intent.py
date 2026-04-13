import pytest
from lib.engine.light_engine import (
    _classify_intent, _classify_windowed,
    _DROP_MIN_DENSITY_ENTER, _DROP_MIN_DENSITY_EXIT,
    _PEAK_MIN_BPM_ENTER, _PEAK_MIN_BPM_EXIT,
    _BREAKDOWN_MAX_DENSITY_ENTER, _BREAKDOWN_MAX_DENSITY_EXIT,
)
from lib.engine.effect_definitions import LightIntent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window(densities: list[float], bpm: float = 128.0, sub_bass: float = 0.0) -> list[tuple]:
    """Build a fake window of BeatRecords (5-tuples) at evenly spaced monotonic times."""
    return [(float(i), d, bpm, sub_bass, 0.5) for i, d in enumerate(densities)]


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


# ---------------------------------------------------------------------------
# _classify_windowed
# ---------------------------------------------------------------------------

def test_windowed_drop_requires_sustained_density():
    # A single spike surrounded by normal density → median stays below DROP threshold → GROOVE
    densities = [4.0, 4.0, 9.5, 4.0, 4.0]
    assert _classify_windowed(_window(densities), bpm=128.0) != LightIntent.DROP


def test_windowed_drop_on_sustained_high_density():
    # Genuine DROP: all beats in window have high density
    densities = [9.0, 9.5, 10.0, 9.2, 8.8]
    assert _classify_windowed(_window(densities), bpm=128.0) == LightIntent.DROP


def test_windowed_buildup_detected_via_forward_context():
    # Past half: low density; future half: high density → forward trend ≥ 1.3 → BUILDUP
    densities = [3.0, 3.2, 5.0, 5.5, 6.0]
    assert _classify_windowed(_window(densities), bpm=120.0) == LightIntent.BUILDUP


def test_windowed_stable_groove_not_classified_as_buildup():
    # Flat density across the window → trend ≈ 1.0 → GROOVE
    densities = [4.5, 4.5, 4.5, 4.5, 4.5]
    assert _classify_windowed(_window(densities), bpm=120.0) == LightIntent.GROOVE


def test_windowed_empty_window_returns_groove():
    assert _classify_windowed([], bpm=128.0) == LightIntent.GROOVE


def test_windowed_breakdown_on_sustained_low_density():
    densities = [1.0, 1.2, 0.8, 1.1, 0.9]
    assert _classify_windowed(_window(densities), bpm=128.0) == LightIntent.BREAKDOWN


# ---------------------------------------------------------------------------
# Hysteresis tests
# ---------------------------------------------------------------------------

def test_drop_entry_threshold():
    # Entry threshold: density must reach _DROP_MIN_DENSITY_ENTER (8.5) to enter DROP
    assert _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER) == LightIntent.DROP
    assert _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER - 0.1) != LightIntent.DROP


def test_drop_hysteresis_stays_in_drop_above_exit_threshold():
    # When currently in DROP, the exit threshold (_DROP_MIN_DENSITY_EXIT = 7.0) applies.
    # Density between 7.0 and 8.5 should STAY in DROP (below entry, above exit).
    mid_density = (_DROP_MIN_DENSITY_EXIT + _DROP_MIN_DENSITY_ENTER) / 2  # e.g. 7.75
    assert _classify_intent(128.0, mid_density, current_intent=LightIntent.DROP) == LightIntent.DROP


def test_drop_hysteresis_exits_below_exit_threshold():
    # Density below exit threshold should leave DROP even when currently in DROP.
    below_exit = _DROP_MIN_DENSITY_EXIT - 0.5
    result = _classify_intent(128.0, below_exit, current_intent=LightIntent.DROP)
    assert result != LightIntent.DROP


def test_drop_cold_entry_requires_higher_threshold():
    # Without current_intent=DROP, mid-zone density should NOT enter DROP.
    mid_density = (_DROP_MIN_DENSITY_EXIT + _DROP_MIN_DENSITY_ENTER) / 2
    assert _classify_intent(128.0, mid_density) != LightIntent.DROP


def test_peak_entry_threshold():
    assert _classify_intent(_PEAK_MIN_BPM_ENTER, 4.0) == LightIntent.PEAK
    assert _classify_intent(_PEAK_MIN_BPM_ENTER - 1.0, 4.0) != LightIntent.PEAK


def test_peak_hysteresis_stays_in_peak_above_exit_threshold():
    mid_bpm = (_PEAK_MIN_BPM_EXIT + _PEAK_MIN_BPM_ENTER) / 2  # e.g. 137.5
    assert _classify_intent(mid_bpm, 4.0, current_intent=LightIntent.PEAK) == LightIntent.PEAK


def test_peak_hysteresis_exits_below_exit_threshold():
    below_exit_bpm = _PEAK_MIN_BPM_EXIT - 1.0
    result = _classify_intent(below_exit_bpm, 4.0, current_intent=LightIntent.PEAK)
    assert result != LightIntent.PEAK


def test_breakdown_entry_threshold():
    # density just below entry threshold enters BREAKDOWN
    assert _classify_intent(128.0, _BREAKDOWN_MAX_DENSITY_ENTER - 0.1) == LightIntent.BREAKDOWN
    # density at entry threshold stays out of BREAKDOWN
    assert _classify_intent(128.0, _BREAKDOWN_MAX_DENSITY_ENTER) != LightIntent.BREAKDOWN


def test_breakdown_hysteresis_stays_in_breakdown_below_exit_threshold():
    # When currently in BREAKDOWN, density must exceed exit threshold (3.5) to leave.
    mid_density = (_BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_MAX_DENSITY_EXIT) / 2  # e.g. 3.25
    assert _classify_intent(128.0, mid_density, current_intent=LightIntent.BREAKDOWN) == LightIntent.BREAKDOWN


def test_breakdown_hysteresis_exits_above_exit_threshold():
    above_exit = _BREAKDOWN_MAX_DENSITY_EXIT + 0.1
    result = _classify_intent(128.0, above_exit, current_intent=LightIntent.BREAKDOWN)
    assert result != LightIntent.BREAKDOWN
