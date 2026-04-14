import pytest
from lib.engine.light_engine import (
    _classify_intent, _classify_windowed,
    _DROP_MIN_DENSITY_ENTER, _DROP_MIN_DENSITY_EXIT,
    _DROP_MIN_SUB_BASS_RATIO,
    _PEAK_MIN_BPM_ENTER, _PEAK_MIN_BPM_EXIT,
    _PEAK_MIN_RMS, _PEAK_MIN_DENSITY_FOR_RMS_PEAK,
    _BREAKDOWN_MAX_DENSITY_ENTER, _BREAKDOWN_MAX_DENSITY_EXIT,
    _KICK_PRESENCE_THRESHOLD, _CENTROID_BUILDUP_TREND,
    _BUILDUP_MIN_TREND,
    _BREAKDOWN_NO_KICK_MAX_DENSITY,
    _BREAKDOWN_MAX_DENSITY_WITH_LOW_SUBBASS, _BREAKDOWN_MAX_SUB_BASS,
)
from lib.engine.effect_definitions import LightIntent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window(densities: list[float], bpm: float = 128.0, sub_bass: float = 0.0,
            kick: float = 2.0, centroid_trend: float = 1.0, spectral_flux: float = 0.0,
            rms: float = 0.0) -> list[tuple]:
    """Build a fake window of BeatRecords (8-tuples) at evenly spaced monotonic times.

    rms defaults to 0.0 so the RMS PEAK gate does not fire unless tests explicitly set it.
    """
    return [(float(i), d, bpm, sub_bass, rms, kick, centroid_trend, spectral_flux) for i, d in enumerate(densities)]


def test_drop_on_density_spike_at_dance_bpm():
    assert _classify_intent(128.0, 12.0, sub_bass_ratio=_DROP_MIN_SUB_BASS_RATIO) == LightIntent.DROP


def test_drop_requires_bpm_floor():
    # High density but BPM below 100 — should not trigger DROP
    result = _classify_intent(80.0, 10.0)
    assert result != LightIntent.DROP


def test_drop_beats_peak_at_high_bpm_high_density():
    # 140 BPM + 12 density: density spike wins, DROP before PEAK
    assert _classify_intent(140.0, 12.0, sub_bass_ratio=_DROP_MIN_SUB_BASS_RATIO) == LightIntent.DROP


def test_peak_at_high_bpm_moderate_density():
    assert _classify_intent(140.0, 4.0) == LightIntent.PEAK


def test_breakdown_on_sparse_density():
    # density < 3.0 → BREAKDOWN regardless of BPM
    assert _classify_intent(128.0, 1.5) == LightIntent.BREAKDOWN


def test_buildup_on_rising_trend():
    # density above BREAKDOWN threshold and trend >= _BUILDUP_MIN_TREND → BUILDUP
    assert _classify_intent(120.0, 5.0, density_trend=_BUILDUP_MIN_TREND) == LightIntent.BUILDUP


def test_no_buildup_without_rising_trend():
    # density >= 3.0 but trend stable → GROOVE, not BUILDUP
    assert _classify_intent(120.0, 5.0, density_trend=1.0) == LightIntent.GROOVE


def test_groove_is_default_at_moderate_conditions():
    # density above BREAKDOWN threshold, no rising trend, BPM below PEAK → GROOVE
    assert _classify_intent(100.0, _BREAKDOWN_MAX_DENSITY_ENTER + 1.0, density_trend=1.0) == LightIntent.GROOVE


def test_atmospheric_never_returned_by_classifier():
    # ATMOSPHERIC is set via beat-absence only, never by _classify_intent
    cases = [
        (60.0, 0.0), (80.0, 1.0), (100.0, 5.0), (130.0, 3.5), (145.0, 2.0),
    ]
    for bpm, density in cases:
        assert _classify_intent(bpm, density) != LightIntent.ATMOSPHERIC


def test_buildup_trend_threshold_boundary():
    # trend exactly at _BUILDUP_MIN_TREND fires BUILDUP
    assert _classify_intent(120.0, 5.0, density_trend=_BUILDUP_MIN_TREND) == LightIntent.BUILDUP
    # trend just below threshold falls to GROOVE
    assert _classify_intent(120.0, 5.0, density_trend=_BUILDUP_MIN_TREND - 0.001) == LightIntent.GROOVE


# ---------------------------------------------------------------------------
# _classify_windowed
# ---------------------------------------------------------------------------

def test_windowed_drop_requires_sustained_density():
    # A single spike surrounded by normal density → median stays below DROP threshold → GROOVE
    densities = [4.0, 4.0, 9.5, 4.0, 4.0]
    assert _classify_windowed(_window(densities), bpm=128.0) != LightIntent.DROP


def test_windowed_drop_on_sustained_high_density():
    # Genuine DROP: all beats in window have high density and sufficient sub-bass
    densities = [_DROP_MIN_DENSITY_ENTER, _DROP_MIN_DENSITY_ENTER + 0.5,
                 _DROP_MIN_DENSITY_ENTER + 1.0, _DROP_MIN_DENSITY_ENTER + 0.2,
                 _DROP_MIN_DENSITY_ENTER + 0.1]
    assert _classify_windowed(_window(densities, sub_bass=_DROP_MIN_SUB_BASS_RATIO), bpm=128.0) == LightIntent.DROP


def test_windowed_buildup_detected_via_forward_context():
    # Past half: low density; future half: high density → forward trend ≥ _BUILDUP_MIN_TREND → BUILDUP
    # Use a large ratio (future ~3x past) to ensure trend exceeds the threshold
    densities = [2.0, 2.0, 5.5, 6.0, 6.5]
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
    # Entry threshold: density must reach _DROP_MIN_DENSITY_ENTER (with all other gates met)
    assert _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER,
                            sub_bass_ratio=_DROP_MIN_SUB_BASS_RATIO) == LightIntent.DROP
    assert _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER - 0.1,
                            sub_bass_ratio=_DROP_MIN_SUB_BASS_RATIO) != LightIntent.DROP


def test_drop_hysteresis_stays_in_drop_above_exit_threshold():
    # When currently in DROP, the exit threshold applies.
    # Density between exit and entry thresholds should STAY in DROP.
    mid_density = (_DROP_MIN_DENSITY_EXIT + _DROP_MIN_DENSITY_ENTER) / 2
    assert _classify_intent(128.0, mid_density, current_intent=LightIntent.DROP,
                            sub_bass_ratio=_DROP_MIN_SUB_BASS_RATIO) == LightIntent.DROP


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
    # With groove-level sub-bass (no compound-rule interference), density above the
    # exit threshold must leave BREAKDOWN → normal hysteresis still works.
    above_exit = _BREAKDOWN_MAX_DENSITY_EXIT + 0.1
    result = _classify_intent(128.0, above_exit, current_intent=LightIntent.BREAKDOWN,
                              sub_bass_ratio=0.25)
    assert result != LightIntent.BREAKDOWN


# ---------------------------------------------------------------------------
# Kick-strength tests
# ---------------------------------------------------------------------------

def test_drop_requires_kick_presence():
    # High density but no kick → should NOT be DROP
    no_kick = _KICK_PRESENCE_THRESHOLD - 0.1
    result = _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER + 1, kick_strength=no_kick)
    assert result != LightIntent.DROP


def test_drop_with_kick_present():
    kick = _KICK_PRESENCE_THRESHOLD + 0.2
    assert _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER + 1, kick_strength=kick,
                            sub_bass_ratio=_DROP_MIN_SUB_BASS_RATIO) == LightIntent.DROP


def test_breakdown_at_moderate_density_with_no_kick():
    # Density between BREAKDOWN and BREAKDOWN_NO_KICK max — kick absent → BREAKDOWN
    mid_density = (_BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_NO_KICK_MAX_DENSITY) / 2
    no_kick = _KICK_PRESENCE_THRESHOLD - 0.1
    assert _classify_intent(128.0, mid_density, kick_strength=no_kick) == LightIntent.BREAKDOWN


def test_groove_at_moderate_density_with_kick():
    # Same density as above but kick present → GROOVE, not BREAKDOWN
    mid_density = (_BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_NO_KICK_MAX_DENSITY) / 2
    kick = _KICK_PRESENCE_THRESHOLD + 0.2
    assert _classify_intent(128.0, mid_density, kick_strength=kick) == LightIntent.GROOVE


def test_high_density_no_kick_above_breakdown_no_kick_max_stays_groove():
    # If density exceeds the no-kick BREAKDOWN ceiling, GROOVE wins even without kick
    above_max = _BREAKDOWN_NO_KICK_MAX_DENSITY + 0.5
    no_kick = _KICK_PRESENCE_THRESHOLD - 0.1
    result = _classify_intent(128.0, above_max, kick_strength=no_kick)
    # Not DROP (no kick), not BREAKDOWN (density too high) — should be GROOVE or BUILDUP
    assert result not in (LightIntent.DROP, LightIntent.BREAKDOWN)


# ---------------------------------------------------------------------------
# Spectral centroid trend tests
# ---------------------------------------------------------------------------

def test_buildup_via_centroid_trend_without_density_trend():
    # Rising centroid alone (density trend neutral) → BUILDUP
    rising = _CENTROID_BUILDUP_TREND + 0.05
    result = _classify_intent(120.0, 5.0, density_trend=1.0, centroid_trend=rising)
    assert result == LightIntent.BUILDUP


def test_groove_when_centroid_trend_is_neutral():
    # Neutral centroid trend + neutral density trend → GROOVE
    assert _classify_intent(120.0, 5.0, density_trend=1.0, centroid_trend=1.0) == LightIntent.GROOVE


def test_buildup_via_either_trend_signal():
    # Either rising density OR rising centroid is sufficient for BUILDUP
    below_density_threshold = _CENTROID_BUILDUP_TREND - 0.05  # density trend not rising
    above_centroid_threshold = _CENTROID_BUILDUP_TREND + 0.05
    assert _classify_intent(120.0, 5.0, density_trend=below_density_threshold,
                            centroid_trend=above_centroid_threshold) == LightIntent.BUILDUP


# ---------------------------------------------------------------------------
# Windowed: kick and centroid propagate through the window
# ---------------------------------------------------------------------------

def test_windowed_drop_blocked_without_kick():
    # All beats have high density but no kick — should NOT classify as DROP
    no_kick = _KICK_PRESENCE_THRESHOLD - 0.1
    densities = [9.0, 9.5, 10.0, 9.2, 8.8]
    assert _classify_windowed(_window(densities, kick=no_kick), bpm=128.0) != LightIntent.DROP


def test_windowed_buildup_via_rising_centroid():
    # Flat density (would be GROOVE without centroid) but rising centroid → BUILDUP
    rising = _CENTROID_BUILDUP_TREND + 0.1
    densities = [4.5, 4.5, 4.5, 4.5, 4.5]
    assert _classify_windowed(_window(densities, centroid_trend=rising), bpm=120.0) == LightIntent.BUILDUP


# ---------------------------------------------------------------------------
# Compound BREAKDOWN rule tests
# (moderate density + absent sub-bass → BREAKDOWN for stripped arrangements)
# ---------------------------------------------------------------------------

def test_compound_breakdown_fires_for_stripped_arrangement():
    # density in gap above BREAKDOWN_MAX_DENSITY_ENTER with low sub-bass → BREAKDOWN
    mid_density = (_BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_MAX_DENSITY_WITH_LOW_SUBBASS) / 2
    low_sub = _BREAKDOWN_MAX_SUB_BASS - 0.05
    assert _classify_intent(128.0, mid_density, sub_bass_ratio=low_sub) == LightIntent.BREAKDOWN


def test_compound_breakdown_does_not_fire_with_bass_present():
    # Same density range but sub-bass at or above threshold → GROOVE
    mid_density = (_BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_MAX_DENSITY_WITH_LOW_SUBBASS) / 2
    assert _classify_intent(128.0, mid_density, sub_bass_ratio=_BREAKDOWN_MAX_SUB_BASS) != LightIntent.BREAKDOWN


def test_compound_breakdown_extends_stay_with_low_subbass():
    # When in BREAKDOWN with low sub-bass, compound rule keeps us there even above
    # the normal sparse-density exit threshold — prevents density-fluctuation oscillation.
    above_exit = _BREAKDOWN_MAX_DENSITY_EXIT + 0.1
    low_sub = _BREAKDOWN_MAX_SUB_BASS - 0.05
    result = _classify_intent(128.0, above_exit, current_intent=LightIntent.BREAKDOWN,
                              sub_bass_ratio=low_sub)
    assert result == LightIntent.BREAKDOWN


def test_compound_breakdown_superseded_by_buildup():
    # Rising density trend fires BUILDUP before compound BREAKDOWN check.
    mid_density = (_BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_MAX_DENSITY_WITH_LOW_SUBBASS) / 2
    low_sub = _BREAKDOWN_MAX_SUB_BASS - 0.05
    assert _classify_intent(128.0, mid_density, density_trend=_BUILDUP_MIN_TREND,
                            sub_bass_ratio=low_sub) == LightIntent.BUILDUP


def test_compound_breakdown_does_not_fire_above_density_ceiling():
    # density at or above the compound rule ceiling → not caught → GROOVE
    at_ceiling = _BREAKDOWN_MAX_DENSITY_WITH_LOW_SUBBASS
    low_sub = _BREAKDOWN_MAX_SUB_BASS - 0.05
    assert _classify_intent(128.0, at_ceiling, sub_bass_ratio=low_sub) != LightIntent.BREAKDOWN
