import pytest
from lib.engine.light_engine import (
    BeatRecord, _classify_intent, _classify_windowed,
    _DROP_MIN_DENSITY_ENTER, _DROP_MIN_DENSITY_EXIT,
    _BREAKDOWN_MAX_DENSITY_ENTER, _BREAKDOWN_MAX_DENSITY_EXIT,
    _KICK_PRESENCE_THRESHOLD, _CENTROID_BUILDUP_TREND,
    _BREAKDOWN_NO_KICK_MARGIN, _BUILDUP_MIN_DENSITY,
)
from lib.analyser.music_analyser import KICK_UNKNOWN
from lib.engine.effect_definitions import LightIntent


KICK_PRESENT = _KICK_PRESENCE_THRESHOLD + 0.5
KICK_ABSENT  = _KICK_PRESENCE_THRESHOLD - 0.1
GROOVE_DENSITY = (_BREAKDOWN_MAX_DENSITY_EXIT + _DROP_MIN_DENSITY_ENTER) / 2


def _window(densities: list[float], bpm: float = 128.0, sub_bass: float = 0.0,
            kick: float = KICK_PRESENT, centroid_trend: float = 1.0) -> list[BeatRecord]:
    return [BeatRecord(float(i), d, bpm, sub_bass, 0.5, kick, centroid_trend)
            for i, d in enumerate(densities)]


def test_drop_on_density_spike_at_dance_bpm():
    assert _classify_intent(128.0, 9.0, kick_strength=KICK_PRESENT) == LightIntent.DROP


def test_unmeasured_kick_reads_as_absent():
    assert _classify_intent(128.0, 9.0) != LightIntent.DROP
    assert _classify_intent(128.0, 9.0, kick_strength=KICK_UNKNOWN) != LightIntent.DROP


def test_drop_requires_bpm_floor():
    result = _classify_intent(80.0, 10.0)
    assert result != LightIntent.DROP


def test_peak_never_returned_by_pure_classifier():
    for density in [0.5, 3.0, 4.5, 6.0, 9.0]:
        assert _classify_intent(160.0, density) != LightIntent.PEAK


def test_breakdown_on_sparse_density():
    assert _classify_intent(128.0, 1.5) == LightIntent.BREAKDOWN


def test_buildup_on_rising_trend():
    assert _classify_intent(120.0, GROOVE_DENSITY, density_trend=1.5) == LightIntent.BUILDUP


def test_no_buildup_without_rising_trend():
    assert _classify_intent(120.0, GROOVE_DENSITY, density_trend=1.0,
                            kick_strength=KICK_PRESENT) == LightIntent.GROOVE


def test_groove_is_default_at_moderate_conditions():
    assert _classify_intent(100.0, GROOVE_DENSITY, density_trend=1.0,
                            kick_strength=KICK_PRESENT) == LightIntent.GROOVE


def test_atmospheric_never_returned_by_classifier():
    cases = [
        (60.0, 0.0), (80.0, 1.0), (100.0, 5.0), (130.0, 3.5), (145.0, 2.0),
    ]
    for bpm, density in cases:
        assert _classify_intent(bpm, density) != LightIntent.ATMOSPHERIC


def test_buildup_trend_threshold_boundary():
    assert _classify_intent(120.0, GROOVE_DENSITY, density_trend=1.3) == LightIntent.BUILDUP
    assert _classify_intent(120.0, GROOVE_DENSITY, density_trend=1.29,
                            kick_strength=KICK_PRESENT) == LightIntent.GROOVE


def test_windowed_drop_requires_sustained_density():
    base = GROOVE_DENSITY
    densities = [base, base, base * 3, base, base]
    assert _classify_windowed(_window(densities), bpm=128.0) != LightIntent.DROP


def test_windowed_drop_on_sustained_high_density():
    densities = [9.0, 9.5, 10.0, 9.2, 8.8]
    assert _classify_windowed(_window(densities), bpm=128.0) == LightIntent.DROP


def test_windowed_buildup_detected_via_forward_context():
    densities = [2.5, 2.7, 3.8, 4.2, 4.4]
    assert _classify_windowed(_window(densities), bpm=120.0) == LightIntent.BUILDUP


def test_windowed_stable_groove_not_classified_as_buildup():
    densities = [GROOVE_DENSITY] * 5
    assert _classify_windowed(_window(densities), bpm=120.0) == LightIntent.GROOVE


def test_an_empty_window_is_not_a_classification():
    assert _classify_windowed([], bpm=128.0) is None
    assert _classify_windowed([], bpm=128.0, current_intent=LightIntent.DROP) is None


def test_windowed_breakdown_on_sustained_low_density():
    densities = [1.0, 1.2, 0.8, 1.1, 0.9]
    assert _classify_windowed(_window(densities), bpm=128.0) == LightIntent.BREAKDOWN


def test_drop_entry_threshold():
    assert _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER,
                            kick_strength=KICK_PRESENT) == LightIntent.DROP
    assert _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER - 0.1,
                            kick_strength=KICK_PRESENT) != LightIntent.DROP


def test_drop_hysteresis_stays_in_drop_above_exit_threshold():
    mid_density = (_DROP_MIN_DENSITY_EXIT + _DROP_MIN_DENSITY_ENTER) / 2
    assert _classify_intent(128.0, mid_density, current_intent=LightIntent.DROP,
                            kick_strength=KICK_PRESENT) == LightIntent.DROP


def test_drop_hysteresis_exits_below_exit_threshold():
    below_exit = _DROP_MIN_DENSITY_EXIT - 0.5
    result = _classify_intent(128.0, below_exit, current_intent=LightIntent.DROP,
                              kick_strength=KICK_PRESENT)
    assert result != LightIntent.DROP


def test_peak_inherits_drop_hysteresis():
    mid_density = (_DROP_MIN_DENSITY_EXIT + _DROP_MIN_DENSITY_ENTER) / 2
    assert _classify_intent(128.0, mid_density, current_intent=LightIntent.PEAK,
                            kick_strength=KICK_PRESENT) == LightIntent.DROP


def test_peak_hysteresis_releases_below_exit_threshold():
    below_exit = _DROP_MIN_DENSITY_EXIT - 0.5
    assert _classify_intent(128.0, below_exit, current_intent=LightIntent.PEAK,
                            kick_strength=KICK_PRESENT) != LightIntent.DROP


def test_drop_cold_entry_requires_higher_threshold():
    mid_density = (_DROP_MIN_DENSITY_EXIT + _DROP_MIN_DENSITY_ENTER) / 2
    assert _classify_intent(128.0, mid_density, kick_strength=KICK_PRESENT) != LightIntent.DROP


def test_breakdown_entry_threshold():
    assert _classify_intent(128.0, _BREAKDOWN_MAX_DENSITY_ENTER - 0.1,
                            kick_strength=KICK_PRESENT) == LightIntent.BREAKDOWN
    assert _classify_intent(128.0, _BREAKDOWN_MAX_DENSITY_ENTER,
                            kick_strength=KICK_PRESENT) != LightIntent.BREAKDOWN


def test_breakdown_hysteresis_stays_in_breakdown_below_exit_threshold():
    mid_density = (_BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_MAX_DENSITY_EXIT) / 2
    assert _classify_intent(128.0, mid_density, current_intent=LightIntent.BREAKDOWN,
                            kick_strength=KICK_PRESENT) == LightIntent.BREAKDOWN


def test_breakdown_hysteresis_exits_above_exit_threshold():
    above_exit = _BREAKDOWN_MAX_DENSITY_EXIT + 0.1
    result = _classify_intent(128.0, above_exit, current_intent=LightIntent.BREAKDOWN,
                              kick_strength=KICK_PRESENT)
    assert result != LightIntent.BREAKDOWN


def test_drop_requires_kick_presence():
    result = _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER + 1, kick_strength=KICK_ABSENT)
    assert result != LightIntent.DROP


def test_drop_with_kick_present():
    assert _classify_intent(128.0, _DROP_MIN_DENSITY_ENTER + 1,
                            kick_strength=KICK_PRESENT) == LightIntent.DROP


def test_breakdown_band_widens_by_the_margin_when_kick_is_absent():
    mid_density = _BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_NO_KICK_MARGIN / 2
    assert _classify_intent(128.0, mid_density, kick_strength=KICK_ABSENT) == LightIntent.BREAKDOWN


def test_groove_at_moderate_density_with_kick():
    mid_density = _BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_NO_KICK_MARGIN / 2
    assert _classify_intent(128.0, mid_density, kick_strength=KICK_PRESENT) == LightIntent.GROOVE


def test_high_density_no_kick_above_the_margin_stays_groove():
    above_max = _BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_NO_KICK_MARGIN + 0.5
    result = _classify_intent(128.0, above_max, kick_strength=KICK_ABSENT)
    assert result not in (LightIntent.DROP, LightIntent.BREAKDOWN)


def test_no_kick_margin_keeps_the_hysteresis_dead_zone():
    entry = _BREAKDOWN_MAX_DENSITY_ENTER + _BREAKDOWN_NO_KICK_MARGIN
    exit_ = _BREAKDOWN_MAX_DENSITY_EXIT + _BREAKDOWN_NO_KICK_MARGIN
    assert exit_ > entry
    dead_zone = (entry + exit_) / 2
    assert _classify_intent(128.0, dead_zone, kick_strength=KICK_ABSENT) == LightIntent.GROOVE
    assert _classify_intent(128.0, dead_zone, current_intent=LightIntent.BREAKDOWN,
                            kick_strength=KICK_ABSENT) == LightIntent.BREAKDOWN


def test_buildup_via_centroid_trend_without_density_trend():
    rising = _CENTROID_BUILDUP_TREND + 0.05
    result = _classify_intent(120.0, GROOVE_DENSITY, density_trend=1.0, centroid_trend=rising)
    assert result == LightIntent.BUILDUP


def test_groove_when_centroid_trend_is_neutral():
    assert _classify_intent(120.0, GROOVE_DENSITY, density_trend=1.0, centroid_trend=1.0,
                            kick_strength=KICK_PRESENT) == LightIntent.GROOVE


def test_buildup_via_either_trend_signal():
    below_density_threshold = _CENTROID_BUILDUP_TREND - 0.05
    above_centroid_threshold = _CENTROID_BUILDUP_TREND + 0.05
    assert _classify_intent(120.0, GROOVE_DENSITY, density_trend=below_density_threshold,
                            centroid_trend=above_centroid_threshold) == LightIntent.BUILDUP


def test_buildup_wins_over_no_kick_breakdown():
    rising = _CENTROID_BUILDUP_TREND + 0.05
    assert _classify_intent(128.0, GROOVE_DENSITY, kick_strength=KICK_ABSENT,
                            centroid_trend=rising) == LightIntent.BUILDUP


def test_sparse_riser_below_density_floor_stays_breakdown():
    rising = _CENTROID_BUILDUP_TREND + 0.05
    assert _classify_intent(128.0, _BUILDUP_MIN_DENSITY - 0.5,
                            centroid_trend=rising) == LightIntent.BREAKDOWN


def test_buildup_floor_meets_breakdown_ceiling():
    assert _BUILDUP_MIN_DENSITY >= _BREAKDOWN_MAX_DENSITY_ENTER


def test_windowed_drop_blocked_without_kick():
    densities = [9.0, 9.5, 10.0, 9.2, 8.8]
    assert _classify_windowed(_window(densities, kick=KICK_ABSENT), bpm=128.0) != LightIntent.DROP


def test_windowed_buildup_via_rising_centroid():
    rising = _CENTROID_BUILDUP_TREND + 0.1
    densities = [GROOVE_DENSITY] * 5
    assert _classify_windowed(_window(densities, centroid_trend=rising), bpm=120.0) == LightIntent.BUILDUP


def test_unknown_density_holds_the_current_intent():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    from lib.engine.effect_definitions import LightIntent
    from lib.engine.light_engine import _classify_intent

    for held in (LightIntent.DROP, LightIntent.GROOVE, LightIntent.BUILDUP,
                 LightIntent.PEAK, LightIntent.BREAKDOWN):
        assert _classify_intent(128.0, DENSITY_UNKNOWN, current_intent=held) is held


def test_unknown_density_without_a_current_intent_is_groove():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    from lib.engine.effect_definitions import LightIntent
    from lib.engine.light_engine import _classify_intent
    assert _classify_intent(128.0, DENSITY_UNKNOWN, current_intent=None) is LightIntent.GROOVE


def test_windowed_classification_drops_unmeasured_beats():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    from lib.engine.effect_definitions import LightIntent
    from lib.engine.light_engine import _classify_windowed

    def row(t, density):
        return BeatRecord(t, density, 128.0, 0.35, 0.2, 4.0, 1.0)

    # Outnumbered by sentinels: a median that counted them lands on one and
    # reads a drop as a groove.
    measured = [row(i, 9.0) for i in range(4)]
    shed = [row(i + 4, DENSITY_UNKNOWN) for i in range(5)]
    assert _classify_windowed(measured + shed, 128.0, LightIntent.GROOVE) is LightIntent.DROP


def test_a_window_with_nothing_measurable_holds():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    from lib.engine.effect_definitions import LightIntent
    from lib.engine.light_engine import _classify_windowed
    window = [BeatRecord(i, DENSITY_UNKNOWN, 128.0, 0.35, 0.2, 4.0, 1.0) for i in range(6)]
    assert _classify_windowed(window, 128.0, LightIntent.DROP) is LightIntent.DROP


def test_report_metrics_exclude_unmeasured_beats():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    from lib.engine.event_buffer import EventBuffer

    buffer = EventBuffer(window_sec=float('inf'))
    buffer.start()
    buffer.add_beat(bpm=128.0, onset_density=4.0, change=False)
    buffer.add_beat(bpm=128.0, onset_density=DENSITY_UNKNOWN, change=False)
    report = buffer.to_report()
    assert report['metrics']['onset_density_mean'] == 4.0
    assert report['beats'][1]['onset_density'] == DENSITY_UNKNOWN
    assert report['beats'][1]['strength'] == 0.0


def test_unknown_density_never_holds_atmospheric():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    from lib.engine.effect_definitions import LightIntent
    from lib.engine.light_engine import _classify_intent

    assert _classify_intent(128.0, DENSITY_UNKNOWN,
                            current_intent=LightIntent.ATMOSPHERIC) is LightIntent.GROOVE


def test_unknown_density_holds_every_other_intent():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    from lib.engine.effect_definitions import LightIntent
    from lib.engine.light_engine import _classify_intent

    for held in (LightIntent.DROP, LightIntent.GROOVE, LightIntent.BUILDUP,
                 LightIntent.PEAK, LightIntent.BREAKDOWN):
        assert _classify_intent(128.0, DENSITY_UNKNOWN, current_intent=held) is held


def test_a_windowed_classification_with_nothing_measurable_also_leaves_atmospheric():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    from lib.engine.effect_definitions import LightIntent
    from lib.engine.light_engine import _classify_windowed

    window = [BeatRecord(i, DENSITY_UNKNOWN, 128.0, 0.35, 0.2, 4.0, 1.0) for i in range(6)]
    assert _classify_windowed(window, 128.0,
                              LightIntent.ATMOSPHERIC) is LightIntent.GROOVE


def test_publishable_bpm_never_emits_zero_as_a_tempo():
    from lib.engine.light_engine import LightEngine

    hold = LightEngine._publishable_bpm
    state = {}
    assert hold(state, 0.0) == 0.0
    assert hold(state, 128.0) == 128.0
    assert hold(state, 0.0) == 128.0
    assert hold(state, 0.0) == 128.0
    assert hold(state, 174.0) == 174.0
    assert hold(state, 0.0) == 174.0
