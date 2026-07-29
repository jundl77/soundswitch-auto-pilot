import pytest
from collections import deque
from unittest.mock import AsyncMock, MagicMock

from lib.engine.light_engine import (
    BeatRecord,
    LightEngine,
    _VOTE_BUFFER_SIZE,
    _MIN_DWELL_BEATS,
    _PEAK_PROMOTION_BEATS,
    _INVALID_TRANSITIONS,
    _DROP_MIN_DENSITY_ENTER,
    _DROP_MIN_DENSITY_EXIT,
    _CENTROID_BUILDUP_TREND,
    _BREAKDOWN_MAX_DENSITY_ENTER,
    _BREAKDOWN_MAX_DENSITY_EXIT,
    _KICK_PRESENCE_THRESHOLD,
)
from lib.engine.effect_definitions import LightIntent
from lib.engine.event_buffer import EventBuffer


def _make_engine(look_ahead_sec: float = 1.0,
                 event_buffer: EventBuffer | None = None) -> LightEngine:
    effect_controller = MagicMock()
    effect_controller.change_effect = AsyncMock()
    engine = LightEngine(
        midi_client=MagicMock(),
        os2l_client=MagicMock(),
        overlay_client=MagicMock(),
        effect_controller=effect_controller,
        command_queue=None,
        event_buffer=event_buffer,
        look_ahead_sec=look_ahead_sec,
    )
    analyser = MagicMock()
    analyser.is_song_playing.return_value = True
    analyser.get_seconds_since_last_beat.return_value = 0.0
    engine.set_analyser(analyser)
    return engine


KICK_PRESENT = _KICK_PRESENCE_THRESHOLD + 0.5
GROOVE_DENSITY = (_BREAKDOWN_MAX_DENSITY_ENTER + _DROP_MIN_DENSITY_EXIT) / 2


def _seed_beat_history(engine: LightEngine, density: float, bpm: float = 128.0,
                       kick: float = KICK_PRESENT, centroid_trend: float = 1.0, n: int = 7):
    now = engine._clock.monotonic()
    half = engine._look_ahead_sec * 0.9
    for i in range(n):
        t = now - half + i * (2 * half / max(n - 1, 1))
        engine._beat_history.append(BeatRecord(t, density, bpm, 0.0, 0.5, kick, centroid_trend))
    return now


@pytest.mark.asyncio
async def test_single_vote_does_not_switch():
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.GROOVE
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_unanimous_votes_triggers_switch():
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.DROP
    engine.effect_controller.change_effect.assert_awaited_once_with(LightIntent.DROP)


@pytest.mark.asyncio
async def test_mixed_votes_do_not_switch():
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    from collections import deque
    engine._intent_vote_buffer = deque(
        [LightIntent.GROOVE, LightIntent.DROP, LightIntent.GROOVE],
        maxlen=_VOTE_BUFFER_SIZE,
    )

    enqueue_time = _seed_beat_history(engine, density=GROOVE_DENSITY)
    await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.GROOVE
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_dwell_prevents_early_switch():
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = 0

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)

    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.GROOVE
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_switch_allowed_after_dwell_met():
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS - 1

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)

    await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.GROOVE

    for _ in range(_VOTE_BUFFER_SIZE - 1):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.DROP
    engine.effect_controller.change_effect.assert_awaited_once_with(LightIntent.DROP)


@pytest.mark.asyncio
async def test_invalid_transition_atmospheric_to_drop_blocked():
    assert (LightIntent.ATMOSPHERIC, LightIntent.DROP) in _INVALID_TRANSITIONS

    engine = _make_engine()
    engine._current_intent = LightIntent.ATMOSPHERIC
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.ATMOSPHERIC
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_transition_groove_to_drop_allowed():
    assert (LightIntent.GROOVE, LightIntent.DROP) not in _INVALID_TRANSITIONS

    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.DROP
    engine.effect_controller.change_effect.assert_awaited_once_with(LightIntent.DROP)


@pytest.mark.asyncio
async def test_invalid_transition_atmospheric_to_buildup_blocked():
    assert (LightIntent.ATMOSPHERIC, LightIntent.BUILDUP) in _INVALID_TRANSITIONS
    assert (LightIntent.ATMOSPHERIC, LightIntent.PEAK) in _INVALID_TRANSITIONS

    engine = _make_engine()
    engine._current_intent = LightIntent.ATMOSPHERIC
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER - 0.5,
                                      centroid_trend=_CENTROID_BUILDUP_TREND + 0.1)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.ATMOSPHERIC
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_vote_buffer_cleared_after_switch():
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.DROP
    assert len(engine._intent_vote_buffer) == 0
    assert engine._beats_in_current_intent == 0


@pytest.mark.asyncio
async def test_sustained_drop_promotes_to_peak():
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)

    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.DROP
    assert engine._beats_in_current_intent == 0

    for _ in range(_PEAK_PROMOTION_BEATS - 1):
        await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.DROP

    await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.PEAK
    engine.effect_controller.change_effect.assert_awaited_with(LightIntent.PEAK)
    assert engine._beats_in_current_intent == 0
    assert len(engine._intent_vote_buffer) == 0

    awaits_after_promotion = engine.effect_controller.change_effect.await_count
    for _ in range(_PEAK_PROMOTION_BEATS + 1):
        await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.PEAK
    assert engine.effect_controller.change_effect.await_count == awaits_after_promotion


@pytest.mark.asyncio
async def test_peak_absorbs_drop_votes():
    engine = _make_engine()
    engine._current_intent = LightIntent.PEAK
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    for _ in range(_VOTE_BUFFER_SIZE * 2):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.PEAK
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_peak_holds_through_mid_hysteresis_density_dip():
    engine = _make_engine()
    engine._current_intent = LightIntent.PEAK
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    mid_density = (_DROP_MIN_DENSITY_EXIT + _DROP_MIN_DENSITY_ENTER) / 2
    enqueue_time = _seed_beat_history(engine, density=mid_density)
    for _ in range(_VOTE_BUFFER_SIZE * 2):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.PEAK
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_peak_timeline_stays_peak_through_absorbed_drop_votes():
    buffer = EventBuffer()
    buffer.start()
    engine = _make_engine(event_buffer=buffer)
    engine._current_intent = LightIntent.DROP
    engine._beats_in_current_intent = _PEAK_PROMOTION_BEATS - 1

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)

    await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.PEAK
    assert buffer.snapshot()['intent'] == LightIntent.PEAK.value

    for _ in range(_VOTE_BUFFER_SIZE * 2):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.PEAK
    assert buffer.snapshot()['intent'] == LightIntent.PEAK.value
    assert [e['intent'] for e in buffer.snapshot()['intents']] == [LightIntent.PEAK.value]


@pytest.mark.asyncio
async def test_peak_exits_to_groove_on_consensus():
    assert (LightIntent.PEAK, LightIntent.GROOVE) not in _INVALID_TRANSITIONS

    engine = _make_engine()
    engine._current_intent = LightIntent.PEAK
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=GROOVE_DENSITY)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.GROOVE
    engine.effect_controller.change_effect.assert_awaited_once_with(LightIntent.GROOVE)


from lib.clock import VirtualClock


async def test_beat_history_uses_injected_clock():
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient
    from lib.engine.effect_controller import EffectController
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.light_engine import LightEngine

    clock = VirtualClock()
    midi = StubMidiClient(clock=clock)
    engine = LightEngine(
        midi, StubOs2lClient(clock=clock), StubOverlayClient(clock=clock),
        EffectController(midi, clock=clock),
        DelayedCommandQueue(2.5, clock=clock),
        look_ahead_sec=2.5,
        clock=clock,
    )

    class _FakeAnalyser:
        def get_song_current_duration(self):
            import datetime
            return datetime.timedelta(seconds=clock.monotonic())
        def get_onset_density(self): return 5.0
        def get_onset_density_trend(self): return 1.0
        def get_sub_bass_ratio(self): return 0.3
        def get_rms_energy(self): return 0.1
        def get_kick_strength(self): return KICK_PRESENT
        def get_spectral_centroid_trend(self): return 1.0
        def is_song_playing(self): return True

    engine.set_analyser(_FakeAnalyser())

    clock.advance(10.0)
    await engine.on_beat(beat_number=1, bpm=128.0, bpm_changed=False)
    assert engine._beat_history[-1].at == 10.0
