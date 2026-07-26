"""
Unit tests for intent-stability mechanisms in LightEngine._commit_intent:
  - Vote buffer: requires _VOTE_BUFFER_SIZE consecutive identical votes
  - Minimum dwell: requires _MIN_DWELL_BEATS beats in current intent before switching
  - Invalid-transition guard: blocks musically impossible jumps
  - PEAK promotion: a DROP held for _PEAK_PROMOTION_BEATS commit-beats becomes PEAK,
    and PEAK absorbs further DROP votes

These tests drive _commit_intent directly with a synthetic _beat_history,
bypassing on_beat() and the audio pipeline entirely.
"""

import pytest
from collections import deque
from unittest.mock import AsyncMock, MagicMock

from lib.engine.light_engine import (
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(look_ahead_sec: float = 1.0,
                 event_buffer: EventBuffer | None = None) -> LightEngine:
    """Build a minimal LightEngine backed by mock clients for unit testing.

    Pass a real EventBuffer when the test asserts on what the intent timeline
    (report / visualizer) records, rather than only on engine state.
    """
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


# Feature levels derived from the constants, so retuning a threshold cannot
# silently change what these tests assert.
KICK_PRESENT = _KICK_PRESENCE_THRESHOLD + 0.5
# Below DROP's exit threshold but above BREAKDOWN's entry: classifies as GROOVE
# even while DROP or PEAK is the committed intent.
GROOVE_DENSITY = (_BREAKDOWN_MAX_DENSITY_ENTER + _DROP_MIN_DENSITY_EXIT) / 2


def _seed_beat_history(engine: LightEngine, density: float, bpm: float = 128.0,
                       kick: float = KICK_PRESENT, centroid_trend: float = 1.0, n: int = 7):
    """Fill _beat_history with beats spread symmetrically around the engine clock's now.

    All beats land within look_ahead_sec of now so they are included in the
    window when _commit_intent(enqueue_time=now, ...) is called immediately after.
    """
    now = engine._clock.monotonic()  # the engine's own clock, whatever it is
    half = engine._look_ahead_sec * 0.9
    for i in range(n):
        t = now - half + i * (2 * half / max(n - 1, 1))
        engine._beat_history.append((t, density, bpm, 0.0, 0.5, kick, centroid_trend))
    return now  # use as enqueue_time


# ---------------------------------------------------------------------------
# Vote-buffer tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_vote_does_not_switch():
    """One unanimous vote is not enough — buffer must be full."""
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10  # bypass dwell

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.GROOVE
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_unanimous_votes_triggers_switch():
    """_VOTE_BUFFER_SIZE identical votes with sufficient dwell → intent switch."""
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
    """A mix of DROP and GROOVE votes prevents a switch even after the buffer is full.

    We inject votes directly into the buffer to isolate the voting logic from
    the windowed classifier (which depends on a correctly seeded beat history).
    """
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    # Pre-load a mixed vote buffer: [GROOVE, DROP, GROOVE]
    from collections import deque
    engine._intent_vote_buffer = deque(
        [LightIntent.GROOVE, LightIntent.DROP, LightIntent.GROOVE],
        maxlen=_VOTE_BUFFER_SIZE,
    )

    # Call _commit_intent with a window that classifies as GROOVE.
    # The vote buffer is full but not unanimous → no switch.
    enqueue_time = _seed_beat_history(engine, density=GROOVE_DENSITY)
    await engine._commit_intent(enqueue_time, 128.0)

    # The new vote (GROOVE) overwrites oldest (GROOVE): [DROP, GROOVE, GROOVE] — still mixed.
    assert engine._current_intent == LightIntent.GROOVE
    engine.effect_controller.change_effect.assert_not_awaited()


# ---------------------------------------------------------------------------
# Minimum-dwell tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dwell_prevents_early_switch():
    """Unanimous votes cannot switch until _MIN_DWELL_BEATS beats have elapsed."""
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = 0  # just entered GROOVE

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)

    # Fill vote buffer with unanimous DROP votes — but dwell counter is too low
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    # After _VOTE_BUFFER_SIZE calls, dwell = _VOTE_BUFFER_SIZE which is < _MIN_DWELL_BEATS
    assert engine._current_intent == LightIntent.GROOVE
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_switch_allowed_after_dwell_met():
    """Once _MIN_DWELL_BEATS beats have elapsed, a unanimous vote switches intent."""
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS - 1  # one beat short

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)

    # First call brings dwell to _MIN_DWELL_BEATS — still can't switch (buffer not full).
    await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.GROOVE

    # Remaining calls fill the vote buffer; dwell is now satisfied.
    for _ in range(_VOTE_BUFFER_SIZE - 1):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.DROP
    engine.effect_controller.change_effect.assert_awaited_once_with(LightIntent.DROP)


# ---------------------------------------------------------------------------
# Invalid-transition tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_transition_atmospheric_to_drop_blocked():
    """ATMOSPHERIC → DROP is an invalid transition and must be blocked."""
    assert (LightIntent.ATMOSPHERIC, LightIntent.DROP) in _INVALID_TRANSITIONS

    engine = _make_engine()
    engine._current_intent = LightIntent.ATMOSPHERIC
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    # Should remain ATMOSPHERIC despite DROP votes
    assert engine._current_intent == LightIntent.ATMOSPHERIC
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_transition_groove_to_drop_allowed():
    """GROOVE → DROP is a valid transition and should proceed when all checks pass."""
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
    """ATMOSPHERIC → BUILDUP is an invalid transition.

    (ATMOSPHERIC → PEAK is equally invalid but no longer drivable through the
    classifier: PEAK is an engine-level promotion, never a pure classification.
    The membership assertion below still pins the guard's contents.)
    """
    assert (LightIntent.ATMOSPHERIC, LightIntent.BUILDUP) in _INVALID_TRANSITIONS
    assert (LightIntent.ATMOSPHERIC, LightIntent.PEAK) in _INVALID_TRANSITIONS

    engine = _make_engine()
    engine._current_intent = LightIntent.ATMOSPHERIC
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    # Moderate density (below DROP entry) with a rising spectral centroid → BUILDUP
    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER - 0.5,
                                      centroid_trend=_CENTROID_BUILDUP_TREND + 0.1)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.ATMOSPHERIC
    engine.effect_controller.change_effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_vote_buffer_cleared_after_switch():
    """After a successful intent switch the vote buffer is cleared (fresh start)."""
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.DROP
    # Vote buffer should be empty after the switch
    assert len(engine._intent_vote_buffer) == 0
    # Dwell counter should be reset
    assert engine._beats_in_current_intent == 0


# ---------------------------------------------------------------------------
# PEAK promotion tests — PEAK is an engine-level promotion, never a classification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sustained_drop_promotes_to_peak():
    """A committed DROP that survives _PEAK_PROMOTION_BEATS commit-beats becomes PEAK.

    The classifier can never return PEAK (see _classify_intent) — "sustained
    maximum energy after the drop" is temporal, so the engine's dwell counter
    is what promotes it.  The promotion deliberately bypasses the
    invalid-transition guard; this test pins that it actually fires.
    """
    engine = _make_engine()
    engine._current_intent = LightIntent.GROOVE
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)

    # Phase 1: unanimous DROP votes commit DROP and reset the dwell counter.
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.DROP
    assert engine._beats_in_current_intent == 0

    # Phase 2: one commit-beat short of the promotion threshold — still DROP.
    for _ in range(_PEAK_PROMOTION_BEATS - 1):
        await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.DROP

    # Phase 3: the _PEAK_PROMOTION_BEATS-th commit-beat promotes DROP → PEAK.
    await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.PEAK
    engine.effect_controller.change_effect.assert_awaited_with(LightIntent.PEAK)
    # Promotion resets stability state so PEAK starts with a fresh dwell window.
    assert engine._beats_in_current_intent == 0
    assert len(engine._intent_vote_buffer) == 0

    # Phase 4: promotion fires exactly once — continued DROP votes do not re-fire it.
    awaits_after_promotion = engine.effect_controller.change_effect.await_count
    for _ in range(_PEAK_PROMOTION_BEATS + 1):
        await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.PEAK
    assert engine.effect_controller.change_effect.await_count == awaits_after_promotion


@pytest.mark.asyncio
async def test_peak_absorbs_drop_votes():
    """While in PEAK, unanimous DROP consensus must not switch back to DROP.

    Easing from PEAK to plain DROP is not a show change — allowing it would let
    the pair oscillate (DROP → promote → PEAK → DROP → promote → ...).
    """
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
    """PEAK inherits DROP's exit threshold — a dip below entry does not eject it.

    A windowed density between _DROP_MIN_DENSITY_EXIT and _DROP_MIN_DENSITY_ENTER
    is exactly the dip DROP's hysteresis exists to ride out.  PEAK is sustained
    DROP, so it must be at least as sticky: the window still votes DROP, and
    absorption keeps the show in PEAK.
    """
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
    """The reported intent timeline must agree with the lights during a PEAK hold.

    Absorbed DROP votes must not be surfaced to the EventBuffer: the report,
    visualizer, and training table all read that timeline, and while PEAK is
    committed the show is in PEAK, not DROP.
    """
    buffer = EventBuffer()
    buffer.start()
    engine = _make_engine(event_buffer=buffer)
    engine._current_intent = LightIntent.DROP
    engine._beats_in_current_intent = _PEAK_PROMOTION_BEATS - 1  # one beat from promotion

    enqueue_time = _seed_beat_history(engine, density=_DROP_MIN_DENSITY_ENTER + 1)

    # Promotion surfaces PEAK to the timeline.
    await engine._commit_intent(enqueue_time, 128.0)
    assert engine._current_intent == LightIntent.PEAK
    assert buffer.snapshot()['intent'] == LightIntent.PEAK.value

    # Every subsequent beat reaches DROP consensus and is absorbed — the timeline
    # must keep reading 'peak' and must not gain a 'drop' block.
    for _ in range(_VOTE_BUFFER_SIZE * 2):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.PEAK
    assert buffer.snapshot()['intent'] == LightIntent.PEAK.value
    assert [e['intent'] for e in buffer.snapshot()['intents']] == [LightIntent.PEAK.value]


@pytest.mark.asyncio
async def test_peak_exits_to_groove_on_consensus():
    """PEAK is not a trap: any non-DROP consensus exits through the normal pipeline."""
    assert (LightIntent.PEAK, LightIntent.GROOVE) not in _INVALID_TRANSITIONS

    engine = _make_engine()
    engine._current_intent = LightIntent.PEAK
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    # Density eased below DROP's exit threshold, no riser → classifies as GROOVE
    enqueue_time = _seed_beat_history(engine, density=GROOVE_DENSITY)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 128.0)

    assert engine._current_intent == LightIntent.GROOVE
    engine.effect_controller.change_effect.assert_awaited_once_with(LightIntent.GROOVE)


# ---------------------------------------------------------------------------
# Virtual clock — beat history timestamps come from the injected clock
# ---------------------------------------------------------------------------

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
    assert engine._beat_history[-1][0] == 10.0  # virtual monotonic, not wall time
