"""
Unit tests for intent-stability mechanisms in LightEngine._commit_intent:
  - Vote buffer: requires _VOTE_BUFFER_SIZE consecutive identical votes
  - Minimum dwell: requires _MIN_DWELL_BEATS beats in current intent before switching
  - Invalid-transition guard: blocks musically impossible jumps

These tests drive _commit_intent directly with a synthetic _beat_history,
bypassing on_beat() and the audio pipeline entirely.
"""

import time
import pytest
from collections import deque
from unittest.mock import AsyncMock, MagicMock

from lib.engine.light_engine import (
    LightEngine,
    _VOTE_BUFFER_SIZE,
    _MIN_DWELL_BEATS,
    _INVALID_TRANSITIONS,
    _DROP_MIN_DENSITY_ENTER,
)
from lib.engine.effect_definitions import LightIntent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(look_ahead_sec: float = 1.0) -> LightEngine:
    """Build a minimal LightEngine backed by mock clients for unit testing."""
    effect_controller = MagicMock()
    effect_controller.change_effect = AsyncMock()
    engine = LightEngine(
        midi_client=MagicMock(),
        os2l_client=MagicMock(),
        overlay_client=MagicMock(),
        effect_controller=effect_controller,
        command_queue=None,
        event_buffer=None,
        look_ahead_sec=look_ahead_sec,
    )
    analyser = MagicMock()
    analyser.is_song_playing.return_value = True
    analyser.get_seconds_since_last_beat.return_value = 0.0
    engine.set_analyser(analyser)
    return engine


def _seed_beat_history(engine: LightEngine, density: float, bpm: float = 128.0, n: int = 7):
    """Fill _beat_history with beats spread symmetrically around time.monotonic().

    All beats land within look_ahead_sec of now so they are included in the
    window when _commit_intent(enqueue_time=now, ...) is called immediately after.
    """
    now = time.monotonic()
    half = engine._look_ahead_sec * 0.9
    for i in range(n):
        t = now - half + i * (2 * half / max(n - 1, 1))
        engine._beat_history.append((t, density, bpm, 0.0, 0.5))
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
    enqueue_time = _seed_beat_history(engine, density=4.0)
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
async def test_invalid_transition_atmospheric_to_peak_blocked():
    """ATMOSPHERIC → PEAK is an invalid transition."""
    assert (LightIntent.ATMOSPHERIC, LightIntent.PEAK) in _INVALID_TRANSITIONS

    engine = _make_engine()
    engine._current_intent = LightIntent.ATMOSPHERIC
    engine._beats_in_current_intent = _MIN_DWELL_BEATS + 10

    # High BPM, low density → classifies as PEAK
    enqueue_time = _seed_beat_history(engine, density=4.0, bpm=145.0)
    for _ in range(_VOTE_BUFFER_SIZE):
        await engine._commit_intent(enqueue_time, 145.0)

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
