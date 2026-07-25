"""Fast-runner behavior: virtual pacing, exhaustion stop, look-ahead flush."""
import time

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE
from lib.clock import VirtualClock
from simulate.fake_audio_client import BeepAudioClient
from simulate.runner import build_simulation, run_simulation, LOOK_AHEAD_SEC


async def test_fast_run_is_much_faster_than_real_time():
    clock = VirtualClock()
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    components, command_queue = build_simulation(audio_client, clock=clock)

    wall_start = time.monotonic()
    await run_simulation(components, duration_sec=10.0, clock=clock)
    wall_elapsed = time.monotonic() - wall_start

    assert wall_elapsed < 5.0, f'10 virtual seconds took {wall_elapsed:.1f}s wall time'
    # Clock ends past duration + flush tail
    assert clock.monotonic() >= 10.0 + LOOK_AHEAD_SEC - 0.1


async def test_fast_run_flushes_lookahead_commands():
    """Beat commands enqueued near the end must still fire (flush tail)."""
    clock = VirtualClock()
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    components, command_queue = build_simulation(audio_client, clock=clock)
    await run_simulation(components, duration_sec=10.0, clock=clock)

    log = command_queue.get_timing_log()
    assert log, 'expected commands to fire'
    # Commands enqueued near the end can only fire during the flush tail
    # (fire time past the 10.0s loop end) — prove the tail actually drains.
    assert any(e['actual_fire_time'] > 10.0 for e in log), \
        'no command fired during the flush tail'
    for entry in log:
        # In virtual time the delay is exact to one buffer quantum (~5.8 ms)
        assert abs(entry['actual_delta_sec'] - entry['target_delta_sec']) <= 0.010
