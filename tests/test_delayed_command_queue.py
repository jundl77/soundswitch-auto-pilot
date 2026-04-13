import asyncio
import pytest
from lib.engine.delayed_command_queue import DelayedCommandQueue


async def test_zero_delay_fires_on_next_drain():
    q = DelayedCommandQueue(0.0)
    fired = []

    async def cmd():
        fired.append(1)

    await q.enqueue('a', cmd)
    await q.drain()
    assert fired == [1]


async def test_positive_delay_holds_until_due():
    q = DelayedCommandQueue(60.0)  # 1 minute — won't fire in this test
    fired = []

    async def cmd():
        fired.append(1)

    await q.enqueue('a', cmd)
    await q.drain()
    assert fired == []


async def test_fires_in_chronological_order():
    q = DelayedCommandQueue(0.0)
    order = []

    async def a(): order.append('a')
    async def b(): order.append('b')
    async def c(): order.append('c')

    await q.enqueue('a', a)
    await q.enqueue('b', b)
    await q.enqueue('c', c)
    await q.drain()
    assert order == ['a', 'b', 'c']


async def test_timing_log_records_label_and_delta():
    q = DelayedCommandQueue(0.0)

    async def noop():
        pass

    await q.enqueue('my_cmd', noop)
    await q.drain()

    log = q.get_timing_log()
    assert len(log) == 1
    assert log[0]['label'] == 'my_cmd'
    assert log[0]['target_delta_sec'] == 0.0
    assert 'actual_delta_sec' in log[0]
    assert log[0]['actual_delta_sec'] >= 0.0


async def test_undrained_commands_not_in_log():
    q = DelayedCommandQueue(60.0)

    async def noop():
        pass

    await q.enqueue('pending', noop)
    assert q.get_timing_log() == []


async def test_command_fires_only_after_delay_has_elapsed():
    """Commands must not fire early; they must fire once the delay is past.

    asyncio.sleep() precision is intentionally not tested here — it is
    hardware/scheduler-dependent and produces flaky results on macOS. The
    integration test (test_simulation.py) covers end-to-end timing with a
    real pipeline loop. This test only verifies the logical invariants:
      1. The command does NOT fire before the delay expires.
      2. The command DOES fire after the delay expires.
      3. The recorded actual_delta_sec is always >= the configured delay.
    """
    delay = 0.05
    q = DelayedCommandQueue(delay)
    fired = []

    async def cmd():
        fired.append(1)

    await q.enqueue('timed', cmd)

    # Drain immediately — command must NOT fire yet.
    await q.drain()
    assert fired == [], "command fired before delay elapsed"

    # Wait well past the delay (generous factor to survive slow CI), then drain.
    await asyncio.sleep(delay * 5)
    await q.drain()
    assert fired == [1], "command did not fire after delay elapsed"

    log = q.get_timing_log()
    assert len(log) == 1
    assert log[0]['actual_delta_sec'] >= delay, "command fired before configured delay"
