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
    q = DelayedCommandQueue(60.0)
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
    # No sleep-precision assertions here — scheduler jitter makes them flaky.
    delay = 0.05
    q = DelayedCommandQueue(delay)
    fired = []

    async def cmd():
        fired.append(1)

    await q.enqueue('timed', cmd)

    await q.drain()
    assert fired == [], "command fired before delay elapsed"

    await asyncio.sleep(delay * 5)
    await q.drain()
    assert fired == [1], "command did not fire after delay elapsed"

    log = q.get_timing_log()
    assert len(log) == 1
    assert log[0]['actual_delta_sec'] >= delay, "command fired before configured delay"


from lib.clock import VirtualClock


async def test_virtual_clock_command_fires_only_after_virtual_delay():
    clock = VirtualClock()
    q = DelayedCommandQueue(2.5, clock=clock)
    fired = []

    async def cmd():
        fired.append(True)

    await q.enqueue('x', cmd)
    await q.drain()
    assert fired == []

    clock.advance(2.4)
    await q.drain()
    assert fired == []

    clock.advance(0.2)
    await q.drain()
    assert fired == [True]


async def test_virtual_clock_timing_log_is_exact():
    clock = VirtualClock()
    q = DelayedCommandQueue(2.5, clock=clock)

    async def cmd():
        pass

    await q.enqueue('x', cmd)
    clock.advance(2.5)
    await q.drain()
    log = q.get_timing_log()
    assert len(log) == 1
    assert log[0]['actual_delta_sec'] == 2.5


async def test_pending_counts_unfired_commands():
    clock = VirtualClock()
    q = DelayedCommandQueue(2.5, clock=clock)

    async def cmd():
        pass

    assert q.pending == 0
    await q.enqueue('a', cmd)
    await q.enqueue('b', cmd)
    assert q.pending == 2
    clock.advance(2.5)
    await q.drain()
    assert q.pending == 0
