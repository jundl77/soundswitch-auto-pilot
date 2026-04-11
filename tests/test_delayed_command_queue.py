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


async def test_timing_error_is_bounded_by_drain_cadence():
    """
    The queue fires commands in drain(), which is called every ~5.8ms in production.
    So the worst-case overshoot in a tight drain loop is one drain cadence (~6ms).
    This test uses a 10ms sleep buffer and asserts error < 20ms, giving comfortable
    headroom for CI without being meaningless. Real precision is validated by the
    integration test (test_simulation.py) which runs the full pipeline for seconds.
    """
    delay = 0.1
    q = DelayedCommandQueue(delay)

    async def noop():
        pass

    await q.enqueue('timed', noop)
    await asyncio.sleep(delay + 0.010)  # 10ms past due — comfortably in range
    await q.drain()

    log = q.get_timing_log()
    assert len(log) == 1
    error_ms = abs(log[0]['actual_delta_sec'] - delay) * 1000
    assert error_ms < 20, f"timing error {error_ms:.1f}ms exceeds one-drain-cadence bound"
