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


async def test_a_command_may_carry_its_own_delay():
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)
    fired = []

    async def beat():
        fired.append('beat')

    async def intent():
        fired.append('intent')

    await q.enqueue('beat', beat)
    await q.enqueue('intent', intent, delay_sec=0.34)

    clock.advance(0.4)
    await q.drain()
    assert fired == ['intent']

    clock.advance(14.0)
    await q.drain()
    assert fired == ['intent', 'beat']


async def test_the_timing_log_records_the_delay_each_command_asked_for():
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)

    async def noop():
        pass

    await q.enqueue('beat', noop)
    await q.enqueue('intent', noop, delay_sec=0.34)
    clock.advance(0.34)
    await q.drain()
    clock.advance(14.0 - 0.34)
    await q.drain()

    log = q.get_timing_log()
    assert {e['label']: e['target_delta_sec'] for e in log} == {'beat': 14.0,
                                                                'intent': 0.34}
    for entry in log:
        assert entry['actual_delta_sec'] == pytest.approx(
            entry['target_delta_sec'], abs=1e-9)


async def test_a_shrinking_delay_cannot_reorder_a_stream():
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)
    fired = []

    await q.enqueue('intent', _record(fired, 'first'), delay_sec=5.0)
    clock.advance(0.5)
    await q.enqueue('intent', _record(fired, 'second'), delay_sec=0.1)

    clock.advance(20.0)
    await q.drain()
    assert fired == ['first', 'second']


async def test_streams_are_clamped_independently():
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)
    fired = []

    await q.enqueue('beat', _record(fired, 'beat'))
    await q.enqueue('intent', _record(fired, 'intent'), delay_sec=0.2)
    clock.advance(0.3)
    await q.drain()
    assert fired == ['intent']


async def test_dropping_pending_commands_takes_only_the_ones_after():
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)
    fired = []

    await q.enqueue('intent', _record(fired, 'early'), delay_sec=1.0)
    await q.enqueue('intent', _record(fired, 'late'), delay_sec=5.0)
    await q.enqueue('beat', _record(fired, 'beat'), delay_sec=5.0)

    assert q.drop_pending('intent', clock.monotonic() + 4.9) == 1
    clock.advance(20.0)
    await q.drain()
    assert fired == ['early', 'beat']


async def test_dropping_pending_commands_releases_the_clamp_they_were_holding():
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)
    fired = []

    await q.enqueue('intent', _record(fired, 'stale'), delay_sec=9.0)
    q.drop_pending('intent', clock.monotonic())
    await q.enqueue('intent', _record(fired, 'fresh'), delay_sec=0.5)

    clock.advance(1.0)
    await q.drain()
    assert fired == ['fresh']


def _record(sink, name):
    async def command():
        sink.append(name)
    return command


async def test_a_command_at_exactly_the_same_fire_time_is_a_predecessor_not_a_restatement():
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)
    fired = []

    await q.enqueue('intent', _record(fired, 'bar10'), delay_sec=0.0)
    assert q.drop_pending('intent', clock.monotonic()) == 0
    await q.enqueue('intent', _record(fired, 'bar11'), delay_sec=0.0)

    clock.advance(1.0)
    await q.drain()
    assert fired == ['bar10', 'bar11']


async def test_an_inclusive_drop_takes_the_equal_fire_time_the_default_keeps():
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)
    fired = []

    await q.enqueue('refresh', _record(fired, 'refresh'), delay_sec=0.0)
    assert q.drop_pending('refresh', clock.monotonic()) == 0
    assert q.drop_pending('refresh', clock.monotonic(), inclusive=True) == 1

    clock.advance(1.0)
    await q.drain()
    assert fired == []


async def test_one_failing_command_does_not_take_the_batch_or_the_show_with_it():
    clock = VirtualClock()
    queue = DelayedCommandQueue(0.0, clock=clock)
    fired: list = []

    async def boom():
        raise RuntimeError('the midi port went away')

    async def note(which):
        fired.append(which)

    queue.schedule('intent', lambda: note('before'))
    queue.schedule('intent', boom)
    queue.schedule('overlay', lambda: note('after'))
    clock.advance(1.0)

    await queue.drain()

    assert fired == ['before', 'after']
    assert queue.pending == 0
