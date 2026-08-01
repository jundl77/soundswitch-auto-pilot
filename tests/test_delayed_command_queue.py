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


# --------------------------------------------------------------------------- #
# B1: one queue, one delay per STREAM
# --------------------------------------------------------------------------- #


async def test_a_command_may_carry_its_own_delay():
    """The queue holds a command until the audience hears the audio that caused
    it, and the two streams reach it at different ages.

    A beat is detected as the audio arrives; a decoder decision is ~13.7 s
    behind it.  Delaying both by the playback delay would put the OS2L wire a
    whole chain-latency ahead of the room, so the delay is
    ``playback_delay - chain_latency`` PER STREAM rather than per queue.
    """
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
    """A single queue-wide target would report every intent as 13.7 s late."""
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
    """The intent delay tracks the measured chain latency, so it moves.

    If it shrinks by more than the gap between two commits, the later decision
    would otherwise overtake the earlier one and the show would run backwards
    for one change.  Order within a stream is preserved by clamping, which is
    cheaper than being sorry.
    """
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
    """A beat enqueued long ago must not drag the next intent out to its own
    fire time -- that is the whole point of the per-stream delay."""
    from lib.clock import VirtualClock

    clock = VirtualClock()
    q = DelayedCommandQueue(14.0, clock=clock)
    fired = []

    await q.enqueue('beat', _record(fired, 'beat'))
    await q.enqueue('intent', _record(fired, 'intent'), delay_sec=0.2)
    clock.advance(0.3)
    await q.drain()
    assert fired == ['intent']


def _record(sink, name):
    async def command():
        sink.append(name)
    return command
