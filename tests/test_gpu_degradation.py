"""D11/#144: what the show does when the GPU stops, and how loudly.

There is no second classifier and none is wanted.  `NN_SHED` means: stop
consuming posteriors, hold the intent the room is already looking at, keep
beats and the silence timer running, say so at WARNING with a rate limit,
reinitialise the extractor on a capped backoff, and resume by clearing every
piece of stale state.  If the GPU never comes back, that state is the show for
the rest of the night -- and it must be a show, not an outage.

One drill per failure mode #143 names.  Three of the four are the same
mechanism reached by different exceptions, and saying so is the point: the
handling is uniform BECAUSE a raised CUDA fault cannot be told apart from an
out-of-memory or a dead context without reading its message, and a policy that
branched on message text would be a policy about strings.  The fourth -- a pass
that never returns -- is genuinely different, because there is no exception to
catch and nothing on the GPU thread left to notice it.

Injection goes through seams that already exist: the encoder interface, the
injectable session, the typed ring overrun, and the `reinit` callable the
shipped chain passes `load_encoder` into.  No production path exists here only
to be tested.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest

from lib.analyser import mert_stream as M
from lib.analyser.drift_watchdog import DriftWatchdog, ShedLevel
from lib.analyser.gpu_stage import GpuStage
from tests.test_drift_watchdog import FakeClock
from tests.test_gpu_stage import (BUFFER_SEC, EMPTY, SlowEncoder, buffers,
                                  chain, noise, settle)
from tests.test_section_model import mean, tiny  # noqa: F401


class FaultyEncoder(SlowEncoder):
    """An encoder that fails the way a GPU does, and can be replaced.

    ``fail_until`` is a generation number: a fresh encoder (what `reinit`
    returns) carries a higher one, which is exactly how a dead CUDA context
    behaves -- the objects are invalid and only rebuilding them helps.
    """

    def __init__(self, error, *, generation=0, fail_until=10 ** 9, **kwargs):
        super().__init__(**kwargs)
        self.error = error
        self.generation = generation
        self.fail_until = fail_until
        self.attempts = 0

    def encode(self, segment, **kwargs):
        self.attempts += 1
        if self.generation < self.fail_until:
            raise self.error
        return super().encode(segment, **kwargs)


def faulty(tiny, mean, error, *, clock=None, repairable=True, **kwargs):  # noqa: F811
    """A stage whose encoder fails.

    ``repairable`` decides what `reinit` hands back: a working encoder (a dead
    context, which a rebuild fixes) or another broken one (a GPU that is simply
    gone, which is the case the backoff and the rate limit exist for).
    """
    built = []

    def reinit():
        built.append(FaultyEncoder(
            error, generation=len(built) + 1,
            fail_until=1 if repairable else 10 ** 9))
        return built[-1]

    encoder = FaultyEncoder(error, **kwargs)
    watchdog = DriftWatchdog(BUFFER_SEC)
    worker = GpuStage(chain(encoder, tiny, mean, margin=1.0, hop=0.5,
                            buffer=20.0),
                      watchdog, clock=clock or FakeClock(), reinit=reinit,
                      pass_timeout_sec=1.0)
    worker.start()
    return worker, watchdog, encoder, built


def feed(worker, seconds=3.0, seed=0):
    for block in buffers(noise(seconds, seed)):
        worker.push_audio(block)


def drain(worker, ticks=200, clock=None, step=0.0):
    """Spin the consumer the way the audio loop does, optionally moving time."""
    out = []
    for _ in range(ticks):
        if clock is not None and step:
            clock.advance(step)
        out.extend(worker.push_audio(EMPTY).posteriors)
        time.sleep(0.002)
    return out


def wait_for_one_more_fault(worker, clock, sec, patience=1.0):
    """Move the backoff clock by ``sec`` and give the worker time to act on it.

    Fake time is the point -- a capped backoff at half a minute a try cannot be
    waited out in a test -- but the worker runs on real time, so each step has
    to be handed over rather than assumed.
    """
    before = worker.faults
    clock.advance(sec)
    deadline = time.monotonic() + patience
    while worker.faults == before and time.monotonic() < deadline:
        worker.push_audio(EMPTY)
        time.sleep(0.002)
    return worker.faults > before


CUDA_ERROR = RuntimeError('CUDA error: an illegal memory access was encountered')
CONTEXT_LOST = RuntimeError('CUDA error: invalid device context '
                            '(the device was reset while suspended)')
OUT_OF_MEMORY = RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB '
                             '(GPU 0; 8.00 GiB total capacity)')


# --------------------------------------------------------------------------- #
# (a) a raised CUDA error mid-pass
# --------------------------------------------------------------------------- #


def test_a_cuda_error_mid_pass_holds_logs_and_comes_back(tiny, mean, caplog):  # noqa: F811
    clock = FakeClock()
    worker, watchdog, encoder, _built = faulty(tiny, mean, CUDA_ERROR,
                                               clock=clock, fail_until=1)
    try:
        with caplog.at_level(logging.WARNING):
            feed(worker, 4.0)
            settle(lambda: worker.faults > 0, why='the fault never surfaced')
            assert watchdog.level is ShedLevel.NN_SHED or worker.faults > 0
            drain(worker, 50, clock=clock, step=0.05)
            feed(worker, 4.0, seed=1)
            settle(lambda: worker.passes > 0, why='the stage never came back')
    finally:
        worker.stop()

    text = caplog.text
    assert 'illegal memory access' in text, 'the fault was not named'
    assert any(record.levelno >= logging.WARNING for record in caplog.records)
    assert worker.resyncs >= 1, 'it resumed without clearing the gap'


# --------------------------------------------------------------------------- #
# (b) a pass that never returns
# --------------------------------------------------------------------------- #


def test_a_hung_pass_is_noticed_from_the_consumer_side(tiny, mean, caplog):  # noqa: F811
    """The one mode with no exception to catch.

    The GPU thread is inside the call, so nothing there can notice; the audio
    thread is the only part of the show still running, and it is the part that
    knows how long the pass has been in flight.
    """
    clock = FakeClock()
    encoder = SlowEncoder()
    encoder.gate.clear()
    watchdog = DriftWatchdog(BUFFER_SEC)
    worker = GpuStage(chain(encoder, tiny, mean, margin=1.0, hop=0.5,
                            buffer=20.0),
                      watchdog, clock=clock, pass_timeout_sec=1.0)
    worker.start()
    try:
        with caplog.at_level(logging.WARNING):
            feed(worker, 3.0)
            settle(lambda: encoder.threads, why='no pass started')
            assert watchdog.level is ShedLevel.NONE, 'shed before the timeout'

            clock.advance(2.0)
            worker.push_audio(EMPTY)
            assert watchdog.fault == 'hung_pass'
            assert worker.push_audio(EMPTY).posteriors == [], 'not holding'

            encoder.gate.set()
            settle(lambda: watchdog.fault is None,
                   why='the returning pass did not clear the fault')
            feed(worker, 4.0, seed=2)
            settle(lambda: worker.passes > 1, why='the stage never came back')
    finally:
        encoder.gate.set()
        worker.stop()
    assert 'NN_SHED' in caplog.text


def test_a_pass_that_never_returns_leaves_a_show_running_on_beats(tiny, mean):  # noqa: F811
    """The contract when the GPU is simply gone: hold forever, cost nothing,
    and never make the audio thread wait for it."""
    clock = FakeClock()
    encoder = SlowEncoder()
    encoder.gate.clear()
    watchdog = DriftWatchdog(BUFFER_SEC)
    worker = GpuStage(chain(encoder, tiny, mean, margin=1.0, hop=0.5,
                            buffer=20.0),
                      watchdog, clock=clock, pass_timeout_sec=1.0)
    worker.start()
    try:
        feed(worker, 3.0)
        settle(lambda: encoder.threads, why='no pass started')
        clock.advance(2.0)

        started = time.perf_counter()
        for minute in range(30):
            clock.advance(60.0)
            feed(worker, 1.0, seed=minute)
        elapsed = time.perf_counter() - started
        assert watchdog.level is ShedLevel.NN_SHED
        assert worker.queued == 0
        assert elapsed < 10.0, 'the audio thread waited on a dead GPU'
    finally:
        encoder.gate.set()
        worker.stop()


# --------------------------------------------------------------------------- #
# (c) sleep/resume: the encoder objects are invalid until they are rebuilt
# --------------------------------------------------------------------------- #


def test_a_lost_context_is_repaired_by_reinitialising_the_extractor(tiny, mean):  # noqa: F811
    """Nothing but a new encoder helps, and nothing but the encoder may be new:
    the ring, the schedule and the sample index are still true, and the sample
    index is the clock every cell in the show is stamped against."""
    clock = FakeClock()
    worker, watchdog, encoder, built = faulty(tiny, mean, CONTEXT_LOST,
                                              clock=clock)
    try:
        feed(worker, 4.0)
        settle(lambda: worker.faults > 0, why='the fault never surfaced')
        for _ in range(40):
            clock.advance(2.0)
            drain(worker, 5)
            feed(worker, 1.0, seed=7)
            if worker.passes:
                break
        settle(lambda: worker.passes > 0, why='the stage never came back')
    finally:
        worker.stop()

    assert worker.reinits >= 1, 'it never tried a new encoder'
    assert built, 'the reinit seam was never used'
    assert worker.posteriors.stream.samples_seen > 0, 'the clock was restarted'


# --------------------------------------------------------------------------- #
# (d) VRAM pressure
# --------------------------------------------------------------------------- #


def test_an_allocation_failure_is_the_same_contract(tiny, mean, caplog):  # noqa: F811
    """Deliberately the same handling as (a) and (c): an OOM cannot be told from
    a dead context without reading its message, and a policy that branched on
    message text would be a policy about strings."""
    clock = FakeClock()
    worker, watchdog, encoder, _built = faulty(tiny, mean, OUT_OF_MEMORY,
                                               clock=clock, fail_until=1)
    try:
        with caplog.at_level(logging.WARNING):
            feed(worker, 4.0)
            settle(lambda: worker.faults > 0, why='the fault never surfaced')
            drain(worker, 50, clock=clock, step=0.05)
            feed(worker, 4.0, seed=3)
            settle(lambda: worker.passes > 0, why='the stage never came back')
    finally:
        worker.stop()
    assert 'out of memory' in caplog.text
    assert worker.resyncs >= 1


# --------------------------------------------------------------------------- #
# A persistent fault is not its own outage
# --------------------------------------------------------------------------- #


def test_a_permanent_fault_logs_at_a_rate_not_at_a_frequency(tiny, mean, caplog):  # noqa: F811
    clock = FakeClock()
    worker, watchdog, encoder, _built = faulty(tiny, mean, CUDA_ERROR,
                                               clock=clock, repairable=False)
    try:
        with caplog.at_level(logging.WARNING):
            feed(worker, 4.0)
            settle(lambda: worker.faults > 0, why='the fault never surfaced')
            for _ in range(8):
                feed(worker, 0.6, seed=4)
                wait_for_one_more_fault(worker, clock, 4.0)
            assert worker.faults > 3, 'it stopped retrying'
    finally:
        worker.stop()

    named = [r for r in caplog.records if 'illegal memory access' in r.message]
    assert named, 'the fault was never named at all'
    assert len(named) <= 3, \
        f'{len(named)} fault lines in 24s of a permanent fault'
    assert worker.faults > len(named), 'the rate limit did nothing'


def test_the_backoff_is_capped_so_retrying_never_stops(tiny, mean):  # noqa: F811
    clock = FakeClock()
    worker, watchdog, encoder, _built = faulty(tiny, mean, CUDA_ERROR,
                                               clock=clock, repairable=False)
    try:
        feed(worker, 4.0)
        settle(lambda: worker.faults > 0, why='the fault never surfaced')
        for _ in range(20):
            feed(worker, 0.6, seed=5)
            assert wait_for_one_more_fault(worker, clock, 60.0),                 f'no retry after {worker.faults} faults — the backoff is uncapped'
        assert worker.faults > 10
        assert worker.reinits > 5, 'it stopped trying a new encoder'
    finally:
        worker.stop()


# --------------------------------------------------------------------------- #
# What the whole run looks like from the outside
# --------------------------------------------------------------------------- #


async def test_a_full_shed_run_is_the_degradation_state_the_digest_names(
        tiny, mean):  # noqa: F811
    """D13's state, produced by a dead GPU instead of by a half-built branch.

    `held_start_to_end` is the predicate the branch has already been asserted
    from three sides -- False on master's rule engine, True after the
    demolition, False once the decoder drove the show.  This is the fourth, and
    the one it will keep meaning after the branch lands: it is what a live
    NN_SHED looks like from the outside.

    **The loop's 100 ms callback is part of the drill**, not scaffolding.  Both
    of the engine's non-decoder producers live there, and without it this run
    committed nothing at all -- an unlit rig that the predicate then blessed,
    which is exactly the report shape the degradation contract must refuse.
    """
    import sys
    from pathlib import Path

    training = str(Path(__file__).resolve().parents[1] / 'training')
    if training not in sys.path:
        sys.path.insert(0, training)
    from pipeline_digest import degradation_digest, held_start_to_end

    from lib.audio_config import SAMPLE_RATE
    from lib.clock import VirtualClock
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.event_buffer import EventBuffer
    from lib.engine.light_engine import LightEngine
    from lib.engine.section_decoder import DecodeParams, SectionDecoder
    from simulate.stub_clients import (StubMidiClient, StubOs2lClient,
                                       StubOverlayClient)
    from tests.test_light_engine import FakeAnalyser
    from tests.test_nn_decoder import toy_priors

    clock = VirtualClock()
    worker, watchdog, _encoder, _built = faulty(tiny, mean, CUDA_ERROR,
                                                clock=FakeClock(),
                                                repairable=False)
    midi = StubMidiClient(clock=clock)
    events = EventBuffer(window_sec=float('inf'), clock=clock,
                         look_ahead_sec=14.0)
    events.start()
    queue = DelayedCommandQueue(14.0, clock=clock)
    engine = LightEngine(midi, StubOs2lClient(clock=clock),
                         StubOverlayClient(clock=clock),
                         EffectController(midi, clock=clock, event_buffer=events),
                         queue, event_buffer=events, playback_delay_sec=14.0,
                         section_chain=worker,
                         section_decoder=SectionDecoder(
                             toy_priors(floor=2),
                             DecodeParams(lag_bars=1, min_coverage=1)),
                         clock=clock)
    engine.set_analyser(FakeAnalyser())

    try:
        block = np.zeros(256, dtype=np.float32)
        for tick in range(4000):
            await engine.on_audio(block)
            clock.advance(256 / SAMPLE_RATE)
            if tick % 80 == 0:
                await engine.on_beat(tick // 80 + 1, 128.0, False)
            if tick % 17 == 0:
                await engine.on_100ms_callback()
            await queue.drain()
    finally:
        worker.stop()

    events.mark_end()
    report = events.to_report(queue.get_timing_log())
    assert report['metrics']['beats_detected'] > 10, 'beats stopped'
    digest = degradation_digest(report)
    assert digest['effect_changes'] >= 1, 'the rig was dark for the whole run'
    assert held_start_to_end(digest), \
        'a dead GPU produced something other than a held show'


# --------------------------------------------------------------------------- #
# The fifth mode: a card that works every other time
# --------------------------------------------------------------------------- #


class FlappingEncoder(SlowEncoder):
    """Fails every other pass.  Not one of #143's four -- it is what those four
    look like when the fault is intermittent, and it is the shape that disarmed
    both of the mechanisms built for them."""

    def __init__(self, error, **kwargs):
        super().__init__(**kwargs)
        self.flap = error
        self.calls = 0

    def encode(self, segment, **kwargs):
        self.calls += 1
        if self.calls % 2 == 1:
            raise self.flap
        return super().encode(segment, **kwargs)


def flapping(tiny, mean, error):  # noqa: F811
    clock = FakeClock()
    encoder = FlappingEncoder(error, )
    watchdog = DriftWatchdog(BUFFER_SEC)
    worker = GpuStage(chain(encoder, tiny, mean, margin=1.0, hop=0.5,
                            buffer=20.0),
                      watchdog, clock=clock, reinit=None, pass_timeout_sec=1.0)
    worker.start()
    return worker, encoder


def test_a_flapping_gpu_does_not_disarm_the_backoff(tiny, mean, caplog):  # noqa: F811
    """`_attempts` used to reset on ANY pass that returned, and a card failing
    every other pass returns one every other pass.  The backoff then restarted
    from its immediate rung forever: 38 faults, 38 decoder resets and 76
    watchdog transitions inside three seconds of a clock that never moved,
    while both the backoff and the rate limit read as working."""
    caplog.set_level(logging.WARNING)
    worker, encoder = flapping(tiny, mean, CUDA_ERROR)
    try:
        feed(worker, seconds=6.0)
        drain(worker, ticks=250)
    finally:
        worker.stop()

    assert worker.faults <= 4, f'{worker.faults} faults on a held backoff'
    assert worker.resyncs <= 2, f'{worker.resyncs} decoder resets'
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) <= 8, f'{len(warnings)} WARNING lines'


def test_one_good_pass_is_not_a_recovery(tiny, mean):  # noqa: F811
    """What "the GPU came back" has to mean, stated where it is decided."""
    from lib.analyser.gpu_stage import _HEALTHY_PASSES

    assert _HEALTHY_PASSES > 1
    worker, encoder = flapping(tiny, mean, CUDA_ERROR)
    try:
        feed(worker, seconds=6.0)
        drain(worker, ticks=250)
        assert worker.passes >= 1, 'the flap never let a pass through'
        assert worker._attempts >= 1, 'a single pass cleared the backoff'
    finally:
        worker.stop()


def test_a_sustained_run_of_passes_does_clear_the_backoff(tiny, mean):  # noqa: F811
    """The other side of it: a card that genuinely recovers must not stay on a
    half-minute retry for the rest of the night."""
    from lib.analyser.gpu_stage import _HEALTHY_PASSES

    clock = FakeClock()
    encoder = SlowEncoder()
    watchdog = DriftWatchdog(BUFFER_SEC)
    worker = GpuStage(chain(encoder, tiny, mean, margin=1.0, hop=0.5,
                            buffer=20.0),
                      watchdog, clock=clock, reinit=None, pass_timeout_sec=1.0)
    worker._attempts = 5
    worker.start()
    try:
        feed(worker, seconds=float(_HEALTHY_PASSES) + 4.0)
        drain(worker, ticks=250)
        assert worker.passes >= _HEALTHY_PASSES
        assert worker._attempts == 0
    finally:
        worker.stop()
