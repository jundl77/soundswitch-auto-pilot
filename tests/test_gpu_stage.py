"""The GPU stage on its own thread (D3/B3), and the shed ladder's one rung.

Three claims, and they fail differently.

**The audio thread never waits for the GPU.** That is the whole reason for the
thread: a pass is ~81 ms against a 5.805 ms buffer period and ~210 ms at p95
under contention, and the audio input drops rather than queues, so an inline
pass throws away fourteen buffers of audio once a second. These tests hold a
pass open far longer than any buffer and assert the loop walked past it.

**Threading may move when a decision arrives, never which decision it is.** The
same audio goes through the threaded stage and through the synchronous one and
the committed decisions are compared whole -- a hand-off that dropped,
duplicated or re-ordered one pass changes the observation a bar is assembled
from, and nothing else here would see it.

**A gap is a discontinuity, not a pause.** Audio from before a shed must never
decode as current, so both edges clear; and the extractor's sample index IS song
time, so it has to survive the shed or every later cell is stamped into the
wrong part of the track, silently and for the rest of the song.
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
from lib.analyser.section_model import PosteriorStream, SectionModel
from tests.test_drift_watchdog import FakeClock
from tests.test_mert_stream import FakeEncoder
from tests.test_section_model import FakeSession, mean, tiny  # noqa: F401

BUF = 256
BUFFER_SEC = BUF / M.SOURCE_SAMPLE_RATE
CELL = 0.25
EMPTY = np.zeros(0, dtype=np.float32)


class SlowEncoder(FakeEncoder):
    """A pass that takes as long as it is told to, and records which thread ran it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threads = []
        self.gate = threading.Event()
        self.gate.set()
        self.raising = 0
        self.error = RuntimeError('CUDA error: an illegal memory access')

    def encode(self, segment, **kwargs):
        self.threads.append(threading.current_thread().name)
        self.gate.wait(20.0)
        if self.raising:
            self.raising -= 1
            raise self.error
        return super().encode(segment, **kwargs)


def geometry(margin=1.0, hop=0.5, buffer=6.0):
    return M.StreamGeometry(model_id="fake", layers=(6, 22), margin_sec=margin,
                            hop_sec=hop, buffer_sec=buffer, label_frame_sec=CELL)


def chain(encoder, graph, corpus_mean, **kwargs):
    return PosteriorStream(
        M.MertStream(encoder, geometry=geometry(**kwargs)),
        SectionModel(graph, mean=corpus_mean,
                     session_factory=lambda _path: FakeSession()))


def noise(seconds, seed=0):
    return np.random.default_rng(seed).normal(
        size=int(seconds * M.SOURCE_SAMPLE_RATE)).astype(np.float32) * 0.1


def buffers(audio):
    return [audio[i:i + BUF] for i in range(0, len(audio) - BUF, BUF)]


def pump(worker, seconds, seed=0, sink=None):
    sink = [] if sink is None else sink
    for block in buffers(noise(seconds, seed)):
        sink.extend(worker.push_audio(block).posteriors)
    return sink


def quiesce(worker, sink=None, timeout=10.0):
    """Let the thread finish what it started, draining as it goes.

    An empty buffer is a pure drain: nothing is resampled and nothing reaches
    the ring, so waiting cannot itself create the passes it is waiting for.
    """
    sink = [] if sink is None else sink
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sink.extend(worker.push_audio(EMPTY).posteriors)
        if worker.idle:
            return sink
        time.sleep(0.002)
    pytest.fail('the stage never went idle')


def settle(predicate, timeout=10.0, why=''):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    pytest.fail(f'timed out waiting: {why}')


class Pacing:
    """Sheds and restores through the drift input, on a clock the test owns.

    The other input cannot be used as a lever: a fault is the stage's own to
    clear and it clears it on its own backoff, which is the behaviour the
    degradation drills are about and the wrong instrument for asking what a
    shed does to the stream.
    """

    def __init__(self):
        self.clock = FakeClock()
        self.watchdog = DriftWatchdog(BUFFER_SEC, clock=self.clock)
        self.calm(4)

    def calm(self, buffers_=2000):
        for _ in range(buffers_):
            self.clock.advance(BUFFER_SEC)
            self.watchdog.observe()

    def stall(self, sec=2.0):
        self.clock.advance(sec)
        self.watchdog.observe()


@pytest.fixture
def stage(tiny, mean):  # noqa: F811
    started = []

    def build(encoder=None, watchdog=None, margin=1.0, hop=0.5, buffer=6.0,
              **kwargs):
        worker = GpuStage(chain(encoder or SlowEncoder(), tiny, mean,
                                margin=margin, hop=hop, buffer=buffer),
                          watchdog or DriftWatchdog(BUFFER_SEC), **kwargs)
        worker.start()
        started.append(worker)
        return worker

    yield build
    for worker in started:
        worker.stop()


# --------------------------------------------------------------------------- #
# The audio thread never waits for the GPU
# --------------------------------------------------------------------------- #


def test_a_pass_runs_somewhere_other_than_the_thread_that_fed_the_audio(stage):
    encoder = SlowEncoder()
    worker = stage(encoder)
    pump(worker, 3.0)
    settle(lambda: encoder.threads, why='no pass ran')
    assert threading.current_thread().name not in encoder.threads


def test_the_audio_loop_walks_past_a_pass_that_takes_hundreds_of_buffers(stage):
    encoder = SlowEncoder()
    encoder.gate.clear()
    worker = stage(encoder)
    pump(worker, 2.0)
    settle(lambda: encoder.threads, why='no pass started')

    blocks = buffers(noise(4.0, seed=1))
    started = time.perf_counter()
    for block in blocks:
        worker.push_audio(block)
    elapsed = time.perf_counter() - started
    encoder.gate.set()
    assert elapsed < len(blocks) * BUFFER_SEC, \
        f'{elapsed:.3f}s of audio-thread time for {len(blocks)} buffers'


def test_whole_passes_arrive_in_order_and_only_once(stage):
    worker = stage()
    seen = quiesce(worker, pump(worker, 6.0))
    indices = [item.index for item in seen]
    assert len(indices) > 8
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))


def test_the_hand_off_queue_is_bounded_and_overflow_is_a_shed(stage):
    """A consumer that stopped draining must cost memory nothing and must be
    visible; the one thing it may never do is stall the audio thread."""
    encoder = SlowEncoder()
    encoder.gate.clear()
    watchdog = DriftWatchdog(BUFFER_SEC)
    worker = stage(encoder, watchdog=watchdog, queue_passes=1, buffer=20.0)
    pump(worker, 8.0)
    encoder.gate.set()

    settle(lambda: worker.overflows > 0, why='the queue never overflowed')
    assert worker.queued <= 1
    assert watchdog.level is ShedLevel.NN_SHED or worker.faults > 0


# --------------------------------------------------------------------------- #
# Threading moves when, never what
# --------------------------------------------------------------------------- #


def test_the_threaded_stage_decides_exactly_what_the_synchronous_one_decides(
        tiny, mean):  # noqa: F811
    from lib.engine.section_decoder import DecodeParams, SectionDecoder
    from tests.test_nn_decoder import toy_priors

    blocks = buffers(noise(14.0, seed=3))
    beat_every = 32

    def run(stream, quiesce_after=None):
        decoder = SectionDecoder(toy_priors(floor=2),
                                 DecodeParams(lag_bars=1, min_coverage=1))
        decisions, cells = [], []

        def absorb(drained):
            assert not drained.gap, 'an undisturbed run must not gap'
            for item in drained.posteriors:
                cells.append((item.index, round(item.time_sec, 6)))
                decisions.extend(decoder.push_posterior(
                    item.time_sec, item.posterior, item.boundary))

        for n, block in enumerate(blocks):
            absorb(stream.push_audio(block))
            if n % beat_every == 0:
                decisions.extend(decoder.push_beat(n * BUFFER_SEC))
        if quiesce_after is not None:
            deadline = time.monotonic() + 10.0
            while not stream.idle and time.monotonic() < deadline:
                absorb(stream.push_audio(EMPTY))
                time.sleep(0.002)
            absorb(stream.push_audio(EMPTY))
        return decisions, cells

    inline = run(chain(FakeEncoder(), tiny, mean))

    threaded = GpuStage(chain(FakeEncoder(), tiny, mean),
                        DriftWatchdog(BUFFER_SEC))
    threaded.start()
    try:
        found = run(threaded, quiesce_after=True)
    finally:
        threaded.stop()

    assert len(inline[1]) > 30 and len(inline[0]) > 0, 'nothing was compared'
    assert found[1] == inline[1], 'the cell stream diverged'
    assert found[0] == inline[0], 'the decisions diverged'


# --------------------------------------------------------------------------- #
# A song boundary
# --------------------------------------------------------------------------- #


def test_a_song_boundary_reset_is_marshalled_onto_the_gpu_thread(stage):
    """The ring cannot be zeroed under a snapshot in flight, so the request is
    handed over rather than performed on the caller's thread."""
    worker = stage()
    pump(worker, 4.0)
    settle(lambda: worker.passes > 0, why='no pass ran')

    worker.reset()
    settle(lambda: not worker.reset_pending, why='the reset was never taken')
    assert worker.posteriors.stream.samples_seen == 0
    assert worker.posteriors.stream.passes == 0


def test_no_audio_is_written_while_a_reset_is_pending(stage):
    encoder = SlowEncoder()
    encoder.gate.clear()
    worker = stage(encoder)
    pump(worker, 3.0)
    settle(lambda: encoder.threads, why='no pass started')

    worker.reset()
    written = worker.posteriors.stream.samples_seen
    pump(worker, 1.0, seed=9)
    assert worker.posteriors.stream.samples_seen == written
    encoder.gate.set()


# --------------------------------------------------------------------------- #
# Shed and restore
# --------------------------------------------------------------------------- #


def test_a_shed_keeps_feeding_the_ring_because_the_index_is_song_time(stage):
    """The tempting economy that would break the show: a stage that stops taking
    audio comes back with a clock the beat grid disagrees with, for the rest of
    the song and with nothing to say so."""
    pacing = Pacing()
    worker = stage(watchdog=pacing.watchdog, buffer=20.0)
    pump(worker, 3.0)
    settle(lambda: worker.passes > 0, why='no pass ran')

    pacing.stall()
    settle(lambda: worker.shed, why='the stage never noticed the shed')
    ran, written = worker.passes, worker.posteriors.stream.samples_seen
    pump(worker, 3.0, seed=5)
    assert worker.posteriors.stream.samples_seen > written, 'song time stopped'
    assert worker.passes == ran, 'a pass ran during a shed'


def test_nothing_reaches_the_consumer_while_shed(stage):
    pacing = Pacing()
    worker = stage(watchdog=pacing.watchdog, buffer=20.0)
    pacing.stall()
    settle(lambda: worker.shed, why='never shed')
    assert pump(worker, 6.0) == []


def test_both_edges_of_a_shed_tell_the_consumer_to_drop_its_decoder_state(stage):
    pacing = Pacing()
    worker = stage(watchdog=pacing.watchdog, buffer=20.0)
    pump(worker, 3.0)

    pacing.stall()
    settle(lambda: worker.shed, why='never shed')
    assert _gap_seen(worker), 'entering a shed did not signal the gap'

    pacing.calm()
    settle(lambda: not worker.shed, why='never restored')
    assert _gap_seen(worker), 'leaving a shed did not signal the gap'


def _gap_seen(worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.push_audio(EMPTY).gap:
            return True
        time.sleep(0.002)
    return False


def test_restoring_rejoins_the_live_edge_rather_than_replaying_the_gap(stage):
    """`resync` is why the recovered span is one ordinary hop: the cells the gap
    swallowed would have seen the whole buffer of future audio, a geometry the
    student was never trained under."""
    pacing = Pacing()
    worker = stage(watchdog=pacing.watchdog)
    pump(worker, 4.0)
    settle(lambda: worker.passes > 1, why='no pass ran')

    pacing.stall()
    settle(lambda: worker.shed, why='never shed')
    pump(worker, 20.0, seed=2)
    pacing.calm()
    settle(lambda: not worker.shed, why='never restored')

    seen = quiesce(worker, pump(worker, 6.0, seed=4))
    edge = worker.posteriors.stream.samples_seen / M.ENCODER_SAMPLE_RATE
    assert seen, 'nothing decoded after the restore'
    assert seen[0].time_sec > 20.0, 'the stream replayed the gap'
    assert seen[0].time_sec <= edge, 'a cell was stamped past the live edge'


def test_a_ring_overrun_is_a_shed_and_a_resync_rather_than_a_crash(stage):
    """The typed overrun from Task 6, wired: a pass whose audio the ring no
    longer holds is backpressure, and the way back is the live edge."""
    encoder = SlowEncoder()
    encoder.gate.clear()
    watchdog = DriftWatchdog(BUFFER_SEC)
    worker = stage(encoder, watchdog=watchdog, margin=0.5, hop=0.5, buffer=2.0)
    pump(worker, 2.0)
    settle(lambda: encoder.threads, why='no pass started')
    pump(worker, 20.0, seed=6)
    encoder.gate.set()

    settle(lambda: worker.resyncs > 0, why='the overrun never surfaced')
    assert worker.faults == 1, 'the overrun was not the fault, or was not one'
    before = worker.passes
    pump(worker, 8.0, seed=7)
    settle(lambda: not worker.shed and worker.passes > before,
           why='the stage never came back')


def test_every_transition_is_logged(stage, caplog):
    pacing = Pacing()
    worker = stage(watchdog=pacing.watchdog, buffer=20.0)
    with caplog.at_level(logging.INFO):
        pacing.stall()
        settle(lambda: worker.shed, why='never shed')
        pacing.calm()
        settle(lambda: not worker.shed, why='never restored')
    messages = ' | '.join(record.message for record in caplog.records)
    assert 'NONE -> NN_SHED' in messages and 'NN_SHED -> NONE' in messages
    assert 'drift' in messages
    assert messages.count('[gpu]') >= 2, 'the stage said nothing either way'


@pytest.mark.integration
def test_a_watchdog_is_what_puts_the_shipped_chain_on_a_thread(nn_artifacts,
                                                              anchor_mp3):
    """The wiring switch, and the WDDM measurement that goes with it.

    Reserved bytes are recorded rather than merely bounded because the trap is
    silent: under pressure the driver spills to host memory and raises no OOM at
    all, so a run that has started crawling looks exactly like one that has not.
    During the Tasks 8+9 acceptance this box reached 7.8 of 8.0 GB.
    """
    from lib import section_chain
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from lib.clients.pyaudio_client import PyAudioClient  # noqa: F401
    from simulate.fake_audio_client import FileAudioClient

    watchdog = DriftWatchdog(BUFFER_SIZE / SAMPLE_RATE)
    chain = section_chain.build_section_chain(watchdog=watchdog)
    assert isinstance(chain.stream, GpuStage)
    try:
        audio = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, anchor_mp3)
        audio.start_streams()
        reserved = [_reserved()]
        for _ in range(int(70.0 * SAMPLE_RATE / BUFFER_SIZE)):
            if audio.exhausted:
                break
            chain.stream.push_audio(audio.read())
            # The file client hands over audio as fast as the disk allows, which
            # is not a pace the GPU has to keep: unthrottled, the ring overruns
            # inside four passes.  Live audio arrives at 1x; this waits for the
            # pass instead, which is the same relation and finishes sooner.
            while not chain.stream.idle:
                time.sleep(0.001)
            if chain.stream.passes > len(reserved) - 1:
                reserved.append(_reserved())
        audio.close()
    finally:
        chain.stop()

    assert chain.stream.passes > 20, 'not enough passes to say anything'
    assert watchdog.fault is None, f'the stage faulted: {watchdog.fault}'

    # The pool is EXPECTED to climb while the ring fills: a pass encodes what
    # the ring holds, so activations grow with the buffer until it reaches its
    # 30 s length.  What must not happen is growth after that -- the WDDM trap
    # is a pool that keeps climbing and spills to host memory without ever
    # raising an OOM, so a run that has started crawling reads as a healthy one.
    settled = reserved[len(reserved) * 2 // 3:]
    print(f'\n[task10] {chain.stream.passes} passes | cuda reserved '
          f'{reserved[1] / 1e6:.0f}MB after pass 1, '
          f'{max(reserved) / 1e6:.0f}MB peak, '
          f'{(max(settled) - min(settled)) / 1e6:+.1f}MB over the last third '
          f'({len(settled)} passes)')
    assert max(settled) - min(settled) < 256e6, \
        'the reserved pool was still climbing once the ring was full'


def _reserved() -> int:
    from lib.analyser.gpu_stage import reserved_bytes

    return reserved_bytes() or 0


def test_stopping_the_stage_ends_its_thread(tiny, mean):  # noqa: F811
    worker = GpuStage(chain(SlowEncoder(), tiny, mean),
                      DriftWatchdog(BUFFER_SEC))
    worker.start()
    worker.push_audio(np.zeros(BUF, dtype=np.float32))
    worker.stop()
    assert not worker.running
