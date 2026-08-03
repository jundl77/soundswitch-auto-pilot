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
    encoder = SlowEncoder()
    encoder.gate.clear()
    watchdog = DriftWatchdog(BUFFER_SEC)
    worker = stage(encoder, watchdog=watchdog, queue_passes=1, buffer=20.0)
    pump(worker, 8.0)
    encoder.gate.set()

    settle(lambda: worker.overflows > 0, why='the queue never overflowed')
    assert worker.queued <= 1

    gaps: list = []

    def cycled() -> bool:
        gaps.append(worker.push_audio(np.zeros(0, dtype=np.float32)).gap)
        return worker.resyncs > 0

    settle(cycled, why='the overflow never became a shed')
    assert any(gaps), 'the consumer was never told about the hole'


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


def test_a_song_boundary_reset_is_marshalled_onto_the_gpu_thread(stage):
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


def test_a_shed_keeps_feeding_the_ring_because_the_index_is_song_time(stage):
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
def test_the_shipped_chain_runs_on_its_own_thread_and_its_vram_pool_settles(
        nn_artifacts, anchor_mp3):
    from lib import section_chain
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from lib.clients.pyaudio_client import PyAudioClient  # noqa: F401
    from simulate.fake_audio_client import FileAudioClient

    watchdog = DriftWatchdog(BUFFER_SIZE / SAMPLE_RATE)
    chain = section_chain.build_section_chain(watchdog=watchdog)
    assert isinstance(chain.stream, GpuStage)

    def wait_for_the_pass_the_disk_outran():
        while not chain.stream.idle:
            time.sleep(0.001)

    try:
        audio = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, anchor_mp3)
        audio.start_streams()
        reserved = [_reserved()]
        for _ in range(int(70.0 * SAMPLE_RATE / BUFFER_SIZE)):
            if audio.exhausted:
                break
            chain.stream.push_audio(audio.read())
            wait_for_the_pass_the_disk_outran()
            if chain.stream.passes > len(reserved) - 1:
                reserved.append(_reserved())
        audio.close()
    finally:
        chain.stop()

    assert chain.stream.passes > 20, 'not enough passes to say anything'
    assert watchdog.fault is None, f'the stage faulted: {watchdog.fault}'

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
