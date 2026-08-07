"""#142 retired the engine's classifier: decoder decisions alone drive the lights."""
import logging
from collections import deque

import numpy as np
import pytest

from lib.audio_config import SAMPLE_RATE
from lib.engine.effect_definitions import LightIntent
from lib.engine.delayed_command_queue import DelayedCommandQueue
from lib.engine.effect_controller import EffectController
from lib.engine.event_buffer import SILENCE_TRIGGER, STOP_PERSISTENCE_SEC
from lib.engine.light_engine import LightEngine
from lib.engine.section_decoder import BarDecision, BarObservation
from lib.clock import VirtualClock
from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient


class FakeAnalyser:
    def __init__(self):
        self.playing = True
        self.since_beat = 0.0
        self.bpm = 128.0

    def is_song_playing(self):
        return self.playing

    def get_seconds_since_last_beat(self):
        return self.since_beat

    def get_bpm(self):
        return self.bpm

    def get_rms_energy(self):
        return 0.1

    def get_song_current_duration(self):
        import datetime
        return datetime.timedelta(seconds=1.0)


class FakeChain:
    def __init__(self, posteriors=(), gap=False):
        from lib.analyser.section_model import Drained

        self._drained = Drained
        self.pending = list(posteriors)
        self.gap = gap
        self.samples = 0
        self.resets = 0

    def push_audio(self, samples):
        self.samples += len(samples)
        out, self.pending = self.pending, []
        gap, self.gap = self.gap, False
        return self._drained(gap, out)

    def reset(self):
        self.resets += 1
        self.pending = []


class FakeDecoder:
    chain_latency_sec = 13.66
    feature_latency_sec = 7.9938
    bar_sec = 1.8898
    classes = ('intro', 'buildup', 'breakdown', 'drop', 'outro')

    @property
    def bar_edges(self):
        return list(self._edges)

    @property
    def first_bar(self):
        return self._first_bar

    class params:
        lag_bars = 2

    def __init__(self, script=None):
        self.beats: list = []
        self.cells: list = []
        self.resets = 0
        self.cold_starts: list = []
        self.recent_observations = deque(maxlen=4)
        self._edges: list = []
        self._first_bar = 0
        self._script = list(script or [])

    def push_beat(self, at_sec):
        self.beats.append(at_sec)
        return self._next()

    def push_posterior(self, at_sec, posterior, boundary):
        self.cells.append((at_sec, boundary))
        return self._next()

    def _next(self):
        return self._script.pop(0) if self._script else []

    def reset(self, *, cold_start=True):
        self.resets += 1
        self.cold_starts.append(cold_start)


def engine(*, decoder=None, chain=None, playback_delay_sec=14.0, clock=None,
           queue=True, events=False):
    clock = clock or VirtualClock()
    midi = StubMidiClient(clock=clock)
    command_queue = DelayedCommandQueue(playback_delay_sec, clock=clock) \
        if queue else None
    buffer = None
    if events:
        from lib.engine.event_buffer import EventBuffer
        buffer = EventBuffer(window_sec=float('inf'), clock=clock,
                             look_ahead_sec=playback_delay_sec)
        buffer.start()
    light = LightEngine(midi, StubOs2lClient(clock=clock),
                        StubOverlayClient(clock=clock),
                        EffectController(midi, clock=clock, event_buffer=buffer),
                        command_queue, event_buffer=buffer,
                        playback_delay_sec=playback_delay_sec,
                        section_chain=chain, section_decoder=decoder,
                        clock=clock)
    light.set_analyser(FakeAnalyser())
    return light, command_queue, clock, midi


def decisions(*labels, start=0):
    return [BarDecision(start + i, label, (start + i) * 1.9)
            for i, label in enumerate(labels)]


async def elapse(light, clock, sec: float) -> None:
    await light.on_audio(np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32))
    clock.advance(sec)


async def hold_silence(light, clock,
                       sec: float = STOP_PERSISTENCE_SEC + 0.1) -> None:
    remaining = sec
    while remaining > 1e-9:
        step = min(0.1, remaining)
        clock.advance(step)
        await light.on_audio(np.zeros(int(step * SAMPLE_RATE), dtype=np.float32))
        remaining -= step


async def hold_silence_on_a_real_clock(light) -> None:
    light._bypass_due_at = light._clock.monotonic()
    await light.on_audio(np.zeros(1, dtype=np.float32))


async def _noop() -> None:
    return None


async def commit(light, decoder, clock, label, *, age_sec=13.7, bar=0):
    decoder._script.append([BarDecision(bar, label, light.audio_sec - age_sec)])
    await light.on_beat(1, 128.0, False)


async def bars(light, decoder, clock, *labels, age_sec=13.7, bar_sec=1.9):
    for index, label in enumerate(labels):
        await elapse(light, clock, bar_sec)
        await commit(light, decoder, clock, label, age_sec=age_sec, bar=index)


def queued_intents(queue):
    return [item for item in queue._queue if item[4] == 'intent']


async def settle(light, clock, queue, sec=20.0, step=0.25):
    for _ in range(int(sec / step)):
        await elapse(light, clock, step)
        await queue.drain()


@pytest.mark.parametrize('label,intent', [
    ('intro', LightIntent.ATMOSPHERIC),
    ('outro', LightIntent.ATMOSPHERIC),
    ('buildup', LightIntent.BUILDUP),
    ('breakdown', LightIntent.BREAKDOWN),
    ('drop', LightIntent.DROP),
])
async def test_a_decision_commits_the_intent_its_class_maps_to(label, intent):
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, label)
    await settle(light, clock, queue)
    assert light.current_intent is intent


async def test_the_same_class_twice_does_not_re_roll_the_effect():
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, 'drop', 'drop', 'drop')
    await settle(light, clock, queue)
    lit = [e for e in midi.events
           if e['label'] in ('set_autoloop', 'set_special_effect')]
    assert len(lit) == 1
    assert light.intent_commits == 1


async def test_a_sustained_drop_stays_drop_with_no_engine_derived_promotion():
    """The intent layer is a pure mapped image of the class space -- a long drop
    evolves through effect refresh inside DROP's banks, not a family change."""
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, *(['drop'] * 12))
    await settle(light, clock, queue)
    assert light.current_intent is LightIntent.DROP
    assert light.intent_commits == 1


async def test_the_peak_promotion_is_gone_not_merely_unreachable():
    import lib.engine.light_engine as light_engine
    assert not hasattr(light_engine, 'PEAK_PROMOTION_BARS')
    assert not hasattr(LightIntent, 'PEAK')
    assert 'peak' not in {intent.value for intent in LightIntent}


async def test_on_beat_feeds_the_bar_grid_in_the_audio_time_base():
    decoder = FakeDecoder()
    chain = FakeChain()
    light, _, _, _ = engine(decoder=decoder, chain=chain)
    buffer = np.zeros(SAMPLE_RATE, dtype=np.float32)
    await light.on_audio(buffer)
    await light.on_beat(1, 128.0, False)
    await light.on_audio(buffer)
    await light.on_beat(2, 128.0, False)
    assert decoder.beats == pytest.approx([1.0, 2.0])


async def test_on_audio_drives_the_chain_into_the_decoder():
    chain = FakeChain([_posterior(0.9288, 0.3), _posterior(1.0217, 0.4)])
    decoder = FakeDecoder()
    light, _, _, _ = engine(decoder=decoder, chain=chain)
    await light.on_audio(np.zeros(256, dtype=np.float32))
    assert decoder.cells == [(0.9288, 0.3), (1.0217, 0.4)]


async def test_a_gap_from_the_feature_stage_clears_the_decoder_before_it_is_fed():
    chain = FakeChain([_posterior(0.9288, 0.3)], gap=True)
    decoder = FakeDecoder()
    light, _, _, _ = engine(decoder=decoder, chain=chain)
    await light.on_audio(np.zeros(256, dtype=np.float32))
    assert decoder.resets == 1
    assert decoder.cells == [(0.9288, 0.3)]
    assert decoder.cold_starts == [False], \
        'the feature stage gapped, not the beat source: madmom kept running and ' \
        'pays no warm-up, so the grid must not re-apply the first-beat anchor'


async def test_a_shed_and_unshed_cycle_leaves_the_grid_on_the_phase_it_had():
    from tests.test_section_decoder import decoder as section

    live, chain = section(lag_bars=2), FakeChain()
    light, _, clock, _ = engine(decoder=live, chain=chain)

    pushed = []
    for index in range(12):
        if index == 6:
            chain.gap = True
            await light.on_audio(np.zeros(0, dtype=np.float32))
        await elapse(light, clock, 0.5)
        await light.on_beat(index + 1, 128.0, False)
        pushed.append(light.audio_sec)

    uninterrupted = section(lag_bars=2)
    for beat in pushed:
        uninterrupted.push_beat(beat)

    assert live.bar_edges == [pytest.approx(line)
                              for line in uninterrupted.bar_edges
                              if line > pushed[5]]


async def test_a_beat_inside_the_sound_start_margin_leaves_the_anchor_alone():
    """9 of 215 val tracks put their first beat under the 0.3 s margin."""
    from tests.test_section_decoder import decoder as section

    live = section(lag_bars=2)
    light, _, clock, _ = engine(decoder=live, chain=FakeChain())

    await elapse(light, clock, 0.2)
    await light.on_beat(1, 128.0, False)
    light.on_sound_start()

    for index in range(4):
        await elapse(light, clock, 0.5)
        await light.on_beat(index + 2, 128.0, False)

    assert live.bar_edges == [pytest.approx(2.0)]


async def test_a_missing_chain_is_the_degradation_state_rather_than_a_crash():
    """#144: missing artifacts are a degraded show, not a reason to die."""
    light, _, _, _ = engine(decoder=None, chain=None)
    await light.on_audio(np.zeros(256, dtype=np.float32))
    await light.on_beat(1, 128.0, False)
    assert light.current_intent is None


async def test_a_song_boundary_resets_both_stages_and_the_grid():
    chain, decoder = FakeChain(), FakeDecoder()
    light, _, _, _ = engine(decoder=decoder, chain=chain)
    await light.on_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    light.on_sound_stop()
    assert (chain.resets, decoder.resets) == (1, 1)
    await light.on_beat(1, 128.0, False)
    assert decoder.beats == [0.0]


async def test_a_beat_waits_the_whole_playback_delay():
    light, queue, clock, _ = engine(decoder=FakeDecoder())
    await light.on_beat(1, 128.0, False)
    clock.advance(13.9)
    await queue.drain()
    assert queue.pending == 1
    clock.advance(0.2)
    await queue.drain()
    assert queue.pending == 0


async def test_an_intent_waits_only_what_the_chain_has_not_already_spent():
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', age_sec=13.7)
    intent = [e for e in queue._queue if e[4] == 'intent'][0]
    assert intent[3] == pytest.approx(14.0 - 13.7)
    assert intent[0] == pytest.approx(20.0 + 0.3)


async def test_a_chain_slower_than_the_budget_fires_at_once_and_says_so(caplog):
    """#154 accepted lateness on slow tempos rather than growing the budget."""
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 30.0)
    with caplog.at_level(logging.WARNING):
        await commit(light, decoder, clock, 'drop', age_sec=16.4)
        await commit(light, decoder, clock, 'breakdown', age_sec=16.4)
    intent = [e for e in queue._queue if e[4] == 'intent'][0]
    assert intent[3] == 0.0
    assert 'late' in caplog.text.lower()
    assert len([r for r in caplog.records if 'late' in r.message.lower()]) == 1, \
        'one line per transition, not one per bar'


def test_the_startup_line_names_both_halves_of_the_measured_chain(caplog):
    with caplog.at_level(logging.INFO):
        engine(decoder=FakeDecoder())
    assert 'chain latency' in caplog.text.lower()
    assert '7.99' in caplog.text and '14.0' in caplog.text


async def test_beat_absence_still_commits_atmospheric():
    light, queue, clock, _ = engine(decoder=FakeDecoder())
    light.analyser.since_beat = 3.0
    await light.on_100ms_callback()
    clock.advance(14.0)
    await queue.drain()
    assert light.current_intent is LightIntent.ATMOSPHERIC


async def test_a_decoder_decision_after_silence_takes_the_stage_back():
    decoder = FakeDecoder([decisions('drop')])
    light, queue, clock, _ = engine(decoder=decoder)
    light.analyser.since_beat = 3.0
    await light.on_100ms_callback()
    clock.advance(14.0)
    await queue.drain()
    assert light.current_intent is LightIntent.ATMOSPHERIC

    light.analyser.since_beat = 0.0
    await light.on_beat(1, 128.0, False)
    clock.advance(14.0)
    await queue.drain()
    assert light.current_intent is LightIntent.DROP


async def test_a_stale_silence_atmospheric_cannot_take_a_stage_the_decoder_owns():
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', age_sec=13.7)
    await elapse(light, clock, 0.4)
    await queue.drain()
    assert light.current_intent is LightIntent.DROP

    light.analyser.since_beat = 3.0
    await light.on_100ms_callback()
    light.analyser.since_beat = 0.0

    await elapse(light, clock, 3.0)
    await commit(light, decoder, clock, 'drop', age_sec=13.7)

    for _ in range(40):
        await elapse(light, clock, 0.5)
        await queue.drain()
    assert light.current_intent is LightIntent.DROP, \
        'the timer clobbered the stage the committer owns'
    assert queued_intents(queue) == []


async def test_a_run_of_real_intent_changes_is_not_swallowed_by_superseding():
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder)
    await elapse(light, clock, 30.0)
    decoder._script.append([BarDecision(0, 'breakdown', 30.0 - 13.9),
                            BarDecision(1, 'drop', 30.0 - 12.0)])
    await light.on_beat(1, 128.0, False)
    assert len(queued_intents(queue)) == 2, 'a later bar cancelled an earlier one'

    seen = []
    for _ in range(30):
        await elapse(light, clock, 0.25)
        await queue.drain()
        if light.current_intent is not None and (
                not seen or seen[-1] is not light.current_intent):
            seen.append(light.current_intent)
    assert seen == [LightIntent.BREAKDOWN, LightIntent.DROP]


async def test_the_stream_is_deduplicated_against_what_it_will_show():
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', age_sec=13.0)
    await elapse(light, clock, 0.2)
    await commit(light, decoder, clock, 'drop', age_sec=13.2)
    assert len(queued_intents(queue)) == 1


async def test_the_stage_goes_dark_once_the_detected_silence_has_persisted():
    light, queue, clock, midi = engine(decoder=FakeDecoder())
    blackout = []
    midi.on_sound_stop = lambda: blackout.append(clock.monotonic())
    await elapse(light, clock, 5.0)
    at_stop = clock.monotonic()
    light.on_sound_stop()

    await hold_silence(light, clock)
    await queue.drain()
    assert blackout == [
        pytest.approx(at_stop + STOP_PERSISTENCE_SEC, abs=0.2)], \
        'the blackout did not wait for the silence, or waited for the room'


async def test_the_engines_own_bookkeeping_at_a_boundary_is_not_delayed():
    chain, decoder = FakeChain(), FakeDecoder()
    light, _queue, clock, _ = engine(decoder=decoder, chain=chain)
    await elapse(light, clock, 5.0)
    light.on_sound_stop()
    assert (chain.resets, decoder.resets) == (1, 1)


async def test_the_overlay_light_bar_is_room_aligned_like_everything_seen():
    light, queue, clock, _ = engine(decoder=FakeDecoder())
    chase = [e for e in light.overlay_client.events
             if e['label'] == 'overlay_update']
    await light.on_note()
    assert chase == [], 'the chase ran fourteen seconds early'

    await elapse(light, clock, 14.1)
    await queue.drain()
    assert len([e for e in light.overlay_client.events
                if e['label'] == 'overlay_update']) == 1


async def test_a_block_records_the_song_instant_it_describes():
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder, events=True)
    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', age_sec=13.7)
    await settle(light, clock, queue)

    block = light.event_buffer.to_report()['intents'][0]
    assert block['song_t'] == pytest.approx(20.0 - 13.7, abs=1e-6)
    assert block['t'] == pytest.approx(20.3, abs=0.25)


def _posterior(time_sec, boundary):
    class P:
        pass

    p = P()
    p.time_sec = time_sec
    p.posterior = np.full(5, 0.2)
    p.boundary = boundary
    return p


class SettlingMidi(StubMidiClient):
    SETTLE_SEC = 0.2

    def on_sound_stop(self):
        import time
        time.sleep(self.SETTLE_SEC)
        super().on_sound_stop()


class ForgivingWatchdog:
    def __init__(self):
        self.forgiven: list = []

    def forgive(self, sec):
        self.forgiven.append(sec)


async def test_the_midi_settle_is_forgiven_where_it_actually_runs():
    """A virtual clock cannot see a sleeping thread, and 0.2 s clears the 0.15 s door."""
    from lib.clock import SYSTEM_CLOCK

    watchdog = ForgivingWatchdog()
    midi = SettlingMidi(clock=SYSTEM_CLOCK)
    light = LightEngine(midi, StubOs2lClient(clock=SYSTEM_CLOCK),
                        StubOverlayClient(clock=SYSTEM_CLOCK),
                        EffectController(midi, clock=SYSTEM_CLOCK),
                        None, playback_delay_sec=0.0, watchdog=watchdog,
                        clock=SYSTEM_CLOCK)
    light.set_analyser(FakeAnalyser())

    light.on_sound_stop()
    await hold_silence_on_a_real_clock(light)

    assert watchdog.forgiven, 'the settle reached the watchdog as lost lead'
    assert max(watchdog.forgiven) >= SettlingMidi.SETTLE_SEC


async def test_the_sound_start_settle_is_forgiven_too():
    from lib.clock import SYSTEM_CLOCK

    watchdog = ForgivingWatchdog()
    midi = SettlingMidi(clock=SYSTEM_CLOCK)
    light = LightEngine(midi, StubOs2lClient(clock=SYSTEM_CLOCK),
                        StubOverlayClient(clock=SYSTEM_CLOCK),
                        EffectController(midi, clock=SYSTEM_CLOCK),
                        None, playback_delay_sec=0.0, watchdog=watchdog,
                        clock=SYSTEM_CLOCK)
    light.set_analyser(FakeAnalyser())

    light.on_sound_start()

    assert watchdog.forgiven == [pytest.approx(0.0, abs=0.05)]


async def test_the_settle_is_forgiven_from_the_queue_not_from_the_handler():
    clock = VirtualClock()
    watchdog = ForgivingWatchdog()
    midi = StubMidiClient(clock=clock)
    queue = DelayedCommandQueue(14.0, clock=clock)
    light = LightEngine(midi, StubOs2lClient(clock=clock),
                        StubOverlayClient(clock=clock),
                        EffectController(midi, clock=clock), queue,
                        playback_delay_sec=14.0, watchdog=watchdog, clock=clock)
    light.set_analyser(FakeAnalyser())

    light.on_sound_stop()
    await hold_silence(light, clock)
    assert watchdog.forgiven == []

    await queue.drain()
    assert len(watchdog.forgiven) == 1


async def test_two_clamped_decisions_both_reach_the_stage():
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await elapse(light, clock, 40.0)

    decoder._script.append([BarDecision(10, 'drop', light.audio_sec - 15.6),
                            BarDecision(11, 'breakdown', light.audio_sec - 15.2)])
    await light.on_beat(1, 128.0, False)
    await settle(light, clock, queue)

    assert [block['intent'] for block
            in light.event_buffer.snapshot()['intents']] == ['drop', 'breakdown']


async def test_a_clamped_pair_is_delivered_in_commit_order():
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await elapse(light, clock, 40.0)

    decoder._script.append([BarDecision(10, 'drop', light.audio_sec - 15.6),
                            BarDecision(11, 'breakdown', light.audio_sec - 15.2)])
    await light.on_beat(1, 128.0, False)
    await settle(light, clock, queue)

    assert light.current_intent is LightIntent.BREAKDOWN


async def test_a_statement_about_later_audio_still_supersedes():
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await elapse(light, clock, 40.0)
    await bars(light, decoder, clock, 'drop')
    await settle(light, clock, queue)

    light.analyser.since_beat = 3.0
    await light.on_100ms_callback()
    assert light.decided_intent is LightIntent.ATMOSPHERIC

    light.analyser.since_beat = 0.0
    await bars(light, decoder, clock, 'breakdown')
    await settle(light, clock, queue)
    assert light.current_intent is LightIntent.BREAKDOWN


async def tick_100ms(light, clock, sec, step=0.1):
    for _ in range(int(sec / step)):
        await elapse(light, clock, step)
        await light.on_100ms_callback()


async def test_a_committer_that_never_speaks_still_lights_the_rig():
    from lib.engine.light_engine import COLD_START_FLOOR_MARGIN_SEC

    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await tick_100ms(light, clock,
                     decoder.chain_latency_sec + COLD_START_FLOOR_MARGIN_SEC + 1.0)
    await settle(light, clock, queue)

    assert light.current_intent is LightIntent.ATMOSPHERIC
    assert [e['label'] for e in midi.events
            if e['label'] == 'set_autoloop'], 'the rig stayed dark'


async def test_the_floor_fires_once_and_not_once_per_callback():
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await tick_100ms(light, clock, decoder.chain_latency_sec + 20.0)
    await settle(light, clock, queue)

    assert len(light.event_buffer.snapshot()['intents']) == 1


async def test_a_decision_that_arrives_in_time_beats_the_floor():
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await tick_100ms(light, clock, 2.0)
    await bars(light, decoder, clock, 'drop')
    await tick_100ms(light, clock, decoder.chain_latency_sec + 20.0)
    await settle(light, clock, queue)

    assert [b['intent'] for b in light.event_buffer.snapshot()['intents']] == ['drop']


async def test_the_floor_describes_the_audio_the_room_is_hearing_now():
    from lib.engine.light_engine import COLD_START_FLOOR_MARGIN_SEC

    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await tick_100ms(light, clock,
                     decoder.chain_latency_sec + COLD_START_FLOOR_MARGIN_SEC + 0.5)

    pending = queued_intents(queue)
    assert len(pending) == 1
    assert pending[0][3] == pytest.approx(0.0, abs=1e-9)


async def test_a_new_song_re_arms_the_floor_and_commits_a_second_time():
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await tick_100ms(light, clock, decoder.chain_latency_sec + 10.0)
    await settle(light, clock, queue)
    commits = light.intent_commits
    assert commits == 1

    light.on_sound_stop()
    light.on_sound_start()
    await tick_100ms(light, clock, decoder.chain_latency_sec + 10.0)
    await settle(light, clock, queue)

    assert light.intent_commits == commits + 1


async def test_a_machine_with_no_committer_at_all_lights_the_floor_too():
    from lib.engine.light_engine import COLD_START_FLOOR_MARGIN_SEC

    light, queue, clock, midi = engine(events=True)
    await tick_100ms(light, clock, COLD_START_FLOOR_MARGIN_SEC + 1.0)
    await settle(light, clock, queue)

    assert light.current_intent is LightIntent.ATMOSPHERIC


async def boundary(light, chain, clock, score, *, age_sec=8.0, sec=0.5):
    from lib.analyser.section_model import Posterior

    chain.pending.append(Posterior(0, light.audio_sec + sec - age_sec,
                                   np.zeros(5), score))
    await elapse(light, clock, sec)


def autoloops(midi):
    return [e for e in midi.events
            if e['label'] in ('set_autoloop', 'set_special_effect')]


async def held(light, decoder, chain, clock, queue, label='drop'):
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, label)
    await settle(light, clock, queue)


async def test_a_boundary_inside_a_held_intent_re_rolls_the_effect():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue)
    before = len(autoloops(midi))
    intent = light.current_intent

    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)

    assert len(autoloops(midi)) == before + 1
    assert light.current_intent is intent


async def test_a_refresh_does_not_appear_on_the_intent_timeline():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue)
    blocks = len(light.event_buffer.snapshot()['intents'])

    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)

    assert len(light.event_buffer.snapshot()['intents']) == blocks


async def test_a_quiet_boundary_changes_nothing():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue)
    before = len(autoloops(midi))

    await boundary(light, chain, clock, 0.0)
    await settle(light, clock, queue)

    assert len(autoloops(midi)) == before


async def test_a_refresh_re_rolls_from_the_intent_the_stage_is_showing():
    from lib.engine.effect_definitions import INTENT_EFFECTS

    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue, label='breakdown')

    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)

    pool = {effect.midi_channel.name
            for effect in INTENT_EFFECTS[LightIntent.BREAKDOWN]}
    assert autoloops(midi)[-1]['channel'] in pool


async def test_a_boundary_before_anything_is_committed_lights_nothing():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await elapse(light, clock, 20.0)

    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)

    assert autoloops(midi) == []


async def test_a_second_boundary_inside_the_cooldown_is_ignored():
    """The retired mechanism's cooldown_time_window_sec = 10, transferred verbatim."""
    from lib.engine.light_engine import REFRESH_COOLDOWN_SEC

    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue)
    before = len(autoloops(midi))

    await boundary(light, chain, clock, 1.0)
    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)
    assert len(autoloops(midi)) == before + 1

    await elapse(light, clock, REFRESH_COOLDOWN_SEC)
    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)
    assert len(autoloops(midi)) == before + 2


async def test_a_refresh_never_displaces_a_pending_intent_change():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue, label='drop')

    await bars(light, decoder, clock, 'breakdown', age_sec=0.5)
    pending = len(queued_intents(queue))
    assert pending == 1

    await boundary(light, chain, clock, 1.0)
    assert len(queued_intents(queue)) == pending

    await settle(light, clock, queue)
    assert light.current_intent is LightIntent.BREAKDOWN


async def test_a_refresh_queued_across_an_intent_change_is_dropped():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue, label='drop')
    before = len(autoloops(midi))

    await boundary(light, chain, clock, 1.0)
    await bars(light, decoder, clock, 'breakdown', age_sec=13.7)
    await settle(light, clock, queue)

    assert len(autoloops(midi)) == before + 1
    assert light.current_intent is LightIntent.BREAKDOWN


async def test_a_refresh_lands_when_the_room_hears_the_audio_that_caused_it():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue)
    await boundary(light, chain, clock, 1.0, age_sec=8.0)

    refresh = [item for item in queue._queue if item[4] == 'refresh']
    assert len(refresh) == 1
    assert refresh[0][3] == pytest.approx(14.0 - 8.0, abs=1e-6)


async def test_a_gap_clears_the_refresh_cooldown():
    from lib.engine.light_engine import REFRESH_COOLDOWN_SEC

    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue)
    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)
    before = len(autoloops(midi))

    chain.gap = True
    await elapse(light, clock, 0.5)
    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)

    assert len(autoloops(midi)) == before + 1
    assert REFRESH_COOLDOWN_SEC > 1.0


async def test_the_refresh_threshold_sits_inside_the_score_the_head_emits():
    from lib.engine.light_engine import BOUNDARY_REFRESH_SCORE

    assert 0.0 < BOUNDARY_REFRESH_SCORE < 1.0


async def test_a_song_boundary_clears_the_refresh_cooldown():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue)
    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)

    light.on_sound_stop()
    light.on_sound_start()
    await settle(light, clock, queue)
    await bars(light, decoder, clock, 'drop')
    await settle(light, clock, queue)
    before = len(autoloops(midi))

    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)

    assert len(autoloops(midi)) == before + 1


def test_the_refresh_threshold_was_priced_against_the_retired_ceiling():
    """Re-price with training/nn_boundary_refresh_rate.py if either constant moves."""
    import json
    from pathlib import Path

    from lib.engine.light_engine import (BOUNDARY_REFRESH_SCORE,
                                         REFRESH_COOLDOWN_SEC)

    record = json.loads(
        (Path(__file__).resolve().parents[1] / 'training'
         / 'nn_boundary_refresh_rate.json').read_text(encoding='utf-8'))

    assert record['chosen_threshold'] == BOUNDARY_REFRESH_SCORE
    assert record['cooldown_sec'] == REFRESH_COOLDOWN_SEC
    ceiling = record['retired_ceiling_per_minute']
    assert ceiling == 60.0 / REFRESH_COOLDOWN_SEC
    assert 0.0 < record['realised_per_minute']['max'] < ceiling / 2.0
    assert record['tracks'], 'the record was cut against no tracks'
    for track in record['tracks'].values():
        rate = track['refreshes_per_minute'][str(BOUNDARY_REFRESH_SCORE)]
        assert 0.0 < rate < ceiling


async def test_the_decoder_state_reaches_the_event_buffer():
    decoder = FakeDecoder()
    light, _, clock, _ = engine(decoder=decoder, events=True)
    decoder.recent_observations.append(
        BarObservation(7, 13.2, 15.1, np.array([0.05, 0.1, 0.1, 0.7, 0.05]), 0.62))

    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', bar=5)

    state = light.event_buffer.snapshot()['decoder']
    assert state['classes'] == list(FakeDecoder.classes)
    assert state['posterior'] == [0.05, 0.1, 0.1, 0.7, 0.05]
    assert (state['observed_bar'], state['committed_bar']) == (7, 5)
    assert state['committed_label'] == 'drop'
    assert state['lag_bars'] == 2
    assert state['chain_latency_sec'] == FakeDecoder.chain_latency_sec


async def test_a_bar_with_no_evidence_says_so_rather_than_reading_as_silence():
    decoder = FakeDecoder()
    light, _, clock, _ = engine(decoder=decoder, events=True)
    decoder.recent_observations.append(
        BarObservation(7, 13.2, 15.1, None, float('nan')))

    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', bar=5)

    assert light.event_buffer.snapshot()['decoder']['posterior'] is None


async def test_the_decoder_state_stays_out_of_the_report():
    decoder = FakeDecoder()
    light, _, clock, _ = engine(decoder=decoder, events=True)
    decoder.recent_observations.append(
        BarObservation(7, 13.2, 15.1, np.array([0.2] * 5), 0.1))

    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', bar=5)

    report = light.event_buffer.to_report()
    assert 'decoder' not in report
    assert 'decoder' not in report['metrics']


async def test_a_song_boundary_forgets_the_commit_cursor():
    decoder = FakeDecoder()
    light, _, clock, _ = engine(decoder=decoder, events=True)
    decoder.recent_observations.append(
        BarObservation(7, 13.2, 15.1, np.array([0.2] * 5), 0.1))
    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', bar=5)

    light.on_sound_start()
    decoder.recent_observations.clear()
    await light.on_beat(1, 128.0, False)

    state = light.event_buffer.snapshot()['decoder']
    assert state['committed_bar'] is None
    assert state['lag_bars'] is None


async def test_a_feature_gap_forgets_the_commit_cursor():
    decoder = FakeDecoder()
    chain = FakeChain()
    light, _, clock, _ = engine(decoder=decoder, chain=chain, events=True)
    decoder.recent_observations.append(
        BarObservation(34, 60.0, 62.0, np.array([0.2] * 5), 0.1))
    await elapse(light, clock, 60.0)
    await commit(light, decoder, clock, 'breakdown', bar=32)
    assert light.event_buffer.snapshot()['decoder']['committed_bar'] == 32

    chain.gap = True
    decoder.recent_observations.clear()
    await elapse(light, clock, 1.0)

    state = light.event_buffer.snapshot()['decoder']
    assert state['committed_bar'] is None
    assert state['lag_bars'] is None


async def test_a_song_boundary_clears_the_intents_still_in_flight():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue, label='drop')

    await bars(light, decoder, clock, 'breakdown', age_sec=0.5)
    assert len(queued_intents(queue)) == 1
    assert light.decided_intent is LightIntent.BREAKDOWN

    light.on_sound_stop()

    assert queued_intents(queue) == []
    assert light.decided_intent is None


async def test_a_song_boundary_clears_the_refreshes_still_in_flight():
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue, label='drop')
    await boundary(light, chain, clock, 1.0)
    assert [item for item in queue._queue if item[4] == 'refresh']

    light.on_sound_stop()

    assert [item for item in queue._queue if item[4] == 'refresh'] == []


async def test_a_refresh_sharing_an_intent_change_s_fire_time_is_dropped():
    from lib.analyser.section_model import Posterior

    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue, label='drop')
    before = len(autoloops(midi))

    beyond_the_playback_delay = light.audio_sec - 20.0
    chain.pending = [Posterior(0, beyond_the_playback_delay, np.zeros(5), 1.0),
                     Posterior(1, beyond_the_playback_delay, np.zeros(5), 0.0)]
    decoder._script = [[], [BarDecision(9, 'breakdown', beyond_the_playback_delay)]]
    await elapse(light, clock, 0.5)

    assert [item for item in queue._queue if item[4] == 'refresh'] == []

    await settle(light, clock, queue)
    assert len(autoloops(midi)) == before + 1
    assert light.current_intent is LightIntent.BREAKDOWN


async def test_the_deliberate_stall_forgives_the_settle_and_not_the_rest():
    from lib.analyser.drift_watchdog import DriftWatchdog
    from lib.clients.midi_client import SETTLE_SEC

    clock = VirtualClock()
    watchdog = DriftWatchdog(256 / SAMPLE_RATE, clock=clock)
    light, queue, _clock, midi = engine(clock=clock)
    light._watchdog = watchdog

    forgiven = []
    watchdog.forgive = forgiven.append
    with light._deliberate_stall():
        clock.advance(SETTLE_SEC * 10)

    assert forgiven == [SETTLE_SEC]


def test_the_grid_never_tears_itself_down_before_the_show_has_gone_quiet():
    from lib.engine import light_engine, section_decoder

    assert section_decoder._BEAT_GAP_SEC > light_engine._BEAT_ABSENCE_SEC, \
        'the bar grid re-anchors while the engine still believes a section is ' \
        'playing: the decisions either side of the gap would name bars on two ' \
        'different grids with nothing on screen saying the beat had stopped'


async def test_a_persistent_stop_silences_the_monitor_without_waiting_for_the_room():
    light, queue, clock, _ = engine()
    cut = []
    light._silence_monitor = lambda: cut.append(queue.pending)
    light.on_sound_stop()
    assert cut == [], 'the monitor was cut before the silence had persisted'
    await hold_silence(light, clock)
    assert cut == [0]


async def test_a_show_with_no_monitor_still_stops_cleanly():
    light, _, clock, _ = engine()
    light.on_sound_stop()
    await hold_silence(light, clock)


async def test_a_gap_between_songs_never_cuts_the_tail_the_room_is_hearing():
    light, queue, clock, midi = engine(events=True)
    blackout, cut = [], []
    midi.on_sound_stop = lambda: blackout.append(clock.monotonic())
    light._silence_monitor = lambda: cut.append(clock.monotonic())

    await elapse(light, clock, 5.0)
    light.on_sound_stop()
    await hold_silence(light, clock, 1.5)
    light.on_sound_start()
    await hold_silence(light, clock, 30.0)
    await queue.drain()

    assert (blackout, cut) == ([], []), \
        'an inter-song gap cut the tail the room had not reached yet'
    assert [b for b in light.event_buffer.to_report()['intents']
            if b['trigger'] == SILENCE_TRIGGER] == []


async def test_a_resume_inside_the_window_cancels_the_pending_bypass():
    light, queue, clock, midi = engine(events=True)
    blackout = []
    midi.on_sound_stop = lambda: blackout.append(clock.monotonic())

    await elapse(light, clock, 5.0)
    light.on_sound_stop()
    await hold_silence(light, clock, STOP_PERSISTENCE_SEC - 0.1)
    assert blackout == []

    light.on_sound_start()
    await hold_silence(light, clock, 10.0)
    await queue.drain()
    assert blackout == [], 'a cancelled bypass fired anyway'


async def test_the_whole_bypass_package_waits_on_the_same_gate():
    light, queue, clock, midi = engine(events=True)
    fired = []
    midi.on_sound_stop = lambda: fired.append('floor')
    light._silence_monitor = lambda: fired.append('monitor')

    await elapse(light, clock, 5.0)
    queue.schedule('overlay', _noop)
    light.on_sound_stop()
    await hold_silence(light, clock, STOP_PERSISTENCE_SEC - 0.1)
    await queue.drain()
    assert fired == []
    assert queue.pending == 1, 'the pending lighting was flushed early'

    await hold_silence(light, clock, 0.2)
    await queue.drain()
    assert fired == ['monitor', 'floor']
    assert [b['trigger'] for b in light.event_buffer.to_report()['intents']] \
        == [SILENCE_TRIGGER]


async def test_a_detected_stop_drops_the_lighting_the_room_has_not_seen():
    light, queue, clock, _ = engine(events=True)
    ran = []

    async def stale():
        ran.append('stale')

    queue.schedule('intent', stale)
    queue.schedule('overlay', stale)
    light.on_sound_stop()
    await hold_silence(light, clock)
    clock.advance(30.0)
    await queue.drain()
    assert ran == []


async def test_a_persistent_stop_darkens_the_room_on_the_next_drain():
    light, queue, clock, midi = engine(events=True)
    blackout = []
    midi.on_sound_stop = lambda: blackout.append(clock.monotonic())
    light.on_sound_stop()
    await hold_silence(light, clock)
    assert blackout == []
    await queue.drain()
    assert blackout == [pytest.approx(clock.monotonic(), abs=0.01)]


async def test_a_persistent_stop_records_the_quiet_floor():
    light, _, clock, _ = engine(events=True)
    light.on_sound_stop()
    await hold_silence(light, clock)
    assert light.event_buffer.snapshot()['intent'] == 'atmospheric'


async def test_the_quiet_floor_is_recorded_as_an_operator_action_not_a_classification():
    light, _, clock, _ = engine(events=True)
    light.on_sound_stop()
    await hold_silence(light, clock)
    blocks = light.event_buffer.to_report()['intents']
    assert [(b['intent'], b['trigger']) for b in blocks] == \
        [('atmospheric', 'silence')]


async def test_the_quiet_floor_records_the_song_instant_it_happened_at():
    light, _, clock, _ = engine(events=True)
    light.event_buffer.start()
    clock.advance(37.0)
    light.on_sound_stop()
    await hold_silence(light, clock)
    block = light.event_buffer.to_report()['intents'][-1]
    assert block['trigger'] == 'silence'
    assert block['song_t'] == pytest.approx(block['t'], abs=1e-6)
    assert block['t'] == pytest.approx(37.0 + STOP_PERSISTENCE_SEC, abs=0.2)


async def test_the_engine_publishes_what_the_watchdog_knows_about_shedding():
    from lib.analyser.drift_watchdog import DriftWatchdog

    light, _, clock, _ = engine(events=True)
    light._watchdog = DriftWatchdog(256 / SAMPLE_RATE, clock=clock)

    await elapse(light, clock, 1.0)
    assert light.event_buffer.snapshot()['shed']['level'] == 'NONE'

    light._watchdog.report_fault('hung_pass')
    await elapse(light, clock, 1.0)
    shed = light.event_buffer.snapshot()['shed']
    assert (shed['level'], shed['fault']) == ('NN_SHED', 'hung_pass')
    assert (shed['sheds'], shed['sheds_per_min']) == (1, 1)


async def test_a_shed_shorter_than_the_publish_step_is_still_counted():
    from lib.analyser.drift_watchdog import DriftWatchdog

    light, _, clock, _ = engine(events=True)
    light._watchdog = DriftWatchdog(256 / SAMPLE_RATE, clock=clock)
    await elapse(light, clock, 1.0)

    light._watchdog.report_fault('hung_pass')
    light._watchdog.report_healthy()
    await elapse(light, clock, 1.0)

    shed = light.event_buffer.snapshot()['shed']
    assert shed['level'] == 'NONE', 'the level poll should have missed it'
    assert shed['sheds'] == 1, 'and the counter should not have'


async def test_a_show_with_no_watchdog_publishes_nothing_rather_than_zeroes():
    light, _, clock, _ = engine(events=True)
    await elapse(light, clock, 1.0)
    assert light.event_buffer.snapshot()['shed'] == {}


async def test_the_grid_is_published_on_the_clock_the_beats_are_stamped_on():
    decoder = FakeDecoder()
    light, _, clock, _ = engine(decoder=decoder, events=True)
    await elapse(light, clock, 20.0)
    await light.on_beat(1, 128.0, False)

    beat_at = light.event_buffer.snapshot()['beats'][-1]['t']
    decoder._edges = [light.audio_sec - 4.0, light.audio_sec - 2.0]
    light._publish_decoder_state(None)

    published = light.event_buffer.snapshot()['decoder']['bar_edges']
    assert published == [pytest.approx(beat_at - 4.0, abs=0.05),
                         pytest.approx(beat_at - 2.0, abs=0.05)]


async def test_the_grid_travels_with_the_bar_numbers_that_name_it():
    decoder = FakeDecoder()
    light, _, clock, _ = engine(decoder=decoder, events=True)
    await elapse(light, clock, 20.0)
    decoder._edges = [light.audio_sec - 2.0, light.audio_sec]
    decoder._first_bar = 41
    light._publish_decoder_state(None)

    state = light.event_buffer.snapshot()['decoder']
    assert state['first_bar'] == 41
    assert state['bar_sec'] == pytest.approx(FakeDecoder.bar_sec)
