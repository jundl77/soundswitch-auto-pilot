"""The rewire: decoder decisions are the only thing that says what the lights do.

The classifier the engine used to run is gone (#142), and with it the vote
buffer, the min-dwell counter and the invalid-transition veto -- the decoder's
fitted duration model and -inf transitions are their successor.  What is left in
the engine is exactly three things this file pins:

* **the map** from the model's class space onto the rig (D7), and PEAK, which is
  the one show device with no class behind it (D8);
* **the queue relation** (B1).  The engine no longer runs ahead of the audience,
  it runs behind it, and the delay each stream waits is what closes the gap;
* **the silence timer**, which survived the demolition and now has to share the
  stage with a committer that speaks about audio 13.7 s old.
"""
import logging

import numpy as np
import pytest

from lib.audio_config import SAMPLE_RATE
from lib.engine.effect_definitions import LightIntent
from lib.engine.delayed_command_queue import DelayedCommandQueue
from lib.engine.effect_controller import EffectController
from lib.engine.light_engine import PEAK_PROMOTION_BARS, LightEngine
from lib.engine.section_decoder import BarDecision
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
    """Audio in, posteriors out -- the two NN stages, without the GPU."""

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
    """Records what it was fed and emits whatever it was told to."""

    chain_latency_sec = 13.66
    feature_latency_sec = 7.9938
    bar_sec = 1.8898

    class params:
        lag_bars = 2

    def __init__(self, script=None):
        self.beats: list = []
        self.cells: list = []
        self.resets = 0
        self._script = list(script or [])

    def push_beat(self, at_sec):
        self.beats.append(at_sec)
        return self._next()

    def push_posterior(self, at_sec, posterior, boundary):
        self.cells.append((at_sec, boundary))
        return self._next()

    def _next(self):
        return self._script.pop(0) if self._script else []

    def reset(self):
        self.resets += 1


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
    """Move the audio counter and the clock together, as the loop does."""
    await light.on_audio(np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32))
    clock.advance(sec)


async def commit(light, decoder, clock, label, *, age_sec=13.7, bar=0):
    """One decision describing audio ``age_sec`` old, delivered on a beat."""
    decoder._script.append([BarDecision(bar, label, light.audio_sec - age_sec)])
    await light.on_beat(1, 128.0, False)


async def bars(light, decoder, clock, *labels, age_sec=13.7, bar_sec=1.9):
    """A run of bars as the show really sees them: one decision per bar, with
    the audio counter and the clock moving a bar between each."""
    for index, label in enumerate(labels):
        await elapse(light, clock, bar_sec)
        await commit(light, decoder, clock, label, age_sec=age_sec, bar=index)


def queued_intents(queue):
    return [item for item in queue._queue if item[4] == 'intent']


async def settle(light, clock, queue, sec=20.0, step=0.25):
    for _ in range(int(sec / step)):
        await elapse(light, clock, step)
        await queue.drain()


# --------------------------------------------------------------------------- #
# The class map, and the one intent with no class behind it
# --------------------------------------------------------------------------- #


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
    """Counted across both effect labels: DROP's pool holds a strobe as well as
    two autoloops, so which MIDI call it is depends on the draw."""
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, 'drop', 'drop', 'drop')
    await settle(light, clock, queue)
    lit = [e for e in midi.events
           if e['label'] in ('set_autoloop', 'set_special_effect')]
    assert len(lit) == 1
    assert light.intent_commits == 1


async def test_a_sustained_drop_is_promoted_to_peak_after_the_converted_run():
    """D8, re-denominated: 32 commit-beats became 8 decoder bars.

    The old device counted commits, and a commit was a beat; the decoder commits
    once per bar, so the same musical length is 32 / 4.  It is the same claim --
    "a drop that has lasted" -- measured in the unit the committer now speaks.
    """
    assert PEAK_PROMOTION_BARS == 8
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, *(['drop'] * (PEAK_PROMOTION_BARS + 1)))
    await settle(light, clock, queue)
    assert light.current_intent is LightIntent.PEAK


async def test_a_short_drop_is_not_promoted():
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, *(['drop'] * PEAK_PROMOTION_BARS))
    await settle(light, clock, queue)
    assert light.current_intent is LightIntent.DROP


async def test_peak_absorbs_further_drop_bars_so_the_pair_cannot_oscillate():
    """The old anti-oscillation contract, kept: while PEAK is current a DROP
    decision is swallowed, so the timeline keeps reading PEAK."""
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, *(['drop'] * (PEAK_PROMOTION_BARS + 3)))
    await settle(light, clock, queue)
    assert light.current_intent is LightIntent.PEAK
    assert light.intent_commits == 2   # the drop, then the promotion


async def test_any_other_class_leaves_peak_through_the_normal_path():
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock,
               *(['drop'] * (PEAK_PROMOTION_BARS + 1) + ['breakdown']))
    await settle(light, clock, queue)
    assert light.current_intent is LightIntent.BREAKDOWN


# --------------------------------------------------------------------------- #
# The two live streams reaching the decoder
# --------------------------------------------------------------------------- #


async def test_on_beat_feeds_the_bar_grid_in_the_audio_time_base():
    """Beats are stamped off the engine's own sample counter, not the clock.

    The cells carry an audio-position stamp, so the grid has to be built in the
    same base or the bar lines land where nothing was playing.  The analyser's
    stream time is not that base: it rebases itself every fifteen minutes and
    tells nobody.
    """
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
    """A shed and its restore are discontinuities, not pauses: the cells the
    decoder is holding and the bar it was assembling them into describe audio
    from the other side of one."""
    chain = FakeChain([_posterior(0.9288, 0.3)], gap=True)
    decoder = FakeDecoder()
    light, _, _, _ = engine(decoder=decoder, chain=chain)
    await light.on_audio(np.zeros(256, dtype=np.float32))
    assert decoder.resets == 1
    assert decoder.cells == [(0.9288, 0.3)]


async def test_a_missing_chain_is_the_degradation_state_rather_than_a_crash():
    """#144: no fallback classifier, and no artifacts is not a reason to die.

    Beats, silence and a held intent are a legitimate show -- it is D13's state,
    which the branch already ships fixtures for.
    """
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


# --------------------------------------------------------------------------- #
# B1: the queue relation
# --------------------------------------------------------------------------- #


async def test_a_beat_waits_the_whole_playback_delay():
    """It is detected as the audio arrives, so nothing has been spent yet."""
    light, queue, clock, _ = engine(decoder=FakeDecoder())
    await light.on_beat(1, 128.0, False)
    clock.advance(13.9)
    await queue.drain()
    assert queue.pending == 1
    clock.advance(0.2)
    await queue.drain()
    assert queue.pending == 0


async def test_an_intent_waits_only_what_the_chain_has_not_already_spent():
    """And what it has spent is measured, not modelled: the decision's own age
    is `_audio_sec - start_sec` exactly, so every intent lands at the song
    instant it describes plus the playback delay, whatever the tempo did."""
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', age_sec=13.7)
    intent = [e for e in queue._queue if e[4] == 'intent'][0]
    assert intent[3] == pytest.approx(14.0 - 13.7)
    assert intent[0] == pytest.approx(20.0 + 0.3)


async def test_a_chain_slower_than_the_budget_fires_at_once_and_says_so(caplog):
    """#154 accepted lateness on slow tempos rather than growing the budget.

    A decision older than the whole playback delay commits as soon as it can,
    which is late but not wrong, and the operator is told once rather than every
    bar.
    """
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
    """Measured, not assumed: the feature half is fixed by the artifact geometry
    and the decoder half moves with the tempo, so both are printed."""
    with caplog.at_level(logging.INFO):
        engine(decoder=FakeDecoder())
    assert 'chain latency' in caplog.text.lower()
    assert '7.99' in caplog.text and '14.0' in caplog.text


# --------------------------------------------------------------------------- #
# The silence timer, sharing a stage with the committer
# --------------------------------------------------------------------------- #


async def test_beat_absence_still_commits_atmospheric():
    light, queue, clock, _ = engine(decoder=FakeDecoder())
    light.analyser.since_beat = 3.0
    await light.on_100ms_callback()
    clock.advance(14.0)
    await queue.drain()
    assert light.current_intent is LightIntent.ATMOSPHERIC


async def test_a_decoder_decision_after_silence_takes_the_stage_back():
    """Precedence, stated: the committer outranks the timer whenever it speaks.

    The timer describes NOW and knows only that beats stopped; a decision
    describes audio the room is about to hear.  Both are queued into audience
    time, so the later-firing one wins and the engine does not second-guess it.
    """
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
    """The blocker: two producers, two delays, and the older statement winning.

    A beat dropout (documented, still live) trips the timer, which describes NOW
    and so waits the whole playback delay.  A decision describes audio ~13.7 s
    old and waits what is left of it, so it fires first -- and then the stale
    ATMOSPHERIC landed on top of it and, being what the engine had last decided,
    swallowed every decision that would have repaired it.  The stage sat on the
    quiet look through a drop, for as long as the section lasted.

    One stream fixes it by construction: a command's fire time is the song
    instant it describes plus the playback delay, and a newer statement about
    that instant or later replaces the one already queued.
    """
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
    """The other half of the same rule: superseding is about audio, not about
    arrival.  Cancelling whatever happened to be in flight would delete every
    intent block but the last one whenever the chain sits near its budget."""
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder)
    await elapse(light, clock, 30.0)
    # Two bars out of one `_advance`, which is how a burst of posteriors
    # arrives: both are in flight at once and neither may cancel the other.
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
    """Not against what was last decided: an intent still in flight is what the
    stage is about to be, and re-committing it would re-roll the effect."""
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder)
    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', age_sec=13.0)
    await elapse(light, clock, 0.2)
    await commit(light, decoder, clock, 'drop', age_sec=13.2)
    assert len(queued_intents(queue)) == 1


async def test_the_stage_does_not_go_dark_while_the_room_is_still_hearing_music():
    """The engine hears the song end fourteen seconds before the audience does.

    Blacking out MIDI and the overlay at detection time therefore killed the
    stage over the last fourteen seconds of every track -- and then in-flight
    intents re-lit it.  Inaudible in every report, unmissable in the venue.
    """
    light, queue, clock, midi = engine(decoder=FakeDecoder())
    blackout = []
    midi.on_sound_stop = lambda: blackout.append(clock.monotonic())
    await elapse(light, clock, 5.0)
    light.on_sound_stop()

    await elapse(light, clock, 13.0)
    await queue.drain()
    assert blackout == [], 'the stage went dark before the room heard the end'

    await elapse(light, clock, 2.0)
    await queue.drain()
    assert blackout == [pytest.approx(20.0, abs=1.1)], \
        'the blackout did not land a playback delay after the boundary'


async def test_the_engines_own_bookkeeping_at_a_boundary_is_not_delayed():
    """The other half: the chain, the decoder and the OS2L wire are not things
    the audience looks at, and holding them back would feed the next song's
    audio into the last song's state for fourteen seconds."""
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
    """The report's two time bases, both written down.

    A block's stamp is when the room sees it, and the delay behind it is a
    per-command quantity, so nothing downstream can recover the audio it was
    about -- and a labelled score against the wrong ~14 s of a track is a number
    that looks fine.  The engine knows the instant exactly at commit time;
    recording it is the only place that costs nothing.
    """
    decoder = FakeDecoder()
    light, queue, clock, _ = engine(decoder=decoder, events=True)
    await elapse(light, clock, 20.0)
    await commit(light, decoder, clock, 'drop', age_sec=13.7)
    await settle(light, clock, queue)

    block = light.event_buffer.to_report()['intents'][0]
    assert block['song_t'] == pytest.approx(20.0 - 13.7, abs=1e-6)
    # The fire stamp carries the drain quantum this test drives; the recorded
    # instant does not, which is the whole difference between them.
    assert block['t'] == pytest.approx(20.3, abs=0.25)


def _posterior(time_sec, boundary):
    class P:
        pass

    p = P()
    p.time_sec = time_sec
    p.posterior = np.full(5, 0.2)
    p.boundary = boundary
    return p


# --------------------------------------------------------------------------- #
# The MIDI settle, forgiven where it runs
# --------------------------------------------------------------------------- #


class SettlingMidi(StubMidiClient):
    """A MIDI client that really blocks, the way the real one does."""

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
    """On the SYSTEM clock, because that is the only one that can see it.

    The settle used to sit inside the analyser's forgive bracket; making the
    boundary room-aligned moved it into the drain loop a playback delay later,
    and 0.2 s is over the watchdog's 0.15 s door -- a spurious NN_SHED at every
    track change.  A virtual clock does not advance while a thread sleeps, so
    no fast simulation can ever show this.
    """
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

    assert watchdog.forgiven, 'the settle reached the watchdog as lost lead'
    assert max(watchdog.forgiven) >= SettlingMidi.SETTLE_SEC


async def test_the_sound_start_settle_is_forgiven_too():
    """Same call, same block, same door -- and it runs at every track change."""
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
    """The point of the fix: the forgive has to happen when the command fires,
    a whole playback delay after the handler returned."""
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
    assert watchdog.forgiven == []

    clock.advance(14.1)
    await queue.drain()
    assert len(watchdog.forgiven) == 1


# --------------------------------------------------------------------------- #
# The clamp path, where two song instants become one wall instant
# --------------------------------------------------------------------------- #


async def test_two_clamped_decisions_both_reach_the_stage():
    """A chain older than the playback delay clamps every decision to `now`,
    and the supersede filter then read two consecutive bars as one restatement
    and deleted the first.  It costs a real intent block on every
    slower-than-120-BPM track -- exactly where the clamp is not hypothetical --
    and `intent_changes_count` under-reads by however many it swallowed."""
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
    """The later bar has to be the one the stage ends on."""
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await elapse(light, clock, 40.0)

    decoder._script.append([BarDecision(10, 'drop', light.audio_sec - 15.6),
                            BarDecision(11, 'breakdown', light.audio_sec - 15.2)])
    await light.on_beat(1, 128.0, False)
    await settle(light, clock, queue)

    assert light.current_intent is LightIntent.BREAKDOWN


async def test_a_statement_about_later_audio_still_supersedes():
    """The supersede rule the clamp fix must not undo: the beat-absence timer
    describes NOW and waits the whole delay, so a decision landing sooner has
    to be able to delete it."""
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


# --------------------------------------------------------------------------- #
# The cold-start floor: "hold the intent" is only a show if there is one
# --------------------------------------------------------------------------- #


async def tick_100ms(light, clock, sec, step=0.1):
    for _ in range(int(sec / step)):
        await elapse(light, clock, step)
        await light.on_100ms_callback()


async def test_a_committer_that_never_speaks_still_lights_the_rig():
    """A GPU dead at boot commits nothing, so the rig was dark for the whole
    night while every log line said the show was holding its intent."""
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
    """The cost of the floor being wrong is one extra effect change at the top
    of a set; the healthy path must not pay it."""
    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await tick_100ms(light, clock, 2.0)
    await bars(light, decoder, clock, 'drop')
    await tick_100ms(light, clock, decoder.chain_latency_sec + 20.0)
    await settle(light, clock, queue)

    assert [b['intent'] for b in light.event_buffer.snapshot()['intents']] == ['drop']


async def test_the_floor_describes_the_audio_the_room_is_hearing_now():
    """One playback delay of age, so it fires on the next drain rather than
    waiting a second delay for an instant it cannot name."""
    from lib.engine.light_engine import COLD_START_FLOOR_MARGIN_SEC

    decoder = FakeDecoder()
    light, queue, clock, midi = engine(decoder=decoder, events=True)
    await tick_100ms(light, clock,
                     decoder.chain_latency_sec + COLD_START_FLOOR_MARGIN_SEC + 0.5)

    pending = queued_intents(queue)
    assert len(pending) == 1
    assert pending[0][3] == pytest.approx(0.0, abs=1e-9)


async def test_a_new_song_re_arms_the_floor():
    """Counted in commits rather than in timeline blocks: the room saw one
    continuous ATMOSPHERIC across both songs, which is what the timeline
    records, but the stage was blacked out and re-lit in between."""
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
    """The degradation state a fresh clone runs in is a show, not an outage."""
    from lib.engine.light_engine import COLD_START_FLOOR_MARGIN_SEC

    light, queue, clock, midi = engine(events=True)
    await tick_100ms(light, clock, COLD_START_FLOOR_MARGIN_SEC + 1.0)
    await settle(light, clock, queue)

    assert light.current_intent is LightIntent.ATMOSPHERIC


# --------------------------------------------------------------------------- #
# D9 -- the boundary-triggered effect refresh
# --------------------------------------------------------------------------- #


async def boundary(light, chain, clock, score, *, age_sec=8.0, sec=0.5):
    """One posterior carrying a boundary score for audio ``age_sec`` old."""
    from lib.analyser.section_model import Posterior

    chain.pending.append(Posterior(0, light.audio_sec + sec - age_sec,
                                   np.zeros(5), score))
    await elapse(light, clock, sec)


def autoloops(midi):
    return [e for e in midi.events
            if e['label'] in ('set_autoloop', 'set_special_effect')]


async def held(light, decoder, chain, clock, queue, label='drop'):
    """A committed intent, on the stage, with nothing left in flight."""
    await elapse(light, clock, 20.0)
    await bars(light, decoder, clock, label)
    await settle(light, clock, queue)


async def test_a_boundary_inside_a_held_intent_re_rolls_the_effect():
    """D9: the successor to YAMNet's section-change refresh.  The audience-
    visible behaviour is "the effect changes inside a long same-intent section",
    which no class boundary can express because the class is the same either
    side."""
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
    """intent_changes_count reads the classifier's opinion and
    effect_changes_count reads the show; a re-roll moves only the second."""
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
    """An effect lit with no intent committed is the stage moving on nobody's
    decision, and the digest calls it a violation rather than a pass."""
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await elapse(light, clock, 20.0)

    await boundary(light, chain, clock, 1.0)
    await settle(light, clock, queue)

    assert autoloops(midi) == []


async def test_a_second_boundary_inside_the_cooldown_is_ignored():
    """The one rate number the retired mechanism recorded, transferred
    verbatim: cooldown_time_window_sec = 10."""
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
    """The superseding rule is the intent stream's, and a refresh is not an
    intent: same-intent re-roll only."""
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
    """The change re-picks the effect itself, so a refresh landing behind it is
    a second re-roll the room reads as a flicker."""
    decoder, chain = FakeDecoder(), FakeChain()
    light, queue, clock, midi = engine(decoder=decoder, chain=chain, events=True)
    await held(light, decoder, chain, clock, queue, label='drop')
    before = len(autoloops(midi))

    # A boundary is younger than a decision, so it waits longer: a change
    # decided after it still lands first.
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
    """Everything the stages hold describes audio from before the gap, and the
    instant of the last refresh is one of those things."""
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
    """A threshold at or outside the sigmoid's range is a switch, not a
    trigger."""
    from lib.engine.light_engine import BOUNDARY_REFRESH_SCORE

    assert 0.0 < BOUNDARY_REFRESH_SCORE < 1.0


async def test_a_song_boundary_clears_the_refresh_cooldown():
    """The chain restarts cell time at a boundary, so a refresh instant from
    the last track is a number in the FUTURE of this one -- and left in place
    it holds the cooldown shut for the whole of the next song."""
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
    """D9's threshold evidence, held to the file that measured it.

    The rate YAMNet produced was never measured and cannot be recovered -- the
    simulation stubbed its detector out before any report was ever written --
    so what transfers is the mechanism's own governor, its ten-second floor,
    and the threshold is chosen so the realised rate lands well inside that
    bracket rather than at its ceiling.  Re-price with
    `training/nn_boundary_refresh_rate.py` if either constant moves; this is
    what stops one moving without the other.
    """
    import json
    from pathlib import Path

    from lib.engine.light_engine import (BOUNDARY_REFRESH_SCORE,
                                         REFRESH_COOLDOWN_SEC)

    record = json.loads(
        (Path(__file__).resolve().parents[1] / 'training'
         / 'nn_boundary_refresh_rate.json').read_text(encoding='utf-8'))

    assert record['chosen_threshold'] == BOUNDARY_REFRESH_SCORE
    assert record['cooldown_sec'] == REFRESH_COOLDOWN_SEC
    # The retired mechanism's cooldown, as a rate.  Nothing may exceed it, and
    # the point of the threshold is to sit well under it.
    ceiling = record['retired_ceiling_per_minute']
    assert ceiling == 60.0 / REFRESH_COOLDOWN_SEC
    assert 0.0 < record['realised_per_minute']['max'] < ceiling / 2.0
    assert record['tracks'], 'the record was cut against no tracks'
    for track in record['tracks'].values():
        rate = track['refreshes_per_minute'][str(BOUNDARY_REFRESH_SCORE)]
        assert 0.0 < rate < ceiling
