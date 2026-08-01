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
           queue=True):
    clock = clock or VirtualClock()
    midi = StubMidiClient(clock=clock)
    command_queue = DelayedCommandQueue(playback_delay_sec, clock=clock) \
        if queue else None
    light = LightEngine(midi, StubOs2lClient(clock=clock),
                        StubOverlayClient(clock=clock),
                        EffectController(midi, clock=clock),
                        command_queue,
                        playback_delay_sec=playback_delay_sec,
                        section_chain=chain, section_decoder=decoder,
                        clock=clock)
    light.set_analyser(FakeAnalyser())
    return light, command_queue, clock, midi


def decisions(*labels, start=0):
    return [BarDecision(start + i, label, (start + i) * 1.9)
            for i, label in enumerate(labels)]


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
    decoder = FakeDecoder([decisions(label)])
    light, queue, clock, _ = engine(decoder=decoder)
    await light.on_beat(1, 128.0, False)
    clock.advance(14.0)
    await queue.drain()
    assert light.current_intent is intent


async def test_the_same_class_twice_does_not_re_roll_the_effect():
    """Counted across both effect labels: DROP's pool holds a strobe as well as
    two autoloops, so which MIDI call it is depends on the draw."""
    decoder = FakeDecoder([decisions('drop', 'drop', 'drop')])
    light, queue, clock, midi = engine(decoder=decoder)
    await light.on_beat(1, 128.0, False)
    clock.advance(14.0)
    await queue.drain()
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
    decoder = FakeDecoder([decisions(*(['drop'] * (PEAK_PROMOTION_BARS + 1)))])
    light, queue, clock, _ = engine(decoder=decoder)
    await light.on_beat(1, 128.0, False)
    clock.advance(14.0)
    await queue.drain()
    assert light.current_intent is LightIntent.PEAK


async def test_a_short_drop_is_not_promoted():
    decoder = FakeDecoder([decisions(*(['drop'] * PEAK_PROMOTION_BARS))])
    light, queue, clock, _ = engine(decoder=decoder)
    await light.on_beat(1, 128.0, False)
    clock.advance(14.0)
    await queue.drain()
    assert light.current_intent is LightIntent.DROP


async def test_peak_absorbs_further_drop_bars_so_the_pair_cannot_oscillate():
    """The old anti-oscillation contract, kept: while PEAK is current a DROP
    decision is swallowed, so the timeline keeps reading PEAK."""
    script = [decisions(*(['drop'] * (PEAK_PROMOTION_BARS + 1))),
              decisions('drop', 'drop', start=99)]
    decoder = FakeDecoder(script)
    light, queue, clock, _ = engine(decoder=decoder)
    await light.on_beat(1, 128.0, False)
    await light.on_beat(2, 128.0, False)
    clock.advance(14.0)
    await queue.drain()
    assert light.current_intent is LightIntent.PEAK
    assert light.intent_commits == 2   # the drop, then the promotion


async def test_any_other_class_leaves_peak_through_the_normal_path():
    script = [decisions(*(['drop'] * (PEAK_PROMOTION_BARS + 1))),
              decisions('breakdown', start=99)]
    decoder = FakeDecoder(script)
    light, queue, clock, _ = engine(decoder=decoder)
    await light.on_beat(1, 128.0, False)
    await light.on_beat(2, 128.0, False)
    clock.advance(14.0)
    await queue.drain()
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
    decoder = FakeDecoder([decisions('drop')])
    light, queue, clock, _ = engine(decoder=decoder)
    await light.on_beat(1, 128.0, False)
    intent = [e for e in queue._queue if e[4] == 'intent'][0]
    assert intent[3] == pytest.approx(14.0 - 13.66)


async def test_a_chain_slower_than_the_budget_clamps_to_zero_and_says_so(caplog):
    """#154 accepted lateness on slow tempos rather than growing the budget.

    A 2.8 s bar puts the chain past 14 s; the intent then commits as soon as it
    can, which is late but not wrong, and the operator is told once rather than
    every bar.
    """
    decoder = FakeDecoder([decisions('drop')])
    decoder.chain_latency_sec = 16.4
    light, queue, clock, _ = engine(decoder=decoder)
    with caplog.at_level(logging.WARNING):
        await light.on_beat(1, 128.0, False)
    intent = [e for e in queue._queue if e[4] == 'intent'][0]
    assert intent[3] == 0.0
    assert 'chain latency' in caplog.text.lower()


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


def _posterior(time_sec, boundary):
    class P:
        pass

    p = P()
    p.time_sec = time_sec
    p.posterior = np.full(5, 0.2)
    p.boundary = boundary
    return p
