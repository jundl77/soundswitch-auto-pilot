from __future__ import annotations
import contextlib
import logging
from typing import TYPE_CHECKING
from lib.audio_config import SAMPLE_RATE
from lib.engine.effect_controller import EffectController
from lib.engine.delayed_command_queue import DelayedCommandQueue
from lib.engine.effect_definitions import LightIntent, intent_for_class
from lib.clients.midi_client import MidiClient
from lib.clients.os2l_client import Os2lClient
from lib.clients.overlay_client import OverlayClient, OverlayEffect
from lib.analyser.music_analyser import MusicAnalyser
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.clock import Clock, SYSTEM_CLOCK

if TYPE_CHECKING:
    from lib.engine.event_buffer import EventBuffer

_BEAT_ABSENCE_SEC = 2.5

# D8, converted rather than re-chosen: the retired device promoted a DROP that
# had survived 32 commits, and a commit was a beat.  The decoder commits once
# per bar, so the same musical length is 32 / 4.  PEAK is the one intent no
# class produces -- "a drop that has lasted" is a run length, which no window of
# audio can express.
PEAK_PROMOTION_BARS = 8

# Enough movement in the measured chain to be worth a line; below it the log
# would say the same number every ten seconds with the last digit twitching.
_LATENCY_LOG_STEP_SEC = 0.25


class LightEngine(IMusicAnalyserHandler):
    """Decoder decisions in, a show out.

    The engine no longer decides anything about the music.  It maps the class
    the committer chose onto the rig, keeps the one show device the class space
    cannot express (PEAK), and holds every command until the audience hears the
    audio that caused it.

    That last part inverted with the NN (B1).  The rule engine ran AHEAD of the
    room and the queue held its commands back by the whole playback delay; the
    model runs BEHIND it, so a command waits ``playback_delay - chain_latency``
    -- and since a beat and a decision are different ages when they arrive, the
    delay belongs to the stream rather than to the queue.
    """

    def __init__(self,
                 midi_client: MidiClient,
                 os2l_client: Os2lClient,
                 overlay_client: OverlayClient,
                 effect_controller: EffectController,
                 command_queue: DelayedCommandQueue | None = None,
                 event_buffer: EventBuffer | None = None,
                 playback_delay_sec: float = 0.0,
                 section_chain=None,
                 section_decoder=None,
                 watchdog=None,
                 clock: Clock = SYSTEM_CLOCK):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.overlay_client: OverlayClient = overlay_client
        self.effect_controller: EffectController = effect_controller
        self.command_queue: DelayedCommandQueue | None = command_queue
        self.event_buffer: EventBuffer | None = event_buffer
        self.analyser: MusicAnalyser = None
        self.section_chain = section_chain
        self.section_decoder = section_decoder
        self._watchdog = watchdog
        self._playback_delay_sec: float = playback_delay_sec
        self._clock: Clock = clock
        self._note_counter: int = 0
        self._atmospheric_sent: bool = False
        self._current_intent: LightIntent | None = None
        # The intent commands enqueued and not yet delivered, in fire order.
        # With `_current_intent` they are the whole stream: what the stage shows
        # now, and what it is going to show.
        self._pending_intents: list = []
        self._published_bpm: dict = {}
        self._bars_in_current_intent: int = 0
        self._intent_commits: int = 0
        self._audio_sec: float = 0.0
        self._latency_logged_at: float | None = None
        self._committing_late: bool = False
        self._log_chain_latency()

    def set_analyser(self, analyser: MusicAnalyser):
        self.analyser: MusicAnalyser = analyser

    @property
    def current_intent(self) -> LightIntent | None:
        return self._current_intent

    @property
    def decided_intent(self) -> LightIntent | None:
        """What the stream ends on: the last command still in flight, or what
        the stage is showing when nothing is.

        Deduplicating against "what was last decided" instead was the shape of
        the latch: a command that had been superseded still counted as the
        engine's opinion, so the decision that would have repaired it read as a
        repeat and was dropped.
        """
        return (self._pending_intents[-1][1] if self._pending_intents
                else self._current_intent)

    @property
    def audio_sec(self) -> float:
        return self._audio_sec

    @property
    def intent_commits(self) -> int:
        return self._intent_commits

    def on_sound_start(self):
        """A boundary the engine hears 14 s before the room does.

        So the two halves part company: the engine's own bookkeeping and the
        OS2L wire (which talks to a DJ's software, not to the audience) happen
        now, and everything the room can SEE waits until the room hears what
        caused it.  Before this split the stage blacked out while the last
        fourteen seconds of a track were still playing -- inaudible in every
        report, unmissable in the venue.
        """
        logging.info('[engine] sound start')
        self.os2l_client.on_sound_start(0, 0, 20000, 120)
        if self.event_buffer:
            self.event_buffer.set_playing(True)
        self._at_the_room('sound', self._show_sound_start)

    def on_sound_stop(self):
        logging.info('[engine] sound stop')
        self.os2l_client.on_sound_stop()
        self.effect_controller.reset_state()
        if self.event_buffer:
            self.event_buffer.set_playing(False)
        self._at_the_room('sound', self._show_sound_stop)

    def _show_sound_start(self) -> None:
        with self._deliberate_stall():
            self.midi_client.on_sound_start()
            self.overlay_client.deactivate_all()

    def _show_sound_stop(self) -> None:
        with self._deliberate_stall():
            self.midi_client.on_sound_stop()
            self.overlay_client.deactivate_all()

    @contextlib.contextmanager
    def _deliberate_stall(self):
        """A stall the show chose is not lost lead.

        `MidiClient.on_sound_stop` blocks for 0.2 s giving the rig time to
        settle.  It used to run inside `MusicAnalyser._on_sound_stop`, which
        forgave it; making the boundary room-aligned moved it into the drain
        loop a playback delay later, where nothing did -- and 0.2 s is over the
        watchdog's 0.15 s door, so every track change shed the GPU stage for
        ~10 s and re-warmed the decoder for another 14.  All night, and
        invisible to a virtual clock, which does not advance while a real thread
        sleeps.  So the forgive follows the stall to where it actually runs.
        """
        started = self._clock.monotonic()
        try:
            yield
        finally:
            if self._watchdog is not None:
                self._watchdog.forgive(self._clock.monotonic() - started)

    def _at_the_room(self, label: str, action) -> None:
        """Fire when the audience hears the audio that caused it."""
        if not self.command_queue:
            action()
            return

        async def command():
            action()

        self.command_queue.schedule(label, command)
        self._atmospheric_sent = False
        self._current_intent = None
        self._bars_in_current_intent = 0
        # D10: everything the stages hold describes audio from before the gap,
        # and the audio counter is the time base the grid and the cells share.
        self._audio_sec = 0.0
        if self.section_chain is not None:
            self.section_chain.reset()
        if self.section_decoder is not None:
            self.section_decoder.reset()

    async def on_cycle(self):
        await self.effect_controller.process_effects()
        self.overlay_client.flush_messages()

    async def on_audio(self, audio_signal) -> None:
        """Every buffer, before the rhythm stage reads it.

        The counter runs whether or not a song is playing, because it is what
        stamps the bar grid and the feature stage is being fed the same buffers
        either way.  Both are zeroed at a song boundary, together.
        """
        self._audio_sec += len(audio_signal) / SAMPLE_RATE
        if self.section_chain is None or self.section_decoder is None:
            return
        drained = self.section_chain.push_audio(audio_signal)
        if drained.gap:
            # The feature stage stopped and rejoined the live edge, so every
            # cell the decoder is holding, and the bar it was assembling them
            # into, describe audio from the other side of a discontinuity.
            self.section_decoder.reset()
        for posterior in drained.posteriors:
            await self._commit(self.section_decoder.push_posterior(
                posterior.time_sec, posterior.posterior, posterior.boundary))

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        current_second = self.analyser.get_song_current_duration().total_seconds()
        rms_energy = self.analyser.get_rms_energy()

        logging.info(
            f'[engine] [{current_second:.2f}s] beat #{beat_number}  bpm={bpm:.1f}'
        )
        if self.event_buffer:
            self.event_buffer.add_beat(bpm, bpm_changed, rms=rms_energy)

        self._atmospheric_sent = False

        if self.section_decoder is not None:
            await self._commit(self.section_decoder.push_beat(self._audio_sec))

        published_bpm = self._publishable_bpm(self._published_bpm, bpm)
        if self.command_queue:
            await self.command_queue.enqueue(
                'beat',
                lambda: self.os2l_client.send_beat(change=bpm_changed, pos=beat_number,
                                                   bpm=published_bpm, strength=0.5)
            )
        else:
            await self.os2l_client.send_beat(change=bpm_changed, pos=beat_number,
                                             bpm=published_bpm, strength=0.5)

    @staticmethod
    def _publishable_bpm(state: dict, bpm: float) -> float:
        """Hold the last measured tempo rather than publishing a warm-up 0.0,
        which OS2L consumers read as a tempo of zero rather than "not known yet".

        Not cleared between songs: a new track's first second carries the
        previous track's tempo, which a DJ has beat-matched anyway.
        """
        if bpm > 0:
            state['last'] = bpm
            return bpm
        return state.get('last', 0.0)

    async def _commit(self, decisions) -> None:
        """The successor to the whole stability pipeline.

        There is nothing left to guard: the decoder's fitted duration floors
        replace min-dwell, its -inf transitions replace the veto, and its
        backtrace pruning replaces the vote buffer.  What remains is PEAK, and
        it is deliberately the same device it always was.
        """
        for decision in decisions:
            intent = intent_for_class(decision.label)
            self._bars_in_current_intent += 1
            decided = self.decided_intent

            if (decided is LightIntent.DROP and intent is LightIntent.DROP
                    and self._bars_in_current_intent >= PEAK_PROMOTION_BARS):
                logging.info(f'[engine] sustained DROP over '
                             f'{self._bars_in_current_intent} bars — promoting to PEAK')
                intent = LightIntent.PEAK
            elif decided is LightIntent.PEAK and intent is LightIntent.DROP:
                # Absorbed, so the pair cannot oscillate and the timeline keeps
                # reading the PEAK the room is actually looking at.
                continue

            logging.info(
                f'[engine] bar {decision.bar} @ {decision.start_sec:.2f}s  '
                f'{decision.label} → {intent.name}')
            await self._commit_intent(
                intent, max(0.0, self._audio_sec - decision.start_sec))

    def _log_chain_latency(self) -> None:
        """Both halves, because only one of them can move."""
        decoder = self.section_decoder
        if decoder is None:
            logging.info(f'[engine] no section decoder — holding intent; '
                         f'playback delay {self._playback_delay_sec:.2f}s')
            return
        latency = decoder.chain_latency_sec
        if (self._latency_logged_at is not None
                and abs(latency - self._latency_logged_at) < _LATENCY_LOG_STEP_SEC):
            return
        self._latency_logged_at = latency
        logging.info(
            f'[engine] chain latency {latency:.2f}s = '
            f'{decoder.feature_latency_sec:.2f}s features + '
            # lag + 1, because bar b's observation needs bar b to finish before
            # the commit lands lag bars later.  Printed as it is computed: this
            # is the line an operator reconciles against dmx-enttec-node.
            f'{latency - decoder.feature_latency_sec:.2f}s decoder '
            f'({decoder.params.lag_bars} + 1 lag × {decoder.bar_sec:.3f}s bars) | '
            f'playback delay {self._playback_delay_sec:.2f}s → queue delay '
            f'{max(0.0, self._playback_delay_sec - latency):.2f}s — ensure '
            f'dmx-enttec-node playback_delay_seconds matches')

    async def _commit_intent(self, intent: LightIntent, age_sec: float) -> None:
        """The one path any intent reaches the stage by.

        Both producers describe a SONG instant -- the committer a bar line, the
        beat-absence timer the present moment -- so a command's fire time is
        that instant plus the playback delay and nothing else, and the age is
        measured (`_audio_sec - start_sec`) rather than modelled from a median
        bar.  Two streams with two delays put the older statement last: a
        decision is ~13.7 s old and waited ~0.3 s, a timer trip is new and
        waited the whole 14 s, so a false ATMOSPHERIC landed on top of a drop
        the committer had already called and then swallowed every repair.

        A newer statement about the same instant or later replaces what is
        queued for it; anything about earlier audio is left alone, because
        superseding by arrival order would delete every intent block but the
        last one whenever the chain sits near its budget.
        """
        now = self._clock.monotonic()
        fire_at = now - age_sec + self._playback_delay_sec
        self._note_lateness(now - fire_at)
        fire_at = max(fire_at, now)

        if self.command_queue:
            self.command_queue.drop_pending('intent', fire_at)
        self._pending_intents = [item for item in self._pending_intents
                                 if item[0] < fire_at]
        if intent is self.decided_intent:
            return

        self._bars_in_current_intent = 0
        # The song instant this describes, in the report's own time base, so a
        # consumer never has to reconstruct it from a delay it did not see.
        song_sec = (None if self.event_buffer is None
                    else self.event_buffer.elapsed() - age_sec)
        if not self.command_queue:
            await self._apply_intent(None, intent, song_sec)
            return
        entry = (fire_at, intent)
        self._pending_intents.append(entry)
        await self.command_queue.enqueue(
            'intent', lambda: self._apply_intent(entry, intent, song_sec),
            delay_sec=fire_at - now)

    def _note_lateness(self, late_sec: float) -> None:
        """#154's accepted lateness: one line per transition, not per bar."""
        if late_sec > 0.0:
            if not self._committing_late:
                self._committing_late = True
                logging.warning(
                    f'[engine] the chain is older than the '
                    f'{self._playback_delay_sec:.2f}s playback delay — intents '
                    f'commit {late_sec:.2f}s late (slow tempo, accepted)')
        elif self._committing_late:
            self._committing_late = False
            logging.info('[engine] the chain is back inside the playback delay')

    async def _apply_intent(self, entry, intent: LightIntent,
                            song_sec: float | None) -> None:
        """The single path that moves the stage: timeline and MIDI together, so
        nothing can light an intent the timeline does not know about.

        Runs when the queue fires, which is the instant the room hears the audio
        this was decided from -- hence the split from ``decided_intent``, which
        is where the stream has got to.
        """
        if entry is not None and self._pending_intents \
                and self._pending_intents[0] is entry:
            self._pending_intents.pop(0)
        if self.event_buffer:
            self.event_buffer.set_intent(intent.value, song_sec=song_sec)
        self._current_intent = intent
        self._intent_commits += 1
        await self.effect_controller.change_effect(intent)

    async def on_note(self):
        """The overlay's 24-channel chase, one step per beat.

        Room-aligned like everything else the audience can see: it is driven by
        a beat, which the engine detects as the audio arrives, so it waits the
        whole playback delay.  Un-queued it advanced fourteen seconds ahead of
        the music -- visibly uncorrelated with it, and invisible in a report.
        """
        dmx_data = [0] * 24
        self._note_counter = (self._note_counter + 3) % 24
        dmx_data[self._note_counter] = 100
        if self.command_queue:
            await self.command_queue.enqueue(
                'overlay',
                lambda: self._show_light_bar(dmx_data))
        else:
            self.overlay_client.update_overlay_data(OverlayEffect.LIGHT_BAR_24,
                                                    dmx_data)
        logging.info('[engine] note detected')

    async def _show_light_bar(self, dmx_data: list) -> None:
        self.overlay_client.update_overlay_data(OverlayEffect.LIGHT_BAR_24,
                                                dmx_data)

    async def on_100ms_callback(self):
        if not self.analyser.is_song_playing():
            return
        if self.analyser.get_seconds_since_last_beat() > _BEAT_ABSENCE_SEC:
            if not self._atmospheric_sent:
                self._atmospheric_sent = True
                # The timer describes NOW, so nothing has been spent on it and
                # it waits the whole playback delay -- through the same stream
                # as every decision, which is what keeps the two in order.
                await self._commit_intent(LightIntent.ATMOSPHERIC, 0.0)

    async def on_1sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        if self.event_buffer and self.command_queue:
            self.event_buffer.set_timing_log(self.command_queue.get_timing_log())

    async def on_10sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        bpm = int(self.analyser.get_bpm())
        current_second = int(self.analyser.get_song_current_duration().total_seconds())
        intent_name = self._current_intent.name if self._current_intent else 'None'
        logging.info(f'[engine] == current state ==')
        logging.info(f'[engine]   realtime_bpm:    {bpm}')
        logging.info(f'[engine]   intent:          {intent_name}')
        logging.info(f'[engine]   current_second:  {current_second}')
        logging.info(f'[engine]   last_effect:     {self.effect_controller.last_effect}')
        self._log_chain_latency()
