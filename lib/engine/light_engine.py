from __future__ import annotations
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
        self._playback_delay_sec: float = playback_delay_sec
        self._clock: Clock = clock
        self._note_counter: int = 0
        self._atmospheric_sent: bool = False
        self._current_intent: LightIntent | None = None
        # What the committer has decided, which runs a queue delay ahead of what
        # the stage is showing.
        self._decided_intent: LightIntent | None = None
        self._published_bpm: dict = {}
        self._bars_in_current_intent: int = 0
        self._intent_commits: int = 0
        self._audio_sec: float = 0.0
        self._latency_logged_at: float | None = None
        self._latency_clamped: bool = False
        self._log_chain_latency()

    def set_analyser(self, analyser: MusicAnalyser):
        self.analyser: MusicAnalyser = analyser

    @property
    def current_intent(self) -> LightIntent | None:
        return self._current_intent

    @property
    def intent_commits(self) -> int:
        return self._intent_commits

    def on_sound_start(self):
        logging.info('[engine] sound start')
        self.midi_client.on_sound_start()
        self.overlay_client.deactivate_all()
        self.os2l_client.on_sound_start(0, 0, 20000, 120)
        if self.event_buffer:
            self.event_buffer.set_playing(True)

    def on_sound_stop(self):
        logging.info('[engine] sound stop')
        self.midi_client.on_sound_stop()
        self.os2l_client.on_sound_stop()
        self.effect_controller.reset_state()
        self.overlay_client.deactivate_all()
        if self.event_buffer:
            self.event_buffer.set_playing(False)
        self._atmospheric_sent = False
        self._current_intent = None
        self._decided_intent = None
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

            if (self._decided_intent is LightIntent.DROP
                    and intent is LightIntent.DROP
                    and self._bars_in_current_intent >= PEAK_PROMOTION_BARS):
                logging.info(f'[engine] sustained DROP over '
                             f'{self._bars_in_current_intent} bars — promoting to PEAK')
                intent = LightIntent.PEAK
            elif (self._decided_intent is LightIntent.PEAK
                    and intent is LightIntent.DROP):
                # Absorbed, so the pair cannot oscillate and the timeline keeps
                # reading the PEAK the room is actually looking at.
                continue

            if intent is self._decided_intent:
                continue

            logging.info(
                f'[engine] bar {decision.bar} @ {decision.start_sec:.2f}s  '
                f'{decision.label} → {intent.name}')
            await self._enqueue_or_apply('intent', intent,
                                         delay_sec=self._intent_delay_sec())

    def _intent_delay_sec(self) -> float:
        """B1: what the chain has not already spent of the playback delay.

        Measured rather than assumed -- the decoder's half is proportional to
        bar length, so it moves with the tempo and ranges 12.11 s to 16.37 s
        across the corpus at lag 2.  Clamped at zero, which is #154's accepted
        lateness on slow material rather than a budget nobody can meet.
        """
        if self.section_decoder is None:
            return self._playback_delay_sec
        delay = self._playback_delay_sec - self.section_decoder.chain_latency_sec
        if delay < 0.0:
            if not self._latency_clamped:
                self._latency_clamped = True
                logging.warning(
                    f'[engine] chain latency '
                    f'{self.section_decoder.chain_latency_sec:.2f}s exceeds the '
                    f'{self._playback_delay_sec:.2f}s playback delay — intents '
                    f'commit {-delay:.2f}s late (slow tempo, accepted)')
            return 0.0
        if self._latency_clamped:
            self._latency_clamped = False
            logging.info('[engine] chain latency is back inside the playback delay')
        return delay

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
            f'{latency - decoder.feature_latency_sec:.2f}s decoder '
            f'(lag {decoder.params.lag_bars} × {decoder.bar_sec:.3f}s bars) | '
            f'playback delay {self._playback_delay_sec:.2f}s → queue delay '
            f'{max(0.0, self._playback_delay_sec - latency):.2f}s — ensure '
            f'dmx-enttec-node playback_delay_seconds matches')

    async def _apply_intent(self, intent: LightIntent) -> None:
        """The single path that moves the stage: timeline and MIDI together, so
        nothing can light an intent the timeline does not know about.

        Runs when the queue fires, which is the instant the room hears the audio
        this was decided from -- hence the split from ``_decided_intent``, which
        is where the decision stream has got to.  Deciding against the stage
        would re-commit every bar for a whole playback delay.
        """
        if self.event_buffer:
            self.event_buffer.set_intent(intent.value)
        self._current_intent = intent
        self._intent_commits += 1
        await self.effect_controller.change_effect(intent)

    async def _enqueue_or_apply(self, label: str, intent: LightIntent,
                                delay_sec: float | None = None) -> None:
        self._decided_intent = intent
        self._bars_in_current_intent = 0
        if self.command_queue:
            await self.command_queue.enqueue(
                label, lambda: self._apply_intent(intent), delay_sec=delay_sec)
        else:
            await self._apply_intent(intent)

    async def on_note(self):
        dmx_data = [0] * 24
        self._note_counter = (self._note_counter + 3) % 24
        dmx_data[self._note_counter] = 100
        self.overlay_client.update_overlay_data(OverlayEffect.LIGHT_BAR_24, dmx_data)
        logging.info('[engine] note detected')

    async def on_100ms_callback(self):
        if not self.analyser.is_song_playing():
            return
        if self.analyser.get_seconds_since_last_beat() > _BEAT_ABSENCE_SEC:
            if not self._atmospheric_sent:
                self._atmospheric_sent = True
                # The timer describes NOW and the chain has spent nothing on it,
                # so it waits the whole playback delay.
                await self._enqueue_or_apply('atmospheric', LightIntent.ATMOSPHERIC)

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
