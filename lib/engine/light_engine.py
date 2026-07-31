from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from lib.engine.effect_controller import EffectController
from lib.engine.delayed_command_queue import DelayedCommandQueue
from lib.engine.effect_definitions import LightIntent
from lib.clients.midi_client import MidiClient
from lib.clients.os2l_client import Os2lClient
from lib.clients.overlay_client import OverlayClient, OverlayEffect
from lib.analyser.music_analyser import MusicAnalyser
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.clock import Clock, SYSTEM_CLOCK

if TYPE_CHECKING:
    from lib.engine.event_buffer import EventBuffer

_BEAT_ABSENCE_SEC = 2.5

_PEAK_PROMOTION_BEATS = 32


class LightEngine(IMusicAnalyserHandler):
    def __init__(self,
                 midi_client: MidiClient,
                 os2l_client: Os2lClient,
                 overlay_client: OverlayClient,
                 effect_controller: EffectController,
                 command_queue: DelayedCommandQueue | None = None,
                 event_buffer: EventBuffer | None = None,
                 look_ahead_sec: float = 0.0,
                 clock: Clock = SYSTEM_CLOCK):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.overlay_client: OverlayClient = overlay_client
        self.effect_controller: EffectController = effect_controller
        self.command_queue: DelayedCommandQueue | None = command_queue
        self.event_buffer: EventBuffer | None = event_buffer
        self.analyser: MusicAnalyser = None
        self._look_ahead_sec: float = look_ahead_sec
        self._clock: Clock = clock
        self._note_counter: int = 0
        self._atmospheric_sent: bool = False
        self._current_intent: LightIntent | None = None
        self._published_bpm: dict = {}
        self._beats_in_current_intent: int = 0

    def set_analyser(self, analyser: MusicAnalyser):
        self.analyser: MusicAnalyser = analyser

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
        self._beats_in_current_intent = 0

    async def on_cycle(self):
        await self.effect_controller.process_effects()
        self.overlay_client.flush_messages()

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        current_second = self.analyser.get_song_current_duration().total_seconds()
        rms_energy = self.analyser.get_rms_energy()

        logging.info(
            f'[engine] [{current_second:.2f}s] beat #{beat_number}  bpm={bpm:.1f}'
        )
        if self.event_buffer:
            self.event_buffer.add_beat(bpm, bpm_changed, rms=rms_energy)

        self._atmospheric_sent = False

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

    async def _apply_intent(self, intent: LightIntent) -> None:
        """The single path that moves the stage: timeline, guard state and MIDI
        together, so nothing can light an intent the guards do not know about."""
        if self.event_buffer:
            self.event_buffer.set_intent(intent.value)
        self._beats_in_current_intent = 0
        self._current_intent = intent
        await self.effect_controller.change_effect(intent)

    async def _enqueue_or_apply(self, label: str, intent: LightIntent) -> None:
        if self.command_queue:
            await self.command_queue.enqueue(label, lambda: self._apply_intent(intent))
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
