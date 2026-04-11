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

if TYPE_CHECKING:
    from lib.engine.event_buffer import EventBuffer

# BPM thresholds for intent classification.
# These are arbitrary starting points — tune them with real DJ sets.
_CALM_MAX_BPM = 90.0
_GROOVE_MAX_BPM = 120.0
_ENERGY_MAX_BPM = 145.0


def _bpm_to_intent(bpm: float) -> LightIntent:
    if bpm < _CALM_MAX_BPM:
        return LightIntent.CALM
    if bpm < _GROOVE_MAX_BPM:
        return LightIntent.GROOVE
    if bpm < _ENERGY_MAX_BPM:
        return LightIntent.ENERGY
    return LightIntent.PEAK


class LightEngine(IMusicAnalyserHandler):
    def __init__(self,
                 midi_client: MidiClient,
                 os2l_client: Os2lClient,
                 overlay_client: OverlayClient,
                 effect_controller: EffectController,
                 command_queue: DelayedCommandQueue | None = None,
                 event_buffer: EventBuffer | None = None):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.overlay_client: OverlayClient = overlay_client
        self.effect_controller: EffectController = effect_controller
        self.command_queue: DelayedCommandQueue | None = command_queue
        self.event_buffer: EventBuffer | None = event_buffer
        self.analyser: MusicAnalyser = None
        self._note_counter: int = 0
        self._needs_initial_effect: bool = False

    def set_analyser(self, analyser: MusicAnalyser):
        self.analyser: MusicAnalyser = analyser

    def on_sound_start(self):
        logging.info('[engine] sound start')
        self.midi_client.on_sound_start()
        self.overlay_client.deactivate_all()
        self.os2l_client.on_sound_start(0, 0, 20000, 120)
        if self.event_buffer:
            self.event_buffer.set_playing(True)
        self._needs_initial_effect = True

    def on_sound_stop(self):
        logging.info('[engine] sound stop')
        self.midi_client.on_sound_stop()
        self.os2l_client.on_sound_stop()
        self.effect_controller.reset_state()
        self.overlay_client.deactivate_all()
        if self.event_buffer:
            self.event_buffer.set_playing(False)

    async def on_cycle(self):
        await self.effect_controller.process_effects()
        self.overlay_client.flush_messages()

    async def on_onset(self):
        pass

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        current_second = self.analyser.get_song_current_duration().total_seconds()
        intent = _bpm_to_intent(bpm)
        logging.info(f'[engine] [{current_second:.2f} sec] beat detected, change={bpm_changed}, beat_number={beat_number}, bpm={bpm:.2f}, intent={intent.name}')
        if self.event_buffer:
            self.event_buffer.add_beat(bpm, 0.5, bpm_changed)
            self.event_buffer.set_intent(intent.value)
        if self._needs_initial_effect:
            self._needs_initial_effect = False
            await self.effect_controller.change_effect(intent)
        # Capture locals in closure — they must not be read by reference after enqueue
        _change, _pos, _bpm = bpm_changed, beat_number, bpm
        if self.command_queue:
            await self.command_queue.enqueue(
                'beat',
                lambda: self.os2l_client.send_beat(change=_change, pos=_pos, bpm=_bpm, strength=0.5)
            )
        else:
            await self.os2l_client.send_beat(change=bpm_changed, pos=beat_number, bpm=bpm, strength=0.5)

    async def on_note(self):
        dmx_data = [0] * 24
        self._note_counter = (self._note_counter + 3) % 24
        dmx_data[self._note_counter] = 100
        self.overlay_client.update_overlay_data(OverlayEffect.LIGHT_BAR_24, dmx_data)
        logging.info(f'[engine] note detected')

    async def on_section_change(self) -> None:
        logging.info(f"[engine] audio section change detected")
        bpm = self.analyser.get_bpm()
        intent = _bpm_to_intent(bpm)
        _intent = intent
        if self.command_queue:
            await self.command_queue.enqueue(
                'section_change',
                lambda: self.effect_controller.change_effect(_intent)
            )
        else:
            await self.effect_controller.change_effect(intent)

    async def on_100ms_callback(self):
        if not self.analyser.is_song_playing():
            return

    async def on_1sec_callback(self):
        if not self.analyser.is_song_playing():
            return

    async def on_10sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        bpm = int(self.analyser.get_bpm())
        current_second = int(self.analyser.get_song_current_duration().total_seconds())
        logging.info(f"[engine] == current song info ==")
        logging.info(f"[engine]   realtime_bpm:    {bpm}")
        logging.info(f"[engine]   current_second:  {current_second}")
        logging.info(f"[engine]   last_effect:     {self.effect_controller.last_effect}")
