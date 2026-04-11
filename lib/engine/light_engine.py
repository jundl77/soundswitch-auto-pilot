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

# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------
# These thresholds are a starting point — tune against real DJ sets.
# onset_density = onsets/sec over a 3-second rolling window (from aubio).
#
# Structure of a typical EDM track and how we detect it:
#   ATMOSPHERIC — barely any beats or onsets (intro, full breakdown, outro)
#   BREAKDOWN   — beat present but very sparse onsets (melodic, stripped)
#   GROOVE      — moderate onsets at mid-tempo (main dance-floor loop)
#   BUILDUP     — onset density rising; moderately high BPM (pre-drop tension)
#   DROP        — onset density spikes hard (bass, kick, hat all firing)
#   PEAK        — sustained very high BPM + high density (post-drop peak)

_ATMOSPHERIC_MAX_BPM    = 75.0
_BREAKDOWN_MAX_DENSITY  = 1.2   # onsets/sec — sparse = breakdown feel
_GROOVE_MAX_BPM         = 118.0
_BUILDUP_MAX_BPM        = 138.0
_DROP_MIN_DENSITY       = 4.0   # density spike triggers DROP regardless of BPM
_PEAK_MIN_BPM           = 138.0


def _classify_intent(bpm: float, onset_density: float) -> LightIntent:
    """Map (BPM, onset_density) → LightIntent.

    Priority order matters: density spike always wins (DROP), then BPM
    thresholds narrow down the remaining cases.
    """
    if bpm < _ATMOSPHERIC_MAX_BPM:
        return LightIntent.ATMOSPHERIC
    if onset_density >= _DROP_MIN_DENSITY and bpm >= 100:
        return LightIntent.DROP
    if bpm >= _PEAK_MIN_BPM:
        return LightIntent.PEAK
    if onset_density < _BREAKDOWN_MAX_DENSITY:
        return LightIntent.BREAKDOWN
    if bpm >= _BUILDUP_MAX_BPM:
        return LightIntent.BUILDUP
    if bpm >= _GROOVE_MAX_BPM:
        return LightIntent.BUILDUP
    return LightIntent.GROOVE


# ---------------------------------------------------------------------------
# LightEngine
# ---------------------------------------------------------------------------

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
        onset_density = self.analyser.get_onset_density()
        intent = _classify_intent(bpm, onset_density)
        logging.info(
            f'[engine] [{current_second:.2f}s] beat #{beat_number}  '
            f'bpm={bpm:.1f}  onsets/s={onset_density:.2f}  intent={intent.name}'
        )
        if self.event_buffer:
            self.event_buffer.add_beat(bpm, onset_density / 10.0, bpm_changed)
            self.event_buffer.set_intent(intent.value)
        if self._needs_initial_effect:
            self._needs_initial_effect = False
            await self.effect_controller.change_effect(intent)
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
        logging.info('[engine] note detected')

    async def on_section_change(self) -> None:
        logging.info('[engine] audio section change detected')
        bpm = self.analyser.get_bpm()
        onset_density = self.analyser.get_onset_density()
        intent = _classify_intent(bpm, onset_density)
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
        onset_density = self.analyser.get_onset_density()
        current_second = int(self.analyser.get_song_current_duration().total_seconds())
        intent = _classify_intent(float(bpm), onset_density)
        logging.info(f'[engine] == current state ==')
        logging.info(f'[engine]   realtime_bpm:    {bpm}')
        logging.info(f'[engine]   onset_density:   {onset_density:.2f} /s')
        logging.info(f'[engine]   intent:          {intent.name}')
        logging.info(f'[engine]   current_second:  {current_second}')
        logging.info(f'[engine]   last_effect:     {self.effect_controller.last_effect}')
