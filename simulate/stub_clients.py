"""
Drop-in stub replacements for all hardware-touching clients.
Each stub records timestamped events instead of sending to real hardware,
allowing timing validation after a simulation run.
"""

import time
import logging
from lib.clients.midi_message import MidiChannel
from lib.clients.overlay_definitions import OverlayEffect

log = logging.getLogger(__name__)


def _event(label: str, **kwargs) -> dict:
    return {'label': label, 'time': time.monotonic(), **kwargs}


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------

class StubMidiClient:
    def __init__(self, event_buffer=None):
        self.events: list[dict] = []
        self._pending_effects = []  # kept for interface compat
        self._event_buffer = event_buffer

    def list_devices(self): pass
    def start(self): pass
    def stop(self): pass
    def on_sound_start(self): pass
    def on_sound_stop(self): pass
    def set_all_intensities(self, value: int): pass
    def set_group_intensities(self, group: int, value: int): pass

    async def on_100ms_callback(self): pass

    async def set_autoloop(self, auto_loop: MidiChannel):
        e = _event('set_autoloop', channel=auto_loop.name)
        self.events.append(e)
        if self._event_buffer:
            self._event_buffer.add_effect(auto_loop.name, 'AUTOLOOP')
        log.info(f'[stub_midi] set_autoloop: {auto_loop.name}')

    async def set_special_effect(self, special_effect: MidiChannel, duration_sec: int):
        e = _event('set_special_effect', channel=special_effect.name, duration_sec=duration_sec)
        self.events.append(e)
        if self._event_buffer:
            self._event_buffer.add_effect(special_effect.name, 'SPECIAL_EFFECT')
        log.info(f'[stub_midi] set_special_effect: {special_effect.name}')

    async def set_color_override(self, color: MidiChannel):
        e = _event('set_color_override', channel=color.name)
        self.events.append(e)
        log.info(f'[stub_midi] set_color_override: {color.name}')

    async def clear_color_overrides(self):
        pass


# ---------------------------------------------------------------------------
# OS2L
# ---------------------------------------------------------------------------

class StubOs2lClient:
    def __init__(self, event_buffer=None):
        self.events: list[dict] = []
        self._event_buffer = event_buffer

    def start(self): pass
    def stop(self): pass
    def set_analyser(self, analyser): pass
    def on_sound_start(self, time_elapsed_ms, beats_to_first_downbeat, first_downbeat_ms, bpm): pass
    def on_sound_stop(self): pass

    async def send_beat(self, change: bool, pos: int, bpm: float, strength: float):
        e = _event('beat', change=change, pos=pos, bpm=bpm, strength=strength)
        self.events.append(e)
        if self._event_buffer:
            self._event_buffer.add_beat(bpm, strength, change)
        log.debug(f'[stub_os2l] beat: pos={pos}, bpm={bpm:.1f}, strength={strength:.2f}')


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

class StubOverlayClient:
    def __init__(self):
        self.events: list[dict] = []
        # Mirror the real OverlayClient's effects_to_overlay_index so assertions don't fail
        self.effects_to_overlay_index = {effect: i for i, effect in enumerate(OverlayEffect)}

    def start(self): pass
    def stop(self): pass

    def update_overlay_data(self, effect: OverlayEffect, dmx_data: list):
        e = _event('overlay_update', effect=effect.name)
        self.events.append(e)

    def toggle_overlay(self, effect: OverlayEffect): pass
    def activate_overlay(self, effect: OverlayEffect): pass
    def deactivate_overlay(self, effect: OverlayEffect): pass
    def deactivate_all(self): pass
    def clear_all(self): pass

    def flush_messages(self):
        e = _event('overlay_flush')
        self.events.append(e)


# ---------------------------------------------------------------------------
# Spotify
# ---------------------------------------------------------------------------

class StubSpotifyClient:
    """Returns no track analysis. The pipeline degrades gracefully when Spotify is absent."""

    def __init__(self):
        self._engine = None

    def start(self): pass
    def stop(self): pass
    def set_engine(self, engine): self._engine = engine

    def get_current_track_analysis(self):
        return None

    async def check_for_track_changes(self, track_analysis, current_second: float):
        pass
