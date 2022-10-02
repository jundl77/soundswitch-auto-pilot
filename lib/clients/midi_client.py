import rtmidi
import time
import logging
import asyncio
import datetime
import lib.clients.midi_message as mm
from lib.clients.midi_message import MidiChannel
from typing import List
from enum import Enum


class EffectAction(Enum):
    ACTIVATE = 1,
    DEACTIVATE = 2


class DelayedEffect:
    def __init__(self,
                 start_ts: datetime.datetime,
                 effect: MidiChannel,
                 action: EffectAction,
                 duration: datetime.timedelta):
        self.start_ts: datetime.datetime = start_ts
        self.effect: MidiChannel = effect
        self.action: EffectAction = action
        self.duration: datetime.timedelta = duration
        self.is_done: bool = False

    def __repr__(self):
        return f"DelayedEffect(start_ts={self.start_ts.time()}, effect={self.effect.name}, action={self.action.name}, duration={self.duration.total_seconds()} sec)"


class MidiClient:
    def __init__(self, midi_port_index: int):
        self.midi_out = rtmidi.MidiOut()
        self.available_ports = self.midi_out.get_ports()
        self.midi_port_index: int = midi_port_index
        self.port_name = self.midi_out.get_port_name(self.midi_port_index)
        self.soundswitch_is_paused: bool = True
        self._pending_effects: List[DelayedEffect] = list()

    def list_devices(self):
        print('=== Midi ports ===')
        for i in range(0, len(self.available_ports)):
            print(f'index: {i}, port: {self.available_ports[i]}')

    def start(self):
        assert 0 <= self.midi_port_index < len(self.available_ports), "midi_port_index does not reference a valid port"
        logging.info(f"[midi] using midi port: {self.port_name}")
        self.midi_out.open_port(self.midi_port_index)
        assert self.midi_out.is_port_open(), f"Unable to open midi port '{self.port_name}', (index={self.midi_port_index})"

    def stop(self):
        logging.info(f'[midi] stopping midi client')
        logging.info(f'[midi] setting soundswitch intensities down')
        self.on_sound_stop()

    def on_sound_start(self):
        self._set_intensities(1)
        if self.soundswitch_is_paused:
            self.midi_out.send_message(mm.get_midi_msg_on(MidiChannel.PLAY_PAUSE))  # unpause
            self.soundswitch_is_paused = False

    def on_sound_stop(self):
        self._set_intensities(0)
        if not self.soundswitch_is_paused:
            time.sleep(0.2)  # we need to give soundswitch some time to process the previous message
            self.midi_out.send_message(mm.get_midi_msg_on(MidiChannel.PLAY_PAUSE))  # pause
            self.soundswitch_is_paused = True

    async def on_100ms_callback(self):
        await self._process_delayed_effects()

    async def set_autoloop(self, auto_loop: MidiChannel):
        logging.info(f'[midi] set autoloop: {auto_loop.name}')
        self.midi_out.send_message(mm.get_midi_msg_on(auto_loop))
        await asyncio.sleep(0.01)
        self.midi_out.send_message(mm.get_midi_msg_off(auto_loop))

    async def set_special_effect(self, special_effect: MidiChannel, duration_sec: int):
        logging.info(f'[midi] set special effect: {special_effect.name}')
        now = datetime.datetime.now()
        self.midi_out.send_message(mm.get_midi_msg_on(special_effect))
        self._pending_effects.append(DelayedEffect(start_ts=now,
                                                   effect=special_effect,
                                                   action=EffectAction.DEACTIVATE,
                                                   duration=datetime.timedelta(seconds=duration_sec)))

    async def set_color_override(self, color: MidiChannel):
        await self.clear_color_overrides()
        await asyncio.sleep(0.01)
        self.midi_out.send_message(mm.get_midi_msg_on(color))

    async def clear_color_overrides(self):
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_1))
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_2))
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_3))
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_4))
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_5))
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_6))
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_7))
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_8))
        self.midi_out.send_message(mm.get_midi_msg_off(MidiChannel.COLOR_OVERRIDE_9))

    async def _process_delayed_effects(self):
        now = datetime.datetime.now()
        for effect in self._pending_effects:
            if not effect.is_done and now - effect.start_ts > effect.duration:
                logging.info(f'[midi] activated delayed effect: {effect}')
                if effect.action == EffectAction.ACTIVATE:
                    self.midi_out.send_message(mm.get_midi_msg_on(effect.effect))
                elif effect.action == EffectAction.DEACTIVATE:
                    self.midi_out.send_message(mm.get_midi_msg_off(effect.effect))
                else:
                    raise RuntimeError(f"unknown EffectAction: {effect.action}")
                effect.is_done = True

        # delete all finished coros
        i = 0
        while i < len(self._pending_effects):
            if self._pending_effects[i].is_done:
                del self._pending_effects[i]
            else:
                i += 1

    def _set_intensities(self, value: int):
        assert 0 <= value <= 1, "intensity value should be in [0, 1]"
        self.midi_out.send_message(mm.get_autoloop_intensity_msg(value))
        self.midi_out.send_message(mm.get_scripted_track_intensity_msg(0))
        self.midi_out.send_message(mm.get_group_1_intensity_msg(value))
        self.midi_out.send_message(mm.get_group_2_intensity_msg(value))
        self.midi_out.send_message(mm.get_group_3_intensity_msg(value))
        self.midi_out.send_message(mm.get_group_4_intensity_msg(value))
