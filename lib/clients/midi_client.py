import rtmidi
import time
import lib.clients.midi_message as mm
import logging
import asyncio


class MidiClient:
    def __init__(self, midi_port_index: int):
        self.midi_out = rtmidi.MidiOut()
        self.available_ports = self.midi_out.get_ports()
        self.midi_port_index: int = midi_port_index
        self.port_name = self.midi_out.get_port_name(self.midi_port_index)
        self.is_paused: bool = True

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
        if self.is_paused:
            self.midi_out.send_message(mm.MIDI_MSG_PAUSE_TOGGLE)  # unpause
            self.is_paused = False

    def on_sound_stop(self):
        self._set_intensities(0)
        if not self.is_paused:
            self.midi_out.send_message(mm.MIDI_MSG_PAUSE_TOGGLE)  # pause
            self.is_paused = True

    async def send_beat(self):
        self.midi_out.send_message(mm.MIDI_MSG_BPM_TAP_ON)
        logging.info('[midi] send BPM TAP')
        await asyncio.sleep(0.01)
        self.midi_out.send_message(mm.MIDI_MSG_BPM_TAP_OFF)

    def _set_intensities(self, value: int):
        assert 0 <= value <= 1, "intensity value should be in [0, 1]"
        self.midi_out.send_message(mm.get_autoloop_intensity_msg(value))
        self.midi_out.send_message(mm.get_scripted_track_intensity_msg(0))
        self.midi_out.send_message(mm.get_group_1_intensity_msg(value))
        self.midi_out.send_message(mm.get_group_2_intensity_msg(value))
        self.midi_out.send_message(mm.get_group_3_intensity_msg(value))
        self.midi_out.send_message(mm.get_group_4_intensity_msg(value))
