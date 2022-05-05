import rtmidi
import time
import lib.clients.midi_message as mm


class MidiClient:
    def __init__(self, midi_port_index: int):
        self.midi_out = rtmidi.MidiOut()
        self.available_ports = self.midi_out.get_ports()
        self.midi_port_index: int = midi_port_index
        self.port_name = self.midi_out.get_port_name(self.midi_port_index)

    def list_devices(self):
        print('=== Midi ports ===')
        for i in range(0, len(self.available_ports)):
            print(f'index: {i}, port: {self.available_ports[i]}')

    def start(self):
        assert 0 <= self.midi_port_index < len(self.available_ports), "midi_port_index does not reference a valid port"
        print(f"Using midi port: {self.port_name}")
        self.midi_out.open_port(self.midi_port_index)
        assert self.midi_out.is_port_open(), f"Unable to open midi port '{self.port_name}', (index={self.midi_port_index})"
        self.midi_out.send_message(mm.MIDI_MSG_LINK_TOGGLE)

    def send_beat(self):
        self.midi_out.send_message(mm.MIDI_MSG_LINK_BPM_TAP_ON)
        print(f'[{self.port_name}] send BPM TAP')
        time.sleep(0.01)
        self.midi_out.send_message(mm.MIDI_MSG_LINK_BPM_TAP_OFF)
