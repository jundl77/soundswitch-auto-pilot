import rtmidi
import time

import midi_message as mm

PORT_IDX = 1

midi_out = rtmidi.MidiOut()
available_ports = midi_out.get_ports()
print(available_ports)

midi_out.open_port(PORT_IDX)
midi_out.is_port_open()  # should be True

port_name = midi_out.get_port_name(PORT_IDX)

midi_out.send_message(mm.MIDI_MSG_LINK_TOGGLE)

while True:
    midi_out.send_message(mm.MIDI_MSG_LINK_BPM_TAP_ON)
    print(f'[{port_name}] send BPM TAP')
    time.sleep(0.1)
    midi_out.send_message(mm.MIDI_MSG_LINK_BPM_TAP_OFF)
    time.sleep(0.5)
