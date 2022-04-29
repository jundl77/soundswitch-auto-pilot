import rtmidi
import time
import queue
import threading
import midi_message as mm

PORT_IDX = 1

midi_out = rtmidi.MidiOut()
available_ports = midi_out.get_ports()
print(available_ports)

midi_out.open_port(PORT_IDX)
midi_out.is_port_open()  # should be True

port_name = midi_out.get_port_name(PORT_IDX)

midi_out.send_message(mm.MIDI_MSG_LINK_TOGGLE)

q = queue.Queue()


class MidiClient:
    def start(self):
        t.start()

    def _run(self):
        while True:
            q.get()
            self._send_beat()

    def signal_beat(self):
        q.put(1)

    def _send_beat(self):
        midi_out.send_message(mm.MIDI_MSG_LINK_BPM_TAP_ON)
        print(f'[{port_name}] send BPM TAP')
        time.sleep(0.01)
        midi_out.send_message(mm.MIDI_MSG_LINK_BPM_TAP_OFF)


t = threading.Thread(target=MidiClient()._run)
