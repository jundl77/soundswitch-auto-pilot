from clients.midi_client import MidiClient


class MusicAnalyserHandler:
    def __init__(self, midi_client: MidiClient):
        self.midi_client: MidiClient = midi_client

    def on_new_song(self):
        print('new song')

    def on_onset(self):
        print('onset')

    def on_beat(self, beat: float, current_bpm: float) -> None:
        print(f'beat {beat}, bpm: {current_bpm}')

