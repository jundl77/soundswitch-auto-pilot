from lib.clients.midi_client import MidiClient


class MusicAnalyserHandler:
    def __init__(self, midi_client: MidiClient):
        self.midi_client: MidiClient = midi_client
        self.analyser: MusicAnalyser = None

    def set_analyser(self, analyser: "MusicAnalyser"):
        self.analyser: MusicAnalyser = analyser

    def on_sound_start(self):
        print('sound start')

    def on_sound_stop(self):
        print('sound stop')

    def on_onset(self):
        print('onset')

    def on_beat(self, beat: float) -> None:
        bpm = self.analyser.get_bpm()
        print(f'beat {beat}, bpm: {bpm}')

