import logging
from lib.clients.midi_client import MidiClient


class MusicAnalyserHandler:
    def __init__(self, midi_client: MidiClient):
        self.midi_client: MidiClient = midi_client
        self.analyser: "MusicAnalyser" = None

    def set_analyser(self, analyser: "MusicAnalyser"):
        self.analyser: "MusicAnalyser" = analyser

    def on_sound_start(self):
        logging.info('sound start')

    def on_sound_stop(self):
        logging.info('sound stop')

    async def on_onset(self):
        logging.info('onset')

    async def on_beat(self, beat: float) -> None:
        bpm = self.analyser.get_bpm()
        logging.info(f'beat {beat}, bpm: {bpm}')
        await self.midi_client.send_beat()

