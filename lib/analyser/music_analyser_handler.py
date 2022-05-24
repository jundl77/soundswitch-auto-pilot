import logging
from lib.clients.midi_client import MidiClient
from lib.clients.os2l_client import Os2lClient


class MusicAnalyserHandler:
    def __init__(self, midi_client: MidiClient, os2l_client: Os2lClient):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.analyser: "MusicAnalyser" = None

    def set_analyser(self, analyser: "MusicAnalyser"):
        self.analyser: "MusicAnalyser" = analyser

    def on_sound_start(self):
        logging.info('sound start')
        self.os2l_client.on_sound_start(0, 0, 1000, 120)

    def on_sound_stop(self):
        logging.info('sound stop')
        self.os2l_client.on_sound_stop()

    async def on_onset(self):
        pass

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        await self.os2l_client.send_beat(change=bpm_changed, pos=beat_number, bpm=bpm, strength=0)

