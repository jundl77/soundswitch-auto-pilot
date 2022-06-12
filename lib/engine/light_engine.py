import logging
from lib.clients.midi_client import MidiClient
from lib.clients.os2l_client import Os2lClient
from lib.clients.spotify_client import SpotifyClient
from lib.analyser.music_analyser import MusicAnalyser
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler


class LightEngine(IMusicAnalyserHandler):
    def __init__(self, midi_client: MidiClient, os2l_client: Os2lClient, spotify_client: SpotifyClient):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.spotify_client: SpotifyClient = spotify_client
        self.analyser: MusicAnalyser = None

    def set_analyser(self, analyser: MusicAnalyser):
        self.analyser: MusicAnalyser = analyser

    def on_sound_start(self):
        logging.info('sound start')
        spotify_song_analysis = self.spotify_client.get_current_song_analysis()
        self.midi_client.on_sound_start()
        self.os2l_client.on_sound_start(0, 0, 20000, 120)

    def on_sound_stop(self):
        logging.info('sound stop')
        self.midi_client.on_sound_stop()
        self.os2l_client.on_sound_stop()

    async def on_onset(self):
        pass

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        await self.os2l_client.send_beat(change=bpm_changed, pos=beat_number, bpm=bpm, strength=0)

