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
        logging.info('[engine] sound start')
        spotify_song_analysis = self.spotify_client.get_current_song_analysis()

        first_downbeat_ms = 20000
        bpm = 120
        beats_to_first_downbeat = 0
        time_elapsed_ms = 0
        if spotify_song_analysis:
            first_downbeat_ms = spotify_song_analysis.first_downbeat_ms
            bpm = spotify_song_analysis.bpm
            beats_to_first_downbeat = spotify_song_analysis.beats_to_first_downbeat
            time_elapsed_ms = spotify_song_analysis.progress_ms
            self.analyser.inject_spotify_track_analysis(spotify_song_analysis)

        self.midi_client.on_sound_start()
        self.os2l_client.on_sound_start(time_elapsed_ms, beats_to_first_downbeat, first_downbeat_ms, bpm)

    def on_sound_stop(self):
        logging.info('[engine] sound stop')
        self.midi_client.on_sound_stop()
        self.os2l_client.on_sound_stop()

    async def on_onset(self):
        pass

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        await self.os2l_client.send_beat(change=bpm_changed, pos=beat_number, bpm=bpm, strength=0)

