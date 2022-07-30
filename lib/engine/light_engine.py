import logging
from typing import Optional
from lib.engine.autoloop_controler import AutoloopController
from lib.clients.midi_client import MidiClient
from lib.clients.os2l_client import Os2lClient
from lib.clients.spotify_client import SpotifyClient, SpotifyTrackAnalysis
from lib.analyser.music_analyser import MusicAnalyser
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler


class LightEngine(IMusicAnalyserHandler):
    def __init__(self,
                 midi_client: MidiClient,
                 os2l_client: Os2lClient,
                 spotify_client: SpotifyClient,
                 autoloop_controller: AutoloopController):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.spotify_client: SpotifyClient = spotify_client
        self.autoloop_controller: AutoloopController = autoloop_controller
        self.analyser: MusicAnalyser = None
        self.spotify_track_analysis: Optional[SpotifyTrackAnalysis] = None

    def set_analyser(self, analyser: MusicAnalyser):
        self.analyser: MusicAnalyser = analyser

    def on_sound_start(self):
        logging.info('[engine] sound start')
        self.spotify_track_analysis = self.spotify_client.get_current_track_analysis()

        first_downbeat_ms = 20000
        bpm = 120
        beats_to_first_downbeat = 0
        time_elapsed_ms = 0
        if self.spotify_track_analysis:
            first_downbeat_ms = self.spotify_track_analysis.first_downbeat_ms
            bpm = self.spotify_track_analysis.bpm
            beats_to_first_downbeat = self.spotify_track_analysis.beats_to_first_downbeat
            time_elapsed_ms = self.spotify_track_analysis.progress_ms
            self.analyser.inject_spotify_track_analysis(self.spotify_track_analysis)

        self.midi_client.on_sound_start()
        self.os2l_client.on_sound_start(time_elapsed_ms, beats_to_first_downbeat, first_downbeat_ms, bpm)
        self._log_current_track_info()

    def on_sound_stop(self):
        logging.info('[engine] sound stop')
        self.midi_client.on_sound_stop()
        self.os2l_client.on_sound_stop()
        self.autoloop_controller.reset_state()

    async def on_onset(self):
        pass

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        current_second = self.analyser.get_song_current_duration().total_seconds()
        beat_strength = self._calculate_current_beat_strength(current_second)
        await self.os2l_client.send_beat(change=bpm_changed, pos=beat_number, bpm=bpm, strength=beat_strength)
        logging.info(f'[engine] [{current_second:.2f}] beat detected, change={bpm_changed}, beat_number={beat_number}, bpm={bpm}, strength={beat_strength}')

    async def on_spotify_track_changed(self, spotify_track_analysis: SpotifyTrackAnalysis) -> None:
        logging.info(f"[engine] spotify track change detected")
        self._handle_spotify_state_change(spotify_track_analysis)

    async def on_spotify_track_progress_changed(self, spotify_track_analysis: SpotifyTrackAnalysis) -> None:
        logging.info(f"[engine] spotify track progress change detected")
        self._handle_spotify_state_change(spotify_track_analysis)

    async def on_100ms_callback(self):
        current_second = float(self.analyser.get_song_current_duration().total_seconds())
        await self.autoloop_controller.check_autoloops(current_second, self.spotify_track_analysis)

    async def on_1sec_callback(self):
        pass

    async def on_10sec_callback(self):
        current_second = float(self.analyser.get_song_current_duration().total_seconds())
        await self.spotify_client.check_for_track_changes(self.spotify_track_analysis, current_second)
        self._log_current_track_info()

    def _handle_spotify_state_change(self, spotify_track_analysis: SpotifyTrackAnalysis):
        self.analyser.inject_spotify_track_analysis(spotify_track_analysis)
        self.spotify_track_analysis = spotify_track_analysis
        self.autoloop_controller.reset_state()
        self._log_current_track_info()

    def _calculate_current_beat_strength(self, current_second: float) -> float:
        if not self.spotify_track_analysis:
            return 0
        current_second = int(current_second)
        if len(self.spotify_track_analysis.beat_strengths_by_sec) <= current_second:
            return 0
        return self.spotify_track_analysis.beat_strengths_by_sec[current_second]

    def _log_current_track_info(self):
        if not self.analyser.is_song_playing():
            return

        bpm = int(self.analyser.get_bpm())
        current_second = int(self.analyser.get_song_current_duration().total_seconds())

        logging.info(f"[engine] == current song info ==")
        if self.spotify_track_analysis:
            logging.info(f"[engine]   name:                    {self.spotify_track_analysis.track_name}")
            logging.info(f"[engine]   artists:                 [{', '.join(self.spotify_track_analysis.artists)}]")
            logging.info(f"[engine]   album:                   {self.spotify_track_analysis.album_name}")
            logging.info(f"[engine]   first_downbeat_count:    {self.spotify_track_analysis.beats_to_first_downbeat}")
            logging.info(f"[engine]   first_downbeat_ms:       {self.spotify_track_analysis.first_downbeat_ms}")
            logging.info(f"[engine]   audio_sections_start_ts: {[s.section_start_sec for s in self.spotify_track_analysis.audio_sections]}")
        logging.info(f"[engine]   bpm:                     {bpm}")
        logging.info(f"[engine]   current_second:          {current_second}")

