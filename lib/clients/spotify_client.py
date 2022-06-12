import spotipy
import json
import logging
from typing import Optional, Dict, Tuple
from pathlib import Path
from spotipy.oauth2 import SpotifyOAuth


class SpotifyDetails:
    def __init__(self, spotify_client_id: str, spotify_client_secret: str):
        self.client_id: str = spotify_client_id
        self.client_secret: str = spotify_client_secret


class SpotifyTrackAnalysis:
    def __init__(self,
                 track_name: str,
                 progress_ms: int,
                 bpm: float,
                 beats_to_first_downbeat: int,
                 first_downbeat_ms: int):
        self.track_name: str = track_name
        self.progress_ms: int = progress_ms
        self.bpm: float = bpm
        self.beats_to_first_downbeat: int = beats_to_first_downbeat
        self.first_downbeat_ms: int = first_downbeat_ms


class SpotifyClient:
    def __init__(self):
        spotify_details = self._read_spotify_details_file()
        if spotify_details is not None:
            auth_manager = SpotifyOAuth(client_id=spotify_details.client_id,
                                        client_secret=spotify_details.client_secret,
                                        redirect_uri="http://localhost:8877/callback",
                                        scope="user-read-playback-state")
            self.spotify = spotipy.Spotify(auth_manager=auth_manager)
            self.is_active = True
            logging.info(f"[spotify] spotify song analysis is active")
        else:
            self.is_active = False
            logging.info(f"[spotify] spotify song analysis is inactive")

    def _read_spotify_details_file(self) -> Optional[SpotifyDetails]:
        spotify_details_file_path = (Path(__file__).parent.parent.parent / "spotify_details.json").absolute()
        if not spotify_details_file_path.is_file():
            logging.info(f"[spotify] {spotify_details_file_path} does not exist, will not use spotify")
            return None

        with open(spotify_details_file_path, 'r') as f:
            spotify_details_json = json.load(f)

        return SpotifyDetails(spotify_client_id=spotify_details_json['spotify_client_id'],
                              spotify_client_secret=spotify_details_json['spotify_client_secret'])

    def is_active(self) -> bool:
        return self.is_active

    def get_current_song_analysis(self) -> Optional[SpotifyTrackAnalysis]:
        if not self.is_active:
            return None

        current_playback = self.spotify.current_playback()
        if current_playback is None or not current_playback['is_playing']:
            return None

        track_name = current_playback['item']['name']
        track_id = current_playback['item']['id']
        progress_ms = int(current_playback['progress_ms'])

        audio_features = self.spotify.audio_features(track_id)  # high-level data, e.g. track danceability
        audio_analysis = self.spotify.audio_analysis(track_id)  # low-level data, e.g. all beats in track
        bpm = audio_analysis['track']['tempo']
        beats_to_first_downbeat, first_downbeat_ms = self._calculate_first_downbeat(audio_analysis)

        return SpotifyTrackAnalysis(track_name, progress_ms, bpm, beats_to_first_downbeat, first_downbeat_ms)

    def _calculate_first_downbeat(self, audio_analysis: Dict) -> Tuple[int, int]:
        sections = audio_analysis['sections']
        second_section_start = 0
        if len(sections) <= 1:
            return 0, 20000  # default values, pretend first downbeat is 20sec into song

        second_section_start = sections[1]['start']
        beat_count = 0
        first_downbeat_ms = 0
        for beat in audio_analysis['beats']:
            if beat['start'] < second_section_start:
                beat_count += 1
            else:
                first_downbeat_ms = beat['start'] * 1000
                beat_count += 1
                break
        return beat_count, int(first_downbeat_ms)
