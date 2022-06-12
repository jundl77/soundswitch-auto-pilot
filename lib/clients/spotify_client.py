import spotipy
import json
import logging
from pathlib import Path
from spotipy.oauth2 import SpotifyOAuth


class SpotifyDetails:
    def __init__(self, spotify_client_id: str, spotify_client_secret: str):
        self.client_id: str = spotify_client_id
        self.client_secret: str = spotify_client_secret


class SpotifyTrackAnalysis:
    def __init__(self, track_name: str, bpm: float):
        self.track_name: str = track_name
        self.bpm: float = bpm


class SpotifyClient:
    def __init__(self):
        self.spotify_details = self._read_spotify_details_file()
        auth_manager = SpotifyOAuth(client_id=self.spotify_details.client_id,
                                    client_secret=self.spotify_details.client_secret,
                                    redirect_uri="http://localhost:8877/callback",
                                    scope="user-read-playback-state")
        self.spotify = spotipy.Spotify(auth_manager=auth_manager)

    def _read_spotify_details_file(self) -> SpotifyDetails:
        spotify_details_file_path = (Path(__file__).parent.parent.parent / "spotify_details.json").absolute()
        assert spotify_details_file_path.is_file(), f"{spotify_details_file_path} does not exist"

        with open(spotify_details_file_path, 'r') as f:
            spotify_details_json = json.load(f)

        return SpotifyDetails(spotify_client_id=spotify_details_json['spotify_client_id'],
                              spotify_client_secret=spotify_details_json['spotify_client_secret'])

    def get_current_song_analysis(self) -> SpotifyTrackAnalysis:
        current_playback = self.spotify.current_playback()
        if not current_playback['is_playing']:
            logging.error("[spotify] song is not playing, returning default values")
            return SpotifyTrackAnalysis('', 0)
        track_name = current_playback['item']['name']
        track_id = current_playback['item']['id']
        progress_ms = int(current_playback['item']['progress_ms'])

        audio_features = self.spotify.audio_features(track_id)  # high-level data, e.g. track danceability
        audio_analysis = self.spotify.audio_analysis(track_id)  # low-level data, e.g. all beats in track
        bpm = audio_analysis['track']['tempo']

        return SpotifyTrackAnalysis(track_name, bpm)
