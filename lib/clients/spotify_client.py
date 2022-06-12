import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyClient:
    def __init__(self):
        self.spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    def get_current_playback(self):
        return self.spotify.current_playback()
