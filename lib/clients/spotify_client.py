import spotipy
import json
import logging
import numpy as np
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from spotipy.oauth2 import SpotifyOAuth
from sklearn import preprocessing


class SpotifyDetails:
    def __init__(self, spotify_client_id: str, spotify_client_secret: str):
        self.client_id: str = spotify_client_id
        self.client_secret: str = spotify_client_secret


class SpotifyAudioSection:
    def __init__(self,
                 section_start_sec: float,
                 section_duration_sec: float,
                 section_loudness: float,
                 section_bpm: float,
                 section_key: int,
                 section_mode: int,
                 section_time_signature: int):
        self.section_start_sec: float = section_start_sec
        self.section_duration_sec: float = section_duration_sec
        self.section_loudness: float = section_loudness
        self.section_bpm: float = section_bpm
        self.section_key: int = section_key
        self.section_mode: int = section_mode
        self.section_time_signature: int = section_time_signature


class SpotifyTrackAnalysis:
    def __init__(self,
                 track_name: str,
                 album_name: str,
                 artists: List[str],
                 progress_ms: int,
                 bpm: float,
                 beats_to_first_downbeat: int,
                 first_downbeat_ms: int,
                 current_beat_count: int,
                 beat_strengths_by_sec: List[float],
                 audio_sections: List[SpotifyAudioSection]):
        self.track_name: str = track_name
        self.album_name: str = album_name
        self.artists: List[str] = artists
        self.progress_ms: int = progress_ms
        self.bpm: float = bpm
        self.beats_to_first_downbeat: int = beats_to_first_downbeat
        self.first_downbeat_ms: int = first_downbeat_ms
        self.current_beat_count: int = current_beat_count
        self.beat_strengths_by_sec: List[float] = beat_strengths_by_sec
        self.audio_sections: List[SpotifyAudioSection] = audio_sections


class SpotifyClient:
    def __init__(self):
        spotify_details = self._read_spotify_details_file()
        if spotify_details is not None:
            auth_manager = SpotifyOAuth(client_id=spotify_details.client_id,
                                        client_secret=spotify_details.client_secret,
                                        redirect_uri="http://localhost:8877/callback",
                                        scope="user-read-playback-state")
            self.spotify = spotipy.Spotify(auth_manager=auth_manager)
            self.engine = None
            self.is_active = True
            logging.info(f"[spotify] spotify song analysis is active")
        else:
            self.is_active = False
            logging.info(f"[spotify] spotify song analysis is inactive")

    def set_engine(self, engine: "Engine"):
        self.engine = engine

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

    def get_current_track_analysis(self) -> Optional[SpotifyTrackAnalysis]:
        if not self.is_active:
            return None

        current_playback = self.spotify.current_playback()
        if current_playback is None or not current_playback['is_playing']:
            return None

        track_name = current_playback['item']['name']
        album_name = current_playback['item']['album']['name']
        artists = [artist['name'] for artist in current_playback['item']['artists']]
        track_id = current_playback['item']['id']
        progress_ms = int(current_playback['progress_ms'])

        audio_features = self.spotify.audio_features(track_id)  # high-level data, e.g. track danceability
        audio_analysis = self.spotify.audio_analysis(track_id)  # low-level data, e.g. all beats in track
        bpm = audio_analysis['track']['tempo']
        beats_to_first_downbeat, first_downbeat_ms = self._calculate_first_downbeat(audio_analysis)
        current_beat_count = self._calculate_current_beat_count(progress_ms, audio_analysis) - beats_to_first_downbeat
        beat_strengths_by_sec = self._calculate_beat_strengths_by_sec(audio_analysis)
        audio_sections = self._get_audio_section(audio_analysis)

        return SpotifyTrackAnalysis(track_name,
                                    album_name,
                                    artists,
                                    progress_ms,
                                    bpm,
                                    beats_to_first_downbeat,
                                    first_downbeat_ms,
                                    current_beat_count,
                                    beat_strengths_by_sec,
                                    audio_sections)

    async def check_for_track_changes(self, previous_song: Optional[SpotifyTrackAnalysis], current_second: float):
        assert self.engine is not None, "engine should be set when 'check_for_track_changes' is called"
        if not self.is_active:
            return

        track_analysis = self.get_current_track_analysis()
        if not track_analysis:
            return

        if not previous_song or track_analysis.track_name != previous_song.track_name:
            await self.engine.on_spotify_track_changed(track_analysis)
            return

        if abs(track_analysis.progress_ms - current_second * 1000) > 1000:
            await self.engine.on_spotify_track_progress_changed(track_analysis)
            return

    def _calculate_first_downbeat(self, audio_analysis: Dict) -> Tuple[int, int]:
        sections = audio_analysis['sections']
        if len(sections) <= 1:
            return 0, 20000  # default values, pretend first downbeat is 20sec into song

        # calculate first downbeat using first beat of bar which has the highest confidence around the second section,
        # because the first downbeat is easy to detect so it "should" have the highest confidence (a bit of an assumption)
        second_section_start = sections[1]['start']
        relevant_bars: dict[float, float] = dict()
        for bar in audio_analysis['bars']:
            if abs(bar['start'] - second_section_start) < 3:
                relevant_bars[bar['confidence']] = bar['start']
        first_downbeat_sec = list(dict(sorted(relevant_bars.items(), reverse=True)).values())[0]

        # count all the beats before the first downbeat
        beat_count = 0
        for beat in audio_analysis['beats']:
            if beat['start'] < first_downbeat_sec:
                beat_count += 1
            else:
                break

        return beat_count, int(first_downbeat_sec * 1000)

    def _calculate_current_beat_count(self, progress_ms, audio_analysis: Dict) -> int:
        progress_sec = progress_ms / 1000.0
        beat_count = 0
        for beat in audio_analysis['beats']:
            if beat['start'] < progress_sec:
                beat_count += 1
            else:
                break
        return beat_count

    def _calculate_beat_strengths_by_sec(self, audio_analysis: Dict) -> List[float]:
        audio_strengths_raw_by_seconds: List[float] = list()
        audio_strengths_count_by_seconds: List[float] = list()
        for segment in audio_analysis['segments']:
            second = int(segment['start'])
            if len(audio_strengths_raw_by_seconds) == second + 1:
                audio_strengths_raw_by_seconds[second] += (segment['loudness_max'] * segment['loudness_max_time'])
                audio_strengths_count_by_seconds[second] += 1
            else:
                audio_strengths_raw_by_seconds.append(segment['loudness_max'] * segment['loudness_max_time'])
                audio_strengths_count_by_seconds.append(1)
        audio_strengths_raw_by_seconds = [audio_strengths_raw_by_seconds[i] / audio_strengths_count_by_seconds[i] for i in range(len(audio_strengths_count_by_seconds))]
        quantile_scaler = preprocessing.QuantileTransformer(n_quantiles=len(audio_strengths_raw_by_seconds))
        normalized_audio_strengths_by_sec = np.array(audio_strengths_raw_by_seconds).reshape(-1, 1)
        normalized_audio_strengths_by_sec = quantile_scaler.fit_transform(normalized_audio_strengths_by_sec)
        normalized_audio_strengths_by_sec = [item for sublist in normalized_audio_strengths_by_sec for item in sublist]
        return normalized_audio_strengths_by_sec

    def _get_audio_section(self, audio_analysis: Dict) -> List[SpotifyAudioSection]:
        audio_sections: List[SpotifyAudioSection] = list()
        for raw_section in audio_analysis['sections']:
            section = SpotifyAudioSection(section_start_sec=float(raw_section['start']),
                                          section_duration_sec=float(raw_section['duration']),
                                          section_loudness=float(raw_section['loudness']),
                                          section_bpm=float(raw_section['tempo']),
                                          section_key=int(raw_section['key']),
                                          section_mode=int(raw_section['mode']),
                                          section_time_signature=int(raw_section['time_signature']))
            audio_sections.append(section)
        return audio_sections
