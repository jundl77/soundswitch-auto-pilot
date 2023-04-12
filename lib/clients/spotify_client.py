import spotipy
import json
import logging
import numpy as np
import datetime
import time
from threading import Thread
from datetime import date
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from spotipy.oauth2 import SpotifyOAuth
from sklearn import preprocessing
from requests.exceptions import ReadTimeout
from lib.analyser.lightshow_classifier import LightShowType, classify_track


SPOTIFY_QUERY_INTERVAL = datetime.timedelta(seconds=20)


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
                 duration_ms: int,
                 bpm: float,
                 beats_to_first_downbeat: int,
                 first_downbeat_ms: int,
                 current_beat_count: int,
                 key: int,
                 mode: int,
                 time_signature: int,
                 acousticness: float,
                 danceability: float,
                 energy: float,
                 instrumentalness: float,
                 liveness: float,
                 loudness: float,
                 speechiness: float,
                 valence: float,
                 tempo: float,
                 release_date: date,
                 popularity: int,
                 genres: List[str],
                 light_show_type: LightShowType,
                 beat_strengths_by_sec: List[float],
                 audio_sections: List[SpotifyAudioSection]):
        self.track_name: str = track_name
        self.album_name: str = album_name
        self.artists: List[str] = artists
        self.progress_ms: int = progress_ms
        self.duration_ms: int = duration_ms
        self.bpm: float = bpm
        self.beats_to_first_downbeat: int = beats_to_first_downbeat
        self.first_downbeat_ms: int = first_downbeat_ms
        self.current_beat_count: int = current_beat_count
        self.key: int = key
        self.mode: int = mode
        self.time_signature: int = time_signature
        self.acousticness: float = acousticness
        self.danceability: float = danceability
        self.energy: float = energy
        self.instrumentalness: float = instrumentalness
        self.liveness: float = liveness
        self.loudness: float = loudness
        self.speechiness: float = speechiness
        self.valence: float = valence
        self.tempo: float = tempo
        self.release_date: date = release_date
        self.popularity: int = popularity
        self.genres: List[str] = genres
        self.light_show_type: LightShowType = light_show_type
        self.beat_strengths_by_sec: List[float] = beat_strengths_by_sec
        self.audio_sections: List[SpotifyAudioSection] = audio_sections


class SpotifyClient:
    def __init__(self):
        self.is_running: bool = False
        self.spotify: spotipy.Spotify = None
        self.engine: "Engine" = None
        self.fetching_thread: Thread = Thread(target=self._run_query_thread)
        self.current_analysis: SpotifyTrackAnalysis = None

    def set_engine(self, engine: "Engine"):
        self.engine = engine

    def start(self):
        spotify_details = self._read_spotify_details_file()
        if spotify_details is None:
            logging.info(f"[spotify] spotify song analysis is inactive")
            return

        auth_manager = SpotifyOAuth(client_id=spotify_details.client_id,
                                    client_secret=spotify_details.client_secret,
                                    redirect_uri="http://localhost:8877/callback",
                                    scope="user-read-playback-state")
        self.spotify = spotipy.Spotify(auth_manager=auth_manager)
        self.engine = None
        self.fetching_thread = Thread(target=self._run_query_thread)
        self.is_active = True
        self.current_analysis = None
        logging.info(f"[spotify] spotify song analysis is active")

        self.is_running = True
        self.fetching_thread.start()

    def stop(self):
        logging.info(f'[spotify] stopping query thread')
        if self.is_running:
            self.is_running = False
            self.fetching_thread.join()

    def _run_query_thread(self):
        logging.info(f'[spotify] started query thread')

        last_query = datetime.datetime.now() - SPOTIFY_QUERY_INTERVAL
        while self.is_running:
            now = datetime.datetime.now()
            if now - last_query > SPOTIFY_QUERY_INTERVAL:
                try:
                    self.current_analysis = self._fetch_current_track_analysis()
                except ReadTimeout:
                    logging.info(f'[spotify] query timed out, skipping cycle')
                last_query = now
            time.sleep(0.001)

    def _read_spotify_details_file(self) -> Optional[SpotifyDetails]:
        spotify_details_file_path = (Path(__file__).parent.parent.parent / "spotify_details.json").absolute()
        if not spotify_details_file_path.is_file():
            logging.info(f"[spotify] {spotify_details_file_path} does not exist, will not use spotify")
            return None

        with open(spotify_details_file_path, 'r') as f:
            spotify_details_json = json.load(f)

        return SpotifyDetails(spotify_client_id=spotify_details_json['spotify_client_id'],
                              spotify_client_secret=spotify_details_json['spotify_client_secret'])

    def is_running(self) -> bool:
        return self.is_running

    def get_current_track_analysis(self) -> Optional[SpotifyTrackAnalysis]:
        return self.current_analysis

    def _fetch_current_track_analysis(self) -> Optional[SpotifyTrackAnalysis]:
        if not self.is_running:
            return None

        current_playback = self.spotify.current_playback()
        if current_playback is None or not current_playback['is_playing'] or current_playback['item'] is None:
            return None

        logging.info(f"[spotify] checking current track information")

        track_name = current_playback['item']['name']
        album_name = current_playback['item']['album']['name']
        artist_names = [artist['name'] for artist in current_playback['item']['artists']]
        artist_ids = [artist['id'] for artist in current_playback['item']['artists']]
        track_id = current_playback['item']['id']
        progress_ms = int(current_playback['progress_ms'])
        duration_ms = int(current_playback['item']['duration_ms'])

        artist_data = self.spotify.artists(artist_ids)
        genres = [data['genres'] for data in artist_data['artists']]
        genres = [item for sublist in genres for item in sublist]

        audio_features = self.spotify.audio_features(track_id)  # high-level data, e.g. track danceability
        audio_analysis = self.spotify.audio_analysis(track_id)  # low-level data, e.g. all beats in track
        bpm = audio_analysis['track']['tempo']
        beats_to_first_downbeat, first_downbeat_ms = self._calculate_first_downbeat(audio_analysis)
        current_beat_count = self._calculate_current_beat_count(progress_ms, audio_analysis) - beats_to_first_downbeat
        key = int(audio_features[0]['key'])
        mode = int(audio_features[0]['mode'])
        time_signature = int(audio_features[0]['time_signature'])
        acousticness = float(audio_features[0]['acousticness'])
        danceability = float(audio_features[0]['danceability'])
        energy = float(audio_features[0]['energy'])
        instrumentalness = float(audio_features[0]['instrumentalness'])
        liveness = float(audio_features[0]['liveness'])
        loudness = float(audio_features[0]['loudness'])
        speechiness = float(audio_features[0]['speechiness'])
        valence = float(audio_features[0]['valence'])
        tempo = float(audio_features[0]['tempo'])
        release_date = self._get_release_date_from_string(current_playback['item']['album']['release_date'])
        popularity = int(current_playback['item']['popularity'])
        light_show_type: LightShowType = classify_track(genres, bpm, energy, loudness, danceability)
        beat_strengths_by_sec = self._calculate_beat_strengths_by_sec(audio_analysis)
        audio_sections = self._get_audio_section(audio_analysis)

        return SpotifyTrackAnalysis(track_name=track_name,
                                    album_name=album_name,
                                    artists=artist_names,
                                    progress_ms=progress_ms,
                                    duration_ms=duration_ms,
                                    bpm=bpm,
                                    beats_to_first_downbeat=beats_to_first_downbeat,
                                    first_downbeat_ms=first_downbeat_ms,
                                    current_beat_count=current_beat_count,
                                    key=key,
                                    mode=mode,
                                    time_signature=time_signature,
                                    acousticness=acousticness,
                                    danceability=danceability,
                                    energy=energy,
                                    instrumentalness=instrumentalness,
                                    liveness=liveness,
                                    loudness=loudness,
                                    speechiness=speechiness,
                                    valence=valence,
                                    tempo=tempo,
                                    release_date=release_date,
                                    popularity=popularity,
                                    light_show_type=light_show_type,
                                    genres=genres,
                                    beat_strengths_by_sec=beat_strengths_by_sec,
                                    audio_sections=audio_sections)

    async def check_for_track_changes(self, previous_song: Optional[SpotifyTrackAnalysis], current_second: float):
        assert self.engine is not None, "engine should be set when 'check_for_track_changes' is called"
        if not self.is_running:
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

    def _get_release_date_from_string(self, release_date_string) -> date:
        # try parse normal date format, but some songs only have a year
        try:
            return datetime.datetime.strptime(release_date_string, '%Y-%m-%d')
        except:
            pass

        try:
            return datetime.datetime.strptime(release_date_string, '%Y')
        except:
            pass

        return datetime.datetime.strptime(str(datetime.date.today().year), '%Y')
