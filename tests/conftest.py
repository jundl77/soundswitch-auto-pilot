import datetime
import pytest
from datetime import date

from lib.clients.spotify_client import SpotifyAudioSection, SpotifyTrackAnalysis
from lib.analyser.lightshow_classifier import LightShowType
from lib.engine.effect_controller import EffectController
from simulate.stub_clients import StubMidiClient


def make_section(
    start: float,
    duration: float,
    loudness: float = -10.0,
    bpm: float = 120.0,
) -> SpotifyAudioSection:
    return SpotifyAudioSection(
        section_start_sec=start,
        section_duration_sec=duration,
        section_loudness=loudness,
        section_bpm=bpm,
        section_key=0,
        section_mode=1,
        section_time_signature=4,
    )


def make_track_analysis(
    genres: list[str] | None = None,
    bpm: float = 128.0,
    energy: float = 0.8,
    loudness: float = -6.0,
    danceability: float = 0.8,
    light_show_type: LightShowType = LightShowType.HIGH_INTENSITY,
    audio_sections: list[SpotifyAudioSection] | None = None,
) -> SpotifyTrackAnalysis:
    return SpotifyTrackAnalysis(
        track_name='Test Track',
        album_name='Test Album',
        artists=['Test Artist'],
        analysis_ts=datetime.datetime.now(),
        progress_ms=0,
        duration_ms=180_000,
        bpm=bpm,
        beats_to_first_downbeat=0,
        first_downbeat_ms=0,
        current_beat_count=0,
        key=0,
        mode=1,
        time_signature=4,
        acousticness=0.1,
        danceability=danceability,
        energy=energy,
        instrumentalness=0.0,
        liveness=0.1,
        loudness=loudness,
        speechiness=0.05,
        valence=0.7,
        tempo=bpm,
        release_date=date(2020, 1, 1),
        popularity=80,
        genres=genres if genres is not None else ['house', 'dance'],
        light_show_type=light_show_type,
        beat_strengths_by_sec=[0.5] * 300,
        audio_sections=audio_sections if audio_sections is not None else [],
    )


@pytest.fixture
def stub_midi():
    return StubMidiClient()


@pytest.fixture
def effect_controller(stub_midi):
    return EffectController(stub_midi)
