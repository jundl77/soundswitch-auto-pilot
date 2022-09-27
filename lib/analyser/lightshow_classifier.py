from typing import List
from enum import Enum


LOW_INTENSITY_GENRES: List[str] = ['mellow', 'soft', 'golden', 'trance']
MEDIUM_INTENSITY_GENRES: List[str] = ['pop']
HIGH_INTENSITY_GENRES: List[str] = ['dance', 'hard', 'techno', 'house', 'edm', 'electro', 'latin', 'euro', 'reggaeton']
HIP_HOP_GENRES: List[str] = ['hip hop']


class LightShowType(Enum):
    LOW_INTENSITY = 1
    MEDIUM_INTENSITY = 2
    HIGH_INTENSITY = 3
    HIP_HOP = 4


def classify_track(genres: List[str],
                   bpm: int,
                   energy: float,
                   loudness: float,
                   danceability: float) -> LightShowType:
    genres_string = ' '.join(genres)
    has_low_intensity_genres: bool = any(genre in genres_string for genre in LOW_INTENSITY_GENRES) or not genres
    has_medium_intensity_genres: bool = any(genre in genres_string for genre in MEDIUM_INTENSITY_GENRES) or not genres
    has_high_intensity_genres: bool = any(genre in genres_string for genre in HIGH_INTENSITY_GENRES) or not genres
    has_hip_hop_genres: bool = any(genre in genres_string for genre in HIP_HOP_GENRES) or not genres

    if has_hip_hop_genres and not has_medium_intensity_genres and not has_high_intensity_genres:
        return LightShowType.HIP_HOP

    if has_high_intensity_genres and energy > 0.87 or loudness > -4.5 or danceability > 0.87:
        return LightShowType.HIGH_INTENSITY

    if has_high_intensity_genres and has_low_intensity_genres:
        return LightShowType.MEDIUM_INTENSITY

    if bpm < 90 or energy < 0.4 or danceability < 0.3:
        return LightShowType.LOW_INTENSITY

    return LightShowType.MEDIUM_INTENSITY
