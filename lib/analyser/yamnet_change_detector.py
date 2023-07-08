import time
import logging
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Optional
from collections import deque
from scipy import stats
from scipy.spatial import distance
from lib.clients.spotify_client import SpotifyTrackAnalysis

FIXED_CHANGE_OFFSET_SEC = 1


def detect_outliers_mean_std(full_data, test_data, std_threshold=3):
    elements = np.array(full_data)
    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    outliers = [x for x in test_data if ((x < mean - std_threshold * sd) or (x > mean + std_threshold + sd))]
    return outliers


def detect_outliers_mad(full_data: deque, test_data, threshold=2.5):
    median = np.median(full_data)
    deviations = np.abs(full_data - median)
    mad = np.median(deviations)
    modified_z_scores = 0.6745 * (test_data - median) / mad
    outliers = np.where(np.abs(modified_z_scores) > threshold)[0]
    return outliers


def detect_outliers(full_data: deque, test_data):
    #return detect_outliers_mean_std(full_data, [test_data], std_threshold=2)
    return detect_outliers_mad(full_data, test_data, threshold=2.5)


class ChangeDetectionTracker:
    def __init__(self,
                 min_outliers_required: int,
                 outlier_tracking_time_window_sec: int,
                 similarity_tracking_time_window_sec: int):
        self.min_outliers_required: int = min_outliers_required
        self.outlier_tracking_time_window_sec: int = outlier_tracking_time_window_sec
        self.similarity_tracking_time_window_sec: int = similarity_tracking_time_window_sec
        self.cooldown_time_window_sec: int = 10

        # tracking state
        self.best_previous_similarity: float = 0
        self.similarity_tracking_start: time.time = time.time()
        self.outlier_count = 0
        self.outlier_tracking_start: time.time = time.time()
        self.cooldown_start: time.time = time.time()

    def track_similarity(self, similarity: float, all_similarities: deque):
        now = time.time()
        best_previous = min(self.best_previous_similarity, similarity)
        outliers = detect_outliers(all_similarities, similarity)
        logging.debug(f"similarity: {similarity}, best_previous={best_previous}, outliers={outliers}")

        if now - self.similarity_tracking_start > self.similarity_tracking_time_window_sec:
            self.similarity_tracking_start = time.time()
            self.best_previous_similarity = 1
        if now - self.outlier_tracking_start > self.outlier_tracking_time_window_sec:
            self.outlier_tracking_start = time.time()
            self.outlier_count = 0
        if len(outliers) > 0:
            self.outlier_count += 1

    def is_change(self) -> bool:
        if self.outlier_count > self.min_outliers_required:
            self.outlier_count = 0
            if not self.is_cooldown_active():
                return True
            else:
                logging.info(f"[yamnet] change detected, but in cooldown, ignoring")
        return False

    def start_cooldown(self):
        self.cooldown_start = time.time()

    def is_cooldown_active(self) -> bool:
        return time.time() - self.cooldown_start < self.cooldown_time_window_sec


class YamnetChangeDetector:
    def __init__(self,
                 sample_rate: int,
                 buffer_size: int):
        # params
        self.agg_buffer_size_multiplier: int = 16
        self.embedding_lookback_sec: int = 2               # cannot be more than 4, rolling window stores 5 sec of data
        self.audio_lookback_sec: int = 1                   # cannot be more than 4, rolling window stores 5 sec of data
        self.min_outliers_required: int = 5
        self.outlier_tracking_time_window_sec: int = 1
        self.similarity_tracking_time_window_sec: int = 3

        self.sample_rate: int = sample_rate
        self.buffer_size: int = buffer_size
        self.agg_buffer_size: int = self.buffer_size * self.agg_buffer_size_multiplier
        self.change_tracker: ChangeDetectionTracker = ChangeDetectionTracker(self.min_outliers_required,
                                                                             self.outlier_tracking_time_window_sec,
                                                                             self.similarity_tracking_time_window_sec)
        self.yamnet_model = None
        self.num_blocks_per_sec = round(self.sample_rate / self.agg_buffer_size)
        self.num_blocks_per_100ms = min(1, int(self.num_blocks_per_sec / 10))
        self.elements_per_sec = self.agg_buffer_size * 2 * self.num_blocks_per_sec

        self.agg_buffer: list = list()
        self.rolling_window_audio: list = list()
        self.rolling_window_embeddings: list = list()
        self.rolling_window_similarities: deque = deque(maxlen=100)

    def start(self):
        logging.info('[yamnet] loading yamnet model..')
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        logging.info('[yamnet] loaded yamnet model successfully')

    def reset(self):
        if not self.change_tracker.is_cooldown_active():
            logging.info('[yamnet] resetting state, starting cooldown')
        self.change_tracker.start_cooldown()

    def detect_change(self,
                      audio_signal: np.ndarray,
                      current_song_duration: datetime.timedelta,
                      track_analysis: Optional[SpotifyTrackAnalysis]) -> bool:
        result = False

        # audio signals come in at a smaller size than we need here, so we aggregate
        # them until we have the size we want
        is_buffer_full, agg_buffer = self._build_agg_buffer(audio_signal)
        if not is_buffer_full:
            return False

        # Convert audio data to numpy array
        audio_data = np.frombuffer(agg_buffer, dtype=np.int16).astype(np.float32)

        audio_data = audio_data / np.iinfo(np.int16).max
        self.rolling_window_audio += audio_data.tolist()
        audio_lookback_index = self.elements_per_sec * self.audio_lookback_sec
        previous_data = self.rolling_window_audio[-1 * audio_lookback_index:]

        scores, embedding, spectrogram = self.yamnet_model(previous_data)
        embedding = tf.reduce_mean(embedding, axis=0)
        self.rolling_window_embeddings.append(embedding)

        embedding_lookback_index = self.num_blocks_per_sec * self.embedding_lookback_sec
        if len(self.rolling_window_embeddings) > embedding_lookback_index:
            index = 0
            similarities: list[float] = []
            while index <= embedding_lookback_index:
                previous_embedding = self.rolling_window_embeddings[-1 * (index + 1)]
                #similarity = distance.minkowski(embedding.numpy(), previous_embedding.numpy(), 2)
                similarity = abs(float(tf.keras.losses.cosine_similarity(previous_embedding, embedding)))
                similarities.append(abs(float(similarity)))
                index += self.num_blocks_per_100ms

            best_similarity: float = min(similarities)
            self.rolling_window_similarities.append(best_similarity)
            self.change_tracker.track_similarity(best_similarity, self.rolling_window_similarities)
            if self.change_tracker.is_change() and self._is_in_spotify_range(current_song_duration, track_analysis):
                logging.info('[yamnet] meaningful change detected')
                self.change_tracker.start_cooldown()
                result = True

        # delete old data in rolling windows
        if len(self.rolling_window_audio) > audio_lookback_index * 2:
            del self.rolling_window_audio[:audio_lookback_index * -1]

        if len(self.rolling_window_embeddings) > embedding_lookback_index * 2:
            del self.rolling_window_embeddings[:embedding_lookback_index * -1]

        return result

    def _is_in_spotify_range(self,
                             current_second: datetime.timedelta,
                             track_analysis: Optional[SpotifyTrackAnalysis]) -> bool:
        if not track_analysis:
            return True

        for audio_section in track_analysis.audio_sections:
            section_start_sec = audio_section.section_start_sec - FIXED_CHANGE_OFFSET_SEC
            section_end_sec = section_start_sec + audio_section.section_duration_sec
            if abs(section_start_sec - current_second.total_seconds()) < 2:
                return True
            if abs(section_end_sec - current_second.total_seconds()) < 2:
                return True
        return False

    def _build_agg_buffer(self, audio_signal: np.ndarray) -> tuple[bool, np.ndarray]:
        assert len(audio_signal) == self.buffer_size
        # if the aggregated buffer is not full yet, add to it, otherwise return the built buffer
        if len(self.agg_buffer) != self.agg_buffer_size:
            self.agg_buffer += audio_signal.tolist()
            return False, None
        else:
            full_agg_buffer: np.ndarray = np.array(self.agg_buffer)
            self.agg_buffer.clear()
            return True, full_agg_buffer
