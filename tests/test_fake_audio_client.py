"""Tests for unthrottled fake audio clients, exhaustion, and the decode cache."""
import time

import numpy as np
import pytest

from lib.clock import VirtualClock
from simulate.fake_audio_client import BeepAudioClient, FileAudioClient

SAMPLE_RATE = 44100
BUFFER_SIZE = 256


def test_beep_client_read_does_not_sleep():
    clock = VirtualClock()
    client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    client.start_streams()
    start = time.monotonic()
    for _ in range(2000):  # ~11.6 s of audio — would take 11.6 s if throttled
        client.read()
    assert time.monotonic() - start < 2.0


def test_beep_client_noise_is_deterministic():
    def collect():
        clock = VirtualClock()
        c = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
        c.start_streams()
        return np.concatenate([c.read() for _ in range(50)])

    assert np.array_equal(collect(), collect())


def test_beep_client_never_exhausted():
    client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE)
    assert client.exhausted is False


def test_beep_client_click_log_uses_virtual_clock():
    clock = VirtualClock()
    client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    client.start_streams()  # virtual start time = 0.0
    # 120 BPM → first click at 0.5 s. Read past it.
    for _ in range(200):
        client.read()
    assert client.click_log, 'expected at least one click'
    assert client.click_log[0]['wall_time'] == pytest.approx(0.5, abs=0.01)


@pytest.fixture
def wav_file(tmp_path):
    """1-second 440 Hz sine written as a wav librosa can load."""
    import soundfile as sf
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / 'tone.wav'
    sf.write(str(path), audio, SAMPLE_RATE)
    return str(path)


def test_file_client_reads_unthrottled_and_exhausts(wav_file):
    client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file)
    client.start_streams()
    assert client.exhausted is False
    start = time.monotonic()
    n = 0
    while not client.exhausted:
        buf = client.read()
        assert len(buf) == BUFFER_SIZE
        n += 1
    assert time.monotonic() - start < 2.0  # 1 s of audio, no throttle
    assert n == pytest.approx(SAMPLE_RATE / BUFFER_SIZE, abs=2)


def test_file_client_decode_cache_roundtrip(wav_file):
    import os
    c1 = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file)
    c1.start_streams()
    cache = f'{wav_file}.{SAMPLE_RATE}.npy'
    assert os.path.exists(cache), 'first load must write the .npy cache'

    c2 = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file)
    c2.start_streams()
    assert np.array_equal(c1._audio, c2._audio)


def test_file_client_stale_cache_is_refreshed(wav_file, tmp_path):
    import os
    c1 = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file)
    c1.start_streams()
    cache = f'{wav_file}.{SAMPLE_RATE}.npy'
    # Make the cache look older than the source → must be regenerated, not trusted.
    os.utime(cache, (1, 1))
    np.save(cache, np.zeros(10, dtype=np.float32))
    os.utime(cache, (1, 1))
    c2 = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file)
    c2.start_streams()
    assert len(c2._audio) == len(c1._audio)  # re-decoded, not the 10-sample fake


def test_pyaudio_client_satisfies_exhausted_interface():
    """The runner's loop reads audio_client.exhausted — the REAL client must
    provide it too (regression: mic mode crashed with AttributeError)."""
    from lib.clients.pyaudio_client import PyAudioClient
    assert isinstance(getattr(PyAudioClient, 'exhausted', None), property)
