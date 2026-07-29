import datetime
import pytest
import numpy as np
from collections import deque
from lib.analyser.music_analyser import MusicAnalyser


class _StubHandler:
    def on_sound_start(self): pass
    def on_sound_stop(self): pass
    async def on_cycle(self): pass
    async def on_onset(self): pass
    async def on_beat(self, beat_number, bpm, bpm_changed): pass
    async def on_note(self): pass
    async def on_section_change(self): pass


@pytest.fixture
def analyser():
    return MusicAnalyser(
        sample_rate=44100,
        buffer_size=256,
        handler=_StubHandler(),
    )


def test_trend_returns_one_when_fewer_than_four_samples(analyser):
    analyser._density_samples = deque([1.0, 2.0, 3.0], maxlen=12)
    assert analyser.get_onset_density_trend() == 1.0


def test_trend_rising(analyser):
    analyser._density_samples = deque([1.0, 1.0, 3.0, 3.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(3.0)


def test_trend_stable(analyser):
    analyser._density_samples = deque([4.0, 4.0, 4.0, 4.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(1.0)


def test_trend_falling(analyser):
    analyser._density_samples = deque([4.0, 4.0, 2.0, 2.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(0.5)


def test_trend_returns_one_when_past_mean_is_zero(analyser):
    analyser._density_samples = deque([0.0, 0.0, 2.0, 2.0], maxlen=12)
    assert analyser.get_onset_density_trend() == 1.0


def test_trend_uses_all_twelve_samples(analyser):
    samples = [1.0] * 6 + [2.0] * 6
    analyser._density_samples = deque(samples, maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(2.0)


def test_seconds_since_last_beat_approximately_correct(analyser):
    analyser.last_beat_detected = datetime.datetime.now() - datetime.timedelta(seconds=1.5)
    elapsed = analyser.get_seconds_since_last_beat()
    assert 1.4 < elapsed < 1.7


def test_seconds_since_last_beat_small_when_recent(analyser):
    analyser.last_beat_detected = datetime.datetime.now()
    assert analyser.get_seconds_since_last_beat() < 0.1


from lib.clock import VirtualClock


def _make_analyser(clock):
    return MusicAnalyser(
        sample_rate=44100,
        buffer_size=256,
        handler=_StubHandler(),
        clock=clock,
    )


def test_seconds_since_last_beat_on_virtual_clock():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    analyser.last_beat_detected = clock.now()
    clock.advance(1.5)
    assert analyser.get_seconds_since_last_beat() == 1.5


def test_onset_density_window_prunes_on_virtual_time():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    clock.advance(2.0)
    analyser._onset_times.append(clock.now())
    analyser._onset_times.append(clock.now())
    assert analyser.get_onset_density() == 2 / 1.5
    clock.advance(2.0)
    assert analyser.get_onset_density() == 0.0


def test_is_silence_all_near_zero(analyser):
    assert analyser._is_silence(np.zeros(40, dtype=np.float32)) is True
    assert analyser._is_silence(np.full(40, 0.00009, dtype=np.float64)) is True
    assert analyser._is_silence(np.full(40, -0.00009, dtype=np.float64)) is True


def test_is_silence_boundary_is_exclusive(analyser):
    energies = np.zeros(40)
    energies[7] = 0.0001
    assert analyser._is_silence(energies) is False


def test_is_silence_single_loud_band(analyser):
    energies = np.zeros(40)
    energies[0] = 0.5
    assert analyser._is_silence(energies) is False


def test_dead_accumulation_arrays_removed(analyser):
    assert not hasattr(analyser, 'mfccs')
    assert not hasattr(analyser, 'energies')


def test_fold_bpm_double_tempo_folds_down():
    assert MusicAnalyser._fold_bpm(257.8) == pytest.approx(128.9)


def test_fold_bpm_half_tempo_folds_up():
    assert MusicAnalyser._fold_bpm(64.0) == pytest.approx(128.0)


def test_fold_bpm_in_range_untouched():
    assert MusicAnalyser._fold_bpm(128.0) == pytest.approx(128.0)


def test_fold_bpm_boundary_170_folds_to_85():
    assert MusicAnalyser._fold_bpm(170.0) == pytest.approx(85.0)


def test_fold_bpm_zero_and_negative_return_zero():
    assert MusicAnalyser._fold_bpm(0.0) == 0.0
    assert MusicAnalyser._fold_bpm(-10.0) == 0.0


def test_fold_bpm_non_finite_returns_zero():
    # Halving inf never terminates, and this runs on the live audio thread.
    assert MusicAnalyser._fold_bpm(float('inf')) == 0.0
    assert MusicAnalyser._fold_bpm(float('-inf')) == 0.0
    assert MusicAnalyser._fold_bpm(float('nan')) == 0.0


def _feed_sub_bass(analyser, values: list[float], beat_buffers: tuple[int, ...] = ()) -> None:
    for i, v in enumerate(values):
        analyser._buffer_index += 1
        analyser._all_sub_bass_samples.append(v)
        if i in beat_buffers:
            analyser._pending_kick_beats.append(analyser._buffer_index)
        analyser._resolve_pending_kicks()


def test_kick_captures_transient_after_the_reported_beat(analyser):
    values = [1.0] * 60
    values[30 + 3] = 8.0
    _feed_sub_bass(analyser, values, beat_buffers=(30,))
    analyser._rms_window.append(0.2)
    assert analyser.get_kick_strength() == pytest.approx(8.0)


def test_kick_ignores_transient_outside_the_capture_window(analyser):
    values = [1.0] * 60
    values[10] = 8.0
    _feed_sub_bass(analyser, values, beat_buffers=(30,))
    analyser._rms_window.append(0.2)
    assert analyser.get_kick_strength() == pytest.approx(1.0)


def test_kick_background_is_a_median_not_a_mean(analyser):
    values = [1.0] * 120
    for beat in range(10, 110, 10):
        values[beat + 3] = 6.0
    _feed_sub_bass(analyser, values, beat_buffers=tuple(range(10, 110, 10)))
    analyser._rms_window.append(0.2)
    assert analyser.get_kick_strength() == pytest.approx(6.0)


def test_kick_ratio_capped(analyser):
    values = [0.001] * 60
    values[30 + 3] = 5.0
    _feed_sub_bass(analyser, values, beat_buffers=(30,))
    analyser._rms_window.append(0.2)
    assert analyser.get_kick_strength() == pytest.approx(10.0)


def test_kick_near_silence_returns_unknown(analyser):
    from lib.analyser.music_analyser import KICK_UNKNOWN
    values = [1e-6] * 60
    values[30 + 3] = 1e-3
    _feed_sub_bass(analyser, values, beat_buffers=(30,))
    analyser._rms_window.append(0.001)
    assert analyser.get_kick_strength() == KICK_UNKNOWN


def test_kick_unknown_before_any_beat_is_measured(analyser):
    from lib.analyser.music_analyser import KICK_UNKNOWN
    _feed_sub_bass(analyser, [1.0] * 30)
    analyser._rms_window.append(0.2)
    assert analyser.get_kick_strength() == KICK_UNKNOWN


def test_kick_unknown_reads_as_absent(analyser):
    from lib.analyser.music_analyser import KICK_UNKNOWN
    from lib.engine.light_engine import _KICK_PRESENCE_THRESHOLD
    assert KICK_UNKNOWN < _KICK_PRESENCE_THRESHOLD


# Computed from commit 9749286, before any madmom code existed; regenerating it
# from current code would prove nothing.
_MEL_GOLDEN_SHA256 = '883eae4b50a9ca0dc6c0a02f5772b5bf2ef4fe4f57f557bcf6d3a5fa2d2cac55'


def _mel_probe_signal() -> np.ndarray:
    rng = np.random.default_rng(20260729)
    t = np.arange(256 * 200) / 44100.0
    return (0.6 * np.sin(2 * np.pi * 55 * t)
            + 0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.1 * np.sin(2 * np.pi * 6000 * t)
            + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


def test_the_mel_filterbank_is_byte_identical_to_before_the_migration(analyser):
    import hashlib
    signal = _mel_probe_signal()
    rows = [analyser._compute_mel_energies(signal[i * 256:(i + 1) * 256])
            for i in range(200)]
    stacked = np.asarray(rows, dtype=np.float32)
    assert stacked.shape == (200, 40)
    assert hashlib.sha256(stacked.tobytes()).hexdigest() == _MEL_GOLDEN_SHA256


def test_bpm_is_unmeasured_until_two_intervals_exist(analyser):
    analyser.is_playing = True
    assert analyser.get_bpm() == 0
    analyser._beat_stream_times.extend([0.0, 0.47])
    assert analyser.get_bpm() == 0, 'one interval is not a measurement'


def test_bpm_is_the_median_interval_not_the_latest(analyser):
    analyser.is_playing = True
    analyser._beat_stream_times.extend([0.0, 0.47, 0.94, 0.95, 1.41, 1.88])
    assert analyser.get_bpm() == pytest.approx(60.0 / 0.47, rel=1e-6)


def test_bpm_is_octave_folded_from_the_beat_stream(analyser):
    analyser.is_playing = True
    analyser._beat_stream_times.extend([0.0, 0.2, 0.4, 0.6, 0.8])
    assert analyser.get_bpm() == pytest.approx(150.0)


def test_an_unmeasured_bpm_is_not_a_bpm_change(analyser):
    analyser.is_playing = True
    analyser.last_bpm = 128.0
    assert analyser._has_bpm_changed(0.0) is False


async def test_a_beep_fires_on_an_onset(analyser):
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([1.23], now) is True


async def test_no_onset_means_no_beep(analyser):
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([], now) is False


async def test_the_beep_refractory_survived_the_migration(analyser):
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([1.0], now) is True
    assert await analyser._track_note([1.01], now + datetime.timedelta(milliseconds=50)) is False
    assert await analyser._track_note([1.1], now + datetime.timedelta(milliseconds=80)) is True


async def test_a_song_reset_does_not_clear_the_drift_watchdog(analyser):
    from lib.analyser.drift_watchdog import ShedLevel
    analyser._drift._level = ShedLevel.SECTION_DETECTION
    analyser._drift.total_drift_sec = 4.2
    analyser._reset_state()
    assert analyser._drift.level is ShedLevel.SECTION_DETECTION
    assert analyser._drift.total_drift_sec == 4.2


async def test_a_song_reset_does_clear_the_rhythm_stack(analyser):
    analyser._beat_stream_times.extend([1.0, 2.0, 3.0])
    analyser._reset_state()
    assert len(analyser._beat_stream_times) == 0


def _fake_yamnet(analyser):
    detector = analyser.yamnet_change_detector
    detector.yamnet_model = object()
    return detector


async def test_restoring_section_detection_clears_its_audio_buffers(analyser):
    detector = _fake_yamnet(analyser)
    detector.agg_buffer.extend([0.1] * 100)
    detector.rolling_window_audio.extend([0.2] * 5000)
    detector.rolling_window_embeddings.extend([object(), object()])
    detector.rolling_window_similarities.extend([0.9, 0.8])

    analyser._set_section_detection_enabled(False)
    analyser._set_section_detection_enabled(True)

    assert not detector.agg_buffer, 'partial block survived the gap'
    assert not detector.rolling_window_audio, 'audio window survived the gap'
    assert not detector.rolling_window_embeddings, 'embeddings survived the gap'
    assert not detector.rolling_window_similarities, 'MAD baseline survived the gap'


async def test_the_splice_is_real_when_the_buffers_are_not_cleared(analyser):
    detector = _fake_yamnet(analyser)
    detector.rolling_window_audio.extend([1.0] * 4000)
    pre_gap_len = len(detector.rolling_window_audio)

    detector.rolling_window_audio.extend([-1.0] * 4000)
    seam = detector.rolling_window_audio[pre_gap_len - 1:pre_gap_len + 1]
    assert seam == [1.0, -1.0], 'expected a discontinuity at the seam'

    detector.reset()
    detector.rolling_window_audio.extend([-1.0] * 4000)
    assert 1.0 not in detector.rolling_window_audio


async def test_shedding_section_detection_does_not_clear_anything(analyser):
    detector = _fake_yamnet(analyser)
    detector.rolling_window_audio.extend([0.2] * 100)
    analyser._set_section_detection_enabled(False)
    assert detector.rolling_window_audio, 'shedding must not clear'


async def test_shedding_onsets_reports_unknown_density_not_zero():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN, density_is_known
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    clock.advance(2.0)
    analyser._onset_times.extend([clock.now()] * 6)
    assert density_is_known(analyser.get_onset_density())

    analyser._rhythm.set_onsets_enabled(False)
    assert analyser.get_onset_density() == DENSITY_UNKNOWN
    assert not density_is_known(analyser.get_onset_density())


async def test_density_stays_unknown_until_the_window_has_refilled():
    from lib.analyser.music_analyser import (DENSITY_UNKNOWN,
                                             _ONSET_DENSITY_WINDOW_SEC)
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    analyser._rhythm.set_onsets_enabled(False)
    analyser._rhythm.set_onsets_enabled(True)
    assert analyser.get_onset_density() == DENSITY_UNKNOWN

    clock.advance(_ONSET_DENSITY_WINDOW_SEC + 0.01)
    assert analyser.get_onset_density() != DENSITY_UNKNOWN


async def test_an_unknown_density_is_never_appended_to_the_trend_samples(analyser):
    from lib.analyser.music_analyser import density_is_known
    analyser._rhythm.set_onsets_enabled(False)
    await analyser._track_beat([1.0, 2.0, 3.0, 4.0], analyser._clock.now())
    assert all(density_is_known(d) for d in analyser._density_samples)


async def test_the_rhythm_heartbeat_survives_repeated_song_resets(caplog):
    import logging
    clock = VirtualClock()
    analyser = _make_analyser(clock)

    with caplog.at_level(logging.INFO):
        for _ in range(40):
            clock.advance(0.3)
            analyser._reset_state()
            analyser._log_rhythm_state(clock.now())
    assert any('[rhythm]' in r.message for r in caplog.records), (
        'the heartbeat never fired across 12 s of repeated song resets')


async def test_the_rhythm_heartbeat_actually_fires(caplog):
    import logging
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    clock.advance(11.0)
    with caplog.at_level(logging.INFO):
        analyser._log_rhythm_state(clock.now())
    assert any('[rhythm]' in r.message for r in caplog.records)


async def test_the_heartbeat_rearms_rather_than_repeating_every_buffer(caplog):
    import logging
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    clock.advance(11.0)
    with caplog.at_level(logging.INFO):
        analyser._log_rhythm_state(clock.now())
        clock.advance(0.1)
        analyser._log_rhythm_state(clock.now())
    assert sum('[rhythm]' in r.message for r in caplog.records) == 1


@pytest.mark.integration
async def test_debug_clicks_are_mixed_into_the_returned_audio():
    from pathlib import Path

    from lib.analyser.music_analyser import MusicAnalyser
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from lib.clock import VirtualClock
    from simulate.fake_audio_client import FileAudioClient

    sample = str(Path(__file__).parent.parent / 'samples'
                 / 'generate_eric_prydz_192k.mp3')

    class _Handler(_StubHandler):
        def __init__(self):
            self.notes = 0

        async def on_note(self):
            self.notes += 1

    async def run(note_clicks: bool):
        client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, sample)
        client.start_streams()
        clock = VirtualClock()
        handler = _Handler()
        analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, handler, clock=clock,
                                 note_clicks=note_clicks)
        analyser.yamnet_change_detector.detect_change = lambda *a, **k: False

        clicked = 0
        for _ in range(int(20 * SAMPLE_RATE / BUFFER_SIZE)):
            if client.exhausted:
                break
            clean = client.read().copy()
            clock.advance(BUFFER_SIZE / SAMPLE_RATE)
            out = await analyser.analyse(clean.copy())
            if not np.array_equal(out, clean):
                clicked += 1
                assert np.allclose(out - clean, analyser.click_sound, atol=1e-5), \
                    'the buffer was modified, but not by the click'
        return handler.notes, clicked

    notes_on, clicked_on = await run(note_clicks=True)
    notes_off, clicked_off = await run(note_clicks=False)

    assert notes_on > 0, 'no beeps fired on 20 s of the bundled track'
    assert clicked_on == notes_on, 'a beep fired without reaching the audio'
    assert clicked_off == 0, 'audio was modified with -d off'


async def test_a_song_reset_reports_unknown_density_during_the_refill():
    from lib.analyser.music_analyser import (DENSITY_UNKNOWN,
                                             _ONSET_DENSITY_WINDOW_SEC)
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    clock.advance(2.0)
    analyser._onset_times.extend([clock.now()] * 6)
    assert analyser.get_onset_density() > 0

    analyser._reset_state()
    assert analyser.get_onset_density() == DENSITY_UNKNOWN, \
        'a reset fabricated a measured density of 0.0'

    clock.advance(_ONSET_DENSITY_WINDOW_SEC + 0.01)
    assert analyser.get_onset_density() != DENSITY_UNKNOWN


async def test_the_reset_refill_uses_the_same_semantics_as_shed_restore():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN
    clock = VirtualClock()
    analyser = _make_analyser(clock)

    analyser._reset_state()
    after_reset = analyser.get_onset_density()
    analyser._rhythm.set_onsets_enabled(False)
    analyser._rhythm.set_onsets_enabled(True)
    after_restore = analyser.get_onset_density()
    assert after_reset == after_restore == DENSITY_UNKNOWN


async def test_a_freshly_built_analyser_has_not_measured_density_yet():
    from lib.analyser.music_analyser import (DENSITY_UNKNOWN,
                                             _ONSET_DENSITY_WINDOW_SEC)
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    assert analyser.get_onset_density() == DENSITY_UNKNOWN
    clock.advance(_ONSET_DENSITY_WINDOW_SEC + 0.01)
    assert analyser.get_onset_density() == 0.0
