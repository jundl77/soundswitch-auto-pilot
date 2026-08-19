import datetime
import pytest
import numpy as np
from lib.analyser.music_analyser import MusicAnalyser


class _StubHandler:
    def on_sound_start(self): pass
    def on_sound_stop(self): pass
    async def on_cycle(self): pass
    async def on_beat(self, beat_number, bpm, bpm_changed): pass
    async def on_note(self): pass


@pytest.fixture
def analyser():
    return MusicAnalyser(
        sample_rate=44100,
        buffer_size=256,
        handler=_StubHandler(),
    )


def test_seconds_since_last_beat_approximately_correct(analyser):
    analyser.last_beat_detected = datetime.datetime.now() - datetime.timedelta(seconds=1.5)
    elapsed = analyser.get_seconds_since_last_beat()
    assert 1.4 < elapsed < 1.7


def test_seconds_since_last_beat_small_when_recent(analyser):
    analyser.last_beat_detected = datetime.datetime.now()
    assert analyser.get_seconds_since_last_beat() < 0.1


from lib.clock import VirtualClock


def _make_analyser(clock, handler=None):
    return MusicAnalyser(
        sample_rate=44100,
        buffer_size=256,
        handler=handler or _StubHandler(),
        clock=clock,
    )


def test_seconds_since_last_beat_on_virtual_clock():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    analyser.last_beat_detected = clock.now()
    clock.advance(1.5)
    assert analyser.get_seconds_since_last_beat() == 1.5


def test_silence_is_decided_on_the_rms_the_loop_already_computes(analyser):
    from lib.analyser.music_analyser import _SILENCE_RMS

    assert analyser._is_silence(0.0) is True
    assert analyser._is_silence(_SILENCE_RMS * 0.99) is True
    assert analyser._is_silence(_SILENCE_RMS) is False
    assert analyser._is_silence(0.2) is False


def test_the_silence_threshold_is_the_matched_one_not_the_retired_gates():
    from lib.analyser.music_analyser import _SILENCE_RMS

    assert 1.4e-4 <= _SILENCE_RMS <= 1.66e-4


async def test_a_quiet_buffer_stops_the_song_and_a_loud_one_keeps_it_playing():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    loud = np.full(256, 0.2, dtype=np.float32)
    for _ in range(60):
        clock.advance(256 / 44100)
        await analyser.analyse(loud.copy())
    assert analyser.is_playing

    for _ in range(60):
        clock.advance(256 / 44100)
        await analyser.analyse(np.zeros(256, dtype=np.float32))
    assert not analyser.is_playing


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
    assert MusicAnalyser._fold_bpm(float('inf')) == 0.0
    assert MusicAnalyser._fold_bpm(float('-inf')) == 0.0
    assert MusicAnalyser._fold_bpm(float('nan')) == 0.0


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


async def test_a_note_event_fires_on_a_beat(analyser):
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([1.23], now) is True


async def test_no_beat_means_no_note_event(analyser):
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([], now) is False


async def test_a_burst_of_beats_in_one_buffer_cannot_strobe_the_overlay_bar(analyser):
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([1.0], now) is True
    assert await analyser._track_note([1.01], now + datetime.timedelta(milliseconds=50)) is False
    assert await analyser._track_note([1.1], now + datetime.timedelta(milliseconds=80)) is True


class _FakeRhythm:
    def __init__(self):
        from lib.analyser.madmom_rhythm import RhythmEvents
        self._events = RhythmEvents
        self.fire_on = set()
        self.calls = 0
        self.resets = 0
        self.pending_latency_sec = 0.0

    def reset(self):
        self.resets += 1

    def process(self, buffer):
        index, self.calls = self.calls, self.calls + 1
        return self._events(beats=[index / 100.0] if index in self.fire_on else [])


async def test_the_overlay_note_event_follows_the_beat_stream():
    class _Recorder(_StubHandler):
        def __init__(self):
            self.beats: list[int] = []
            self.notes: list[int] = []
            self.index = 0

        async def on_beat(self, beat_number, bpm, bpm_changed):
            self.beats.append(self.index)

        async def on_note(self):
            self.notes.append(self.index)

    clock = VirtualClock()
    handler = _Recorder()
    analyser = _make_analyser(clock, handler)
    analyser._rhythm = _FakeRhythm()
    analyser._rhythm.fire_on = {5, 40, 90}
    analyser.is_playing = True

    loud = np.full(256, 0.2, dtype=np.float32)
    for i in range(120):
        handler.index = i
        clock.advance(0.05)
        await analyser.analyse(loud.copy())

    assert handler.beats == [5, 40, 90]
    assert handler.notes == handler.beats


async def test_the_fifteen_minute_roll_is_not_a_song_boundary():
    class _Recorder(_StubHandler):
        def __init__(self):
            self.events: list[str] = []

        def on_sound_start(self):
            self.events.append('start')

        def on_sound_stop(self):
            self.events.append('stop')

    clock = VirtualClock()
    handler = _Recorder()
    analyser = _make_analyser(clock, handler)

    loud = np.full(256, 0.2, dtype=np.float32)
    for _ in range(4):
        clock.advance(0.1)
        await analyser.analyse(loud.copy())
    assert handler.events == ['start']

    analyser.song_start_time = clock.now() - datetime.timedelta(minutes=16)
    for _ in range(4):
        clock.advance(0.1)
        await analyser.analyse(loud.copy())

    assert handler.events == ['start'], \
        'the roll re-announced a song the room never stopped hearing'
    assert analyser.is_song_playing() is True
    assert analyser.get_song_current_duration() < datetime.timedelta(minutes=1), \
        'the song clock was not rolled'


async def test_the_fifteen_minute_roll_keeps_the_beat_lock():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    analyser._rhythm = _FakeRhythm()
    analyser._rhythm.fire_on = {2, 6}

    loud = np.full(256, 0.2, dtype=np.float32)
    for _ in range(8):
        clock.advance(0.1)
        await analyser.analyse(loud.copy())
    assert analyser.beat_count == 2
    bpm_window = list(analyser._beat_stream_times)

    analyser.song_start_time = clock.now() - datetime.timedelta(minutes=16)
    clock.advance(0.1)
    await analyser.analyse(loud.copy())

    assert analyser.get_song_current_duration() < datetime.timedelta(minutes=1), \
        'the song clock was not rolled'
    assert analyser._rhythm.resets == 0, \
        'the roll re-locked madmom mid-audio; a gained or lost beat rotates the bar grid'
    assert analyser.beat_count == 2, \
        'the OS2L beat position re-based mid-song'
    assert list(analyser._beat_stream_times) == bpm_window, \
        'the BPM window was torn down mid-lock'


async def test_a_real_sound_stop_still_relocks_the_tracker():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    analyser._rhythm = _FakeRhythm()

    loud = np.full(256, 0.2, dtype=np.float32)
    for _ in range(8):
        clock.advance(0.1)
        await analyser.analyse(loud.copy())
    assert analyser._rhythm.resets == 0

    for _ in range(8):
        clock.advance(0.1)
        await analyser.analyse(np.zeros(256, dtype=np.float32))
    assert analyser._rhythm.resets > 0, \
        'a song boundary must still re-lock the tracker'


async def test_a_song_reset_does_not_clear_the_drift_watchdog(analyser):
    from lib.analyser.drift_watchdog import ShedLevel
    analyser._drift._level = ShedLevel.NN_SHED
    analyser._drift.total_drift_sec = 4.2
    analyser._reset_state()
    assert analyser._drift.level is ShedLevel.NN_SHED
    assert analyser._drift.total_drift_sec == 4.2


async def test_a_song_reset_does_clear_the_rhythm_stack(analyser):
    from lib.analyser.madmom_rhythm import HOP_SIZE

    analyser._beat_stream_times.extend([1.0, 2.0, 3.0])
    analyser._rhythm.process(np.zeros(HOP_SIZE + 100, dtype=np.float32))

    analyser._reset_state()

    assert len(analyser._beat_stream_times) == 0
    assert analyser._rhythm.pending_latency_sec == 0.0, 'a partial hop survived the song'
    assert analyser._rhythm._hops == 0, 'the rhythm time base was not rebased'


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
    from lib.analyser.music_analyser import MusicAnalyser
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from lib.clock import VirtualClock
    from simulate.fake_audio_client import FileAudioClient
    from tests.conftest import anchor_mp3_path

    sample = anchor_mp3_path()

    class _Handler(_StubHandler):
        def __init__(self):
            self.index = 0
            self.beat_buffers: list[int] = []
            self.note_buffers: list[int] = []

        async def on_beat(self, beat_number, bpm, bpm_changed):
            self.beat_buffers.append(self.index)

        async def on_note(self):
            self.note_buffers.append(self.index)

    async def run(note_clicks: bool):
        client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, sample)
        client.start_streams()
        clock = VirtualClock()
        handler = _Handler()
        analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, handler, clock=clock,
                                 note_clicks=note_clicks)

        clicked: list[int] = []
        for i in range(int(20 * SAMPLE_RATE / BUFFER_SIZE)):
            if client.exhausted:
                break
            handler.index = i
            clean = client.read().copy()
            clock.advance(BUFFER_SIZE / SAMPLE_RATE)
            out = await analyser.analyse(clean.copy())
            if not np.array_equal(out, clean):
                clicked.append(i)
                assert np.allclose(out - clean, analyser.click_sound, atol=1e-5), \
                    'the buffer was modified, but not by the click'
        return handler, clicked

    on, clicked_on = await run(note_clicks=True)
    _, clicked_off = await run(note_clicks=False)

    assert on.beat_buffers, 'no beats on 20 s of the anchor track'
    assert clicked_on == on.beat_buffers, \
        'clicks did not land exactly on the beats'
    assert on.note_buffers == on.beat_buffers, \
        'the overlay light bar is no longer following the beat stream'
    assert not clicked_off, 'audio was modified with -d off'


async def test_beat_position_survives_a_song_reset_landing_mid_read():
    class _ResettingClock(VirtualClock):
        def __init__(self):
            super().__init__()
            self.victim = None

        def now(self):
            victim, self.victim = self.victim, None
            if victim is not None:
                victim._reset_state()
            return super().now()

    clock = _ResettingClock()
    analyser = _make_analyser(clock)
    analyser.is_playing = True
    analyser.time_to_last_beat_sec = 0.47
    analyser.beat_count = 12
    clock.victim = analyser

    assert analyser.get_beat_position() >= 0
