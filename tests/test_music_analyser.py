"""
Unit tests for MusicAnalyser: onset-density trend, beat timing, virtual-clock
behaviour, and the vectorized silence check.
"""
import datetime
import pytest
import numpy as np
from collections import deque
from lib.analyser.music_analyser import MusicAnalyser


class _StubHandler:
    """Minimal handler — satisfies the interface without any real behaviour."""
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


# ---------------------------------------------------------------------------
# get_onset_density_trend
# ---------------------------------------------------------------------------

def test_trend_returns_one_when_fewer_than_four_samples(analyser):
    analyser._density_samples = deque([1.0, 2.0, 3.0], maxlen=12)
    assert analyser.get_onset_density_trend() == 1.0


def test_trend_rising(analyser):
    # past half [1, 1], recent half [3, 3] → ratio 3.0
    analyser._density_samples = deque([1.0, 1.0, 3.0, 3.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(3.0)


def test_trend_stable(analyser):
    analyser._density_samples = deque([4.0, 4.0, 4.0, 4.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(1.0)


def test_trend_falling(analyser):
    # past half [4, 4], recent half [2, 2] → ratio 0.5
    analyser._density_samples = deque([4.0, 4.0, 2.0, 2.0], maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(0.5)


def test_trend_returns_one_when_past_mean_is_zero(analyser):
    # Avoid division by zero — past half is all zeros
    analyser._density_samples = deque([0.0, 0.0, 2.0, 2.0], maxlen=12)
    assert analyser.get_onset_density_trend() == 1.0


def test_trend_uses_all_twelve_samples(analyser):
    # Twelve samples: first 6 = 1.0, last 6 = 2.0 → trend ≈ 2.0
    samples = [1.0] * 6 + [2.0] * 6
    analyser._density_samples = deque(samples, maxlen=12)
    assert analyser.get_onset_density_trend() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# get_seconds_since_last_beat
# ---------------------------------------------------------------------------

def test_seconds_since_last_beat_approximately_correct(analyser):
    analyser.last_beat_detected = datetime.datetime.now() - datetime.timedelta(seconds=1.5)
    elapsed = analyser.get_seconds_since_last_beat()
    assert 1.4 < elapsed < 1.7


def test_seconds_since_last_beat_small_when_recent(analyser):
    analyser.last_beat_detected = datetime.datetime.now()
    assert analyser.get_seconds_since_last_beat() < 0.1


# ---------------------------------------------------------------------------
# Virtual clock
# ---------------------------------------------------------------------------

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
    # Past the start-up refill window first: a freshly built analyser has an
    # empty rolling window and reports UNMEASURED until one has filled.
    clock.advance(2.0)
    analyser._onset_times.append(clock.now())
    analyser._onset_times.append(clock.now())
    assert analyser.get_onset_density() == 2 / 1.5
    clock.advance(2.0)
    assert analyser.get_onset_density() == 0.0


# ---------------------------------------------------------------------------
# _is_silence — vectorized equivalence with the old list-comp semantics
# ---------------------------------------------------------------------------

def test_is_silence_all_near_zero(analyser):
    assert analyser._is_silence(np.zeros(40, dtype=np.float32)) is True
    assert analyser._is_silence(np.full(40, 0.00009, dtype=np.float64)) is True
    assert analyser._is_silence(np.full(40, -0.00009, dtype=np.float64)) is True


def test_is_silence_boundary_is_exclusive(analyser):
    # old semantics: -0.0001 < n < 0.0001 (strict) → exactly 0.0001 is NOT silence
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


# ---------------------------------------------------------------------------
# _fold_bpm — octave folding into [85, 170)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# get_kick_strength — transient capture, ratio cap, silence gate
# ---------------------------------------------------------------------------

def _feed_sub_bass(analyser, values: list[float], beat_buffers: tuple[int, ...] = ()) -> None:
    """Drive the per-buffer sub-bass path the way analyse() does.

    `beat_buffers` are indices into `values` at which a beat is reported.
    """
    for i, v in enumerate(values):
        analyser._buffer_index += 1
        analyser._all_sub_bass_samples.append(v)
        if i in beat_buffers:
            analyser._pending_kick_beats.append(analyser._buffer_index)
        analyser._resolve_pending_kicks()


def test_kick_captures_transient_after_the_reported_beat(analyser):
    # The mel FFT delays the sub-bass peak past the beat, so a spike 3 buffers late is a kick.
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
    # Every beat spikes — a mean denominator would absorb them and flatten the ratio.
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
    analyser._rms_window.append(0.001)  # below _KICK_MIN_RMS
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


# ---------------------------------------------------------------------------
# The aubio filterbank, pinned beat-independently
# ---------------------------------------------------------------------------

# Computed by running the BASE commit's `_compute_mel_energies` (9749286, before
# any madmom code existed) over the deterministic signal below, then reproduced
# byte-for-byte at HEAD. Provenance matters here: a golden generated from the
# migrated code would prove nothing.
_MEL_GOLDEN_SHA256 = '883eae4b50a9ca0dc6c0a02f5772b5bf2ef4fe4f57f557bcf6d3a5fa2d2cac55'


def _mel_probe_signal() -> np.ndarray:
    """Sub-bass, mid and top plus a seeded noise floor — energy in every region
    of the bank, so a change to any band shows up in the hash."""
    rng = np.random.default_rng(20260729)
    t = np.arange(256 * 200) / 44100.0
    return (0.6 * np.sin(2 * np.pi * 55 * t)
            + 0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.1 * np.sin(2 * np.pi * 6000 * t)
            + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


def test_the_mel_filterbank_is_byte_identical_to_before_the_migration(analyser):
    """aubio keeps exactly one job — the 40-band mel bank every trained model
    and every kick/centroid/sub-bass feature is built on. This is the test that
    says the migration did not touch it, and it is deliberately independent of
    the beat grid: no beat source can move this number.
    """
    import hashlib
    signal = _mel_probe_signal()
    rows = [analyser._compute_mel_energies(signal[i * 256:(i + 1) * 256])
            for i in range(200)]
    stacked = np.asarray(rows, dtype=np.float32)
    assert stacked.shape == (200, 40)
    assert hashlib.sha256(stacked.tobytes()).hexdigest() == _MEL_GOLDEN_SHA256


# ---------------------------------------------------------------------------
# BPM, now derived from the beat stream rather than read off a tempo object
# ---------------------------------------------------------------------------

def test_bpm_is_unmeasured_until_two_intervals_exist(analyser):
    analyser.is_playing = True
    assert analyser.get_bpm() == 0
    analyser._beat_stream_times.extend([0.0, 0.47])
    assert analyser.get_bpm() == 0, 'one interval is not a measurement'


def test_bpm_is_the_median_interval_not_the_latest(analyser):
    """A single mistracked beat must not move the reported tempo — the whole
    reason for a median rather than madmom's own one-interval `tempo`."""
    analyser.is_playing = True
    analyser._beat_stream_times.extend([0.0, 0.47, 0.94, 0.95, 1.41, 1.88])
    assert analyser.get_bpm() == pytest.approx(60.0 / 0.47, rel=1e-6)


def test_bpm_is_octave_folded_from_the_beat_stream(analyser):
    analyser.is_playing = True
    # 300 BPM: 0.2 s intervals -> folds to 150.
    analyser._beat_stream_times.extend([0.0, 0.2, 0.4, 0.6, 0.8])
    assert analyser.get_bpm() == pytest.approx(150.0)


def test_an_unmeasured_bpm_is_not_a_bpm_change(analyser):
    """It is also the denominator, so this used to be a latent divide-by-zero
    on the first beat of every track."""
    analyser.is_playing = True
    analyser.last_bpm = 128.0
    assert analyser._has_bpm_changed(0.0) is False


# ---------------------------------------------------------------------------
# The debug beep, now triggered by madmom onsets
# ---------------------------------------------------------------------------

async def test_a_beep_fires_on_an_onset(analyser):
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([1.23], now) is True


async def test_no_onset_means_no_beep(analyser):
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([], now) is False


async def test_the_beep_refractory_survived_the_migration(analyser):
    """Same 75 ms as the aubio note detector had: a dense bar must click, not
    buzz. This is the '-d beeps identical in UX' requirement, pinned."""
    now = analyser._clock.now()
    analyser.last_note_detected = now - datetime.timedelta(seconds=1)
    assert await analyser._track_note([1.0], now) is True
    assert await analyser._track_note([1.01], now + datetime.timedelta(milliseconds=50)) is False
    assert await analyser._track_note([1.1], now + datetime.timedelta(milliseconds=80)) is True


# ---------------------------------------------------------------------------
# Backpressure state is loop-scoped, not song-scoped
# ---------------------------------------------------------------------------

async def test_a_song_reset_does_not_clear_the_drift_watchdog(analyser):
    """_reset_state() fires whenever the input goes quiet for 0.3 s — between
    tracks, and continuously on a silent input. Clearing the watchdog there
    emptied its rolling window over and over, so it re-entered its first shed
    level roughly once a second and logged a degradation every time. Found by
    running the tool against a silent live input, not by a test.
    """
    from lib.analyser.drift_watchdog import ShedLevel
    analyser._drift._level = ShedLevel.SECTION_DETECTION
    analyser._drift.total_drift_sec = 4.2
    analyser._reset_state()
    assert analyser._drift.level is ShedLevel.SECTION_DETECTION
    assert analyser._drift.total_drift_sec == 4.2


async def test_a_song_reset_does_clear_the_rhythm_stack(analyser):
    """The opposite case, and the reason the two are not reset together: madmom
    state IS song-scoped — carrying a tempo lock across a stop would fight the
    next track."""
    analyser._beat_stream_times.extend([1.0, 2.0, 3.0])
    analyser._reset_state()
    assert len(analyser._beat_stream_times) == 0


# ---------------------------------------------------------------------------
# Shedding section detection must not splice audio across the gap
# ---------------------------------------------------------------------------

def _fake_yamnet(analyser):
    """Make the detector run its buffering path without loading TensorFlow."""
    detector = analyser.yamnet_change_detector
    detector.yamnet_model = object()   # not None -> detect_change proceeds
    return detector


async def test_restoring_section_detection_clears_its_audio_buffers(analyser):
    """YAMNet is fed one aggregated block at a time and keeps three things
    between calls: a partial block, a rolling audio window, and a rolling
    embedding window scored against a MAD baseline. While shed it is handed no
    audio at all, so on restore the post-gap audio butt-joins the pre-gap tail
    inside one embedded signal and is compared against embeddings of music that
    stopped seconds ago. Same hazard as the onset chain's frame buffer, one
    level up.
    """
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
    """The mechanism, measured rather than predicted: without the clear, the
    window handed to the model contains pre-gap and post-gap samples adjacent to
    each other. This is what a spurious section change would be built from."""
    detector = _fake_yamnet(analyser)
    detector.rolling_window_audio.extend([1.0] * 4000)      # pre-gap marker
    pre_gap_len = len(detector.rolling_window_audio)

    # No clear: post-gap audio lands directly against the pre-gap tail.
    detector.rolling_window_audio.extend([-1.0] * 4000)     # post-gap marker
    seam = detector.rolling_window_audio[pre_gap_len - 1:pre_gap_len + 1]
    assert seam == [1.0, -1.0], 'expected a discontinuity at the seam'

    # With the clear, no pre-gap sample can reach the model.
    detector.reset()
    detector.rolling_window_audio.extend([-1.0] * 4000)
    assert 1.0 not in detector.rolling_window_audio


async def test_shedding_section_detection_does_not_clear_anything(analyser):
    """Only restoring costs a clear; shedding must not disturb state that a
    recovery seconds later would otherwise still be able to use."""
    detector = _fake_yamnet(analyser)
    detector.rolling_window_audio.extend([0.2] * 100)
    analyser._set_section_detection_enabled(False)
    assert detector.rolling_window_audio, 'shedding must not clear'


# ---------------------------------------------------------------------------
# A shed onset chain must report UNKNOWN density, never zero
# ---------------------------------------------------------------------------

async def test_shedding_onsets_reports_unknown_density_not_zero():
    """Zero is a measurement: it says the music went sparse. Shedding the onset
    detector produces the same number for a completely different reason, and
    within 1.5 s the rolling window empties and every consumer sees a genuine
    silence that is not happening -- pinning the classifier to BREAKDOWN with
    BUILDUP and DROP unreachable, while the beats keep flowing so nothing else
    contradicts it.
    """
    from lib.analyser.music_analyser import DENSITY_UNKNOWN, density_is_known
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    clock.advance(2.0)                    # past the start-up refill window
    analyser._onset_times.extend([clock.now()] * 6)
    assert density_is_known(analyser.get_onset_density())

    analyser._rhythm.set_onsets_enabled(False)
    assert analyser.get_onset_density() == DENSITY_UNKNOWN
    assert not density_is_known(analyser.get_onset_density())


async def test_density_stays_unknown_until_the_window_has_refilled():
    """Restoring the detector does not restore the measurement: the rolling
    window is empty and needs a full window of audio before its rate means
    anything. Reporting the partial count would be a low reading, not a missing
    one -- the same lie one second later."""
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
    """The trend is a ratio of two halves of this deque. A sentinel inside it
    would come back out as a number."""
    from lib.analyser.music_analyser import density_is_known
    analyser._rhythm.set_onsets_enabled(False)
    await analyser._track_beat([1.0, 2.0, 3.0, 4.0], analyser._clock.now())
    assert all(density_is_known(d) for d in analyser._density_samples)


# ---------------------------------------------------------------------------
# The rhythm heartbeat is loop-scoped, not song-scoped
# ---------------------------------------------------------------------------

async def test_the_rhythm_heartbeat_survives_repeated_song_resets(caplog):
    """_reset_state() fires after 0.3 s of quiet, so on a silent input it runs
    over and over as time passes. Scheduling the heartbeat there pushed its
    deadline forward on every reset, so it never printed -- precisely the
    situation where a heartbeat is the only evidence the loop is alive. Found on
    a live run against a silent cable: 33 s produced 67 drift lines and zero
    rhythm lines.

    The clock must ADVANCE between the resets for this to discriminate. A first
    draft reset in a tight loop at a frozen instant, where `now + interval` is
    the same deadline every time, and passed against the pre-fix code.
    """
    import logging
    clock = VirtualClock()
    analyser = _make_analyser(clock)

    with caplog.at_level(logging.INFO):
        for _ in range(40):          # 40 x 0.3 s = 12 s, past one interval
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


# ---------------------------------------------------------------------------
# The -d click actually reaches the audio (end to end)
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_debug_clicks_are_mixed_into_the_returned_audio():
    """`-d` is a feature the owner asked for by name, and the line that
    implements it -- `audio_signal += self.click_sound` in analyse() -- had no
    test. Everything around it was covered: the trigger logic, the refractory,
    the onset stream. A change that stopped the click ever reaching the audio
    would have left the whole suite green.

    Asserts the click is present, is exactly the click, lands only on beeps, and
    is absent entirely when the flag is off.
    """
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


# ---------------------------------------------------------------------------
# A song reset leaves density UNMEASURED, not measured-zero
# ---------------------------------------------------------------------------

async def test_a_song_reset_reports_unknown_density_during_the_refill():
    """_reset_state() empties the onset window, so for one window afterwards
    there is nothing to measure. It was reporting a confident 0.0 instead --
    fabricated data at every song start and every 15-minute reset, on the one
    path that runs constantly on a quiet input.

    The shed-restore path already got this right; the reset path swallowed its
    own epoch bump by re-seeding `_onset_epoch_seen` from the freshly-bumped
    counter, so the getter never saw an edge.
    """
    from lib.analyser.music_analyser import (DENSITY_UNKNOWN,
                                             _ONSET_DENSITY_WINDOW_SEC)
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    clock.advance(2.0)                    # past the start-up refill window
    analyser._onset_times.extend([clock.now()] * 6)
    assert analyser.get_onset_density() > 0

    analyser._reset_state()
    assert analyser.get_onset_density() == DENSITY_UNKNOWN, \
        'a reset fabricated a measured density of 0.0'

    clock.advance(_ONSET_DENSITY_WINDOW_SEC + 0.01)
    assert analyser.get_onset_density() != DENSITY_UNKNOWN


async def test_the_reset_refill_uses_the_same_semantics_as_shed_restore():
    """Two paths, one rule -- an empty window is unmeasured until it has had a
    window's worth of audio to fill."""
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
    """The same rule as reset and shed-restore, applied to the one case that
    happens on every single start: an empty rolling window has nothing in it,
    so a rate read off it is fabricated rather than sparse."""
    from lib.analyser.music_analyser import (DENSITY_UNKNOWN,
                                             _ONSET_DENSITY_WINDOW_SEC)
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    assert analyser.get_onset_density() == DENSITY_UNKNOWN
    clock.advance(_ONSET_DENSITY_WINDOW_SEC + 0.01)
    assert analyser.get_onset_density() == 0.0
