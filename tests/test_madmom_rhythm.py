import numpy as np
import pytest

from lib.analyser.madmom_rhythm import HOP_SIZE, MadmomRhythm

SR = 44100


class FakeStage:
    # Stamps events from its OWN frame count, exactly as madmom's decoders do —
    # which is why a stage that is skipped for a while reports stale times.
    def __init__(self):
        self.hops = []
        self.fire_on = set()
        self.resets = 0
        self.built = 1

    def __call__(self, hop):
        self.hops.append(np.asarray(hop).copy())
        if len(self.hops) - 1 in self.fire_on:
            return np.array([(len(self.hops) - 1) / 100.0], dtype=float)
        return np.zeros(0)

    def reset(self):
        self.resets += 1
        self.hops = []


def _rhythm(**kw):
    beats, onsets = FakeStage(), FakeStage()
    return MadmomRhythm(SR, beat_stage=beats, onset_stage=onsets, **kw), beats, onsets


def _ramp(n, start=0):
    return np.arange(start, start + n, dtype=np.float32)


def test_buffers_are_accumulated_into_exact_hops():
    r, beats, _ = _rhythm()
    for i in range(0, 441 * 4, 256):
        r.process(_ramp(256, i))
    assert all(len(h) == HOP_SIZE for h in beats.hops)


def test_no_sample_is_lost_or_repeated_across_the_hop_boundary():
    r, beats, _ = _rhythm()
    fed = []
    for i in range(0, 256 * 40, 256):
        buf = _ramp(256, i)
        fed.append(buf)
        r.process(buf)
    seen = np.concatenate(beats.hops)
    expected = np.concatenate(fed)[:len(seen)]
    assert np.array_equal(seen, expected)


def test_both_stages_see_the_same_hops():
    r, beats, onsets = _rhythm()
    for i in range(0, 256 * 20, 256):
        r.process(_ramp(256, i))
    assert len(beats.hops) == len(onsets.hops)
    assert all(np.array_equal(a, b) for a, b in zip(beats.hops, onsets.hops))


def test_events_are_reported_in_the_buffer_whose_hop_produced_them():
    r, beats, onsets = _rhythm()
    beats.fire_on = {2}
    onsets.fire_on = {2, 3}
    fired_beats, fired_onsets = [], []
    for i in range(0, 256 * 20, 256):
        result = r.process(_ramp(256, i))
        fired_beats.append(result.beats)
        fired_onsets.append(result.onsets)
    assert sum(len(b) for b in fired_beats) == 1
    assert sum(len(o) for o in fired_onsets) == 2


def test_a_buffer_that_completes_no_hop_reports_nothing():
    r, _, _ = _rhythm()
    result = r.process(_ramp(256))
    assert not result.beats and not result.onsets


def test_a_buffer_larger_than_a_hop_still_drains_completely():
    r, beats, _ = _rhythm()
    r.process(_ramp(HOP_SIZE * 3 + 7))
    assert len(beats.hops) == 3


def test_shedding_onsets_stops_the_onset_work_but_not_the_beat_work():
    r, beats, onsets = _rhythm()
    r.set_onsets_enabled(False)
    for i in range(0, 256 * 20, 256):
        r.process(_ramp(256, i))
    assert beats.hops, 'beat tracking must survive shedding'
    assert not onsets.hops, 'the shed stage must not be called at all'


def test_shedding_is_reversible():
    r, _, onsets = _rhythm()
    r.set_onsets_enabled(False)
    for i in range(0, 256 * 20, 256):
        r.process(_ramp(256, i))
    r.set_onsets_enabled(True)
    for i in range(0, 256 * 20, 256):
        r.process(_ramp(256, i))
    assert onsets.hops


def _feed(r, buffers, start=0):
    out = []
    for i in range(buffers):
        out.append(r.process(_ramp(256, start + i * 256)))
    return out


def test_event_times_come_from_the_adapter_not_from_the_stages():
    # A shed chain's frame counter stops while the other's runs on, so the two
    # streams silently stop sharing a time base unless the adapter owns the clock.
    r, beats, onsets = _rhythm()
    r.set_onsets_enabled(False)
    _feed(r, 200)
    r.set_onsets_enabled(True)
    beats.fire_on = {len(beats.hops)}
    onsets.fire_on = {len(onsets.hops)}
    results = _feed(r, 20, start=256 * 200)
    fired = [e for e in results if e.beats or e.onsets]
    assert fired, 'expected an event after restore'
    event = fired[0]
    assert event.beats and event.onsets, 'both chains should fire on the same hop'
    assert event.beats[0] == pytest.approx(event.onsets[0]), (
        'beat and onset times diverged across a shed — they are not on one clock')


def test_the_adapter_clock_tracks_audio_fed_not_frames_processed():
    r, beats, _ = _rhythm()
    r.set_onsets_enabled(False)
    hops = 300
    beats.fire_on = {hops - 1}
    results = _feed(r, int(hops * HOP_SIZE / 256) + 2)
    fired = [e for e in results if e.beats]
    assert fired
    assert fired[0].beats[0] == pytest.approx((hops - 1) / 100.0, abs=0.011)


def test_restoring_a_shed_chain_clears_its_stale_state():
    r, _, onsets = _rhythm()
    r.set_onsets_enabled(False)
    _feed(r, 200)
    resets_before = onsets.resets
    r.set_onsets_enabled(True)
    assert onsets.resets == resets_before + 1, 'restore must clear the shed chain'


def test_shedding_does_not_reset_anything():
    r, beats, onsets = _rhythm()
    _feed(r, 20)
    beat_resets = beats.resets
    r.set_onsets_enabled(False)
    assert beats.resets == beat_resets


def test_reset_clears_stage_state_without_rebuilding_the_models():
    r, beats, onsets = _rhythm()
    built_before = (beats.built, onsets.built)
    r.process(_ramp(256))
    r.reset()
    assert beats.resets == 1 and onsets.resets == 1
    assert (beats.built, onsets.built) == built_before


def test_reset_also_drops_the_partial_hop():
    r, beats, _ = _rhythm()
    r.process(_ramp(256))
    r.reset()
    r.process(_ramp(HOP_SIZE, 10_000))
    assert np.array_equal(beats.hops[0], _ramp(HOP_SIZE, 10_000))


def test_the_caller_may_mutate_its_buffer_after_handing_it_over():
    r, beats, _ = _rhythm()
    held = _ramp(256)
    r.process(held)
    held += 1000.0
    r.process(_ramp(HOP_SIZE, 256))
    assert beats.hops[0][0] == 0.0, 'the adapter kept a live reference to the caller'


def test_a_wrong_sample_rate_is_refused_rather_than_silently_resampled():
    with pytest.raises(ValueError, match='sample rate'):
        MadmomRhythm(48000, beat_stage=FakeStage(), onset_stage=FakeStage())


def test_pending_latency_is_reported_and_bounded_by_one_hop():
    r, _, _ = _rhythm()
    assert r.pending_latency_sec == 0.0
    r.process(_ramp(256))
    assert 0 < r.pending_latency_sec < HOP_SIZE / SR
    r.process(_ramp(256))
    assert r.pending_latency_sec < HOP_SIZE / SR


@pytest.mark.integration
def test_the_real_stack_streams_and_is_deterministic():
    from pathlib import Path

    from lib.audio_config import BUFFER_SIZE
    from simulate.fake_audio_client import FileAudioClient
    from tests.conftest import anchor_mp3_path

    mp3 = Path(anchor_mp3_path())
    client = FileAudioClient(SR, BUFFER_SIZE, str(mp3))
    client.start_streams()
    audio = np.concatenate([client.read() for _ in range(SR * 20 // BUFFER_SIZE)])

    def run():
        r = MadmomRhythm(SR)
        beats, onsets = [], []
        for i in range(0, len(audio) - 256, 256):
            out = r.process(audio[i:i + 256])
            beats.extend(out.beats)
            onsets.extend(out.onsets)
        return beats, onsets

    first, second = run(), run()
    assert first == second
    assert first[0], 'expected beats on 20 s of real dance music'


def test_the_shipped_onset_threshold_is_the_calibrated_one():
    import json
    from pathlib import Path
    from lib.analyser.madmom_rhythm import ONSET_THRESHOLD

    evidence = json.loads(
        (Path(__file__).resolve().parents[1] / 'training' / 'onset_operating_point.json').read_text())
    assert ONSET_THRESHOLD == evidence['chosen_threshold']
    assert evidence['stability']['mode'] == evidence['chosen_threshold']


def test_a_multi_event_decoder_return_still_counts_one_onset_per_hop():
    from lib.analyser.madmom_rhythm import HOP_SIZE

    class BurstStage(FakeStage):
        def __call__(self, hop):
            self.hops.append(np.asarray(hop).copy())
            return np.array([0.01, 0.02, 0.03], dtype=float)

    rhythm, _, _ = _rhythm()
    rhythm._onsets = BurstStage()
    events = rhythm.process(np.zeros(HOP_SIZE, dtype=np.float32))
    assert len(events.onsets) == 1
