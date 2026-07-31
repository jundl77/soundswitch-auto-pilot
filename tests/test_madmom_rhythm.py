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
    beats = FakeStage()
    return MadmomRhythm(SR, beat_stage=beats, **kw), beats


def _ramp(n, start=0):
    return np.arange(start, start + n, dtype=np.float32)


def test_buffers_are_accumulated_into_exact_hops():
    r, beats = _rhythm()
    for i in range(0, 441 * 4, 256):
        r.process(_ramp(256, i))
    assert all(len(h) == HOP_SIZE for h in beats.hops)


def test_no_sample_is_lost_or_repeated_across_the_hop_boundary():
    r, beats = _rhythm()
    fed = []
    for i in range(0, 256 * 40, 256):
        buf = _ramp(256, i)
        fed.append(buf)
        r.process(buf)
    seen = np.concatenate(beats.hops)
    expected = np.concatenate(fed)[:len(seen)]
    assert np.array_equal(seen, expected)


def test_events_are_reported_in_the_buffer_whose_hop_produced_them():
    r, beats = _rhythm()
    beats.fire_on = {2}
    fired_beats = []
    for i in range(0, 256 * 20, 256):
        fired_beats.append(r.process(_ramp(256, i)).beats)
    assert sum(len(b) for b in fired_beats) == 1
    # hop 2 completes in the first buffer that has accumulated three hops
    assert fired_beats[-(-3 * HOP_SIZE // 256) - 1]


def test_a_buffer_that_completes_no_hop_reports_nothing():
    r, _ = _rhythm()
    assert not r.process(_ramp(256)).beats


def test_a_buffer_larger_than_a_hop_still_drains_completely():
    r, beats = _rhythm()
    r.process(_ramp(HOP_SIZE * 3 + 7))
    assert len(beats.hops) == 3


def _feed(r, buffers, start=0):
    out = []
    for i in range(buffers):
        out.append(r.process(_ramp(256, start + i * 256)))
    return out


def test_the_adapter_clock_tracks_audio_fed_not_frames_processed():
    r, beats = _rhythm()
    hops = 300
    beats.fire_on = {hops - 1}
    results = _feed(r, int(hops * HOP_SIZE / 256) + 2)
    fired = [e for e in results if e.beats]
    assert fired
    assert fired[0].beats[0] == pytest.approx((hops - 1) / 100.0, abs=0.011)


def test_reset_clears_stage_state_without_rebuilding_the_models():
    r, beats = _rhythm()
    built_before = beats.built
    r.process(_ramp(256))
    r.reset()
    assert beats.resets == 1
    assert beats.built == built_before


def test_reset_also_drops_the_partial_hop():
    r, beats = _rhythm()
    r.process(_ramp(256))
    r.reset()
    r.process(_ramp(HOP_SIZE, 10_000))
    assert np.array_equal(beats.hops[0], _ramp(HOP_SIZE, 10_000))


def test_the_caller_may_mutate_its_buffer_after_handing_it_over():
    r, beats = _rhythm()
    held = _ramp(256)
    r.process(held)
    held += 1000.0
    r.process(_ramp(HOP_SIZE, 256))
    assert beats.hops[0][0] == 0.0, 'the adapter kept a live reference to the caller'


def test_a_wrong_sample_rate_is_refused_rather_than_silently_resampled():
    with pytest.raises(ValueError, match='sample rate'):
        MadmomRhythm(48000, beat_stage=FakeStage())


def test_pending_latency_is_reported_and_bounded_by_one_hop():
    r, _ = _rhythm()
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
        beats = []
        for i in range(0, len(audio) - 256, 256):
            beats.extend(r.process(audio[i:i + 256]).beats)
        return beats

    first, second = run(), run()
    assert first == second
    assert first, 'expected beats on 20 s of real dance music'


