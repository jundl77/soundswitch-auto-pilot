"""The buffer-to-hop adapter: madmom's framing is its problem, not the analyser's.

The live pipeline reads 256-sample buffers; madmom's online models are trained
at 441-sample hops. Everything about that mismatch lives in one place, and these
tests pin the properties the analyser above it is entitled to assume.

The unit tests drive fake processors so they neither load eight pickled LSTMs
nor take a second each. One integration test streams the real stack.
"""

import numpy as np
import pytest

from lib.analyser.madmom_rhythm import HOP_SIZE, MadmomRhythm

SR = 44100


class FakeStage:
    """Stands in for one madmom chain: records every hop, fires on demand."""

    def __init__(self):
        self.hops = []
        self.fire_on = set()
        self.resets = 0
        self.built = 1

    def __call__(self, hop):
        self.hops.append(np.asarray(hop).copy())
        if len(self.hops) - 1 in self.fire_on:
            return np.array([len(self.hops) - 1], dtype=float)
        return np.zeros(0)

    def reset(self):
        self.resets += 1


def _rhythm(**kw):
    beats, onsets = FakeStage(), FakeStage()
    return MadmomRhythm(SR, beat_stage=beats, onset_stage=onsets, **kw), beats, onsets


def _ramp(n, start=0):
    return np.arange(start, start + n, dtype=np.float32)


def test_buffers_are_accumulated_into_exact_hops():
    r, beats, _ = _rhythm()
    for i in range(0, 441 * 4, 256):
        r.process(_ramp(256, i))
    # 441*4 = 1764 samples fed in 256-sample buffers -> six buffers = 1536,
    # which is three whole hops with 213 samples still held.
    assert all(len(h) == HOP_SIZE for h in beats.hops)


def test_no_sample_is_lost_or_repeated_across_the_hop_boundary():
    """A resampling or off-by-one bug here would be invisible downstream — the
    beat stream would simply be slightly wrong forever."""
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
    result = r.process(_ramp(256))   # 256 < 441
    assert not result.beats and not result.onsets


def test_a_buffer_larger_than_a_hop_still_drains_completely():
    """Buffer size is a config value; the adapter must not assume it is small."""
    r, beats, _ = _rhythm()
    r.process(_ramp(HOP_SIZE * 3 + 7))
    assert len(beats.hops) == 3


def test_shedding_onsets_stops_the_onset_work_but_not_the_beat_work():
    """What the drift watchdog buys: the expensive half can be dropped while
    beat tracking keeps running."""
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


def test_reset_clears_stage_state_without_rebuilding_the_models():
    """`MusicAnalyser` resets every 15 minutes and on every sound stop.
    Rebuilding would reload eight pickled LSTMs mid-show."""
    r, beats, onsets = _rhythm()
    built_before = (beats.built, onsets.built)
    r.process(_ramp(256))
    r.reset()
    assert beats.resets == 1 and onsets.resets == 1
    assert (beats.built, onsets.built) == built_before


def test_reset_also_drops_the_partial_hop():
    """Carrying 200 stale samples across a sound-stop would splice unrelated
    audio into the first frame of the next track."""
    r, beats, _ = _rhythm()
    r.process(_ramp(256))            # 256 samples held, no hop yet
    r.reset()
    r.process(_ramp(HOP_SIZE, 10_000))
    assert np.array_equal(beats.hops[0], _ramp(HOP_SIZE, 10_000))


def test_the_caller_may_mutate_its_buffer_after_handing_it_over():
    """`MusicAnalyser.analyse` mixes the debug click into the very buffer it
    just passed here. If the adapter aliased it, every click would be fed back
    into the onset detector that triggered it."""
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
    """The adapter's own contribution to look-ahead spend, stated rather than
    assumed. It can never exceed one hop minus one sample."""
    r, _, _ = _rhythm()
    assert r.pending_latency_sec == 0.0
    r.process(_ramp(256))
    assert 0 < r.pending_latency_sec < HOP_SIZE / SR
    r.process(_ramp(256))
    assert r.pending_latency_sec < HOP_SIZE / SR


@pytest.mark.integration
def test_the_real_stack_streams_and_is_deterministic():
    """Two fresh adapters over identical audio must agree exactly — the whole
    determinism story rests on madmom's online path having no hidden RNG."""
    from pathlib import Path
    npy = (Path(__file__).parent.parent / 'samples'
           / 'generate_eric_prydz_192k.mp3.44100.npy')
    if not npy.exists():
        pytest.skip('decode cache absent; run the sim once to create it')
    audio = np.load(npy)[: SR * 20]

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
    assert first[0], 'expected beats on 20 s of four-on-the-floor'
