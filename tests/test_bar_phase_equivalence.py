"""The online fusion against the offline reference it ports.

The #330 operating point was measured by cc_fuse_score.py (campaign dir):
per-beat vote aggregation, gated log-odds evidence, `phase_tracking`'s
fixed-lag Viterbi and forward-only application.  The runtime's
``BarPhaseFusion`` re-states that pipeline one beat at a time, so this test
builds the offline reference out of ``phase_tracking``'s own functions plus a
verbatim port of cc_fuse_score's emission builder, drives the online form in
arrival order, and demands the identical committed position stream.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "training" / "phase_tracking",):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import phase_tracking as pt  # noqa: E402

from lib.analyser.bar_tracker import FPS, TrackerChunk  # noqa: E402
from lib.engine.bar_phase import (AGG_HALFWIDTH, BETA, GAMMA,  # noqa: E402
                                  GATE_T, LAG_BEATS, MAX_ADVANCE, SIGMA, ETA,
                                  BarPhaseFusion)

STRIDE_FRAMES = 125
MARGIN_FRAMES = 6


# ---- the offline reference: cc_fuse_score.py's functions, verbatim ---------- #
def _logit(x: float) -> float:
    return float(np.log(x / (1.0 - x)))


def per_beat_votes(beats, db_logit, avail_frame, halfwidth):
    logits = np.clip(db_logit.astype(np.float64), -30.0, 30.0)
    sig = 1.0 / (1.0 + np.exp(-logits))
    n_frames = sig.size
    frames = np.round(beats * FPS).astype(np.int64)
    p = np.full(beats.size, np.nan)
    avail_t = np.full(beats.size, np.inf)
    for j in range(beats.size):
        lo = max(0, frames[j] - halfwidth)
        hi = min(n_frames, frames[j] + halfwidth + 1)
        if lo >= hi:
            continue
        p[j] = sig[lo:hi].max()
        avail_t[j] = avail_frame[lo:hi].max() / FPS
    return p, avail_t


def build_emission(beats, p, avail_t, gate_t, beta):
    n = beats.size
    emission = np.zeros((n, 4), dtype=np.float64)
    for j in range(n):
        pj = p[j]
        if not np.isfinite(pj) or pj < gate_t:
            continue
        strength = _logit(min(pj, 1.0 - GAMMA)) - _logit(gate_t)
        r = int(np.searchsorted(beats, avail_t[j], side="left"))
        if r >= n:
            continue
        emission[r, (r - j) % 4] += beta * strength
    return emission


def offline_committed(beats, db_logit, avail_frame):
    p, avail_t = per_beat_votes(beats, db_logit, avail_frame, AGG_HALFWIDTH)
    emission = build_emission(beats, p, avail_t, GATE_T, BETA)
    ratios = pt.interval_ratios(beats, window=8)
    log_transition = pt.advance_log_weights(ratios[1:], sigma=SIGMA, eta=ETA,
                                            max_advance=MAX_ADVANCE)
    log_start = np.log(np.asarray(pt.LIVE_START_PRIOR, dtype=np.float64))
    decided = pt.fixed_lag_viterbi(log_transition, emission, log_start,
                                   LAG_BEATS)
    base = np.arange(beats.size, dtype=np.int64) % 4
    committed, _ = pt.apply_forward_only(base, decided, LAG_BEATS)
    return committed


# ---- the synthetic session -------------------------------------------------- #
def make_session(seed, duration_sec=240.0, bpm=128.0, slip_every=40):
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    beats = []
    t = 0.35
    k = 0
    while t < duration_sec - 12.0:
        beats.append(t + rng.normal(0.0, 0.012))
        t += period
        k += 1
        if slip_every and k % slip_every == 0:
            t += period if k % (2 * slip_every) else -period * 0.5
    beats = np.asarray(sorted(beats))

    n_frames = int(duration_sec * FPS)
    logits = rng.normal(-6.0, 1.5, n_frames)
    for j, beat in enumerate(beats):
        if j % 4 == 0:
            frame = int(np.round(beat * FPS))
            if 0 <= frame < n_frames:
                logits[frame] = rng.normal(4.0, 2.0)
    logits = logits.astype(np.float32)

    chunks = []
    # Float frame stamps so the reference's avail_t (max/FPS) is exactly the
    # runtime chunks' avail_sec rather than a rounded frame count.
    avail_frame = np.zeros(n_frames, dtype=np.float64)
    prev_safe = 0
    p = 1
    while True:
        end = p * STRIDE_FRAMES
        if end > n_frames:
            break
        safe = end - MARGIN_FRAMES
        if safe > prev_safe:
            avail_sec = end / FPS + 0.0264
            chunks.append(TrackerChunk(prev_safe, safe,
                                       logits[prev_safe:safe], avail_sec))
            avail_frame[prev_safe:safe] = avail_sec * FPS
            prev_safe = safe
        p += 1
    return beats, logits, chunks, avail_frame, prev_safe


class _FakeTracker:
    alive = True


def online_committed(beats, chunks):
    fusion = BarPhaseFusion(_FakeTracker(), None)
    events = ([("chunk", c.avail_sec, c) for c in chunks]
              + [("beat", t, t) for t in beats])
    events.sort(key=lambda item: (item[1], item[0] == "beat"))
    out = []
    for kind, _, payload in events:
        if kind == "chunk":
            fusion.push_chunk(payload)
        else:
            out.append(fusion.push_beat(payload))
    return np.asarray(out, dtype=np.int64)


@pytest.mark.parametrize("seed,slip_every", [(1, 0), (2, 40), (3, 25), (4, 60)])
def test_the_online_fusion_commits_the_offline_references_stream(seed,
                                                                 slip_every):
    beats, logits, chunks, avail_frame, coverage = make_session(
        seed, slip_every=slip_every)
    assert beats[-1] * FPS < coverage - FPS, "beats must end inside coverage"
    expected = offline_committed(beats, logits[:coverage],
                                 avail_frame[:coverage])
    got = online_committed(beats, chunks)
    assert np.array_equal(got, expected), (
        f"first disagreement at beat "
        f"{int(np.flatnonzero(got != expected)[0])} of {beats.size}")


def test_the_fusion_with_no_evidence_converges_to_the_anchor1_count():
    """The start prior IS the warm-up anchor: with every vote gated out, the
    trellis decides position 1 for the first beat (147/215 val tracks) and the
    one forward-only correction leaves exactly the shipping counting grid."""
    beats = 0.5 + 0.46 * np.arange(200)
    fusion = BarPhaseFusion(_FakeTracker(), None)
    got = [fusion.push_beat(t) for t in beats]
    assert got[:3] == [0, 1, 2]
    assert got[3:] == [(j + 1) % 4 for j in range(3, 200)]
    assert fusion.corrections == 1


def test_a_dead_tracker_freezes_the_offset_and_keeps_counting():
    beats, logits, chunks, _, _ = make_session(2, slip_every=40)
    tracker = _FakeTracker()
    fusion = BarPhaseFusion(tracker, None)
    events = ([("chunk", c.avail_sec, c) for c in chunks]
              + [("beat", t, t) for t in beats])
    events.sort(key=lambda item: (item[1], item[0] == "beat"))
    out = []
    for k, (kind, _, payload) in enumerate(events):
        if k == len(events) // 2:
            tracker.alive = False
        if kind == "chunk":
            fusion.push_chunk(payload)
        else:
            out.append(fusion.push_beat(payload))
    tail = out[len(out) // 2:]
    steps = {(b - a) % 4 for a, b in zip(tail, tail[1:])}
    assert steps == {1}, "after the tracker died every advance must be +1"


def test_a_reanchor_rebirths_at_position_zero():
    fusion = BarPhaseFusion(_FakeTracker(), None)
    for j, t in enumerate(0.5 * np.arange(20)):
        fusion.push_beat(t)
    assert fusion.reanchor(30.0) == 0
    assert fusion.push_beat(30.5) == 1
