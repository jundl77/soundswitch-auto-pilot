"""
Integration tests: run the full simulation pipeline without hardware on a
virtual clock — seconds of song time complete in well under a second of wall
time, and results are fully deterministic.

Marked @pytest.mark.integration so you can skip with:
  pytest -m "not integration"
"""

import pytest

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE
from lib.clock import VirtualClock
from simulate.fake_audio_client import BeepAudioClient
from simulate.runner import (
    build_simulation,
    run_fast_simulation,
    run_simulation,
    print_timing_report,
)


@pytest.mark.integration
async def test_simulation_runs_without_error():
    """Smoke test: the full pipeline runs for 5 virtual seconds without raising."""
    clock = VirtualClock()
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    components, command_queue = build_simulation(audio_client, clock=clock)
    await run_simulation(components, duration_sec=5.0, clock=clock)


@pytest.mark.integration
async def test_simulation_timing_passes():
    """
    Timing validation: beat commands enqueued at T must fire within 10 ms of
    T + delay. In virtual time the error cannot exceed one buffer quantum
    (256/44100 ≈ 5.8 ms), so 10 ms is a meaningful regression guard.
    """
    clock = VirtualClock()
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    components, command_queue = build_simulation(audio_client, clock=clock)
    await run_simulation(components, duration_sec=8.0, clock=clock)

    log = command_queue.get_timing_log()
    assert log, 'expected beat commands on the virtual clock (deterministic input)'

    passed = print_timing_report(command_queue, tolerance_sec=0.010)
    assert passed, 'one or more beat commands exceeded 10 ms timing tolerance'


async def _run_fast_beep_sim(duration_sec: float) -> dict:
    """One fast, seeded, virtual-clock run → full report dict.

    Uses run_fast_simulation so this test exercises the exact same determinism
    contract (seed + virtual clock + infinite window) as the CLI's fast mode.
    """
    _, event_buffer, command_queue = await run_fast_simulation(
        lambda clock: BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock),
        duration_sec=duration_sec,
    )
    return event_buffer.to_report(command_queue.get_timing_log())


@pytest.mark.integration
async def test_fast_simulation_is_deterministic():
    """Two identical fast runs must produce byte-identical reports."""
    report_a = await _run_fast_beep_sim(20.0)
    report_b = await _run_fast_beep_sim(20.0)
    assert report_a == report_b


@pytest.mark.integration
async def test_report_duration_matches_run_not_flush_tail():
    """The flush tail advances the clock past the end; the report must not
    inflate duration_sec by that tail (it would bias beat_detection_rate)."""
    report = await _run_fast_beep_sim(20.0)
    assert report['duration_sec'] == pytest.approx(20.0, abs=0.02)


async def test_fast_simulation_rejects_endless_source_with_infinite_duration():
    """Default duration is infinite; a source that never exhausts must be
    rejected up front instead of spinning the loop forever."""
    with pytest.raises(ValueError, match='duration_sec'):
        await run_fast_simulation(
            lambda clock: BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
        )
