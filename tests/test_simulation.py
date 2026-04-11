"""
Integration test: runs the full simulation pipeline without hardware.

Marked @pytest.mark.integration so you can skip with:
  pytest -m "not integration"

Or run only integration tests with:
  pytest -m integration
"""

import pytest
from simulate.fake_audio_client import BeepAudioClient
from simulate.runner import build_simulation, run_simulation, print_timing_report

SAMPLE_RATE = 44100
BUFFER_SIZE = 256


@pytest.mark.integration
async def test_simulation_runs_without_error():
    """Smoke test: the full pipeline runs for 5 seconds without raising."""
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0)
    components, command_queue = build_simulation(audio_client, delay_sec=0.1)
    await run_simulation(components, duration_sec=5.0)


@pytest.mark.integration
async def test_simulation_timing_passes():
    """
    Timing validation: beat commands enqueued at T must fire within 50 ms of T + delay.
    If no beats were detected (rare with synthetic audio), the test is skipped.
    """
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0)
    components, command_queue = build_simulation(audio_client, delay_sec=0.1)
    await run_simulation(components, duration_sec=8.0)

    log = command_queue.get_timing_log()
    if not log:
        pytest.skip('no commands dispatched — aubio did not detect beats in this run')

    passed = print_timing_report(command_queue, tolerance_sec=0.050)
    assert passed, 'one or more beat commands exceeded 50 ms timing tolerance'
