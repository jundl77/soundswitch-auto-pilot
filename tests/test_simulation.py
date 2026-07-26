"""
Simulation tests: run the bundled sample track through the exact production
pipeline on the virtual clock — real MP3, real DSP, real scoring. There is one
simulation mode (real audio files, paced real-time or sped-up); these tests
cover it end-to-end. The run executes once and is shared across assertions,
plus one more full run to prove determinism.

Marked @pytest.mark.integration so you can skip with:
  pytest -m "not integration"
"""

import time
from pathlib import Path

import pytest

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE
from simulate.fake_audio_client import FileAudioClient
from simulate.runner import run_fast_simulation

SAMPLE_SONG = str(Path(__file__).parent.parent / 'samples' / 'generate_eric_prydz_192k.mp3')

# One full run shared by the assertions below (module-level cache rather than a
# fixture so each test's event loop stays independent).
_run_cache: dict = {}


async def _sample_run() -> dict:
    if not _run_cache:
        wall_start = time.monotonic()
        audio_client, event_buffer, command_queue = await run_fast_simulation(
            FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, SAMPLE_SONG)
        )
        _run_cache.update(
            report=event_buffer.to_report(command_queue.get_timing_log()),
            wall_elapsed=time.monotonic() - wall_start,
            song_sec=audio_client.duration_sec,
            pending_after=command_queue.pending,
        )
    return _run_cache


@pytest.mark.integration
async def test_sample_song_evaluation_passes():
    """The evaluator's PASS verdict on the bundled track — the same scoring
    `auto_pilot simulate file` prints."""
    from simulate.evaluator import evaluate
    run = await _sample_run()
    result = evaluate(run['report'])
    assert result['passed'], f'sample-song evaluation failed: {result["criteria"]}'


@pytest.mark.integration
async def test_command_timing_is_exact_in_virtual_time():
    """Every queued command fires within 10 ms of its look-ahead target (one
    buffer quantum ≈ 5.8 ms is the max possible error on the virtual clock)."""
    run = await _sample_run()
    log = run['report']['timing_log']
    assert log, 'expected commands in the timing log'
    for entry in log:
        assert abs(entry['actual_delta_sec'] - entry['target_delta_sec']) <= 0.010


@pytest.mark.integration
async def test_flush_tail_drains_queue():
    """Commands enqueued in the final look-ahead window still fire: nothing is
    left pending after the run."""
    run = await _sample_run()
    assert run['pending_after'] == 0


@pytest.mark.integration
async def test_report_duration_matches_song_not_flush_tail():
    """The flush tail advances the clock past the end; duration_sec must not
    inflate (it would bias beat_detection_rate)."""
    run = await _sample_run()
    assert run['report']['duration_sec'] == pytest.approx(run['song_sec'], abs=0.02)


@pytest.mark.integration
async def test_runs_much_faster_than_real_time():
    """The track must simulate in a small fraction of its length — a regression
    guard against accidental wall-clock pacing."""
    run = await _sample_run()
    assert run['wall_elapsed'] < run['song_sec'] / 4, (
        f"{run['song_sec']:.0f}s of audio took {run['wall_elapsed']:.1f}s wall"
    )


@pytest.mark.integration
async def test_simulation_is_deterministic():
    """A second full run produces a byte-identical report (and checksum)."""
    from simulate.evaluator import report_checksum
    first = (await _sample_run())['report']
    _, event_buffer, command_queue = await run_fast_simulation(
        FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, SAMPLE_SONG)
    )
    second = event_buffer.to_report(command_queue.get_timing_log())
    assert second == first
    assert report_checksum(second) == report_checksum(first)
