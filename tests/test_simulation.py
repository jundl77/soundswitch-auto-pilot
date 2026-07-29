import time
from pathlib import Path

import pytest

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE
from simulate.fake_audio_client import FileAudioClient
from simulate.runner import run_fast_simulation

SAMPLE_SONG = str(Path(__file__).parent.parent / 'samples' / 'generate_eric_prydz_192k.mp3')

# Module-level cache rather than a fixture, so each test keeps its own event loop.
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
    from simulate.evaluator import evaluate
    run = await _sample_run()
    result = evaluate(run['report'])
    assert result['passed'], f'sample-song evaluation failed: {result["criteria"]}'


@pytest.mark.integration
async def test_command_timing_is_exact_in_virtual_time():
    run = await _sample_run()
    log = run['report']['timing_log']
    assert log, 'expected commands in the timing log'
    for entry in log:
        assert abs(entry['actual_delta_sec'] - entry['target_delta_sec']) <= 0.010


@pytest.mark.integration
async def test_flush_tail_drains_queue():
    run = await _sample_run()
    assert run['pending_after'] == 0


@pytest.mark.integration
async def test_report_duration_matches_song_not_flush_tail():
    run = await _sample_run()
    assert run['report']['duration_sec'] == pytest.approx(run['song_sec'], abs=0.02)


@pytest.mark.integration
async def test_runs_much_faster_than_real_time():
    # 2x, not 4x: madmom's online nets measure ~3.8x real time here (aubio was ~46x).
    run = await _sample_run()
    assert run['wall_elapsed'] < run['song_sec'] / 2, (
        f"{run['song_sec']:.0f}s of audio took {run['wall_elapsed']:.1f}s wall"
    )


@pytest.mark.integration
async def test_simulation_is_deterministic():
    from simulate.evaluator import report_checksum
    first = (await _sample_run())['report']
    _, event_buffer, command_queue = await run_fast_simulation(
        FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, SAMPLE_SONG)
    )
    second = event_buffer.to_report(command_queue.get_timing_log())
    assert second == first
    assert report_checksum(second) == report_checksum(first)
