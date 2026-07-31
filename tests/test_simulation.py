import sys
import time
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import run_eval_set  # noqa: E402  (needs the path insert above)
from select_eval_set import EVAL_SET_FILE, load_eval_set  # noqa: E402

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE  # noqa: E402
from simulate.fake_audio_client import FileAudioClient  # noqa: E402
from simulate.runner import run_fast_simulation  # noqa: E402

# The eval set itself is committed, so this is safe at import time; only the
# audio it names can be missing.
EVAL_SET = load_eval_set(EVAL_SET_FILE)
DATA_DIR = run_eval_set.corpus_dir()

# Named through the selector rather than by literal id: if the set is ever
# re-frozen, the wall-time budget these protect follows the new durations.
SAMPLE_TRACK_ID = run_eval_set.shortest_track_ids(EVAL_SET, 1)[0]
BENCH_TRACK_IDS = run_eval_set.shortest_track_ids(EVAL_SET, 3)


def _require_audio(track_ids: list) -> list:
    """The eval-set records for ``track_ids``, or one clear failure line."""
    tracks = run_eval_set.select_tracks(EVAL_SET, track_ids)
    problems = run_eval_set.missing_inputs(DATA_DIR, tracks)
    if problems:
        # One line, everything that is missing: a fresh clone should not have to
        # re-run the suite three times to learn it needs three files.
        pytest.fail("; ".join(problems))
    return tracks


# Module-level cache rather than a fixture, so each test keeps its own event loop.
_run_cache: dict = {}


async def _sample_run() -> dict:
    if not _run_cache:
        track = _require_audio([SAMPLE_TRACK_ID])[0]
        mp3 = str(run_eval_set.audio_path(DATA_DIR, track["youtube_id"]))
        wall_start = time.monotonic()
        audio_client, event_buffer, command_queue = await run_fast_simulation(
            FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, mp3)
        )
        _run_cache.update(
            report=event_buffer.to_report(command_queue.get_timing_log()),
            wall_elapsed=time.monotonic() - wall_start,
            song_sec=audio_client.duration_sec,
            pending_after=command_queue.pending,
        )
    return _run_cache


@pytest.mark.integration
@pytest.mark.xfail(strict=True, reason="D13: the demolished branch commits no "
                                        "intent, so the plumbing verdict's four "
                                        "show criteria cannot pass until Task 9 "
                                        "rewires the decoder")
async def test_sample_song_evaluation_passes():
    """The plumbing verdict -- "did anything happen at all".

    Between the demolition and the rewire the answer is deliberately no: beats,
    silence and a held intent are the whole show (D13), and this asks for two
    intent changes, two distinct intents, two channels and one effect change.
    Strict, so it fails the moment Task 9 makes it true again and the marker
    cannot be left behind.
    """
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
    track = _require_audio([SAMPLE_TRACK_ID])[0]
    mp3 = str(run_eval_set.audio_path(DATA_DIR, track["youtube_id"]))
    first = (await _sample_run())['report']
    _, event_buffer, command_queue = await run_fast_simulation(
        FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, mp3)
    )
    second = event_buffer.to_report(command_queue.get_timing_log())
    assert second == first
    assert report_checksum(second) == report_checksum(first)


@pytest.mark.integration
@pytest.mark.xfail(strict=True, reason="the demolition moved all three report "
                                        "checksums and zeroed the label-aligned "
                                        "scores; Task 15 re-cuts the committed "
                                        "baseline, once")
def test_eval_set_head_matches_the_committed_baseline():
    """Three eval-set tracks through the benchmark runner, compare mode.

    This is the gate the old plumbing-PASS verdict used to be: it fails on a
    changed report checksum (the pipeline behaves differently) and on a score
    that fell below the committed baseline (the show got worse).  A subset run,
    so the ten-track aggregate is not compared -- see run_eval_set.compare.

    Read the printed table on failure: it names the track, the metric, and the
    direction, and says whether re-cutting the baseline is the right answer.

    Marked xfail through the demolition, and strict on purpose.  The report
    schema lost four beat columns and one metric, so every checksum moved; the
    engine commits nothing, so every label-aligned score is 0.  Re-cutting the
    baseline here would bless the intermediate state as the benchmark, which is
    why the plan gives that job to Task 15 and gives it once.  When the rewire
    lands and the baseline is re-cut this XPASSes, which pytest reports as a
    failure until the marker comes off.
    """
    _require_audio(BENCH_TRACK_IDS)
    exit_code = run_eval_set.main([
        "--only", ",".join(BENCH_TRACK_IDS),
        "--data-dir", str(DATA_DIR),
        "--quiet",
    ])
    assert exit_code == 0, (
        f"eval-set benchmark failed on {', '.join(BENCH_TRACK_IDS)} -- see the "
        f"table above; re-cut with 'uv run python training/run_eval_set.py "
        f"--write-baseline' if the change is intended"
    )
