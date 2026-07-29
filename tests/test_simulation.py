"""Simulation tests: run real, expert-labeled music through the exact
production pipeline on the virtual clock -- real MP3, real DSP, real scoring.

The benchmark is the FROZEN EVAL SET (``training/eval_set.json``, ten Raveform
tracks spanning 117-174 BPM), not a single bundled track, and it is excluded
from every neural training and validation split.  Two layers:

* one track, run once and shared across the assertions below, pins the
  *mechanism* -- command timing on the virtual clock, the flush tail, report
  duration, speed, and byte-identical determinism.  Those are track-agnostic,
  so this uses the SHORTEST track of the set: the whole file has a wall-time
  budget and a longer track buys nothing here.
* three tracks through ``training/run_eval_set.py`` in compare mode pins the
  *behaviour* -- per-track report checksums and label-aligned scores against the
  committed baseline.  Also the shortest three, for the same reason (the full
  ten-track benchmark is a ~2 min manual run, not a test).

Both things a run reads are COMMITTED -- the ten mp3s under derived, opaque
names in ``training/eval_audio/`` and their labels in ``training/eval_labels.json``
(see ``training/eval_assets.py``) -- so these run from a fresh clone with no
corpus and no downloads.  If either is ever pruned, they fail with one line
naming both places to get it back; deliberately a failure and not a skip, since
a benchmark nobody notices has been skipped is not a benchmark.

Marked @pytest.mark.integration so you can skip with:
  pytest -m "not integration"
"""

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


# One full run shared by the assertions below (module-level cache rather than a
# fixture so each test's event loop stays independent).
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
async def test_sample_song_evaluation_passes():
    """The evaluator's PASS verdict on a real track -- the same plumbing
    scoring `auto_pilot simulate file` prints.  Musical quality is the eval
    set's job (below); this only says the pipeline produced a show at all."""
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
    guard against accidental wall-clock pacing.

    The bound moved from 4x to 2x when the rhythm front-end became madmom's
    online networks: the same pipeline measured 46.3x on this box with aubio and
    3.79x with madmom, because state-of-the-art online beat and onset tracking
    costs 25.7 % of a core against aubio's 1.4 %. Accidental wall-clock pacing —
    the thing this test exists to catch — reads 1.0x, so a 2x bound still
    catches it decisively while leaving room for a slower machine.
    """
    run = await _sample_run()
    assert run['wall_elapsed'] < run['song_sec'] / 2, (
        f"{run['song_sec']:.0f}s of audio took {run['wall_elapsed']:.1f}s wall"
    )


@pytest.mark.integration
async def test_simulation_is_deterministic():
    """A second full run produces a byte-identical report (and checksum)."""
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
def test_eval_set_head_matches_the_committed_baseline():
    """Three eval-set tracks through the benchmark runner, compare mode.

    This is the gate the old plumbing-PASS verdict used to be: it fails on a
    changed report checksum (the pipeline behaves differently) and on a score
    that fell below the committed baseline (the show got worse).  A subset run,
    so the ten-track aggregate is not compared -- see run_eval_set.compare.

    Read the printed table on failure: it names the track, the metric, and the
    direction, and says whether re-cutting the baseline is the right answer.
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
