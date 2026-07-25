# Fast Simulation Mode — Design

**Date:** 2026-07-25
**Status:** Approved

## Goal

`auto_pilot simulate file <song>` must complete in **under 10 seconds** for a typical
3-minute track, while remaining **deterministic** and **semantically identical to
production** (sim/prod match). The existing real-time Dash visualization mode and the
realtime microphone mode are preserved unchanged.

## Background / Measurements

The file simulation currently runs at exactly 1× real-time because:

1. `FileAudioClient.read()` sleeps to pace 256-sample buffers at wall-clock speed.
2. Every time-based mechanism in the pipeline reads the wall clock directly
   (`datetime.datetime.now()` / `time.monotonic()`): onset-density windows, silence
   detection, beat-absence timer, note debounce, the 2.5 s look-ahead command queue,
   beat-history windowing, event-buffer timestamps, color-override cooldown.
   Removing the sleep alone would corrupt all of these (e.g. onset density inflates
   by the speed-up factor).

Benchmarks on the 165 s sample track (Windows, unthrottled pipeline, YAMNet disabled
as it already is in sim):

| Configuration | Pipeline time | Throughput |
|---|---|---|
| As-is (dead vstack bounded) | 11.6 s | 14.2× real-time |
| cProfile breakdown | 4.1 s in `_track_song_duration` silence list-comp; ~1.5 s in dead `mfccs`/`energies` vstack | — |

With the two hot spots fixed, expected throughput is ~25–30× real-time:
**~6 s first run (+~3 s mp3 decode), ~4 s on repeat runs** with the decode cache.

## Requirements (user-confirmed)

- Fast headless mode is the **default** for `simulate file`: run at max speed, write
  the JSON report, print the evaluation, exit 0=PASS / 1=FAIL.
- The current **real-time visualization mode stays** available via `--ui`
  (real-time pacing + live Dash timeline, `--play-audio` still valid there).
- `simulate realtime` (microphone) is unchanged.
- Fast mode must be **deterministic**: two runs of the same file produce identical
  intent timelines and reports.
- **Sim/prod match**: the same pipeline code path runs in both; only the time source
  differs.
- Integration tests move to the virtual clock (suite ~15 s → ~2–3 s, deterministic).

## Design

### 1. Clock abstraction (`lib/clock.py`)

```
Clock (protocol): now() -> datetime, monotonic() -> float
SystemClock:  delegates to datetime.datetime.now() / time.monotonic()  [prod default]
VirtualClock: starts at a fixed epoch; advance(dt_sec) moves both readings
```

A module-level `SYSTEM_CLOCK` singleton is the default constructor argument
everywhere, so production and visualization mode are byte-for-byte unchanged.

Consumers that gain an optional `clock` parameter:

| Component | Wall-clock uses converted |
|---|---|
| `MusicAnalyser` | onset timestamps (density window), beat timestamps (BPM interval, beat absence), silence/sound-state tracking, note debounce, 15-min full reset |
| `LightEngine` | beat-history append/prune and symmetric window filter |
| `DelayedCommandQueue` | enqueue/fire scheduling of the 2.5 s look-ahead |
| `EventBuffer` | timeline timestamps (`start`, `_now`) |
| `EffectController` | color-override cooldown |
| stub clients (`simulate/stub_clients.py`) | event record timestamps |
| fake audio clients | click log timestamps; **throttling removed** (pacing moves to the runner, see below) |

Real hardware clients (`midi_client`, `os2l_client`/`os2l_sender`, `overlay_client`)
never run in simulation and keep direct system time. `YamnetChangeDetector` stays
stubbed out in sim (existing behavior; its wall-clock cooldowns are prod-only).

### 2. Runner (`simulate/runner.py`)

Fast mode loop — single asyncio task, no thread, no sleeps:

```
for each 256-sample buffer until source exhausted or duration_sec reached:
    clock.advance(256 / 44100)
    analyse(buffer)
    command_queue.drain()
    fire on_100ms_callback / on_1sec_callback on virtual boundaries
then: advance clock 2.5 s further in buffer-sized steps, draining, so pending
      look-ahead commands land in the report
```

`run_simulation(components, duration_sec)` keeps its signature; under a
`VirtualClock` the bound means **virtual (song) seconds**. Endless sources
(`BeepAudioClient`, used by the integration tests) rely on it; file mode runs to
end-of-file.

- Callback cadence is driven by virtual elapsed time → deterministic (every 0.1 s /
  1.0 s of song time exactly).
- `random.seed(<fixed>)` at fast-sim start → deterministic effect selection.
  Visualization and prod keep unseeded randomness.
- Report/event timestamps become **song-position seconds**, so the intent timeline
  aligns directly with track structure (improves the tuning workflow documented in
  `lib/analyser/CLAUDE.md`).

Pacing responsibility moves out of the audio clients: the visualization path keeps
real-time pacing (sleep against the wall clock) at the runner level, driven by the
same sample-position arithmetic the clients use today.

### 3. CLI (`simulate/cli.py`)

- `simulate file song.mp3` → fast headless (default): report + evaluation + exit code.
  `--report FILE` kept (default `report.json`).
- `simulate file song.mp3 --ui` → today's behavior exactly: real-time pacing, live
  Dash timeline on `--port`, `--play-audio` allowed.
- `--no-ui` flag is removed (headless is now the default; `--ui` is the opt-in).
- `--play-audio` without `--ui` is rejected with a clear error (cannot play audio at
  30× speed).
- `simulate realtime` unchanged.

### 4. Pipeline performance fixes (also relieve the live rig's hot path)

1. **Silence check** in `MusicAnalyser._track_song_duration`: replace the per-buffer
   Python list-comp over 40 numpy floats with `bool(np.all(np.abs(energies) < 1e-4))`
   — identical semantics (`abs(n) < t ⇔ -t < n < t`), ~4 s saved per 165 s track.
2. **Delete dead accumulation**: `self.mfccs = np.vstack(...)` /
   `self.energies = np.vstack(...)` grow per-buffer and are never read anywhere
   (the matplotlib visualizer keeps its own buffer). Remove the accumulation;
   keep per-buffer outputs local.
3. **Decode cache**: after `librosa.load`, save the mono 44.1 kHz float32 array as
   `<audiofile>.npy` beside the source. The cache is valid iff the `.npy` exists and
   its mtime is newer than the source file's. On the next run, load the `.npy`
   (~0.1 s) instead of decoding (~3 s). Transparent, no flag. Cache files are
   gitignored.

### 5. Timing-report semantics

In virtual time the 2.5 s delay is exact to one buffer quantum (±5.8 ms), so the
fast-mode report validates **scheduling logic** (correct delay, ordering, no drops)
rather than OS scheduler jitter. Real jitter remains observable in `--ui` and
`simulate realtime` modes, which still run on the wall clock. The report's
`timing_error_max_ms` threshold (50 ms) stays meaningful in both.

## Testing

- **Unit**: `VirtualClock` behavior (advance, epoch stability, monotonic/now
  consistency); silence-check equivalence on boundary values.
- **Integration** (`tests/test_simulation.py`): convert to virtual clock —
  `BeepAudioClient` unthrottled, clock advanced by the test loop. Timing assertions
  unchanged (they check enqueue→fire deltas, which are clock-source-agnostic).
- **New determinism test**: two fast runs over the same audio produce fully
  identical reports (beat list, intent timeline, effect selections, timing log).
- **Full suite** must pass per project policy: `uv run pytest`.

## Out of scope

- Enabling YAMNet in simulation (pre-existing sim/prod divergence, unchanged).
- Removing the unused `pitch_o` computation (measured ~1 % — not worth touching here).
- The uncommitted Windows environment fixes (`pyproject.toml`/`uv.lock`) ride on this
  branch as a separate commit but are not part of this design.

## Documentation impact (per CLAUDE.md policy)

- Root `CLAUDE.md`: simulate usage (fast default, `--ui` mode), look-ahead note
  (`LOOK_AHEAD_SEC` definition sites unchanged), known-issues if any.
- `lib/analyser/CLAUDE.md`: evaluation strategy section — report timestamps are now
  song-position seconds; fast tuning loop.
