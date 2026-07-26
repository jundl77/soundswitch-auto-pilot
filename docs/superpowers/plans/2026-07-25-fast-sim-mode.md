# Fast Simulation Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `auto_pilot simulate file <song>` run a full track in under 10 seconds, deterministically, with unchanged pipeline semantics; keep the real-time `--ui` visualization mode and `simulate realtime` untouched.

**Architecture:** An injectable `Clock` abstraction (`SystemClock` default everywhere, `VirtualClock` in fast sim) replaces every direct `datetime.datetime.now()` / `time.monotonic()` call in the pipeline components that run in simulation. The sim runner advances the virtual clock by exactly one buffer duration (256/44100 s) per audio buffer, with no sleeps. Two hot-spot fixes in `MusicAnalyser` (vectorized silence check, deletion of dead vstack accumulation) raise throughput from ~14× to ~25-30× real-time.

**Tech Stack:** Python 3.11, aubio, numpy, librosa, pytest (asyncio_mode=auto), uv.

**Spec:** `docs/superpowers/specs/2026-07-25-fast-sim-mode-design.md`

## Global Constraints

- Never commit to `master` — all work happens on branch `add_fast_sim_mode` (already checked out).
- Run tests with `uv run pytest` (unit-only: `uv run pytest -m "not integration"`). The FULL suite must pass before the PR.
- Every constructor change MUST use a keyword default `clock: Clock = SYSTEM_CLOCK` so production code (`lib/main.py`) and existing tests need no changes.
- Commit messages: lowercase imperative subject matching repo style (e.g. `add clock abstraction`), and append the footer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` on its own line.
- `LOOK_AHEAD_SEC = 2.5` stays defined in `lib/main.py` and `simulate/runner.py`; do not move it.
- Production behavior must be byte-for-byte unchanged: no edits to `lib/main.py`, `lib/clients/midi_client.py`, `lib/clients/os2l_client.py`, `lib/clients/os2l_sender.py`, `lib/clients/overlay_client.py`, `lib/analyser/yamnet_change_detector.py`.
- pytest is configured with `asyncio_mode = "auto"` — async test functions need no decorator.

---

### Task 1: Clock abstraction

**Files:**
- Create: `lib/clock.py`
- Test: `tests/test_clock.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Clock` (base class with `now() -> datetime.datetime`, `monotonic() -> float`), `SystemClock(Clock)`, `VirtualClock(Clock)` (adds `advance(dt_sec: float) -> None`), module singleton `SYSTEM_CLOCK: SystemClock`. All later tasks import `from lib.clock import Clock, SYSTEM_CLOCK` (and `VirtualClock` in sim/tests).

- [ ] **Step 1: Write the failing test**

Create `tests/test_clock.py`:

```python
"""Tests for the injectable Clock abstraction (lib/clock.py)."""
import datetime
import time

from lib.clock import SystemClock, VirtualClock, SYSTEM_CLOCK


def test_system_clock_tracks_real_time():
    clock = SystemClock()
    assert abs((clock.now() - datetime.datetime.now()).total_seconds()) < 0.5
    assert abs(clock.monotonic() - time.monotonic()) < 0.5


def test_system_clock_singleton_is_system_clock():
    assert isinstance(SYSTEM_CLOCK, SystemClock)


def test_virtual_clock_starts_at_zero_and_fixed_epoch():
    clock = VirtualClock()
    assert clock.monotonic() == 0.0
    assert clock.now() == datetime.datetime(2000, 1, 1)


def test_virtual_clock_advance_moves_both_readings():
    clock = VirtualClock()
    clock.advance(2.5)
    clock.advance(0.5)
    assert clock.monotonic() == 3.0
    assert clock.now() == datetime.datetime(2000, 1, 1, 0, 0, 3)


def test_virtual_clock_is_deterministic_across_instances():
    a, b = VirtualClock(), VirtualClock()
    for _ in range(1000):
        a.advance(256 / 44100)
        b.advance(256 / 44100)
    assert a.monotonic() == b.monotonic()
    assert a.now() == b.now()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_clock.py -v`
Expected: FAIL (collection error) with `ModuleNotFoundError: No module named 'lib.clock'`

- [ ] **Step 3: Write minimal implementation**

Create `lib/clock.py`:

```python
"""
Injectable time source for the pipeline.

Every component that measures time (windows, delays, cooldowns, timelines)
takes a Clock so that simulation can run faster than real-time on a virtual
clock while production runs on the system clock. Default is always
SYSTEM_CLOCK — production wiring never passes a clock explicitly.
"""
import datetime
import time


class Clock:
    """Time source interface: wall-clock datetime plus a monotonic float."""

    def now(self) -> datetime.datetime:
        raise NotImplementedError

    def monotonic(self) -> float:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> datetime.datetime:
        return datetime.datetime.now()

    def monotonic(self) -> float:
        return time.monotonic()


class VirtualClock(Clock):
    """Deterministic clock advanced manually by the simulation loop.

    Starts at a fixed epoch so runs are reproducible; monotonic() is the
    virtual elapsed seconds (song position in file simulation).
    """

    _EPOCH = datetime.datetime(2000, 1, 1)

    def __init__(self):
        self._elapsed_sec: float = 0.0

    def now(self) -> datetime.datetime:
        return self._EPOCH + datetime.timedelta(seconds=self._elapsed_sec)

    def monotonic(self) -> float:
        return self._elapsed_sec

    def advance(self, dt_sec: float) -> None:
        if dt_sec < 0:
            raise ValueError(f'cannot advance clock backwards ({dt_sec})')
        self._elapsed_sec += dt_sec


SYSTEM_CLOCK = SystemClock()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_clock.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add lib/clock.py tests/test_clock.py
git commit -m "add injectable clock abstraction (system + virtual)"
```

---

### Task 2: DelayedCommandQueue on the clock

**Files:**
- Modify: `lib/engine/delayed_command_queue.py`
- Test: `tests/test_delayed_command_queue.py` (append new tests; existing tests unchanged)

**Interfaces:**
- Consumes: `Clock`, `SYSTEM_CLOCK`, `VirtualClock` from `lib/clock`.
- Produces: `DelayedCommandQueue(delay_sec: float, clock: Clock = SYSTEM_CLOCK)`. `enqueue`/`drain`/`get_timing_log`/`delay_sec` signatures unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_delayed_command_queue.py`:

```python
# ---------------------------------------------------------------------------
# Virtual clock
# ---------------------------------------------------------------------------

from lib.clock import VirtualClock


async def test_virtual_clock_command_fires_only_after_virtual_delay():
    clock = VirtualClock()
    q = DelayedCommandQueue(2.5, clock=clock)
    fired = []

    async def cmd():
        fired.append(True)

    await q.enqueue('x', cmd)
    await q.drain()
    assert fired == []  # no virtual time has passed

    clock.advance(2.4)
    await q.drain()
    assert fired == []  # still 0.1s short

    clock.advance(0.2)
    await q.drain()
    assert fired == [True]


async def test_virtual_clock_timing_log_is_exact():
    clock = VirtualClock()
    q = DelayedCommandQueue(2.5, clock=clock)

    async def cmd():
        pass

    await q.enqueue('x', cmd)
    clock.advance(2.5)
    await q.drain()
    log = q.get_timing_log()
    assert len(log) == 1
    assert log[0]['actual_delta_sec'] == 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_delayed_command_queue.py -v`
Expected: the two new tests FAIL with `TypeError: DelayedCommandQueue.__init__() got an unexpected keyword argument 'clock'`; all pre-existing tests still pass.

- [ ] **Step 3: Write minimal implementation**

In `lib/engine/delayed_command_queue.py`:

Replace the imports at the top:

```python
import logging
from typing import Callable, Awaitable
from lib.clock import Clock, SYSTEM_CLOCK
```

(the `import time` line is removed — no other use of `time` remains in this file.)

Replace `__init__`:

```python
    def __init__(self, delay_sec: float, clock: Clock = SYSTEM_CLOCK):
        self._delay_sec = delay_sec
        self._clock = clock
        # (enqueue_time, fire_at, label, factory)
        self._queue: list[tuple[float, float, str, CommandFactory]] = []
        self._timing_log: list[dict] = []
```

In `enqueue`, replace `enqueue_time = time.monotonic()` with:

```python
        enqueue_time = self._clock.monotonic()
```

In `drain`, replace `now = time.monotonic()` with:

```python
        now = self._clock.monotonic()
```

and replace `actual_fire_time = time.monotonic()` with:

```python
            actual_fire_time = self._clock.monotonic()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_delayed_command_queue.py -v`
Expected: all tests pass (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add lib/engine/delayed_command_queue.py tests/test_delayed_command_queue.py
git commit -m "inject clock into delayed command queue"
```

---

### Task 3: EventBuffer on the clock

**Files:**
- Modify: `lib/engine/event_buffer.py`
- Test: `tests/test_event_buffer.py` (new file)

**Interfaces:**
- Consumes: `Clock`, `SYSTEM_CLOCK` from `lib/clock`.
- Produces: `EventBuffer(window_sec: float = 60.0, clock: Clock = SYSTEM_CLOCK)`. All other methods unchanged. `window_sec=float('inf')` must keep every event (needed later by the fast CLI path so >2 min tracks are not pruned from reports).

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_buffer.py`:

```python
"""Tests for EventBuffer virtual-clock timestamps and infinite window."""
from lib.clock import VirtualClock
from lib.engine.event_buffer import EventBuffer


def test_timestamps_use_injected_clock():
    clock = VirtualClock()
    buf = EventBuffer(clock=clock)
    buf.start()
    clock.advance(12.5)
    buf.add_beat(bpm=128.0, onset_density=4.0, change=False)
    snap = buf.snapshot()
    assert snap['now'] == 12.5
    assert snap['beats'][0]['t'] == 12.5


def test_infinite_window_keeps_old_events():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    buf.set_intent('GROOVE')
    clock.advance(500.0)  # far beyond the default 60 s window
    buf.set_intent('DROP')
    report = buf.to_report()
    assert [e['intent'] for e in report['intents']] == ['GROOVE', 'DROP']
    assert report['intents'][0]['t'] == 0.0


def test_default_window_prunes_old_effects():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=60.0, clock=clock)
    buf.start()
    buf.add_effect('CH_A', 'AUTOLOOP')
    clock.advance(500.0)
    buf.add_effect('CH_B', 'AUTOLOOP')
    clock.advance(1.0)
    buf.add_effect('CH_C', 'AUTOLOOP')
    report = buf.to_report()
    channels = [e['channel'] for e in report['effects']]
    assert 'CH_A' not in channels  # pruned — ended far outside 2× window
    assert 'CH_B' in channels and 'CH_C' in channels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_buffer.py -v`
Expected: FAIL with `TypeError: EventBuffer.__init__() got an unexpected keyword argument 'clock'`

- [ ] **Step 3: Write minimal implementation**

In `lib/engine/event_buffer.py`:

Replace the imports:

```python
import threading
from collections import deque

from lib.clock import Clock, SYSTEM_CLOCK
```

(the `import time` line is removed.)

Replace the `__init__` signature and first lines:

```python
    def __init__(self, window_sec: float = 60.0, clock: Clock = SYSTEM_CLOCK):
        self._lock = threading.Lock()
        self._window_sec = window_sec
        self._clock = clock
```

(rest of `__init__` unchanged.)

Replace `start`, `elapsed`, and `_now`:

```python
    def start(self) -> None:
        with self._lock:
            self._start_time = self._clock.monotonic()

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return self._clock.monotonic() - self._start_time

    def _now(self) -> float:
        if self._start_time is None:
            return 0.0
        return self._clock.monotonic() - self._start_time
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_event_buffer.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add lib/engine/event_buffer.py tests/test_event_buffer.py
git commit -m "inject clock into event buffer"
```

---

### Task 4: EffectController on the clock

**Files:**
- Modify: `lib/engine/effect_controller.py`
- Test: `tests/test_effect_controller.py` (append one test)

**Interfaces:**
- Consumes: `Clock`, `SYSTEM_CLOCK` from `lib/clock`.
- Produces: `EffectController(midi_client, event_buffer: EventBuffer | None = None, clock: Clock = SYSTEM_CLOCK)`. All other methods unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_effect_controller.py` (reuse whatever stub MIDI client fixture the file already defines — check `tests/conftest.py`, which builds `EffectController(stub_midi)`; import style should match the existing file):

```python
# ---------------------------------------------------------------------------
# Virtual clock — color override cooldown runs on injected time
# ---------------------------------------------------------------------------

from lib.clock import VirtualClock
from lib.engine.effect_controller import EffectController, APPLY_COLOR_OVERRIDE_INTERVAL_SEC


async def test_color_override_cooldown_uses_injected_clock(stub_midi):
    clock = VirtualClock()
    controller = EffectController(stub_midi, clock=clock)

    # Immediately after construction the cooldown is active.
    await controller._apply_color_override_if_due()
    first_time = controller.last_color_override_time

    # Advance virtual time past the cooldown — the override must now fire.
    clock.advance(APPLY_COLOR_OVERRIDE_INTERVAL_SEC + 1)
    await controller._apply_color_override_if_due()
    assert controller.last_color_override_time != first_time
```

Note: if `tests/test_effect_controller.py` has no `stub_midi` fixture of its own, it comes from `tests/conftest.py`. If the conftest fixture has a different name, adapt the parameter name to match — do not create a duplicate stub.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_effect_controller.py -v`
Expected: new test FAILS with `TypeError: EffectController.__init__() got an unexpected keyword argument 'clock'`; existing tests pass.

- [ ] **Step 3: Write minimal implementation**

In `lib/engine/effect_controller.py`:

Add to the imports (keep existing imports; `datetime` stays — `timedelta` is still used):

```python
from lib.clock import Clock, SYSTEM_CLOCK
```

Replace the `__init__` signature and the timestamp line:

```python
    def __init__(self, midi_client: MidiClient, event_buffer: EventBuffer | None = None,
                 clock: Clock = SYSTEM_CLOCK):
        self.midi_client: MidiClient = midi_client
        self.event_buffer: EventBuffer | None = event_buffer
        self._clock: Clock = clock
        self.last_effect: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1A)
        self.last_special_effect: Effect = Effect(type=EffectType.SPECIAL_EFFECT, source=EffectSource.MIDI, midi_channel=MidiChannel.SPECIAL_EFFECT_STROBE)
        self.last_color_override: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_1)
        self.last_color_override_time: datetime.datetime = self._clock.now()
```

In `reset_state`, replace `self.last_color_override_time = datetime.datetime.now()` with:

```python
        self.last_color_override_time = self._clock.now()
```

In `_apply_color_override_if_due`, replace `now = datetime.datetime.now()` with:

```python
        now = self._clock.now()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_effect_controller.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lib/engine/effect_controller.py tests/test_effect_controller.py
git commit -m "inject clock into effect controller"
```

---

### Task 5: MusicAnalyser on the clock

**Files:**
- Modify: `lib/analyser/music_analyser.py`
- Test: `tests/test_music_analyser.py` (append tests)

**Interfaces:**
- Consumes: `Clock`, `SYSTEM_CLOCK` from `lib/clock`.
- Produces: `MusicAnalyser(sample_rate, buffer_size, handler, visualizer_updater, clock: Clock = SYSTEM_CLOCK)`. All getters unchanged; every internal `datetime.datetime.now()` now reads `self._clock.now()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_music_analyser.py`:

```python
# ---------------------------------------------------------------------------
# Virtual clock
# ---------------------------------------------------------------------------

from lib.clock import VirtualClock


def _make_analyser(clock):
    return MusicAnalyser(
        sample_rate=44100,
        buffer_size=256,
        handler=_StubHandler(),
        visualizer_updater=None,
        clock=clock,
    )


def test_seconds_since_last_beat_on_virtual_clock():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    analyser.last_beat_detected = clock.now()
    clock.advance(1.5)
    assert analyser.get_seconds_since_last_beat() == 1.5


def test_onset_density_window_prunes_on_virtual_time():
    clock = VirtualClock()
    analyser = _make_analyser(clock)
    # Two onsets now, then advance beyond the 1.5 s rolling window.
    analyser._onset_times.append(clock.now())
    analyser._onset_times.append(clock.now())
    assert analyser.get_onset_density() == 2 / 1.5
    clock.advance(2.0)
    assert analyser.get_onset_density() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_music_analyser.py -v`
Expected: 2 new tests FAIL with `TypeError: MusicAnalyser.__init__() got an unexpected keyword argument 'clock'`; existing tests pass.

- [ ] **Step 3: Write minimal implementation**

In `lib/analyser/music_analyser.py`:

Add the import:

```python
from lib.clock import Clock, SYSTEM_CLOCK
```

Change `__init__` signature and set the clock **before** `_reset_state()` is called (`_reset_state` reads it):

```python
    def __init__(self,
                 sample_rate: int,
                 buffer_size: int,
                 handler: IMusicAnalyserHandler,
                 visualizer_updater: VisualizerUpdater,
                 clock: Clock = SYSTEM_CLOCK):
        self._clock: Clock = clock
        self.sample_rate: int = sample_rate
```

(rest of `__init__` unchanged.)

Then replace every `datetime.datetime.now()` in this file with `self._clock.now()`. There are exactly 9 sites:

1. `_reset_state`: `self.song_start_time`, `self.song_current_time`, `self.silence_period_start`, `self.last_mfcc_sample_time`, `self.last_beat_detected`, `self.last_note_detected` (6 sites) — each becomes `self._clock.now()`.
2. `get_beat_position`: `time_to_current_beat_sec = (self._clock.now() - self.last_beat_detected).total_seconds()`
3. `get_onset_density`: `now = self._clock.now()`
4. `get_seconds_since_last_beat`: `return (self._clock.now() - self.last_beat_detected).total_seconds()`
5. `analyse`: `now = self._clock.now()`
6. `_track_onset`: `self._onset_times.append(self._clock.now())`

(Sites 1–6 above total 11 replacements across 9 code locations; `_track_beat`, `_track_note`, `_track_song_duration` already receive `now` as a parameter and need no change. `import datetime` stays — `timedelta` is still used.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_music_analyser.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full unit suite to catch regressions**

Run: `uv run pytest -m "not integration"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lib/analyser/music_analyser.py tests/test_music_analyser.py
git commit -m "inject clock into music analyser"
```

---

### Task 6: MusicAnalyser hot-spot fixes (silence check + dead vstack)

**Files:**
- Modify: `lib/analyser/music_analyser.py`
- Test: `tests/test_music_analyser.py` (append tests)

**Interfaces:**
- Consumes: nothing new.
- Produces: `MusicAnalyser._is_silence(energies: np.ndarray) -> bool` (extracted for testability). The attributes `self.mfccs` and `self.energies` are **deleted** (they were write-only; verified unused by grep across the repo).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_music_analyser.py`:

```python
# ---------------------------------------------------------------------------
# _is_silence — vectorized equivalence with the old list-comp semantics
# ---------------------------------------------------------------------------

import numpy as np


def test_is_silence_all_near_zero(analyser):
    assert analyser._is_silence(np.zeros(40, dtype=np.float32)) is True
    assert analyser._is_silence(np.full(40, 0.00009, dtype=np.float64)) is True
    assert analyser._is_silence(np.full(40, -0.00009, dtype=np.float64)) is True


def test_is_silence_boundary_is_exclusive(analyser):
    # old semantics: -0.0001 < n < 0.0001 (strict) → exactly 0.0001 is NOT silence
    energies = np.zeros(40)
    energies[7] = 0.0001
    assert analyser._is_silence(energies) is False


def test_is_silence_single_loud_band(analyser):
    energies = np.zeros(40)
    energies[0] = 0.5
    assert analyser._is_silence(energies) is False


def test_dead_accumulation_arrays_removed(analyser):
    assert not hasattr(analyser, 'mfccs')
    assert not hasattr(analyser, 'energies')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_music_analyser.py -v`
Expected: new tests FAIL with `AttributeError: 'MusicAnalyser' object has no attribute '_is_silence'` (and the removal test fails because the attributes still exist).

- [ ] **Step 3: Write minimal implementation**

In `lib/analyser/music_analyser.py`:

1. Delete these two lines from `_reset_state`:

```python
        self.mfccs = np.zeros([self.mfcc_coeffs,])
        self.energies = np.zeros((40,))
```

2. Delete these two lines from `_compute_mfcc`:

```python
        self.mfccs = np.vstack((self.mfccs, mfcc_out))
        self.energies = np.vstack([self.energies, energies_out])
```

3. Add the helper method (place it right above `_track_song_duration`):

```python
    @staticmethod
    def _is_silence(energies: np.ndarray) -> bool:
        """All mel bands within ±1e-4 of zero (strict) — vectorized hot path."""
        return bool(np.all(np.abs(energies) < 0.0001))
```

4. In `_track_song_duration`, replace:

```python
        is_silence_now: bool = len([n for n in energies if -0.0001 < n < 0.0001]) == len(energies)
```

with:

```python
        is_silence_now: bool = self._is_silence(energies)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_music_analyser.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lib/analyser/music_analyser.py tests/test_music_analyser.py
git commit -m "vectorize silence check and remove dead mfcc/energy accumulation"
```

---

### Task 7: LightEngine on the clock

**Files:**
- Modify: `lib/engine/light_engine.py`
- Test: `tests/test_intent_stability.py` (append one test)

**Interfaces:**
- Consumes: `Clock`, `SYSTEM_CLOCK` from `lib/clock`.
- Produces: `LightEngine(midi_client, os2l_client, overlay_client, effect_controller, command_queue=None, event_buffer=None, look_ahead_sec=0.0, clock: Clock = SYSTEM_CLOCK)`. Beat-history timestamps (`BeatRecord[0]`) now come from `clock.monotonic()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_intent_stability.py` (match the file's existing construction helpers — it already builds a `LightEngine` around line 34 with stub clients; reuse the same stubs/fixtures and add):

```python
# ---------------------------------------------------------------------------
# Virtual clock — beat history timestamps come from the injected clock
# ---------------------------------------------------------------------------

from lib.clock import VirtualClock


async def test_beat_history_uses_injected_clock():
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient
    from lib.engine.effect_controller import EffectController
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.light_engine import LightEngine

    clock = VirtualClock()
    midi = StubMidiClient()
    engine = LightEngine(
        midi, StubOs2lClient(), StubOverlayClient(),
        EffectController(midi, clock=clock),
        DelayedCommandQueue(2.5, clock=clock),
        look_ahead_sec=2.5,
        clock=clock,
    )

    class _FakeAnalyser:
        def get_song_current_duration(self):
            import datetime
            return datetime.timedelta(seconds=clock.monotonic())
        def get_onset_density(self): return 5.0
        def get_onset_density_trend(self): return 1.0
        def get_sub_bass_ratio(self): return 0.3
        def get_rms_energy(self): return 0.1
        def get_kick_strength(self): return 2.0
        def get_spectral_centroid_trend(self): return 1.0
        def is_song_playing(self): return True

    engine.set_analyser(_FakeAnalyser())

    clock.advance(10.0)
    await engine.on_beat(beat_number=1, bpm=128.0, bpm_changed=False)
    assert engine._beat_history[-1][0] == 10.0  # virtual monotonic, not wall time
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_intent_stability.py -v`
Expected: new test FAILS with `TypeError: LightEngine.__init__() got an unexpected keyword argument 'clock'`; existing tests pass.

- [ ] **Step 3: Write minimal implementation**

In `lib/engine/light_engine.py`:

Add the import (keep `import time` removal in mind: after this change `time` is no longer used — remove `import time` from the top):

```python
from lib.clock import Clock, SYSTEM_CLOCK
```

Extend `__init__`:

```python
    def __init__(self,
                 midi_client: MidiClient,
                 os2l_client: Os2lClient,
                 overlay_client: OverlayClient,
                 effect_controller: EffectController,
                 command_queue: DelayedCommandQueue | None = None,
                 event_buffer: EventBuffer | None = None,
                 look_ahead_sec: float = 0.0,
                 clock: Clock = SYSTEM_CLOCK):
```

and add inside the body:

```python
        self._clock: Clock = clock
```

In `on_beat`, replace `now_mono = time.monotonic()` with:

```python
        now_mono = self._clock.monotonic()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_intent_stability.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest -m "not integration"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lib/engine/light_engine.py tests/test_intent_stability.py
git commit -m "inject clock into light engine"
```

---

### Task 8: Fake audio clients — unthrottle, clock, exhaustion, decode cache

**Files:**
- Modify: `simulate/fake_audio_client.py`
- Modify: `.gitignore` (add `*.npy`)
- Test: `tests/test_fake_audio_client.py` (new file)

**Interfaces:**
- Consumes: `Clock`, `SYSTEM_CLOCK` from `lib/clock`.
- Produces:
  - `BeepAudioClient(sample_rate, buffer_size, bpm=120.0, clock: Clock = SYSTEM_CLOCK)` — **no sleeping in `read()`**; internal `np.random.default_rng(12345)` for the noise floor (deterministic); `exhausted` property always `False`.
  - `FileAudioClient(sample_rate, buffer_size, path, clock: Clock = SYSTEM_CLOCK)` — **no sleeping in `read()`**; `exhausted: bool` property (`True` once all samples consumed); decode cache at `<path>.npy` (valid iff it exists and its mtime is newer than the source's).
  - Pacing responsibility moves entirely to the runner (Task 9).

- [ ] **Step 1: Write the failing test**

Create `tests/test_fake_audio_client.py`:

```python
"""Tests for unthrottled fake audio clients, exhaustion, and the decode cache."""
import time

import numpy as np
import pytest

from lib.clock import VirtualClock
from simulate.fake_audio_client import BeepAudioClient, FileAudioClient

SAMPLE_RATE = 44100
BUFFER_SIZE = 256


def test_beep_client_read_does_not_sleep():
    clock = VirtualClock()
    client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    client.start_streams()
    start = time.monotonic()
    for _ in range(2000):  # ~11.6 s of audio — would take 11.6 s if throttled
        client.read()
    assert time.monotonic() - start < 2.0


def test_beep_client_noise_is_deterministic():
    def collect():
        clock = VirtualClock()
        c = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
        c.start_streams()
        return np.concatenate([c.read() for _ in range(50)])

    assert np.array_equal(collect(), collect())


def test_beep_client_never_exhausted():
    client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE)
    assert client.exhausted is False


def test_beep_client_click_log_uses_virtual_clock():
    clock = VirtualClock()
    client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    client.start_streams()  # virtual start time = 0.0
    # 120 BPM → first click at 0.5 s. Read past it.
    for _ in range(200):
        client.read()
    assert client.click_log, 'expected at least one click'
    assert client.click_log[0]['wall_time'] == pytest.approx(0.5, abs=0.01)


@pytest.fixture
def wav_file(tmp_path):
    """1-second 440 Hz sine written as a wav librosa can load."""
    import soundfile as sf
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / 'tone.wav'
    sf.write(str(path), audio, SAMPLE_RATE)
    return str(path)


def test_file_client_reads_unthrottled_and_exhausts(wav_file):
    clock = VirtualClock()
    client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file, clock=clock)
    client.start_streams()
    assert client.exhausted is False
    start = time.monotonic()
    n = 0
    while not client.exhausted:
        buf = client.read()
        assert len(buf) == BUFFER_SIZE
        n += 1
    assert time.monotonic() - start < 2.0  # 1 s of audio, no throttle
    assert n == pytest.approx(SAMPLE_RATE / BUFFER_SIZE, abs=2)


def test_file_client_decode_cache_roundtrip(wav_file):
    import os
    clock = VirtualClock()
    c1 = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file, clock=clock)
    c1.start_streams()
    cache = wav_file + '.npy'
    assert os.path.exists(cache), 'first load must write the .npy cache'

    c2 = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file, clock=VirtualClock())
    c2.start_streams()
    assert np.array_equal(c1._audio, c2._audio)


def test_file_client_stale_cache_is_refreshed(wav_file, tmp_path):
    import os
    c1 = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file, clock=VirtualClock())
    c1.start_streams()
    cache = wav_file + '.npy'
    # Make the cache look older than the source → must be regenerated, not trusted.
    os.utime(cache, (1, 1))
    np.save(cache, np.zeros(10, dtype=np.float32))
    os.utime(cache, (1, 1))
    c2 = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, wav_file, clock=VirtualClock())
    c2.start_streams()
    assert len(c2._audio) == len(c1._audio)  # re-decoded, not the 10-sample fake
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fake_audio_client.py -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'clock'` and `AttributeError: ... no attribute 'exhausted'`.

- [ ] **Step 3: Write the implementation**

Rewrite `simulate/fake_audio_client.py` (full file — the diff touches most lines):

```python
"""
Fake audio clients for simulation. Two modes:

  BeepAudioClient  — generates a synthetic metronome (configurable BPM) at known
                     timestamps. Deterministic (fixed-seed noise floor). Used for
                     timing validation: clicks occur at predictable positions, so
                     we can assert that downstream light commands fire at exactly
                     beep_time + delay_sec.

  FileAudioClient  — decodes an audio file (MP3/WAV/FLAC) with librosa and feeds
                     256-sample buffers. Caches the decoded array as <path>.npy
                     (valid while newer than the source) so repeat runs skip the
                     multi-second decode.

Neither client throttles: read() returns immediately. Pacing (real-time or
virtual) is the simulation runner's responsibility. Both implement the same
interface as PyAudioClient.read() / start_streams() / close() plus an
`exhausted` property.
"""

import os
import logging
import numpy as np

from lib.clock import Clock, SYSTEM_CLOCK

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
BUFFER_SIZE = 256


# ---------------------------------------------------------------------------
# Beep audio client
# ---------------------------------------------------------------------------

class BeepAudioClient:
    """
    Generates a click at the start of each beat (configurable BPM) embedded in
    a near-silent buffer. Records the clock time of each generated click so the
    simulation can compare against when the corresponding command fired.
    """

    def __init__(self, sample_rate: int, buffer_size: int, bpm: float = 120.0,
                 clock: Clock = SYSTEM_CLOCK):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.bpm = bpm
        self._clock = clock
        self._samples_per_beat = sample_rate * 60.0 / bpm
        self._total_samples = 0
        self._start_time: float | None = None
        self._click = self._make_click()
        # Fixed seed: the noise floor must not introduce run-to-run variance.
        self._rng = np.random.default_rng(12345)
        # (sample_index, clock_time) for each generated click
        self.click_log: list[dict] = []

    def _make_click(self) -> np.ndarray:
        """10 ms Hann-windowed 1 kHz sine burst — easily detected by Aubio tempo."""
        n = int(self.sample_rate * 0.01)
        t = np.arange(n, dtype=np.float32)
        tone = np.sin(2 * np.pi * 1000 * t / self.sample_rate)
        return (tone * np.hanning(n)).astype(np.float32)

    def list_devices(self): pass
    def support_output(self) -> bool: return False

    def start_streams(self, start_stream_out: bool = False):
        self._start_time = self._clock.monotonic()

    def play(self, audio_buffer: np.ndarray): pass
    def close(self): pass

    @property
    def exhausted(self) -> bool:
        return False  # endless synthetic source

    def read(self) -> np.ndarray:
        """Return the next buffer immediately (no throttling)."""
        buf = self._rng.normal(0, 0.0005, self.buffer_size).astype(np.float32)

        # Embed a click whenever a beat boundary falls inside this buffer
        buf_start = self._total_samples
        beat_idx = int(buf_start / self._samples_per_beat)
        next_beat_sample = int((beat_idx + 1) * self._samples_per_beat)
        offset = next_beat_sample - buf_start
        if 0 <= offset < self.buffer_size:
            end = min(offset + len(self._click), self.buffer_size)
            length = end - offset
            buf[offset:end] += self._click[:length] * 0.8
            click_time = self._start_time + next_beat_sample / self.sample_rate
            self.click_log.append({'sample': next_beat_sample, 'wall_time': click_time})
            log.debug(f'[fake_audio] click at sample={next_beat_sample}, t={next_beat_sample / self.sample_rate:.3f}s')

        self._total_samples += self.buffer_size
        return buf


# ---------------------------------------------------------------------------
# File audio client
# ---------------------------------------------------------------------------

class FileAudioClient:
    """Decodes an audio file and feeds 256-sample buffers (no throttling)."""

    def __init__(self, sample_rate: int, buffer_size: int, path: str,
                 clock: Clock = SYSTEM_CLOCK):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.path = path
        self._clock = clock
        self._audio: np.ndarray | None = None
        self._pos = 0

    def list_devices(self): pass
    def support_output(self) -> bool: return False

    def start_streams(self, start_stream_out: bool = False):
        cache_path = self.path + '.npy'
        src_mtime = os.path.getmtime(self.path)
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) > src_mtime:
            log.info(f'[fake_audio] loading decode cache {cache_path}')
            self._audio = np.load(cache_path)
        else:
            import librosa
            log.info(f'[fake_audio] decoding {self.path} ...')
            audio, _ = librosa.load(self.path, sr=self.sample_rate, mono=True)
            self._audio = audio.astype(np.float32)
            np.save(cache_path, self._audio)
            log.info(f'[fake_audio] decode cache written → {cache_path}')
        self._pos = 0
        log.info(f'[fake_audio] loaded {len(self._audio) / self.sample_rate:.1f}s of audio')

    def play(self, audio_buffer: np.ndarray): pass
    def close(self): pass

    @property
    def exhausted(self) -> bool:
        return self._audio is not None and self._pos >= len(self._audio)

    def read(self) -> np.ndarray:
        end = self._pos + self.buffer_size
        if end > len(self._audio):
            # Pad last buffer with silence
            buf = np.zeros(self.buffer_size, dtype=np.float32)
            remaining = len(self._audio) - self._pos
            if remaining > 0:
                buf[:remaining] = self._audio[self._pos:]
        else:
            buf = self._audio[self._pos:end].copy()
        self._pos = min(end, len(self._audio))
        return buf

    @property
    def duration_sec(self) -> float:
        return len(self._audio) / self.sample_rate if self._audio is not None else 0.0
```

Also append to `.gitignore`:

```
*.npy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fake_audio_client.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run unit suite (integration will be converted in Task 10)**

Run: `uv run pytest -m "not integration"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add simulate/fake_audio_client.py tests/test_fake_audio_client.py .gitignore
git commit -m "unthrottle fake audio clients, add exhaustion and decode cache"
```

---

### Task 9: Stub clients + runner virtual loop

**Files:**
- Modify: `simulate/stub_clients.py`
- Modify: `simulate/runner.py`
- Test: covered by existing integration tests (converted next task) plus a new runner unit test in `tests/test_runner_fast.py`

**Interfaces:**
- Consumes: `Clock`, `SYSTEM_CLOCK`, `VirtualClock` from `lib/clock`; `exhausted` property from Task 8.
- Produces:
  - `StubMidiClient(event_buffer=None, clock: Clock = SYSTEM_CLOCK)`, `StubOs2lClient(event_buffer=None, clock: Clock = SYSTEM_CLOCK)`, `StubOverlayClient(clock: Clock = SYSTEM_CLOCK)` — event timestamps from the clock.
  - `build_simulation(audio_client, clock: Clock = SYSTEM_CLOCK)` and `build_visualizer_simulation(audio_client, event_buffer, clock: Clock = SYSTEM_CLOCK)` — thread the clock into every component they construct.
  - `run_simulation(components, duration_sec, clock: Clock = SYSTEM_CLOCK, pace_real_time: bool = False)` — virtual clocks are advanced by `BUFFER_SIZE / SAMPLE_RATE` per buffer and flushed `LOOK_AHEAD_SEC` past the end; `pace_real_time=True` sleeps against the wall clock to reproduce today's real-time pacing (used by `--ui` file mode).

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_fast.py`:

```python
"""Fast-runner behavior: virtual pacing, exhaustion stop, look-ahead flush."""
import time

from lib.clock import VirtualClock
from simulate.fake_audio_client import BeepAudioClient
from simulate.runner import build_simulation, run_simulation, LOOK_AHEAD_SEC

SAMPLE_RATE = 44100
BUFFER_SIZE = 256


async def test_fast_run_is_much_faster_than_real_time():
    clock = VirtualClock()
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    components, command_queue = build_simulation(audio_client, clock=clock)

    wall_start = time.monotonic()
    await run_simulation(components, duration_sec=10.0, clock=clock)
    wall_elapsed = time.monotonic() - wall_start

    assert wall_elapsed < 5.0, f'10 virtual seconds took {wall_elapsed:.1f}s wall time'
    # Clock ends past duration + flush tail
    assert clock.monotonic() >= 10.0 + LOOK_AHEAD_SEC - 0.1


async def test_fast_run_flushes_lookahead_commands():
    """Beat commands enqueued near the end must still fire (flush tail)."""
    clock = VirtualClock()
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    components, command_queue = build_simulation(audio_client, clock=clock)
    await run_simulation(components, duration_sec=10.0, clock=clock)

    log = command_queue.get_timing_log()
    assert log, 'expected commands to fire'
    for entry in log:
        # In virtual time the delay is exact to one buffer quantum (~5.8 ms)
        assert abs(entry['actual_delta_sec'] - entry['target_delta_sec']) <= 0.010
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner_fast.py -v`
Expected: FAIL with `TypeError: build_simulation() got an unexpected keyword argument 'clock'`.

- [ ] **Step 3: Write the implementation**

**`simulate/stub_clients.py`** — replace the header and `_event`, and thread the clock:

Replace the imports and `_event` helper:

```python
import logging
from lib.clock import Clock, SYSTEM_CLOCK
from lib.clients.midi_message import MidiChannel
from lib.clients.overlay_definitions import OverlayEffect

log = logging.getLogger(__name__)


def _event(label: str, t: float, **kwargs) -> dict:
    return {'label': label, 'time': t, **kwargs}
```

(the `import time` line is removed.)

`StubMidiClient.__init__` becomes:

```python
    def __init__(self, event_buffer=None, clock: Clock = SYSTEM_CLOCK):
        self.events: list[dict] = []
        self._pending_effects = []  # kept for interface compat
        self._event_buffer = event_buffer
        self._clock = clock
```

and every `_event('label', ...)` call in `StubMidiClient` gains the clock time, e.g.:

```python
        e = _event('set_autoloop', self._clock.monotonic(), channel=auto_loop.name)
```

```python
        e = _event('set_special_effect', self._clock.monotonic(), channel=special_effect.name, duration_sec=duration_sec)
```

```python
        e = _event('set_color_override', self._clock.monotonic(), channel=color.name)
```

`StubOs2lClient.__init__` becomes:

```python
    def __init__(self, event_buffer=None, clock: Clock = SYSTEM_CLOCK):
        self.events: list[dict] = []
        self._event_buffer = event_buffer
        self._clock = clock
```

and in `send_beat`:

```python
        e = _event('beat', self._clock.monotonic(), change=change, pos=pos, bpm=bpm, strength=strength)
```

`StubOverlayClient.__init__` becomes:

```python
    def __init__(self, clock: Clock = SYSTEM_CLOCK):
        self.events: list[dict] = []
        self._clock = clock
        # Mirror the real OverlayClient's effects_to_overlay_index so assertions don't fail
        self.effects_to_overlay_index = {effect: i for i, effect in enumerate(OverlayEffect)}
```

and in `update_overlay_data` / `flush_messages`:

```python
        e = _event('overlay_update', self._clock.monotonic(), effect=effect.name)
```

```python
        e = _event('overlay_flush', self._clock.monotonic())
```

**`simulate/runner.py`** — replace the whole file with:

```python
"""
Simulation runner for soundswitch-auto-pilot.

Replaces all hardware-touching clients with stubs and feeds synthetic or
file-based audio through the full MusicAnalyser → LightEngine →
DelayedCommandQueue pipeline.

Two pacing modes, selected by the caller:

  Virtual (default for `simulate file`): a VirtualClock is advanced by exactly
  one buffer duration (256/44100 s) per buffer — the run completes as fast as
  the CPU allows while every window/delay/threshold sees identical time to a
  real-time run. Deterministic.

  Real-time (`--ui` file mode): pace_real_time=True sleeps against the wall
  clock so the live Dash timeline scrolls in sync with actual playback.

Usage examples:
  # Fast headless evaluation (writes report.json, exits 0=PASS / 1=FAIL)
  python auto_pilot simulate file samples/song.mp3

  # Real music file with live Dash UI (real-time paced)
  python auto_pilot simulate file samples/song.mp3 --ui
"""

import asyncio
import datetime
import logging
import time

from lib.clock import Clock, SYSTEM_CLOCK, VirtualClock

SAMPLE_RATE = 44100
BUFFER_SIZE = 256
TIMING_TOLERANCE_SEC = 0.050  # 50 ms
# Must match LOOK_AHEAD_SEC in lib/main.py and playback_delay_seconds in dmx-enttec-node.
LOOK_AHEAD_SEC = 2.5


def build_simulation(audio_client, clock: Clock = SYSTEM_CLOCK):
    """Wire all components together with stub clients and return (app_components, command_queue)."""
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.light_engine import LightEngine
    from lib.analyser.music_analyser import MusicAnalyser

    midi_client = StubMidiClient(clock=clock)
    os2l_client = StubOs2lClient(clock=clock)
    overlay_client = StubOverlayClient(clock=clock)
    command_queue = DelayedCommandQueue(LOOK_AHEAD_SEC, clock=clock)

    effect_controller = EffectController(midi_client, clock=clock)
    light_engine = LightEngine(
        midi_client, os2l_client, overlay_client,
        effect_controller, command_queue,
        look_ahead_sec=LOOK_AHEAD_SEC,
        clock=clock,
    )

    music_analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, light_engine,
                                   visualizer_updater=None, clock=clock)
    light_engine.set_analyser(music_analyser)
    # Skip YAMNet loading — section detection disabled in simulation for speed.
    # To enable: call music_analyser.start() (requires internet on first run to download model).
    music_analyser.yamnet_change_detector.detect_change = lambda *a, **kw: False

    return {
        'audio_client': audio_client,
        'midi_client': midi_client,
        'os2l_client': os2l_client,
        'overlay_client': overlay_client,
        'command_queue': command_queue,
        'music_analyser': music_analyser,
        'light_engine': light_engine,
    }, command_queue


def build_visualizer_simulation(audio_client, event_buffer, clock: Clock = SYSTEM_CLOCK):
    """Like build_simulation but the engine emits events to the shared EventBuffer."""
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.light_engine import LightEngine
    from lib.analyser.music_analyser import MusicAnalyser

    midi_client = StubMidiClient(clock=clock)
    os2l_client = StubOs2lClient(clock=clock)
    overlay_client = StubOverlayClient(clock=clock)
    command_queue = DelayedCommandQueue(LOOK_AHEAD_SEC, clock=clock)

    effect_controller = EffectController(midi_client, event_buffer=event_buffer, clock=clock)
    light_engine = LightEngine(
        midi_client, os2l_client, overlay_client,
        effect_controller, command_queue,
        event_buffer=event_buffer,
        look_ahead_sec=LOOK_AHEAD_SEC,
        clock=clock,
    )

    music_analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, light_engine,
                                   visualizer_updater=None, clock=clock)
    light_engine.set_analyser(music_analyser)
    music_analyser.yamnet_change_detector.detect_change = lambda *a, **kw: False

    return {
        'audio_client': audio_client,
        'midi_client': midi_client,
        'os2l_client': os2l_client,
        'overlay_client': overlay_client,
        'command_queue': command_queue,
        'music_analyser': music_analyser,
        'light_engine': light_engine,
    }, command_queue


async def run_simulation(components: dict, duration_sec: float,
                         clock: Clock = SYSTEM_CLOCK,
                         pace_real_time: bool = False):
    """Main simulation loop — mirrors SoundSwitchAutoPilot.run().

    duration_sec is measured on `clock`: virtual (song) seconds under a
    VirtualClock, wall seconds otherwise. The loop also stops when the audio
    source reports exhaustion (end of file).

    With a VirtualClock the clock is advanced by exactly one buffer duration
    per buffer, then advanced LOOK_AHEAD_SEC past the end ("flush tail") so
    commands enqueued near the end still fire into the timing log/report.
    """
    audio_client = components['audio_client']
    music_analyser = components['music_analyser']
    command_queue = components['command_queue']

    is_virtual = isinstance(clock, VirtualClock)
    buffer_sec = BUFFER_SIZE / SAMPLE_RATE

    audio_client.start_streams()
    start_mono = clock.monotonic()
    wall_start = time.monotonic()
    buffers_fed = 0

    last_100ms = clock.now()
    last_1s = clock.now()

    logging.info(f'[sim] starting simulation loop for {duration_sec:.1f}s '
                 f'({"virtual" if is_virtual else "wall"} time)')
    while clock.monotonic() - start_mono < duration_sec and not audio_client.exhausted:
        audio_signal = audio_client.read()
        buffers_fed += 1
        if is_virtual:
            clock.advance(buffer_sec)
        if pace_real_time:
            deadline = wall_start + buffers_fed * buffer_sec
            sleep_sec = deadline - time.monotonic()
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)

        await music_analyser.analyse(audio_signal)
        await command_queue.drain()

        now = clock.now()
        if now - last_100ms > datetime.timedelta(milliseconds=100):
            last_100ms = now
            await components['light_engine'].on_100ms_callback()
            await components['midi_client'].on_100ms_callback()

        if now - last_1s > datetime.timedelta(seconds=1):
            last_1s = now
            await components['light_engine'].on_1sec_callback()

    if is_virtual:
        # Flush tail: advance past the look-ahead delay so pending commands fire.
        flush_until = clock.monotonic() + command_queue.delay_sec
        while clock.monotonic() < flush_until:
            clock.advance(buffer_sec)
            await command_queue.drain()

    audio_client.close()
    logging.info('[sim] simulation complete')


def print_timing_report(command_queue, tolerance_sec: float = TIMING_TOLERANCE_SEC):
    """Print a human-readable timing validation report."""
    log = command_queue.get_timing_log()
    if not log:
        print('\n[TIMING REPORT] No commands were dispatched.')
        return

    target = command_queue.delay_sec
    passed = 0
    worst_error_ms = 0.0

    print(f'\n{"─" * 72}')
    print(f'  TIMING REPORT   delay_target={target:.3f}s   tolerance=±{tolerance_sec * 1000:.0f}ms')
    print(f'{"─" * 72}')
    print(f'  {"label":<18} {"actual_delta":>12}  {"error":>8}  {"status":>6}')
    print(f'  {"─"*18} {"─"*12}  {"─"*8}  {"─"*6}')

    for entry in log:
        actual = entry['actual_delta_sec']
        error = actual - target
        error_ms = error * 1000
        ok = abs(error) <= tolerance_sec
        if ok:
            passed += 1
        worst_error_ms = max(worst_error_ms, abs(error_ms))
        status = '✓' if ok else '✗'
        print(f'  {entry["label"]:<18} {actual:>10.3f}s  {error_ms:>+7.1f}ms  {status:>6}')

    total = len(log)
    print(f'{"─" * 72}')
    print(f'  RESULT: {passed}/{total} within ±{tolerance_sec * 1000:.0f}ms  |  worst error: {worst_error_ms:.1f}ms')
    verdict = 'PASS' if passed == total else f'FAIL ({total - passed} command(s) out of tolerance)'
    print(f'  {verdict}')
    print(f'{"─" * 72}\n')
    return passed == total
```

(`print_timing_report` is unchanged from the current file; only the module docstring, imports, build functions, and `run_simulation` change.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runner_fast.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the whole suite (integration tests still on old call style — must still pass via defaults)**

Run: `uv run pytest`
Expected: all pass. The old integration tests call `run_simulation(components, duration_sec=5.0)` without a clock — with `SYSTEM_CLOCK` default and unthrottled Beep audio they still complete (faster than before) and the timing tolerance still holds (drain runs every ~0.5 ms wall time).

- [ ] **Step 6: Commit**

```bash
git add simulate/stub_clients.py simulate/runner.py tests/test_runner_fast.py
git commit -m "add virtual-clock fast loop and real-time pacing to sim runner"
```

---

### Task 10: Integration tests on the virtual clock + determinism test

**Files:**
- Modify: `tests/test_simulation.py`

**Interfaces:**
- Consumes: everything from Tasks 1–9.
- Produces: nothing new — test-only changes.

- [ ] **Step 1: Rewrite the integration tests**

Replace the body of `tests/test_simulation.py` with:

```python
"""
Integration tests: run the full simulation pipeline without hardware on a
virtual clock — seconds of song time complete in well under a second of wall
time, and results are fully deterministic.

Marked @pytest.mark.integration so you can skip with:
  pytest -m "not integration"
"""

import random

import pytest

from lib.clock import VirtualClock
from lib.engine.event_buffer import EventBuffer
from simulate.fake_audio_client import BeepAudioClient
from simulate.runner import (
    build_simulation,
    build_visualizer_simulation,
    run_simulation,
    print_timing_report,
)

SAMPLE_RATE = 44100
BUFFER_SIZE = 256


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
    Timing validation: beat commands enqueued at T must fire within 50 ms of
    T + delay (virtual time — exact to one buffer quantum).
    """
    clock = VirtualClock()
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    components, command_queue = build_simulation(audio_client, clock=clock)
    await run_simulation(components, duration_sec=8.0, clock=clock)

    log = command_queue.get_timing_log()
    assert log, 'expected beat commands on the virtual clock (deterministic input)'

    passed = print_timing_report(command_queue, tolerance_sec=0.050)
    assert passed, 'one or more beat commands exceeded 50 ms timing tolerance'


async def _run_fast_beep_sim(duration_sec: float) -> dict:
    """One fast, seeded, virtual-clock run → full report dict."""
    random.seed(1337)
    clock = VirtualClock()
    audio_client = BeepAudioClient(SAMPLE_RATE, BUFFER_SIZE, bpm=120.0, clock=clock)
    event_buffer = EventBuffer(window_sec=float('inf'), clock=clock)
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer, clock=clock)
    event_buffer.start()
    await run_simulation(components, duration_sec=duration_sec, clock=clock)
    return event_buffer.to_report(command_queue.get_timing_log())


@pytest.mark.integration
async def test_fast_simulation_is_deterministic():
    """Two identical fast runs must produce byte-identical reports."""
    report_a = await _run_fast_beep_sim(20.0)
    report_b = await _run_fast_beep_sim(20.0)
    assert report_a == report_b
```

- [ ] **Step 2: Run the integration tests**

Run: `uv run pytest -m integration -v`
Expected: 3 passed, in roughly 2–5 s total (previously ~13 s of pure sleeping).

Note: if `test_simulation_timing_passes` fails on the `assert log` line, aubio did not detect beats from the synthetic clicks — that would be a real regression introduced by the unthrottling (e.g. the deterministic noise floor changed detection). Debug rather than re-adding the skip: compare against `git stash`-restored throttled behavior.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest`
Expected: all pass, total well under 10 s.

- [ ] **Step 4: Commit**

```bash
git add tests/test_simulation.py
git commit -m "run integration tests on virtual clock, add determinism test"
```

---

### Task 11: CLI — fast headless default, `--ui` opt-in

**Files:**
- Modify: `simulate/cli.py`
- Modify: `simulate/evaluator.py` (docstring only — remove the `--no-ui` reference)

**Interfaces:**
- Consumes: `VirtualClock`, `run_simulation(..., clock=..., pace_real_time=...)`, `EventBuffer(window_sec=..., clock=...)`, `FileAudioClient(..., clock=...)`.
- Produces: `simulate file <audio>` = fast headless (report + evaluation + exit code). `simulate file <audio> --ui` = real-time paced live Dash timeline (previous default behavior). `--no-ui` flag removed. `--play-audio` requires `--ui`. `simulate realtime` unchanged.

- [ ] **Step 1: Rewrite `simulate/cli.py`**

Replace the whole file with:

```python
"""
Simulation CLI handlers — wired into 'auto_pilot simulate' subcommand.

MODES
  file            — FAST headless (default): run the whole file through the
                    pipeline on a virtual clock (~25-30× real-time), write the
                    JSON report, print the evaluation, exit 0=PASS / 1=FAIL.
  file --ui       — real-time paced run with the live Dash timeline (previous
                    default behavior; --play-audio available here).
  realtime        — capture from microphone in real time with Dash timeline.

EXAMPLES
  python auto_pilot simulate file samples/song.mp3
  python auto_pilot simulate file samples/song.mp3 --report out.json
  python auto_pilot simulate file samples/song.mp3 --ui --play-audio
  python auto_pilot simulate realtime --device-index 1
"""

import asyncio
import json
import random
import sys
import threading
import time

SAMPLE_RATE = 44100
BUFFER_SIZE = 256

# Fixed seed for the fast headless mode: effect selection is random by design,
# but fast-sim reports must be reproducible run-to-run.
FAST_SIM_RANDOM_SEED = 1337


def _run_pipeline(components, duration_sec: float, event_buffer, command_queue,
                  pace_real_time: bool):
    from simulate.runner import run_simulation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            run_simulation(components, duration_sec, pace_real_time=pace_real_time)
        )
    finally:
        event_buffer.set_timing_log(command_queue.get_timing_log())
        loop.close()


def _write_report_and_evaluate(event_buffer, command_queue, report_path: str) -> bool:
    from simulate.evaluator import evaluate, print_evaluation
    report = event_buffer.to_report(command_queue.get_timing_log())
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'[simulate] report written → {report_path}')
    result = evaluate(report)
    print_evaluation(result)
    return result['passed']


def run_file(args):
    if args.play_audio and not args.ui:
        print('[simulate] error: --play-audio requires --ui '
              '(audio cannot play at fast-simulation speed)')
        sys.exit(2)
    if args.ui:
        _run_file_realtime_ui(args)
    else:
        _run_file_fast(args)


def _run_file_fast(args):
    """Fast headless mode: virtual clock, no UI, report + evaluation + exit code."""
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import build_visualizer_simulation, run_simulation

    random.seed(FAST_SIM_RANDOM_SEED)
    clock = VirtualClock()
    audio_client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, args.audio, clock=clock)
    # Infinite window: keep the entire song's events — reports must never prune.
    event_buffer = EventBuffer(window_sec=float('inf'), clock=clock)
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer, clock=clock)

    event_buffer.start()
    wall_start = time.monotonic()
    asyncio.run(run_simulation(components, duration_sec=float('inf'), clock=clock))
    wall_elapsed = time.monotonic() - wall_start

    song_sec = audio_client.duration_sec
    speed = song_sec / wall_elapsed if wall_elapsed > 0 else 0.0
    print(f'[simulate] {song_sec:.1f}s of audio processed in {wall_elapsed:.1f}s ({speed:.1f}x real-time)')

    passed = _write_report_and_evaluate(event_buffer, command_queue, args.report)
    sys.exit(0 if passed else 1)


def _run_file_realtime_ui(args):
    """Real-time paced run with the live Dash timeline (previous default behavior)."""
    from lib.engine.event_buffer import EventBuffer
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import build_visualizer_simulation

    audio_client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, args.audio)
    event_buffer = EventBuffer()
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer)

    try:
        import librosa
        duration_sec = librosa.get_duration(path=args.audio)
    except Exception:
        duration_sec = float('inf')

    event_buffer.start()

    thread = threading.Thread(
        target=_run_pipeline,
        args=(components, duration_sec, event_buffer, command_queue, True),
        daemon=True,
    )
    thread.start()

    if args.play_audio:
        try:
            import sounddevice as sd
            import librosa as lr
            audio_data, sr = lr.load(args.audio, sr=SAMPLE_RATE, mono=True)
            sd.play(audio_data, samplerate=sr)
            print('[simulate] audio playback started')
        except ImportError as e:
            print(f'[simulate] warning: {e} — audio playback skipped')

    from simulate.visualizer_app import run_app
    run_app(event_buffer, port=args.port)


def run_realtime(args):
    from lib.engine.event_buffer import EventBuffer
    from lib.clients.pyaudio_client import PyAudioClient
    from simulate.runner import build_visualizer_simulation
    from simulate.visualizer_app import run_app

    audio_client = PyAudioClient(
        sample_rate=SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
        input_device_index=args.device_index,
    )
    event_buffer = EventBuffer()
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer)
    event_buffer.start()

    # Microphone input is hardware-paced — no artificial pacing needed.
    thread = threading.Thread(
        target=_run_pipeline,
        args=(components, float('inf'), event_buffer, command_queue, False),
        daemon=True,
    )
    thread.start()

    run_app(event_buffer, port=args.port)


def add_simulate_subparser(subparsers):
    """Register the 'simulate' subcommand and its sub-subcommands."""
    sim = subparsers.add_parser(
        'simulate',
        help='Run the pipeline against a file (fast, headless) or microphone (live UI)',
    )
    sub = sim.add_subparsers(dest='sim_mode', required=True)

    fp = sub.add_parser('file', help='Simulate from an audio file (fast headless by default)')
    fp.add_argument('audio', help='Path to audio file (MP3 / WAV / FLAC)')
    fp.add_argument('--ui', action='store_true',
                    help='Real-time paced run with live Dash timeline (instead of fast headless)')
    fp.add_argument('--play-audio', action='store_true',
                    help='Play audio from speakers (requires --ui and sounddevice)')
    fp.add_argument('--report', default='report.json',
                    help='Report output path for fast mode (default: report.json)')
    fp.add_argument('--port', type=int, default=8050, help='Dash server port (--ui only)')

    rp = sub.add_parser('realtime', help='Simulate from microphone in real time')
    rp.add_argument('--device-index', type=int, default=None,
                    help='PyAudio input device index (default: system default)')
    rp.add_argument('--port', type=int, default=8050, help='Dash server port')

    sim.set_defaults(func=simulate_cmd)


def simulate_cmd(args):
    if args.sim_mode == 'file':
        run_file(args)
    elif args.sim_mode == 'realtime':
        run_realtime(args)
```

Note: `_run_file_realtime_ui` and `run_realtime` deliberately do not pass a clock — they run on `SYSTEM_CLOCK` defaults, preserving today's behavior exactly (pacing now handled by `run_simulation(pace_real_time=True)` for the file-UI path; the mic path is hardware-paced).

- [ ] **Step 2: Fix the stale docstring in `simulate/evaluator.py`**

Replace the usage lines in the module docstring:

```python
"""
Agentic evaluator: scores a simulation report against configurable criteria.

Usage (headless / CI):
  python auto_pilot simulate file song.mp3 --report report.json
  # exit code 0 = PASS, 1 = FAIL
```

(only the `--no-ui` flag is dropped from the example; everything else in the file is unchanged.)

- [ ] **Step 3: Verify the fast CLI end-to-end (acceptance run)**

Run: `uv run python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3`

Expected:
- Completes in **< 10 s wall time** on the first run (decode + pipeline), and faster on the second run (decode cache hit — look for `loading decode cache` in the log).
- Prints `[simulate] 165.1s of audio processed in X.Xs (NN.Nx real-time)` with NN ≥ 20.
- Writes `report.json`, prints the evaluation table, exits 0 (echo `$LASTEXITCODE` / `echo $?` to confirm; FAIL exit 1 is acceptable only if the evaluation table shows a genuine criteria failure — investigate if so).

Run it twice and diff the reports:

Run: `uv run python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3 --report a.json; uv run python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3 --report b.json`
Then compare `a.json` and `b.json` — they must be byte-identical (determinism). Delete `a.json`/`b.json` afterwards.

- [ ] **Step 4: Verify the `--ui` path still starts**

Run: `uv run python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3 --ui --port 8051` in the background; after ~5 s, `curl http://localhost:8051` (or `Invoke-WebRequest`) must return the Dash page HTML; then kill the process. This confirms the real-time path wires up (full visual check is manual).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add simulate/cli.py simulate/evaluator.py
git commit -m "make fast headless the default file-simulation mode, --ui opt-in"
```

---

### Task 12: Documentation (CLAUDE.md policy) + final verification

**Files:**
- Modify: `CLAUDE.md` (root)
- Modify: `lib/analyser/CLAUDE.md`

**Interfaces:** none — documentation only, per the repo's CLAUDE.md policy (intent/meta level, no thresholds or signatures).

- [ ] **Step 1: Update root `CLAUDE.md`**

Make these edits (intent-level, no code duplication):

1. In the **Key Files** table, add a row after the `lib/main.py` row:

```markdown
| `lib/clock.py` | `Clock` abstraction — `SystemClock` (prod default) vs `VirtualClock` (fast sim); every time-based component takes an injectable clock |
```

and update the `simulate/runner.py` row description to:

```markdown
| `simulate/runner.py` | Simulation runner — stub clients, full pipeline; virtual-clock fast mode (default) or real-time pacing for the live UI |
```

2. In the **Running** section, replace the simulation examples:

```bash
# Simulation (no hardware required)
python auto_pilot simulate file samples/song.mp3          # fast headless: full song in seconds, report + evaluation
python auto_pilot simulate file samples/song.mp3 --ui     # real-time paced with live Dash timeline
python auto_pilot simulate realtime                       # microphone input with live Dash timeline
```

3. Add a paragraph to the section describing simulation (near the LightIntent/look-ahead discussion), at intent level:

```markdown
**Fast simulation:** file simulation runs on a virtual clock driven by audio sample
position instead of the wall clock — the full pipeline (identical code path to
production) processes a track ~25-30× faster than real-time and deterministically:
the same file always produces byte-identical reports (RNG seeded, no wall-clock
jitter). Report timestamps are song-position seconds, so intent timelines align
directly with track structure. The decoded audio is cached beside the source file
(`*.npy`, gitignored) to skip repeat decodes. Real OS scheduler jitter is only
observable in `--ui` / realtime modes, which still run on the system clock.
```

4. In **Known Issues / Gotchas**, add:

```markdown
- **Decode cache**: `simulate file` writes `<song>.npy` beside the audio file (gitignored). Stale caches are detected by mtime; delete the `.npy` to force a re-decode.
```

- [ ] **Step 2: Update `lib/analyser/CLAUDE.md`**

In the **Evaluation Strategy → Running a simulation** section, replace the command and add a sentence:

```markdown
### Running a simulation

```bash
python auto_pilot simulate file samples/song.mp3 --report report.json
```

Fast headless mode is the default: the full track runs through the identical
production pipeline on a virtual clock in seconds, deterministically — rerunning
the same file yields an identical report, so threshold changes show up as clean
diffs. Report timestamps are song-position seconds, so intent blocks can be read
directly against the track structure.
```

(keep the existing "The JSON report contains …" list that follows.)

- [ ] **Step 3: Run the FULL suite one last time**

Run: `uv run pytest`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md lib/analyser/CLAUDE.md
git commit -m "document fast simulation mode in CLAUDE.md"
```

- [ ] **Step 5: Clean up stray artifacts**

Confirm `git status` shows no unintended files (e.g. `report.json`, `a.json`, `b.json`, `samples/*.npy` must not be staged; `report.json` is already expected output — leave it untracked or delete).

---

## Self-Review (completed)

- **Spec coverage:** clock abstraction (T1), all component conversions (T2–T7 — queue, event buffer, effect controller, analyser, engine), perf fixes (T6), unthrottled clients + cache + gitignore (T8), stub timestamps + runner virtual loop + flush + real-time pacing (T9), integration tests + determinism (T10), CLI default flip + `--ui` + play-audio guard + infinite window (T11), docs (T12). Evaluator needs no logic change (works off report `duration_sec`, which is song seconds under virtual clock) — only its docstring (T11).
- **Prod untouched:** verified — no task edits `lib/main.py` or any real client; all new constructor params default to `SYSTEM_CLOCK`.
- **Type consistency:** `Clock`/`SystemClock`/`VirtualClock`/`SYSTEM_CLOCK` names used identically across tasks; `exhausted` property consumed by T9's loop; `pace_real_time` produced in T9, consumed in T11; `EventBuffer(window_sec=float('inf'))` produced in T3, consumed in T10/T11.
