# madmom migration — aubio keeps the filterbank, madmom takes the rhythm

**Branch** `madmom_migration` off `origin/master` (`9749286`, post-PR-7).
**Worktree** `C:\Users\Julian\Projects\soundswitch-madmom-worktree`.
**Charter** `.superpowers/sdd/2026-07-28-madmom-migration/charter.md` (binding, owner-amended).
**Gate** satisfied: decisions #81 — decoded sweep GO on madmom online beats.

The one-line scope: **every aubio object that answers a rhythm question is
replaced by a madmom online processor; the one aubio object that answers a
spectral question stays.**

---

## Task 0 — aubio call-site census (done; this table is the scope contract)

Every `aubio` symbol reachable from the live pipeline, classified. Verified by
`grep -rn aubio --include=*.py` at `9749286`.

### `lib/analyser/music_analyser.py` — the only aubio importer in `lib/`

| # | site | what it answers | verdict |
|---|---|---|---|
| 1 | `aubio.tempo("default", win_s_small, hop_s, sr)` — `self.tempo_o` | beat instants (`_track_beat`) **and** BPM (`get_bpm` → `tempo_o.get_bpm()`) | **MIGRATES** — madmom `RNNBeatProcessor(online=True)` → `DBNBeatTrackingProcessor(online=True)`; BPM derived from the emitted beat stream |
| 2 | `aubio.onset("default", …)` — `self.onset_o` | onset instants (`_track_onset`) feeding `get_onset_density` / `get_onset_density_trend` / `_density_samples` | **MIGRATES** — madmom online onset processor + `OnsetPeakPickingProcessor(online=True)` |
| 3 | `aubio.notes("default", …)` — `self.notes_o` | the `-d` debug click trigger only (`_track_note` → `handler.on_note()` → `click_sound` mixed into the monitored buffer) | **MIGRATES** — retriggered from the madmom onset stream, same 75 ms refractory, same click, same `-d` UX |
| 4 | `aubio.pvoc(win_s, hop_s)` — `self.pvoc_o` | the FFT that feeds the filterbank | **STAYS** (filterbank role; owner amendment #72a) |
| 5 | `aubio.filterbank(40, win_s)` + `set_mel_coeffs_slaney(sr)` — `self.energy_filter` | the 40-band mel bank behind `kick_strength`, `sub_bass_ratio`, `spectral_centroid_trend`, `_is_silence`, and every trained model's sidecars | **STAYS** (owner amendment #72a) |

No other aubio symbol is reachable from `lib/`. `pitch`, `mfcc`, `source`, `sink`
do not appear in the live path.

### Outside `lib/` — not in scope, recorded so the grep proof reads clean

| site | verdict |
|---|---|
| `training/train.py` — `aubio.source/onset/tempo/pvoc/mfcc/filterbank` | **OUT OF SCOPE.** A standalone offline feature-dump script, not the live pipeline, not imported by `lib/` or `simulate/`. The charter's grep proof is scoped to `lib/`. Touching it would change the offline training front-end, which #72/#77 explicitly defer to the champion-generation front-end bake. Recorded as a follow-up. |
| `tests/test_music_analyser.py:188,199` — the word "aubio" in two comments | **prose**, updated to name madmom |
| `lib/engine/light_engine.py:22`, `lib/engine/event_buffer.py:70` | **prose**, updated |

### What the census changes about the charter's assumptions

The charter (written from memory) expected a possible `aubio.pitch`. There is
none. It also expected note detection to be its own feature; it is not — the
note stream has exactly one consumer, the debug beep. That makes site 3 a
trigger swap, not a feature port.

---

## Design decisions taken before any code (rationale, not values)

**D1 — one adapter module, not madmom calls sprayed through `MusicAnalyser`.**
`lib/analyser/madmom_rhythm.py` owns the whole madmom side: sample-rate
bookkeeping, the buffer→hop accumulator, both networks, both decoders, and
reset. `MusicAnalyser` sees one object that takes a buffer and returns the
rhythm events fired inside it. This is the #78 simplicity principle applied to
our own code: the framing mismatch is a property of the adapter, and nothing
above it should ever have to know the hop size.

**D2 — the framing mismatch is the adapter's problem.** The live pipeline reads
256-sample buffers (5.805 ms); madmom's online models are trained at 441-sample
hops (10 ms). The adapter accumulates and emits whole hops, so a sample can wait
up to **one full hop (10 ms)** before the frame containing it is decoded — the
residue after a hop is consumed is 0–440 samples, so the bound is the hop, not
the hop minus a buffer. (An earlier draft of this plan said 4.2 ms; that was the
buffer-quantisation term alone and understated it ~2.4×.) Measured live, the
adapter's held audio ranged 0.4–8.9 ms, consistent with the bound. Against a
2.5 s look-ahead this is 0.4 % of the budget, but it is spent, not free, and the
adapter reports it rather than leaving it implicit.

**D3 — online/causal only, and the decode is forward-only.** `process_online`
with single-frame activations runs the HMM forward algorithm and reports a beat
at the frame it is decided on — no Viterbi, no backtrack, no future frames.
Verified by reading `DBNBeatTrackingProcessor.process_online`. The full 8-model
beat LSTM ensemble is used: the investigation measured the single-model
shortcut at 0.813 agreement with the offline decode against the ensemble's
0.968, and the CPU saving does not buy that back.

**D4 — BPM is derived from the beat stream, not from a tempo object.** madmom's
DBN exposes an instantaneous `tempo` (one inter-beat interval), which is far
noisier than what `aubio.tempo.get_bpm()` returned. BPM is therefore a median
over the recent beat intervals — the smoothing lives in our code where it can
be tested, not in a library attribute.

**D5 — octave folding stays.** `_fold_bpm` was written for aubio's warmup
double-tempo lock. It is not aubio-specific in effect: it makes tempo and its
octave the same number, which is what the DROP branch's BPM gate assumes, and
drum & bass still folds. Removing it would be retuning the rule engine, which
the charter forbids in this PR. Its docstring's aubio attribution gets corrected.

**D6 — the onset peak-picking threshold is the migration's one free parameter,
and it is set by matching the stream it replaces.** madmom's peak picker needs
a threshold; aubio's `onset("default")` had its own baked in. Choosing madmom's
library default would silently move the onset *rate*, and every density
constant in `light_engine.py` is denominated in that rate. Matching the median
onset rate of the aubio stream over a multi-track corpus is the opposite of
retuning the rule engine — it holds the rule engine's input distribution fixed
so the migration can be judged on what actually changed. Both the chosen point
and what madmom's default would have given are disclosed.

**D7 — processors are built once and reset, never rebuilt.** `_reset_state()`
fires every 15 minutes and on every sound stop. Rebuilding the ensembles would
reload eight pickled LSTMs mid-show. The adapter separates construction from
`reset()`.

---

## Measurements already taken (they shaped the plan; re-measured properly in-task)

Bundled sample track, first 120 s, this worktree's venv, single thread:

| stream | cost/buffer | share of one core | rate |
|---|---|---|---|
| aubio rhythm (tempo+onset+notes) | 0.0445 ms | 0.8 % | beats 2.117/s, onsets 3.700/s, notes 3.383/s |
| aubio filterbank (pvoc+mel) — *stays* | 0.0280 ms | 0.5 % | — |
| madmom beats+onsets, streamed | 1.1374 ms | 19.6 % | beats 2.108/s |

Beat rate agrees to 0.4 %. Two fresh runs of the streaming adapter produced
**identical** beat and onset lists — the determinism precondition holds before
any integration work. The cost ratio (~25×) is the headline risk and Task 7
owns it.

---

## Tasks

Each task ends with a commit and an appended report in
`.superpowers/sdd/2026-07-28-madmom-migration/progress.md`.

### Task 1 — golden fixtures FIRST (no behaviour change)

Record what the pipeline does *now*, split into what this migration is allowed
to change and what it must not.

- Commit `training/pipeline_digest.py`: runs the fast sim on a track and emits a
  compact digest — beat count and a hash of beat times (**allowed to change**),
  the report's schema keys and a filterbank fingerprint (**must not change**),
  plus intent/effect counts and wall-clock speed.
- Commit `tests/fixtures/pipeline_digest_baseline.json` for three tracks: the
  bundled sample plus two eval-set tracks whose audio is on disk.
- Commit `tests/test_pipeline_digest.py` pinning the digest's own shape (unit),
  and an integration test asserting the bundled track's **schema + filterbank**
  digest against the baseline.

Red first: the digest test must fail before the digest exists.

**CORRECTED IN FLIGHT.** The first version made the *beat-sampled* filterbank
columns the anchor. That cannot work: those columns are filterbank output read
at beat instants, so a moved beat grid moves them by construction — the anchor
would fail for the expected reason while a genuine filterbank regression hid
inside it. Replaced by a whole-track **fixed-time-grid** fingerprint, which no
beat source can perturb, plus a golden hash of the mel path generated from the
base commit's own code. The beat-sampled section survives as reported evidence
under the honest name `at_beats`, with a test pinning that it is not a gate.

### Task 2 — the dependency, pinned and justified

- `pyproject.toml`: `madmom` at git SHA `27f032e8`, `no-build-isolation-package`
  plus `extra-build-dependencies` (cython/numpy/setuptools). PyPI 0.16.1 cannot
  import on ≥3.10; git main can. Recorded with reasons in the file.
- `uv.lock` regenerated.
- `tests/test_madmom_contract.py`: asserts the online/causal properties of the
  **constructed** processors — no bidirectional layer anywhere in the live path
  (with a control proving madmom still builds them for the offline variants, so
  the assertion cannot go vacuous), forward-only decode, the online frame
  geometry, and that nothing under `lib/` names the offline downbeat tracker.

**CORRECTED IN FLIGHT.** The first version asserted that `online=` was an
accepted keyword — which cannot fail for the reason the test exists. An upgrade
renaming or dropping the flag would have handed the live path bidirectional
networks with every assertion still green.

### Task 3 — `lib/analyser/madmom_rhythm.py`, TDD

The adapter. Unit-tested against synthetic buffers with a fake processor pair so
the tests are fast and do not load models:

- accumulates arbitrary buffer sizes into exact hops, loses no samples;
- emits beats and onsets with the hop index they were decided at;
- `reset()` returns it to the constructed state without rebuilding models;
- refuses non-mono / wrong-sample-rate input rather than silently resampling.

One model-loading integration test proves the real stack streams and is
deterministic across two runs in one process.

### Task 4 — the onset operating point (measurement, then a constant)

Script (committed under `training/`) that sweeps the peak-picking threshold on
the bundled track plus the eval-set tracks, reporting aubio's onset rate and
madmom's per track. Choose the threshold that matches the **median** rate.
Record the sweep table and the residual per-track spread in progress.md. Report
what madmom's library default would have given.

### Task 5 — the swap

`MusicAnalyser` loses `tempo_o`, `onset_o`, `notes_o`; keeps `pvoc_o`,
`energy_filter`. `_track_beat` / `_track_onset` / `_track_note` are driven by
the adapter's per-buffer output. `get_bpm` derives from the beat stream (D4).
The beep keeps its 75 ms refractory, its click, and its `-d` wiring.

Suite must be green here, with skip-count and collected-count unchanged and
explained if not (#47/#59).

### Task 5b — backpressure (added mid-flight by ruling)

Live audio arrives at exactly 1x and the input side DROPS rather than queues, so
falling behind costs audio rather than latency and nothing in the pipeline said
so. `lib/analyser/drift_watchdog.py` measures lost lead over a rolling window
(not cumulatively — dropped samples never arrive, so a cumulative measure would
latch after the first hiccup) and sheds work cheapest-loss-first: section
detection, then onsets, never beats.

Both shed components clear their state on restore. That is not symmetry for its
own sake — a component fed no audio during a gap holds buffers describing
pre-gap music, and butt-joining post-gap audio onto them decodes a seam.

### Task 6 — determinism, cross-process

Two fresh processes, same file, byte-identical reports and equal checksums, on
all three fixture tracks. New checksums are expected and are recorded as the
new anchors. Any RNG found inside madmom is named and pinned.

### Task 7 — realtime

Per-buffer cost of the whole new `analyse()` vs the old, on one track, single
thread, mean/p95/max, expressed against the 5.805 ms buffer period and against
decision #18's ≥5× bar. If the integration suite's speed guard no longer holds,
it is changed **with the new number and the reason in the diff**, never
silently — the guard exists to catch accidental wall-clock pacing and must
still do that.

### Task 8 — deltas, disclosed

Feature and intent deltas on the fixture tracks: onset density distribution,
beat count, BPM, intent timeline, `intent_changes_count` /
`effect_changes_count`. Rule-engine constants are **not** touched.

**Scope note the charter could not know:** `training/eval_set.json`,
`training/eval_set_baseline.json` and `training/run_eval_set.py` live on PR #8's
`eval_pipeline` branch, **not on master**. This branch therefore cannot
regenerate a committed baseline that does not exist here. The eval-set *audio*
is on disk, so the deltas are measured on those tracks through the master-side
`simulate` + `evaluator` path, and the committed-baseline regeneration is
handed to whoever merges PR #8 — stated in the PR body as a merge-time task,
not quietly skipped.

### Task 9 — beeps, verified not asserted

A `-d`-equivalent simulated run that writes the click-mixed monitor buffer and
shows the clicks landing on the reported onsets. Evidence recorded.

### Task 9b — soak (added mid-flight by ruling)

30+ minutes of live capture at real-time pace, over multiple tracks played back
to back so the run exercises song boundaries rather than none. Reports max
backlog, per-buffer p99/tail, and every shed/recover transition.

### Task 10 — docs, grep proof, PR

- Root `CLAUDE.md` and `lib/analyser/CLAUDE.md`: aubio = spectral front-end,
  madmom = rhythm. Intent level only — no thresholds, no signatures (policy).
- `grep -rn aubio lib/` returns only filterbank-role lines.
- PR vs master with: the license note (madmom CC BY-NC-SA accepted per #57;
  aubio GPL unchanged), the ~16 CPU-h corpus report-cache regeneration as a
  documented post-merge cost, the onset-delta findings, the realtime number,
  the new checksums, and the follow-up list.

---

## Follow-ups this PR deliberately does not do

1. **Replace the onset chain with something already computed.** It runs three
   multi-resolution STFTs and an eight-network RNN ensemble — 10.6 % of a core —
   to feed exactly one feature: a count per second. Candidates: spectral flux
   over the aubio filterbank we already compute, or the beat RNN's own
   activation. madmom's *beats* stay untouched. The 49-track calibration
   harness makes the comparison measurable: hold the onset rate matched and
   compare the resulting intent timelines.

   **Record the dead end so nobody re-prices it:** shrinking the ensemble is
   *not* the lever. 8 → 1 models saves 2.5 percentage points, not the ~9 the
   model count suggests, because the cost is the STFTs and their spectrogram
   differences, not the networks — and it costs activation agreement (r = 0.977)
   for a still-failing 4.3×.

2. Rule-engine constant recalibration, if the disclosed density deltas warrant
   it. Note the onset threshold's own 80 % interval is [0.30, 0.40]; anything
   downstream tuned against it inherits that width.
3. Corpus report-cache regeneration — `pipeline_sha` invalidates it by
   construction; expected, not a bug to work around. Per-core sim throughput
   fell ~46× → ~4×, so the regen is ~11× more CPU than before; track
   parallelism recovers most of the wall-clock.
4. `training/train.py`'s offline aubio front-end — belongs to the
   champion-generation front-end bake (#72/#77).
5. Committed eval-set baseline regeneration — blocked on PR #8 (see Task 8).
6. Dropping the interim aubio/madmom split entirely (#77) — a retrain decision.
