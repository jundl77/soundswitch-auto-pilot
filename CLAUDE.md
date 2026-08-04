# SoundSwitch Auto Pilot

Intelligent DJ lighting automation system that synchronizes stage effects to live music in real time. A neural section classifier reads the music -- MERT encoder -> online student -> fixed-lag bar decoder -- and the decoder's per-bar decisions are the only thing permitted to say what the lights are doing. The show reaches SoundSwitch over MIDI and OS2L.

---

## CLAUDE.md Policy

**CLAUDE.md documents intent and architecture, not code.** This applies to every CLAUDE.md in this repo â€” root or subdirectory.

- Do not replicate threshold values, function signatures, or internal variable names.
- Do not duplicate content that is already expressed in code. Point to the file instead.
- CLAUDE.md sits one layer above the code. It explains *why* things work the way they do, not *what* specific values are set to.
- Every PR must update CLAUDE.md to reflect any architectural, interface, or behavioural changes â€” but only at the intent/meta level.

**CLAUDE.md is the agent's source of truth.** Any analysis ideas, design decisions, or specifications must be recorded here â€” not just in code or in PR descriptions. An agent must be able to understand what this system does, why it is designed the way it is, and what direction it is heading by reading CLAUDE.md alone, without scanning all source files. Keep it current at all times.

---

## Development Workflow

**All changes must go through a pull request.** Never commit directly to `master`.

Before opening a PR, all tests must pass:

```bash
# Fast unit tests (run these frequently during development)
uv run pytest -m "not integration"

# Full suite including unit + integration tests (a few minutes: the integration
# tests run whole tracks through the real rhythm networks and the real section
# chain, so this is minutes, not seconds. It is not hung. A cold cell cache
# additionally needs a GPU and roughly doubles it.)
uv run pytest

# Run a single test file
uv run pytest tests/test_delayed_command_queue.py -v
```

The integration tests in `tests/test_simulation.py` run real, expert-labelled music through the full pipeline (identical code path to production). One track pins the *mechanism* — command-timing exactness, flush behaviour, report duration, speed, byte-identical determinism, and the plumbing evaluator's verdict; three more go through `training/run_eval_set.py` in compare mode and pin the *behaviour* — per-track report checksums and label-aligned scores against the committed baseline. If they fail, the pipeline is broken or its output moved. There is one simulation mode — real audio files — paced either sped-up (default) or real-time (`--ui`).

**The audio and labels are committed; the model is not, and cannot be.** The ten eval-set mp3s and the label slice they are scored against are in the repository (see The benchmark), so a fresh clone needs no downloads for them. The shipped encoder is 1.3 GB and lives with the student's graph and priors in the gitignored corpus directory, so **the parts of the suite that need a committer do not run on a checkout that has no model**. The `nn_artifacts` fixture is how a test says it needs one, and it skips with a line naming where the artifacts live; everything the demolition left — beats, silence, the OS2L wire, the digest survivors, every unit test — still runs without it, which is what makes the skip narrow rather than a hole. That includes `test_eval_set_head_matches_the_committed_baseline`: without the model it would simulate the degradation state and compare it against a baseline cut from the neural show, so it takes the fixture too, and the benchmark runner refuses outright rather than scoring a machine that is only missing a download.

The gitignored corpus does not follow `git worktree add`, so a linked worktree finds the main checkout's copy automatically (`training/corpus_root.py` owns that resolution, and it is stdlib-only so a *show* can ask it without pulling the benchmark harness into production startup); `$RAVEFORM_DATA_DIR` overrides.

### Testing philosophy

- **Coverage over completeness**: aim for broad, confident coverage of critical logic â€” not 100% line coverage. Tests should catch real regressions, not just pad numbers.
- **Test the logic, not the wiring**: unit tests target pure functions and isolated methods. Integration tests verify the full pipeline assembles correctly.
- **Missing deps**: if a package is declared in `pyproject.toml` but absent from the venv, run `uv sync --extra dev --extra visualizer` â€” do not mock it.
- **Every PR must pass `uv run pytest`** (the full suite, not just unit tests) before merge.

---

## What It Does

1. Reads audio from a microphone/line input
2. Extracts rhythm via madmom's online neural trackers (beats, BPM). The beat stream drives OS2L, the overlay chase and -- counted in fours -- the live bar grid
3. Resamples the same audio to the encoder's rate and runs a MERT encoder plus an online student over it, emitting one class posterior and one boundary score per label cell (`intro / buildup / breakdown / drop / outro`)
4. Commits one immutable decision per bar through a fixed-lag Viterbi decoder, and maps that class to a `LightIntent` (ATMOSPHERIC / BREAKDOWN / BUILDUP / DROP / PEAK)
5. Selects and sends MIDI lighting effects to SoundSwitch based on intent; also sends OS2L beat events to VirtualDJ and DMX overlays via UDP

---

## Architecture

```
  audio thread                                          GPU thread
  ------------                                          ----------
  PyAudio --> MusicAnalyser (madmom online) --> beats, BPM, silence gate
         \
          -> resample + ring write ==================> MERT encoder
                                                       online student (ONNX)
          <== bounded hand-off queue (whole passes) <==/
          |
          SectionDecoder  (bar grid from beats  +  fixed-lag Viterbi)
                |
          class --> LightIntent --> LightEngine --> DelayedCommandQueue --> MIDI / OS2L / Overlay
                                         ^
                                   DriftWatchdog (drift + stage health -> NONE | NN_SHED)
```

Three threads under `run` and under `--ui`/realtime: the audio loop, the GPU
stage, and `Os2lSender`, plus the snapshot server's thread when a viewer is
attached (the viewer itself is a separate *process*). **The consumer is the audio
loop**, deliberately: the delayed command queue, the MIDI client and the event
buffer already live there, so no show state is shared across threads at all. The
only objects two threads touch are the sample ring (one writer, one reader), the
hand-off queue under a lock held for microseconds, and the event buffer, which
the snapshot server reads under the lock an in-process viewer used to take. Fast simulation runs the same stages
inline on a virtual clock, and `build_section_chain` taking a watchdog *is* the
switch between the two -- the stage reports health to one and reads its shed
level off one, so handing one over is exactly the statement "run this off the
caller's thread".

### Key Files

| Path | Role |
|---|---|
| `auto_pilot` | CLI entry point (`run MIDI_PORT`, `list`, `simulate`) |
| `lib/main.py` | `SoundSwitchAutoPilot` â€” async event loop, per-buffer + 100 ms / 1 s / 10 s callbacks; owns `PLAYBACK_DELAY_SEC` |
| `lib/clock.py` | `Clock` abstraction â€” `SystemClock` (prod default) vs `VirtualClock` (fast sim); every time-based component takes an injectable clock |
| `lib/audio_config.py` | Canonical `SAMPLE_RATE` / `BUFFER_SIZE` â€” single source for live pipeline, simulation, and virtual-clock timing math |
| `lib/ui_bridge.py` | The show's whole UI surface: the event buffer served as JSON on localhost, and the viewer process `--ui` starts beside it and kills on the way out |
| `lib/section_chain.py` | Assembles the show's NN path out of the shipped artifacts, for both entry points, so sim=prod is one wiring rather than two; `artifacts_present` is how a caller asks whether this machine has the model |
| `lib/analyser/madmom_rhythm.py` | `MadmomRhythm` -- madmom's online beat stack, adapted from the pipeline's buffer size to madmom's frame rate; the only place that framing mismatch exists |
| `lib/analyser/mert_stream.py` | The live MERT feature stage: resample, 30 s ring buffer, one encoder pass per hop, label cells out. Every geometry number is read off the shipped artifact |
| `lib/analyser/section_model.py` | The online student, one cell at a time -- a bounded feature ring, the carried forward state, and the pinned single-threaded ONNX session |
| `lib/analyser/gpu_stage.py` | The encoder + student on their own thread (B3), the bounded hand-off, and the degradation contract when the GPU stops |
| `lib/analyser/drift_watchdog.py` | `DriftWatchdog` -- one shed door with two inputs: the loop's lost lead, and the stage's own health, which pacing is structurally blind to |
| `lib/analyser/music_analyser.py` | `MusicAnalyser` â€” per-buffer rhythm, RMS, the silence gate, beat/note events |
| `lib/analyser/CLAUDE.md` | Analysis pipeline detail: what each stage measures, the delay model, and the shed ladder |
| `lib/engine/section_decoder.py` | The live bar grid and the committer that runs on it: `FixedLagViterbi` imported from the offline package, fed from two streams that arrive ~8 s apart |
| `lib/engine/light_engine.py` | `LightEngine` â€” decoder decisions in, a show out; owns PEAK, the boundary refresh, the cold-start floor and the per-stream delay |
| `lib/engine/effect_controller.py` | `EffectController` â€” maps `LightIntent` â†’ non-repetitive random MIDI channel selection |
| `lib/engine/effect_definitions.py` | `LightIntent` enum, `SECTION_CLASS_INTENTS` (class â†’ intent) and `INTENT_EFFECTS` (intent â†’ MIDI) â€” the two places routing changes |
| `lib/engine/event_buffer.py` | Thread-safe beat/effect/intent/decoder-state store; read by Dash every 100 ms and serialised into the report |
| `lib/clients/midi_client.py` | MIDI note-on/off to SoundSwitch; 90+ channels, delayed deactivation |
| `lib/clients/os2l_client.py` | zeroconf discovery of VirtualDJ; bidirectional OS2L JSON |
| `lib/clients/pyaudio_client.py` | Mono 44.1 kHz audio input (and optional debug output passthrough) |
| `lib/clients/overlay_client.py` | UDP binary DMX overlay (hardcoded IP â€” must match venue) |
| `simulate/visualizer_app.py` | The visualizer, in its own process: timeline, intent-based stage simulation, metrics, and the decoder-state row, rendered from snapshots polled off the show; motion is the browser's job (see Visualizer smoothness) |
| `simulate/runner.py` | Simulation runner â€” stub clients, full pipeline; virtual-clock fast mode (default) or real-time pacing with a threaded GPU stage for the live UI |
| `simulate/cell_cache.py` | The extractor's cells, cached beside the audio â€” what makes a warm `simulate file` pure CPU and byte-deterministic |
| `simulate/cli.py` | `auto_pilot simulate file|realtime` subcommands |
| `simulate/ui_wedge_rig.py` | The viewer under a slow callback, with no audio and no GPU: a seeded buffer, the real snapshot server and the real Dash app, with latency injected into every poll — `serve` reproduces the freeze in a browser, `profile` times the server callbacks without one |
| `training/corpus_root.py` | Where the gitignored corpus is on this machine, stdlib-only â€” so a *show* can ask without importing the benchmark harness |
| `training/inspect_report.py` | Report inspector â€” per-10s rms/beat/intent bins + intent timeline; the tool for checking a show against a track's structure |
| `training/run_eval_set.py` | The benchmark â€” the frozen eval set through the sim, scored against its labels; cuts and enforces `training/eval_set_baseline.json` |
| `training/soak_nn.py` | The live soak: a player and a subject, real audio through real hardware for half an hour, sampled at 1 Hz |
| `training/nn_determinism_proof.py` | Four runs per track in four interpreters â€” cold/cold, warm/warm, cold/warm â€” and the bytes compared |
| `training/nn_boundary_refresh_rate.py` | What the boundary head fires at and how often; the evidence behind the effect-refresh threshold |

---

## LightIntent System

`LightIntent` is the semantic bridge between audio analysis and lighting output. It lives in `lib/engine/effect_definitions.py`.

Five intents map to structural moments in an EDM track:

| Intent | Musical moment | Decoder class | MIDI pool |
|---|---|---|---|
| ATMOSPHERIC | Intro, outro, silence â€” quiet, or nothing playing | `intro`, `outro` | BANK_2A/B/C |
| BREAKDOWN | Melodic, stripped, emotional | `breakdown` | BANK_2C/D/E + 2F/G/H |
| BUILDUP | Rising tension pre-drop | `buildup` | BANK_1A/B/C |
| DROP | Maximum impact â€” bass, kick, full arrangement | `drop` | BANK_1D/E + STROBE |
| PEAK | A drop that has lasted (engine promotion, see below) | â€” | BANK_1F/G/H |

**GROOVE is gone, and its banks are not** (D7). The decoder's class space has no
groove, so an enum member no path can produce would be a lie -- but retiring the
intent would also have retired `BANK_2F/G/H` from the show, so the pool merged
into BREAKDOWN's and six banks stay in rotation behind the class the corpus
actually labels. This was an audience-visible choice, not a mechanical
consequence, and it was taken deliberately.

**ATMOSPHERIC is one intent for two classes** because an intent cannot know where
in the arrangement it is: the same sound is `intro` at the start of a track and
`outro` at the end. **Any class the model can decode must light something, and
that is checked when the chain is built** -- a retrained model naming a sixth
class stops construction rather than killing the show at the first bar of that
class, an hour into a set.

### How the show is decided

**One posterior per label cell, one decision per bar, and nothing else votes.**
The MERT encoder runs once per hop over the ring buffer it holds; the online
student turns each emitted cell into a class posterior plus a boundary score; the
decoder averages the cells that fall inside a bar, reads the boundary at the bar
*line*, and commits through a fixed-lag Viterbi over an explicit-duration HSMM.
The offline `FixedLagViterbi` is *imported* rather than copied -- a sweep that
disagreed with the runtime by one line would be measuring the wrong decoder.

**The whole stability pipeline is gone, and it was replaced rather than
loosened.** The retired engine guarded a thresholded branch with a vote buffer, a
minimum-dwell check and an invalid-transition table. The decoder's fitted
duration floors are the successor to min-dwell, its `-inf` transitions are the
successor to the veto, and its bounded backtrace is the successor to the vote
buffer -- each of them fitted to the corpus rather than chosen. What remains in
the engine is PEAK, and it is deliberately the same device it always was.

**A bar is four beats, and the count starts one beat in.** There is no live
downbeat tracker (see the follow-ups), so bars are counted off the beat stream.
The thing that turned out to matter most about that count is where it *starts*:
madmom's online warm-up costs the first annotated beat, so the first beat the
runtime ever sees is already bar position 1, and the shipping rule called it
position 0. That single rotation left the show on the correct phase for a median
**0.7 %** of a track -- a grid wrong from beat one rather than slip damage
accumulating -- and fixing it is worth about **4.5x** what repairing slips on the
wrong anchor was worth. The constant is favourable rather than correct (right on
147 of 215 val tracks); a *measured* live anchor would be strictly better.

The price the fallback still pays is written down: at the old anchor, counting
cost **-0.1396 crispness@0.5 s** against an expert downbeat grid, the anchor
recovers **+0.0510** of that, and **all** of the remainder is placement -- the
class decisions are nearly grid-invariant (contested macro moves within noise,
flicker slightly better); they land at a displaced instant. A perfect live
tracker would recover the rest and no more: a lag-0 phase oracle scores level
with the annotated-beat ceiling, so phase is the *entire* cost of the live grid
and beat-detection quality adds nothing on top. Closing it needs an audio-side
cue that is not the boundary head, which has now lost three times: a third of
slips carry no interval evidence at all, and interval repair as configured
over-triggers 11x. The plan that commissioned the first measurement is
`docs/superpowers/plans/2026-08-01-nn-runtime-integration.md` (committed); the
anchor measurement is `training/phase_tracking/` plus its gate artifact in the
corpus, and `tests/test_section_decoder_equivalence.py` is what holds the
runtime's grid to the grid that was priced.

**Beats can stop while audio keeps arriving**, and that is neither silence nor a
song boundary -- heavy sidechain, a beatless passage, crowd noise between sets.
The grid re-anchors at the next beat rather than closing one bar across the gap,
because averaging minutes of audio into a single observation produces a confident
decision about a section nobody played. **The warm-up anchor is not carried
across that re-anchor**: madmom ran throughout and lost no beat to the gap, so
there is no warm-up to re-pay, and the true bar position of the beat that ends a
*beat* gap is measurably a coin toss, so there is nothing to anchor on either.

**A restart has three flavours and they are not interchangeable.** A cold start
re-applies the warm-up anchor, because the beat source really did begin from
nothing. A beat gap re-anchors at position 0 (above). A gap in the *feature*
stage -- a GPU shed and its recovery -- stopped neither madmom nor the count it
produced, so that restart keeps the bar position it is holding and rebuilds only
the grid, the pending cells and the committer; discarding a position that is
still correct would trade it for a one-in-four guess.

**A restart is a birth, and a birth is not the start of a track.** Whatever the
flavour, the committer that came back used to take the corpus's start-of-track
prior, so a decoder reborn 34 minutes into a set believed it was at bar zero of
an imaginary track: it could commit `intro`, which no fitted transition can
enter, and it owed a whole duration floor before it was allowed to leave. On one
measured track a 5 s hole in the beat stream put the rig near dark for **53.4 s**
over the back half of a breakdown the model was 0.978 sure of. A birth now
carries the class the show was already in, places its mass on each class's
*final* duration state -- a floor is charged for time the grid witnessed, never
for time it was not alive for -- and is preceded by one virtual bar. The three
are one change and ship together: the virtual bar is what gives the first real
bar a transition, which is both how a boundary landing there can be acted on at
all and how a carried belief can be **refused**. Without it a carry cannot be
rejected on the bar it is born on, and forcing every val birth to carry every
class in turn, carry-alone accepted 85 of 85 wrong beliefs against 37 of 85 with
the virtual bar. Priced on val in `models/phase_b/decoder_rebirth/`; the runtime
is held to that reference's own decisions by
`tests/test_section_decoder_rebirth.py`.

**A cold start is a birth too, and that is the change with the widest reach.**
The gate is named after re-anchors, but pre-aging and the virtual bar apply at
the opening of *every* track, so the first run no longer owes its duration floor:
a probe track's `intro` now ends after one bar where the floor is eight. Only 13
of 215 val tracks re-anchor at all, yet every track's report moves, and that --
not the gap repair -- is the dominant reason a benchmark re-cut touches all ten
checksums. Read a moved checksum against that before assuming a gap was involved.

**Two of the four births are unmeasured, and both are deliberate.** The third --
the committer restarted because the grid outran it -- is unreachable offline (the
harness holds the whole grid), so it is correct by construction rather than
measured. The fourth is a feature-stage gap: it carries for the same reason the
other three do, and resetting a mid-track committer to the start-of-track prior
is precisely the defect this package removes, so it was approved on the failure
ranking rather than on a measurement. Both are pinned by unit tests and both are
one line to withdraw. A class the fitted priors never start a track in cannot be
carried into at all -- its whole initial row is `-inf`, and a birth into it would
commit that class forever with nothing able to move it -- so a birth degrades to
the cold spread and says so, rather than raising on the show's thread.

**PEAK is an engine-level promotion, not a class.** "A drop that has lasted" is a
run length, which no window of audio can express, so the engine promotes an
already-committed DROP once the run reaches a fixed number of *bars* -- the same
musical length the retired beat-denominated device used, converted rather than
re-chosen. While PEAK is current, DROP decisions are absorbed so the pair cannot
oscillate and the reported timeline keeps reading the PEAK the room is looking
at. Constant in `lib/engine/light_engine.py`.

**ATMOSPHERIC now has three producers, and only one of them is the old timer.**
The decoder commits it for `intro` and `outro`; the beat-absence timer still
fires it when beats stop; and a **cold-start floor** lights it once if the
committer has never spoken at all. That floor exists because "hold the current
intent" is only a show if there *is* one -- a GPU that is dead at boot, or a
machine with no artifacts, commits nothing, and the rig would stay dark all night
while every log line said the show was holding. Its margin is the bar grid's
warm-up budget on top of the measured chain latency, and ATMOSPHERIC is the
choice because the error asymmetry says so: a quiet default that turns out wrong
reads as a slow start, a guessed high-energy look that turns out wrong reads as a
broken rig.

**The boundary head triggers the effect refresh that YAMNet used to** (D9). An
effect change *inside an unchanged intent* is behaviour no class boundary can
express, because the class is the same either side. The model already emits a
boundary score per cell; a peak in it inside a held intent re-rolls the effect
from the current pool. Same signal, no new model, and trained on section
boundaries rather than on cosine outliers. The rate governor moved across from
the retired detector verbatim; the threshold is measured
(`training/nn_boundary_refresh_rate.py`), because the rate YAMNet actually
produced was never recorded and cannot be recovered -- simulation stubbed it out
from the day fast simulation landed, so it never fired in a report, a fixture or
a training table. This is the one place the project's "match a measured rate,
never pick a number" discipline could not be applied literally, and saying so is
better than inventing a number to match.

### The delay model: the engine now runs BEHIND the room

This inverted with the NN (finding **B1**), and it is the single most
consequential plumbing change on the branch.

    audio -> final posterior      features:  F + hop + the head's future window
    posterior -> committed bar    decoder:   (lag_bars + 1) x the current bar
    ---------------------------------------------------------------------------
    audio -> committed intent     ~13.7 s at the corpus median bar

The rule engine ran *ahead* of the audience and `DelayedCommandQueue` held every
command back by the whole look-ahead. The model runs *behind*, so the correct
relation is `queue_delay = playback_delay - chain_latency`, and it must be >= 0.
`PLAYBACK_DELAY_SEC` is **14.0 s** and must match `playback_delay_seconds` in
dmx-enttec-node; it is defined in `lib/main.py` and mirrored in
`simulate/runner.py`. Local debug playback is delayed by the same amount, so
headphone monitoring stays in sync.

- **`lag_bars = 2` is the chosen point on a measured curve, not a default.** The
  accuracy-versus-lag sweep over the shipped student's val posteriors saturates
  at 2: lag 3 and lag 4 are statistically indistinguishable from it on crispness
  (paired bootstrap, P(delta>0) = 0.79 and 0.56), lag 4 is significantly *worse*
  on contested macro, and each bar costs ~1.9 s of show delay. Lag 2 puts the
  corpus p99 (14.08 s) inside the budget where lag 3's p99 (16.11 s) straddles
  it. **Two owner rulings disagree on this axis and the disagreement is real**:
  under the decoder sweep's own published utility (macro minus a flicker
  penalty), `lag_bars = 0` ranks first by a wide margin -- it flickers about a
  quarter less, switches least, and lands the whole show ~5.7 s earlier. It pays
  for that with crispness and contested macro, both losses statistically solid.
  Crispness was promoted above that utility, and lag 2 is what that promotion
  chooses. If crispness is ever demoted, lag 0 is the pick and it is the cheapest
  option in every other respect too.
- **The delay belongs to the stream, not to the queue.** A beat, a bar decision
  and a boundary refresh are different ages when they arrive, so a command's fire
  time is the *song instant it describes* plus the playback delay, and the age is
  measured rather than modelled from a median bar. Getting this wrong put a
  timer-fired ATMOSPHERIC on top of a drop the committer had already called.
- **Superseding is by audio, not by arrival.** A newer statement about the same
  instant or later replaces what is queued for it; anything about earlier audio
  is left alone. Cancelling by arrival order would delete every intent block but
  the last one whenever the chain sits near its budget, since consecutive bars
  are ~1.9 s apart and can both be in flight.
- **Chain latency is measured per track and logged, never assumed.** The
  decoder's half is proportional to bar length (12.11 s to 16.37 s across the val
  corpus at lag 2), so a slow track can be older than a constant playback delay.
  When that happens the intent commits as soon as it can -- late, not broken --
  and the engine logs it once per transition. This lateness is *accepted*
  (owner ruling) rather than a defect, but it is not free to *move*: the
  benchmark records it per track and compares it exactly, as a count fact rather
  than a score with a tolerance. It is a property of the music and of the grid
  the bars are drawn on, so a run in which it changed measured something
  different, whatever the scores did.
- **Everything the room can see waits; the engine's own bookkeeping does not.**
  Song-boundary resets and the OS2L wire (which talks to a DJ's software, not to
  the audience) happen immediately; MIDI, the overlay chase and the stage go
  through the queue. Before that split the stage blacked out while the last
  fourteen seconds of a track were still playing -- inaudible in every report,
  unmissable in the venue.

### Fast simulation and the cell cache

File simulation runs on a virtual clock driven by audio sample position instead
of the wall clock -- the full pipeline (identical code path to production, same
`build_section_chain` wiring) processes a track faster than real-time and
deterministically. Beat timestamps are song-position seconds; intent and effect
blocks are stamped when the audience hears them, and **each block now records
`song_t`, the instant of audio it describes**, because the per-command delay
means no constant de-shift and no beat-matching rule can recover song time from
the stamp alone (a bar line is not a beat the report carries).

**`simulate/cell_cache.py` is the decode cache one layer up** (D12). The only
part of the path that needs a GPU is the MERT encoder, so the cells it emits are
recorded beside the audio and replayed: every run after the first is pure CPU,
and a machine with no GPU can run the whole pipeline over a track someone else
extracted. The trigger is a *position in the call sequence* (source samples
pushed when a pass ran), not song time, so a replay needs neither the resampler,
the ring nor the schedule, and the reset the engine does at each song boundary
needs no special handling. The key carries the encoder identity, the framing, the
source rate, the audio's size *and* mtime -- and the decode path, twice: the
simulation decodes with librosa and the corpus pipeline with ffmpeg, and the two
move 13.2 % of near-boundary decisions, so a corpus sidecar replayed into a sim
would be a silently different measurement. It is in the filename (collision
structurally impossible) and in the key (a named miss rather than a wrong
answer).

**Determinism is measured, not inherited.** A GPU is the one component whose
answer can depend on which kernels a driver picked, so
`training/nn_determinism_proof.py` runs four processes per track -- cold/cold,
warm/warm, cold/warm -- and compares report bytes *and* extractor sidecar bytes.
On this box every comparison held: the fp16 CUDA forward pass is bitwise
reproducible and the replay reproduces the live extractor exactly, so the
contract is **byte-deterministic, cold or warm**. The script derives its verdict
from the measurement, so a machine where cold runs diverge gets the weaker
sentence ("given cached sidecars") rather than the stronger one. There is no RNG
in the NN path at all; the only RNG in a simulation is the effect pool draw,
seeded before every run.

The threaded paths (`--ui`, `realtime`) are deliberately **not** cached: they
exist to run the real GPU thread, and a replay there would prove nothing about
it. Real OS scheduler jitter is only observable there too.

### DMX migration path

When moving away from SoundSwitch to direct DMX:
- Replace `EffectController._apply_autoloop(effect)` with a `_send_dmx(intent)` call
- Everything above (`MERT -> student -> decoder -> LightIntent -> EventBuffer`) stays unchanged
- `INTENT_EFFECTS` dict in `effect_definitions.py` becomes the only thing to remove

---

## Visualizer smoothness

**The viewer is a separate process, and that is measured rather than tidy.** Dash
used to serve out of the show's own process, so every callback rendered a figure
while holding the GIL the audio loop needs -- and that loop is fed by an input
that drops rather than queues, so what it loses is audio, not latency. One
ordinary viewer, at the same poll rate either way, cost two paced full-track runs
five and four shed transitions; with the render moved out, the same two runs cost
none and one -- and none of the *drift* sheds the contention produced survived at
all. So the show serves exactly one UI-ish thing -- `EventBuffer.snapshot()` as
JSON, over a stdlib HTTP server on a thread, at the poll rate the callback used
to run at -- and `--ui` spawns the Dash app as a child that polls it. `--ui-port`
still means the Dash port and the snapshot port is derived from it, so nothing an
owner types changed. Neither did any rendering: the render functions always took
snapshot dicts, so only the data source swapped. Two things keep the split from
eroding -- `lib/` may not import dash, flask or plotly, which the
dependency-surface probe enforces, and the viewer is killed with `taskkill /T`
because the venv launcher re-execs and the pid the show holds is a parent (#181).
The report path never went near any of this: it is still written in the show.

**The simulation can be monitored on headphones, and it is the same monitor the
live show uses.** `simulate file -o IDX` plays the decoded track out one playback
delay behind the analysis, so the owner hears *room* time and what he hears lines
up with the room-aligned UI beside it rather than running fourteen seconds ahead
of it. The delay machinery is extracted rather than copied — one `DelayedMonitor`
holds the buffer, the arm, and the drop-the-tail-on-silence rule, and both the
live path and the simulation drive it. Copying it is how the two would drift, and
the one bug this code has already had (a monitor delay that collapsed after the
first stop) is exactly the kind that would then need finding twice. The stop gate
applies unchanged: a gap shorter than the persistence window plays through, a
real stop drops the queued tail and re-arms.

`-o` implies real-time pacing, because a monitor is meaningless at fast-sim
speed; passing it without `--ui` paces the run with no viewer. Fast headless is
untouched and stays the default — the monitor is a pure side effect that the
report path never sees, which the digest is the check on. One caveat worth
knowing: the queue drains one buffer per fed buffer, so the true lag is the delay
minus one buffer period. That is the live path's behaviour too, preserved rather
than introduced.

**One Ctrl-C has to end the session, and the obvious way to wait on a child
guarantees it cannot.** `Popen.wait()` on Windows is `WaitForSingleObject` with
an infinite timeout: it releases the GIL without joining CPython's SIGINT event,
so it cannot return early. The interrupt is *delivered and then held* — measured,
not inferred: killing the viewer by hand released the wait and the
`KeyboardInterrupt` fired five seconds late, at the first bytecode afterwards,
while the line straight after the wait never ran at all. That makes the deadlock
exact, because the only thing that kills the viewer is the handler the interrupt
has not reached yet. A second Ctrl-C cannot help; it re-sets a flag that is
already set. So the session waits on the viewer by polling, which costs one
wake-up every fifth of a second and makes the interrupt land in the same
millisecond it is sent. The report is unaffected either way — it is written from
a `finally` around the pipeline, so an interrupt mid-track still writes what the
run had.

**Smoothness is bought in the browser and never from the server**: the poll stays
at its tick, because a faster poll once starved the plotly bundle behind Chrome's
connection limit, and a 10 Hz viewer once cost the audio loop four sheds. The
server publishes an anchor beside the panels; the browser runs one
`requestAnimationFrame` loop off it and interpolates. The one callback behind
that tick has since become two, but the rate did not move and neither did the
number of polls per tick -- what was bought is that the anchor no longer waits
on the figure.

Two things about that loop are measured facts rather than preferences, and both
are easy to undo by accident:

- **Nothing on the per-frame path may touch Plotly.** `Plotly.relayout` is not a
  cheap way to move an axis -- it re-renders the traces, so scrolling by relayout
  spent 40% of the main thread, most of it laying out the same beat markers sixty
  times a second. The window scrolls by translating a wrapper div, which the
  compositor does with pixels that already exist, and the axis is only re-seated
  when the offset would expose the padded strip. That strip is why the timeline is
  drawn past the lead: what translates in from the right has to be future the
  audience has not reached rather than a gap.
- **The offset is derived from the figure's live range, not from a stored
  anchor.** Whether the range last moved because the server pushed a figure or
  because the loop re-seated it, the window on screen is right, so there is no
  seam to get wrong and no ordering to coordinate.

What is animated: the window scroll and both clocks (from their own bases), and
the beat glow, which is a CSS keyframe retriggered per beat with a negative
animation-delay so a beat arriving mid-poll starts its decay where it landed.
What is stepwise **by design** is data arrival -- new beats and new intent blocks
appear at the poll rate, and beat markers are Plotly nodes rebuilt on every
figure push, so there is no stable element for an enter-transition to attach to.
The now-cursor moves at no rate at all: in a window that always ends a fixed lead
past now its screen position is a constant, so it is a plain positioned div
outside the translated layer -- as a plotted shape it rode the transform and
jittered against the very thing it measures.

**The timeline shows a song, not a session, and the song is the room's.** Its
zero is the instant the room *hears* a track start; the axis, its ticks and both
clocks re-zero there and sit at an empty zero between songs. That follows from
the convention the display already keeps -- every event is drawn at the moment
the room experiences it -- so a boundary judged on detection stamps would cut the
ribbon somewhere other than where the events on it were placed. Both sides of
every such comparison are therefore on the room clock, which is the one way this
can silently go wrong: a beat detected *before* a start reaches the room still
belongs to the song that start opens, and a beat detected before a stop belongs
to the song it interrupts. Nothing stored moves -- the re-basing is derived in
the render path from the sound events the snapshot already carries, so the report
and the digest are untouched. Two consequences in the browser layer: a backward
step in the anchor is a re-base rather than elapsed time and must not be
extrapolated from (the existing re-seat guard then absorbs it), and the
last-pulsed beat stamp is cleared at a boundary because song times repeat.

**The snapshot ships the window, never the session.** It carries only what the
viewer can draw: the timeline's span plus the look-ahead, because a record is
stamped when it is detected and drawn one look-ahead later. Storage keeps its own
window -- unbounded when a report is being written, since that has to cover the
whole track -- and the snapshot no longer reads the whole of it, so a poll costs
the length of the view rather than the length of the set. Sound events are exempt
from the *storage* window and stay whole: one record per track boundary rather
than per second, and the song the display counts from can be an hour old. Pruning
them from storage is the subtle version of the same bug -- a live run keeps a
finite window, so the stop that ends a long song would evict the start it is
measured from and the axis would jump rather than reset. The *payload* is not
exempt, because it was the one list a four-hour set grew without bound and the
render path scans it once per beat: it ships the events inside the view plus the
single event before them, which is exactly the record the display needs and no
more. One is enough because only the latest event before the view can still be
in force, and it must be shipped whichever kind it is -- keeping only the start
would read a room that stopped an hour ago as playing. The report and the digest
read the storage directly and never the payload, so neither moved.

**A tick that overlaps its own answer loses it, so the tick is gated.**
dash-renderer evicts an in-flight callback from its `watched` list the moment an
identically-wired one is requested, and the evicted response then fails a
membership check and returns silently: no prop write, no downstream dispatch, no
error, a clean 200 in the network log. So once a refresh takes longer than the
interval, *every* answer is thrown away and the page freezes while the show and
the snapshot endpoint stay perfectly healthy -- at the measured 330 ms against a
250 ms tick that is the steady state, not a transient. The interval stays at 250
ms, because the cliff is the overlap and not the rate: a clientside gate decides
whether a tick is allowed to become a request, and a tick that finds one in
flight is skipped rather than queued. The renderer's `request_pre` hook marks a
stream busy as the request leaves; what frees it again is the answer **landing in
the layout**, watched from the clientside callback the store already feeds.

Freeing it on the *resolution* instead is the trap, and it was measured here
rather than reasoned about: `callback_resolved` runs before the props are
written, so a tick arriving in that window still evicts the answer the gate
existed to protect. It looked correct and lost every second answer -- the browser
showed the anchor resolving in pairs three seconds apart with only the later one
reaching the page. Landing is the only seam strictly after the write. Two escapes
keep that from stranding anything: an errored callback frees its stream at once
because nothing will land, and a stall deadline frees it regardless and says so
-- the worst case is the old behaviour, never a page that stops asking.

**The same seam carries the watchdog, because this failure prints nothing.**
`callback_resolved` sees the fresh answer whether or not the page takes it, so
comparing successive answers would notice nothing; what it compares instead is
the answer the server just gave against the value the page last accepted. Those
agree on every healthy tick and on a genuinely paused show, and diverge only when
answers are being dropped on the floor -- which is the one thing nothing else in
the stack can see.

**The anchor and the figure are two streams now.** They were one callback, so the
clock waited on the timeline: the anchor the animation loop runs off costs under
a millisecond to build and the figure costs an order of magnitude more, on a
payload an hour into a set. Split, the anchor answers every tick and the figure
every other one, and the figure re-uses the snapshot the anchor stream just read
instead of polling again -- so a tick still costs exactly one poll and the panels
still cannot disagree about which snapshot they are drawing.

**Stops are immediate once the silence has persisted; starts wait for the room.**
The asymmetry is the spec, not an oversight: music starting has to travel the
playback delay before anyone hears it, so a start and everything after it stays
room-aligned -- but silence needs no look-ahead, because there is nothing left to
hear. A stop therefore ends the song on the display, cuts the monitored output's
buffered tail instead of playing it out, and discards the beats still in flight,
which were never going to reach the room. Everything mid-play keeps the room
alignment.

What the first version of that got wrong is that **detected silence is not the
same as a stopped set**. The gap between two tracks is silence too, so every
song change threw away a playback delay's worth of tail the room was still
enjoying. The whole bypass package therefore waits for the silence to *persist*:
the monitor cut, the flush of lighting the room has not seen, the quiet floor,
and the display's own reading of the stop are one thing that fires together
after the window, or does not fire at all. A resume inside the window cancels
it entirely and the room hears the natural gap a look-ahead later -- which is
exactly what it would have heard from a DJ mixing on the desk. Nothing else
moves: the stop is still detected when it is detected, its sound event is still
recorded there, and the song-boundary reset and rebirth machinery still run at
once. Those were gap-correct before the bypass existed, and the gate is not
theirs to wear.

**A shed chain and a model opinion look identical on screen, so the screen says
which it is.** The show holds its current intent when the section stage sheds,
and held ATMOSPHERIC through a shed is pixel-for-pixel what a confident
ATMOSPHERIC decode looks like — which cost a day of debugging a model that was
never asked. The snapshot therefore carries the watchdog's own reading: the
level, the fault holding it, a **monotonic** shed count and a sheds-per-minute
rate over the trailing minute. The counter is the load-bearing one: state is
polled at a fixed step and a shed shorter than that step is invisible to a level
poll but is exactly the flapping worth seeing, so the count is incremented at the
watchdog's single transition point rather than sampled. The rate is what
distinguishes a box that stumbled once from one that is failing continuously; the
total is what says a quiet-looking run had a bad five minutes an hour ago.

This is payload only. It is deliberately **not** in the report: the report is
what the room saw, and a shed is a fact about this machine on this night, so
putting it there would make two runs of the same audio on two machines produce
different bytes. The digest is the check that this stayed true.

**The display derives the gate rather than being told about it.** A stop reaches
the room one window after it was recorded, and a stop with a start inside that
window never reached the room at all -- so that pair is dropped from the render
entirely and the song keeps its origin straight through the gap. This is the
same move the look-ahead shift already makes: nothing stored moves, the room's
version is derived from the sound events the snapshot already carries. The
report and the digest are untouched, and the window is written down in exactly
one place for both sides to read.

**That cut applies to the start record too, not only to the beats behind it.** A
start whose audio is still travelling when the bypass fires never reaches the
room either, so it is not a record the display may count a song from. Leaving it in is
what made the remapped list non-monotonic -- starts move by the delay and stops do
not -- and the two obvious ways to ask which record is current then disagreed: the
last by list position, and the last by room time. Position was accidentally right;
room time was confidently wrong, claiming PLAYING with a running clock over
silence that carries no beats, because a play burst shorter than the look-ahead is
audible to nobody. Cutting first makes the two provably the same answer, and that
equivalence is the property worth pinning rather than either reading on its own:
one rule decides, so neither can drift from the other.

**The blackout that follows is recorded, and it says who asked for it.** A stop
drops the lighting still queued and puts the rig on a quiet floor at once, and
that floor is a real thing the room saw, so the report states it rather than
hiding it -- report-equals-truth is what makes the report worth reading, and a
blackout the record omits is invisible to the flicker and boundary streams that
are supposed to notice it. But it is an operator action, not a reading of the
music, so every intent block carries the trigger that produced it (the classifier
by default, silence when the stop path wrote it) and the label-aligned evaluation
takes only the classifier's. Without that field the two are indistinguishable
downstream, and a blackout would be scored as if the classifier had claimed the
section was quiet.

**Dash answers callbacks on threads, and the poller they share is one
connection.** A failing poll clears it; a poll running beside it then re-read
that attribute between its own request and its response, found nothing, and
raised past a handler that only catches transport errors -- so the callback
returned 500 and *every panel* stopped updating until the page was reloaded.
Measured, not theorised: one live session froze for its remaining four minutes
this way. Any shared client here needs a lock and needs its handle held locally
across the round trip.

Measuring this needs a **real visible browser window**. A hidden tab gets no
`requestAnimationFrame` whatsoever, so every cadence reads as zero and every
throttled tab looks like a passing result; a headless one measures software
raster instead of the machine's compositor. And a run whose track has ended reads
frozen for an unrelated reason -- the server clock stops, so the loop correctly
stops extrapolating. Both mistakes were made here before the numbers meant
anything.

**Occluded counts as hidden, and that is the trap on this machine.** Chrome's
native occlusion check reports `visibilityState: hidden` for a window that is
neither minimised nor backgrounded but merely covered -- which is every window in
a session with no foreground desktop, so a viewer meant to be measured under load
polled at 0.5 Hz instead of 4 Hz and reported a *softer* load than a real one.
`--disable-features=CalculateNativeWinOcclusion` (with the backgrounding and
timer-throttling flags) restores the real behaviour rather than faking it, and
`document.visibilityState` is the thing to assert before believing any number.

---

## Running

```bash
# Install dependencies (requires uv: https://github.com/astral-sh/uv).
# torch, transformers and onnxruntime are BASE dependencies now -- they run the
# show, not an offline extra -- so a base install carries the CUDA wheels.
uv sync --extra dev --extra visualizer

# Offline training work only (onnx export, tensorboard).
uv sync --extra dev --extra visualizer --extra training

# The app is installed as an editable console script, so `uv run auto_pilot ...`
# and `python auto_pilot ...` are equivalent.

# List available MIDI and audio devices
python auto_pilot list

# Minimal run (MIDI port 0, default audio device)
python auto_pilot run 0

# Run with real-time Dash visualizer
python auto_pilot run 0 --ui

# Full options
python auto_pilot run 0 -i INPUT_DEVICE_IDX -o OUTPUT_DEVICE_IDX --no-os2l --ui

# Simulation (no hardware; needs the shipped model artifacts, and a GPU only on
# a cold run -- a warm cell cache replays the extractor on CPU)
python auto_pilot simulate file path/to/song.mp3          # fast headless: report + plumbing evaluation
python auto_pilot simulate file path/to/song.mp3 --ui     # real-time paced, threaded GPU stage, live Dash timeline
python auto_pilot simulate file path/to/song.mp3 --ui -o 17  # ...and monitor it on device 17, one delay behind (room time)
python auto_pilot simulate file path/to/song.mp3 -o 17     # monitor with no viewer; -o alone still paces in real time
python auto_pilot simulate realtime                       # microphone input with live Dash timeline

# Inspect a report: per-10s rms/beat bins, intent timeline, distribution
python auto_pilot simulate file path/to/song.mp3 --report report.json
python training/inspect_report.py report.json

# The benchmark: the frozen eval set, scored against its labels
python training/run_eval_set.py                     # compare against the committed baseline
python training/run_eval_set.py --write-baseline    # re-cut it (see below before you do)

# Score the committed show against the whole expert corpus (seconds; needs a
# built table, and the committed table still describes the rule engine)
python training/evaluate_against_labels.py --data-dir training/data/raveform

# The offline v1 NN verdict: decoder search on val, then the side-by-side val
# table (needs mel posterior sidecars + priors; pure CPU). This is the MEL
# generation, which is not what ships -- see the NN section below.
uv run python -m training.nn.sweep --data-dir training/data/raveform
uv run python -m training.nn.evaluate_v1 --data-dir training/data/raveform

# The live-path evidence harnesses
uv run python training/nn_determinism_proof.py <mp3>... --write      # four processes, byte comparison
uv run python training/nn_boundary_refresh_rate.py <mp3>... --write  # the effect-refresh threshold's evidence
uv run python training/soak_nn.py play --device N --gap 3            # the source, in one shell
uv run python training/soak_nn.py run --midi-port 1 --input-device N --minutes 33

# Tests
uv run pytest -m "not integration"   # fast unit tests only
uv run pytest                        # unit + integration (minutes, not seconds)
```

**Flags (`run`):**
- `-i / -o` â€” audio device indices from `list`; passing `-o` enables delayed audio monitoring on that device
- `-d` â€” debug: adds a click on every detected BEAT to the monitored audio (implies monitoring on the default output if `-o` is not given). Beat-triggered by owner preference, and now the only trigger there is â€” the onset stream it used to be able to fire from no longer exists.
- `--no-os2l` â€” disable VirtualDJ connection
- `--ui` â€” launch the real-time visualizer, in its own process, at http://localhost:8050; it stops with the show
- `--ui-port N` â€” change the visualizer's port (the show's snapshot endpoint follows it)
- `--report FILE` â€” write JSON session report on exit

---

## ML / DSP Components

- **madmom** -- all rhythm: beat tracking (a recurrent-network ensemble feeding a
  dynamic Bayesian network) and BPM. **Online mode only**, and that is
  load-bearing: madmom's offline decoders score better and cannot run live, so a
  number produced by one is a number the runtime can never reproduce. Chosen over
  aubio on a measured decoded comparison rather than reputation; the basis and
  the measured effect on the show are in `docs/migration-evidence.md` and
  `training/migration_deltas.json`. The onset stage is **gone** -- it existed
  only to feed the rule classifier's density features, and its calibration
  artifacts went with it -- so what madmom supplies today is beats, and beats are
  also what the bar grid is counted from. It is the one live network that is not
  optional: the shed ladder's only rung is the section stage, and beats are never
  shed, because a show with no beats has no OS2L wire, no overlay chase, no bar
  grid and no silence recovery.
- **MERT (`m-a-p/MERT-v1-330M`, via `transformers`/torch)** -- the acoustic
  front-end. It is run as a *streaming* encoder: pass `k` encodes the ring buffer
  ending at `k*hop` and emits exactly the frames whose centre lies in
  `[previous_hi, T - F)`, so an emitted cell sees a bounded amount of future
  audio and that bound is the first half of the delay budget. Every geometry
  number -- model id, revision, layers, pooling, margin, hop -- is read off the
  shipped artifact rather than retyped, because a constant copied into `lib/`
  drifts silently the first time a model is re-exported. The resample into the
  encoder's rate is part of train==deploy and was *measured* against the
  ffmpeg resampler the offline features were extracted with (D4), not assumed:
  nothing fails loudly when a front-end filter changes, the posteriors just get
  quietly worse.
- **The online student (ONNX, `onnxruntime`)** -- a bidirectional model over a
  bounded window, so a cell is decided from a ring holding a fixed span around it
  plus a carried forward state, which is the whole past of the song at no extra
  cost. The live path therefore holds two things and nothing else: a ring of
  feature cells and one state tensor. The ring is primed from the corpus mean and
  never from zeros -- zero raw features are a confident out-of-distribution input
  after the model's own input affine. The session is pinned single-threaded as a
  *determinism* contract, not a performance choice: a threaded reduction sums in
  whatever order the pool finishes in and float addition is not associative. The
  graph is verified against its recorded sha **and** its geometry re-derived from
  the shapes the graph itself declares, at construction rather than at the first
  beat -- a show that discovers its model is the wrong one halfway through a set
  has already played the wrong lights.
- **The fixed-lag decoder** -- `training/nn/decoder.py`'s `FixedLagViterbi` over
  an explicit-duration HSMM, with the structural graph, duration floors and
  hazards fitted from the corpus (`training/nn/priors.py`). It is imported by
  `lib/`, not copied. It owns stability and latency policy in one place.
- **aubio and YAMNet/TensorFlow are gone.** aubio supplied a 40-band mel
  filterbank whose only remaining consumers were the rule classifier's features
  and the silence gate; YAMNet supplied section-change events for an effect
  refresh the boundary head now triggers. Both dependencies left the tree with
  them (owner rulings #142/#143), and a test asserts the live path imports
  neither. **This is a one-way door and it is disclosed rather than buried:**
  deleting the aubio front-end retires this repo's ability to regenerate the mel
  model generation's inputs. MERT supersedes mel, and aubio's GPL was the tree's
  largest licence exposure, so it was intended -- but existing mel sidecars can
  only be read from here on, never rebuilt.

---

## Training Corpus (Raveform)

The project's direction was to replace hand-tuned classification thresholds with a model trained on real, expert-labelled EDM structure. **That has happened** -- the thresholds are deleted and a model drives the show -- so this corpus is now what the *next* generation trains on and what every evaluation is scored against, rather than a plan. It is **Raveform** (Hugging Face, `taejunkim/raveform`): 1,423 EDM tracks with expert section annotations plus a per-track beat grid. Only the annotations are distributed -- the audio is not, so each track is fetched per YouTube ID for research use.

The pipeline is a set of scripts, each resumable and safe to re-run. Acquisition through the clean manifest lives in `training/raveform/` and is stdlib-only; everything downstream lives in `training/` and `training/nn/`, and the training-table build additionally drives the project's own simulation pipeline (which it treats as read-only). None of it is an installed package -- each script puts the directories it needs on `sys.path` so siblings import by plain module name:

| Script | Role |
|---|---|
| `raveform/raveform_fetch_annotations.py` | pull and schema-validate the annotation archive from Hugging Face |
| `raveform/raveform_manifest.py` | annotations -> `manifest.csv`, plus the label / duration / transition statistics |
| `raveform/raveform_download.py` | manifest -> one mp3 per YouTube ID, sequential and resumable |
| `raveform/raveform_supervisor.py` | unattended patient resume: relaunches the downloader across escalating cool-downs after a refusal wall |
| `raveform/build_clean_manifest.py` | manifest + downloaded audio -> `clean_manifest.csv`, the trusted subset everything downstream reads |
| `raveform/raveform_validate.py` | manifest + audio + download state -> `validation_report.{json,txt}` and `checksums.sha256`: the acquisition-complete verdict |
| `build_training_table.py` | clean manifest + the unmodified fast sim -> `training_table.csv.gz` (one row per labelled beat) and a sim report per track. It **no longer exports mel sidecars**: the exporter went with the analyser's filterbank, so existing sidecars are still read and new ones cannot be produced |
| `evaluate_against_labels.py` | training table -> `baseline_eval.json` + a printed report: the committed show scored against expert labels (confusion, per-class F1, boundary-F1, flicker, worst songs) |
| `select_eval_set.py` | clean manifest + annotations -> the frozen ten-track benchmark at `training/eval_set.json` (committed, tempo-spanning, structurally rich) |
| `eval_assets.py` | the eval set's committed artifacts: the derived opaque mp3 names, the sha-pinned label slice, and the `--cut` that re-makes both |
| `run_eval_set.py` | the frozen eval set -> per-track report checksums and label-aligned scores; cuts and enforces `training/eval_set_baseline.json` |
| `nn/dataset.py` | clean manifest + mel sidecars + annotations -> `splits.json` and the windowed, loss-masked training set the CRNN reads |
| `nn/model.py` | `SectionCRNN` -- the two-head acoustic model (label logits at ~10 Hz, boundary logits at frame rate) |
| `nn/train.py` | windowed dataset -> checkpoints, `training_report.json` and TensorBoard logs under `<data-dir>/models/v1/` |
| `nn/export_onnx.py` | a training checkpoint -> `<data-dir>/models/v1/model.onnx` (dynamic time axis), plus the single pinned onnxruntime session every consumer is required to go through |
| `nn/infer.py` | mel sidecars + `model.onnx` -> one posterior sidecar per track under `<data-dir>/posteriors/`, byte-identical on every regeneration |
| `nn/priors.py` | corpus bar runs -> the fitted structural graph, duration floors and hazards the decoder commits against (`<data-dir>/models/v1/priors.json`) |
| `nn/decoder.py` | posterior sidecar + bar grid -> immutable per-bar class decisions (fixed-lag Viterbi over an explicit-duration HSMM); owns stability and latency policy. **`lib/` imports this**, so it is live code, not offline code |
| `nn/decoder_config.json` | the shipping decoder config, committed as a file rather than synthesised at runtime. Loading a config with a key `DecodeParams` does not know **raises** -- it used to be silently dropped, which would have shipped a decoder nobody chose |
| `nn/evaluate_v1.py` | decoded timelines + training table -> the verdict for one split at `<data-dir>/models/v1/eval_<split>.json` (`eval_val.json` is the tuned reading, `eval_test.json` the selection-clean one): NN and rule classifier side by side, scored by `evaluate_against_labels`' own functions |
| `nn/sweep.py` | cached posteriors -> the decoder parameter search and `<data-dir>/models/v1/decoder_config.json` (best val macro-F1 subject to the baseline's flicker and the latency budget) |
| `nn/downbeat_*.py`, `nn/evaluate_downbeat.py`, `nn/compare_runs.py` | the downbeat chain: a second head, a bar-phase decoder and their verdict harness. Offline and parked -- nothing in `lib/` imports any of it, and the show counts bars instead. Mapped in `training/nn/CLAUDE.md` |
| `phase_tracking/*.py` | the live bar grid priced without a GPU: candidate phase trackers over the cached madmom streams, a no-decode sweep on phase accuracy, and a gate that decodes each candidate on the shipping config against the shipping fallback. It produced the anchor the runtime now uses; `tests/test_section_decoder_equivalence.py` holds the two grids together |

`clean_manifest.csv` is the boundary between "audio we happen to have" and "audio we are willing to learn from". Only its `ok` rows may feed a training table or an evaluation run.

`training_table.csv.gz` is the corpus-wide version of what a single sim report already is: a row per beat, now carrying the expert label for the section that beat falls in, the show state the engine actually committed there, and a per-track z-scored copy of every continuous column. It is the input to both the label-aligned baseline evaluation and the neural section classifier's dataset builder. **Its feature columns are now BPM and RMS and nothing else** -- the rule engine's four (onset density, kick strength, centroid trend, sub-bass ratio) were deleted with the chains that produced them, because a report carries none of those keys and emitting them would write `0.0` on every beat of every track with nothing to say so.

Decisions that belong here rather than in the code:

- **Canonical label vocabulary.** The published labels are a superset of the documented seven. `end` is a tail sentinel, not a musical section, and is dropped; `altintro` folds into `intro` and `bridge` into `breakdown` (too few examples to learn as a class); `altoutro` is kept. Adjacent same-label sections are merged *after* the drop and the fold, so a canonical "section" means one contiguous stretch of one label. The rationale and the mapping live in `raveform_manifest.py`.
- **Politeness over throughput.** Downloads are strictly sequential with a pause between videos -- pulling 1,423 tracks in parallel is indistinguishable from abuse. Bot checks are never worked around: no cookies, credentials or IP tricks. A refusal is recorded, reported, and left as an owner decision, and a run of consecutive refusals aborts rather than burning the manifest into failure records.
- **Resumability is the contract.** The downloader is safe to kill at any point: finished tracks and failures are both persisted, an interrupted track is never recorded as failed, and a track counts as downloaded only when a non-empty mp3 is actually on disk. See the module docstring in `raveform_download.py`.
- **The container header is not evidence.** An mp3 truncated on a frame boundary -- the normal shape of an interrupted download -- decodes with no error at all (the stream just ends) while its Xing header still advertises the original full length, because that header is written at encode time and never revised. Neither "ffmpeg exited clean" nor "the header duration matches" detects it, together or apart. So the gate measures how much audio the decoder actually emitted and judges on that: it must agree with the header (or the file is truncated) and with the annotation (or it is the wrong recording). The measurement is free -- it comes from ffmpeg's own progress output during the decode pass the gate already runs.
- **Cleanliness is a gate, not a cleanup.** A track is admitted only if it decodes without a single error line, decodes to the length it claims, and that length matches the annotation record. A truncated file is `corrupt` (the bytes are damaged); a fully-decodable file of the wrong length is `duration_mismatch` (the wrong video was fetched, so every beat-to-label join would drift). Rejected tracks are recorded with their reason rather than deleted -- a wrong-length track is worth a human's eye. The tolerance is deliberately loose (an absolute floor for normal tracks, proportional for long DJ edits) because it exists to catch a different recording, not encoder padding.
- **The gate reads a moving corpus.** It is designed to be re-run while the downloader is still fetching: tracks not yet on disk are simply absent from the output, and a file whose mtime is younger than the min age is left for the next run because it may still be mid-write. Nothing in `audio/` is ever written, moved or deleted by the gate.
- **Acquisition is done when the arithmetic says so, not when the downloader exits.** The gate is silent about what it never saw, so a corpus that is quietly short looks exactly like a complete one. `raveform_validate.py` closes that hole by reconciling every manifest row against the disk *and* the download state, into exactly one of OK / DURATION_MISMATCH / CORRUPT / MISSING / UNAVAILABLE, and declaring convergence only when `OK + DURATION_MISMATCH + UNAVAILABLE == manifest rows` **and** `MISSING == CORRUPT == 0`. Both halves matter: the zeroes say nothing is in a state we refuse to accept, the sum says nothing fell out of the accounting. `MISSING` is the bucket that exists purely to make "we lost track of this" impossible to confuse with "this is unobtainable" -- the latter must always carry a recorded yt-dlp reason and its error tail.
- **Complete means complete against the annotations, not "1,423 files".** The five buckets only ever describe audio, so two failures cannot appear in them by construction: an orphan mp3 that no manifest row claims, and an annotation that does not reconcile with the manifest -- a missing or unparsable beat grid, a disagreeing YouTube id, a duration that differs by more than the manifest's millisecond rounding. A track whose beat grid never arrived is as untrainable as one whose audio never arrived, and its mp3 looks perfect. So the validator reports two verdicts: convergence (the audio accounting, unchanged) and an **all-clear** that additionally requires no orphans and no annotation issues. The exit code follows the all-clear, because a corpus that adds up but does not reconcile is not one we are done with.
- **What the disk says outranks what the log says.** `failed.jsonl` is append-only, so a track refused on one cycle and fetched on the next is recorded in both it and `downloaded.txt`. Any track with audio on disk is judged by decoding it; the failure log is consulted only for tracks that are not there.
- **Checksums are a baseline, not a verification.** YouTube publishes no canonical hashes, so there is nothing to check the corpus *against*. `checksums.sha256` (sha256sum format, OK files only) records what passed the decode check on the day it passed, so a later re-validation can prove the bytes have not drifted. That baseline plus the decode gate is the entire correctness mechanism this corpus can have.
- **Recovery is selective, not blanket.** A plain downloader re-run skips every recorded failure and a `--retry-failed` re-run re-polls genuinely dead videos forever. So the failure reasons split into *permanent* (the video is gone, private, age-gated or geo-blocked) and *transient* (YouTube refused this client). Only the transient ones — plus `other`, the unclassified bucket, which is far likelier to be something unnamed than a video that vanished silently — are worth re-attempting, and the script derives its own retry hints from that same set so its advice can never be narrower than the advice that works.
- **A permanent condition always outranks a transient one.** YouTube's permanent errors habitually carry transient-looking text: a private video ends with "Use --cookies-from-browser", an age gate opens with "Sign in to confirm your age". Every mis-classification found in this pipeline was a transient bucket sitting above a permanent one and swallowing a dead video, and the cost was never the wrong label alone — retry passes re-poll it forever and a run of them aborts a healthy run. The bucket table is ordered permanent-first for that reason, and tests use the wording yt-dlp really emits, since a strawman fixture is what let those bugs survive.
- **A 403 on the media URL is our problem, not YouTube's policy.** It means a signature/nsig challenge could not be solved, which is a property of the toolchain, not of the video. It gets its own reason bucket rather than folding into `bot_check` because the two demand opposite remedies, and reporting one as the other sends the owner after credentials they do not need. **The remedy that was actually needed is patience**: all 68 recorded 403 events resolved on re-attempt, which is why they sit in a retryable bucket rather than with the dead videos. The diagnosis, for the record: a JS runtime that is installed but not *visible to the running process* explains only 19 of them (a shell started before the install carries a stale PATH — "installed" must be checked from the runner). The other 43 happened while a runtime *was* visible, and yt-dlp names the cause itself in four of those records — its remote challenge-solver components are opt-in and were not enabled. That was never acted on because it never had to be. If a future refresh hits a 403 wall that survives repeated patient re-runs, *then* it becomes a decision; nothing is pre-built for it here.
- **Running it detached, and stopping it.** The long sweeps run as detached OS processes that outlive the session. Two things about that are not guessable and cost real data when guessed wrong. **Stopping requires `taskkill /PID <pid> /T`** — the venv's `python.exe` is a trampoline that re-execs the real interpreter, so killing the recorded PID alone leaves the actual downloader orphaned and still fetching, invisible to the next run's bookkeeping. **Re-issuing the detached launch command truncates `download.log`** — the redirect reopens the file, so a relaunch silently destroys the previous run's evidence. Redirect to a new filename per cycle (the supervisor does this: `download.cycle<N>.log`).
- **The corpus data directory holds *ops copies* of the scripts, and they drift.** Both `raveform_download.py` and `raveform_supervisor.py` are copied next to the corpus, and a supervised refresh runs **those copies, not the branch**. So whenever either changes, refresh both and confirm it: `cmp <data-dir>/raveform_download.py training/raveform/raveform_download.py` and the same for `raveform_supervisor.py` — they must be byte-identical to the branch blob. This is not hypothetical. The supervisor used to live *only* in the gitignored data directory, unversioned and unreviewable, and the ops downloader beside it carried the pre-`http_403` classifier while the supervisor hardcoded `--retry-reasons bot_check`; the next refresh would have stranded every recoverable 403 while the branch looked correct. Both files are now on the branch, and the supervisor reads `RETRYABLE_REASONS` out of the downloader sitting beside it rather than keeping its own copy — but the copy step itself is still manual, which is why the `cmp` is written down.
- **A beat is labelled by its own section, never by a merged run.** Merging adjacent same-label sections answers "how long is a musical section"; it must not answer "what is playing at time t". A merged run's *span* can swallow a dropped `end` sentinel sitting between two members, and that time is explicitly not the surrounding label's. So the beat-to-label join looks up the individual published section (folded and clamped) and uses the merged runs only to find where the labelled region of a track starts and stops.
- **Unlabelled audio is dropped and counted, never absorbed.** Most tracks begin with a stretch of audio before the first annotated section, some end with audio past the last one, and the `end` sentinel is a third such region. Attributing any of it to a neighbouring section would teach a section the annotator never marked -- and would look exactly like a classifier error downstream. Every beat is accounted for: kept, leading, interior gap, or trailing.
- **Two label spaces ship in the same table.** `label_canonical` is the seven-class corpus vocabulary; `label_v1` merges `cooldown` into `breakdown` and `altoutro` into `outro` for the five-class space the neural classifier trains on (a cooldown is defined by *where* it sits, which no single analysis window can see). Keeping both means changing the model's class space costs a re-join, not a re-simulation. `label_raw` is kept alongside so any mapping decision stays auditable from the table alone.
- **Simulate once, join many times.** The batch is two stages: an expensive parallel pass that simulates each track and caches its report, and a cheap serial pass that rebuilds the whole table from those cached reports. Fixing the join or adding a column is then seconds of work rather than hours of re-simulation, the join stays pure and unit-testable, and the reports themselves are the artefact the evaluation harness reads.
- **The report cache is keyed on what the report actually depends on.** A cached report is reused only if the pipeline that produced it (`lib/` + `simulate/`) and the exact audio file (size *and* mtime) are unchanged, and its mel sidecar is still on disk; anything else is a miss with a named reason, and `--force` misses everything. The key is deliberately *not* repo HEAD -- keying on HEAD would discard the whole corpus cache on every commit to a script or a document, which is what the cache exists to prevent -- but uncommitted edits under those paths do invalidate it, so a pipeline change under test never hides behind stale reports. A rebuild with nothing changed therefore costs seconds, and a rebuild after twenty new downloads costs twenty tracks. Two *different* uncommitted edits are two different keys — a constant "dirty" marker would have given a whole afternoon of edits one cache entry, which is the state a pipeline is changed in most often. The mel sidecar is checked for the exporter generation that wrote it and not merely for existence, because an exporter change that keeps the frame rate and the band count changes every number in the file; sidecars from before that stamp are grandfathered rather than re-simulated. **That sidecar condition is now unrepairable**: the exporter is deleted, so a track whose sidecar is absent or of the wrong generation misses on every rebuild and is re-simulated for a report it already has. Harmless while every corpus track has one; a trap the first time a new download does not.
- **The corpus stops short of the analyser's self-reset.** `MusicAnalyser` throws its rolling state away every 15 minutes, and the mel exporter has no such reset, so past that horizon a track's beats and its features describe the same audio from different states — and the join would produce wrong rows with no error and no counter. Tracks at or past it are dropped from the build with a line saying why. The corpus tops out 0.11 s under it, so this is a live edge rather than a hypothetical one.
- **The batch cleans up after itself, and only after itself.** The simulation now leaves *two* derived files beside each mp3 -- the decoded samples (~7.7x the mp3) and D12's extractor cell sidecar (~25 MB a track, ~35 GiB over the corpus) -- and neither buys this batch anything, because it simulates a track at most once. Each worker deletes both as soon as the features are out, in a `finally` so a failed track cannot leak one. Files that existed before the run started are left where they were -- tidying is scoped to what this run created.
- **Feature parity with the runtime used to be enforced by a golden test, and the thing it guarded is gone.** The mel exporter rebuilt the analyser's aubio objects rather than borrowing them, and a unit test fed both sides the same buffers and demanded identical energies. The filterbank is deleted, so the exporter is too: sidecars on disk are now a *record* of the corpus's mel grid rather than something reproducible, and `load_sidecar` refuses one whose recorded geometry disagrees. The live model's own train==deploy question moved to the encoder's resampler, where it is measured the same way (see D4 in the ML/DSP section).
- **Prerequisites.** `yt-dlp` and `ffmpeg` on PATH, plus a JavaScript runtime (Deno or Node) *visible to the running process*. Installed is not the same as visible: a shell or detached process started before the install inherits a stale environment block, so check from the actual runner, not from a fresh terminal. Downloads **can** fail with `HTTP Error 403: Forbidden` — not *will*: the whole 1,387-track corpus was fetched under exactly this setup, and every 403 that occurred was cleared by patient re-attempts.

### Label-aligned evaluation

`evaluate_against_labels.py` replaces the plumbing-only verdict in `simulate/evaluator.py` (which only asks "did anything happen") with musical ground truth. It reads the training table and nothing else -- the intent timelines were already realigned into song time when the table was built, so the evaluator never re-derives them from reports.

- **The intent alphabet and the label alphabet are different languages, so the confusion matrix is not square.** Rows are the intents the engine can commit, columns are the annotator's classes, cells are minutes of show. Flattening one alphabet into the other before looking at the matrix hides exactly the failures worth seeing. The mapping from intent to "the labels this intent is correct for" lives in a single dict at the top of the script and is the thing the owner iterates on. The alphabets are *closer* now than they were -- the show's five intents are four decoder classes plus PEAK -- but they are still not the same alphabet, and the matrix stays rectangular for that reason.
- **Two spaces, one primary.** Everything is reported in the five-class `label_v1` space (the model's target space, so its numbers are what a model gets compared against) and again in the seven-class canonical space as a diagnostic. They disagree in a specific place: `cooldown` is a canonical class that v1 merges into `breakdown`, so v1 flatters whichever intent is claiming that time. (Under the rule engine that intent was GROOVE, which no longer exists; the merge is now BREAKDOWN's.)
- **An intent cannot know where in the track it is.** ATMOSPHERIC describes quiet with no beat, which is `intro` at the start of a track and `outro` at the end -- the same sound, labelled by position. It is therefore scored correct against either, and when it is wrong its false positive is split across the classes it claimed so no single class absorbs the blame for an ambiguous prediction.
- **Both event streams are quantised the same way.** The table is per beat, so a label boundary is only observable as "the first beat carrying the new label" and an intent change as "the first beat carrying the new intent". Using one estimator on both sides makes the quantisation cancel when a change is correctly timed. The residual uncertainty is one beat period, which is why the strict tolerance tier sits at the resolution floor and means "within a beat", not "sample accurate".
- **Flicker is the product metric, and it is not boundary precision.** Flicker counts state changes with no real boundary anywhere near them, per audience-minute; boundary precision additionally punishes a correctly-placed decision made twice. An audience notices the first and not the second, so both are reported and they are not interchangeable.
- **"How often did it change" has two honest answers, and a model may only be scored against one of them.** The *intent stream* counts every committed `LightIntent` change - each one re-picks a lighting effect, so that is the show as the room experiences it and the owner's continuity number. The *class stream* maps into the label space first and then differences, so a DROP-to-PEAK move (different lights, same label class) is not counted. A model that predicts label classes emits a class stream by construction and physically cannot make the difference, so comparing it against intent-stream numbers would credit it for changes it is unable to make. Boundary-F1 and flicker are therefore reported for both streams everywhere, and the report says which is which.
- **A structural ceiling is a wall, not a target.** "Reachable classes over all classes" cannot be exceeded, but it also cannot be reached: the time sitting in classes the vocabulary cannot name is still predicted as *something*, so it lands as false positives on the classes that do exist. The evaluator reports the naive bound and, beside it, the best figure actually achievable - the optimum concentrates all of that damage on the single largest class rather than spreading it, because the objective is convex and its maximum over the allocation simplex is at a vertex. Spreading is the *worst* allocation, which is the trap an "equalise the marginal loss" reading falls straight into.
- **An operator action is not a classification claim, so it is excluded -- and counted where it fell.** A silence-triggered block (the quiet floor a detected stop puts up) is dropped from the scored intent stream: it asserts nothing about what the music was doing, and scoring it against a label would charge the classifier for a blackout it never proposed. The span it vacated is left as a *hole* rather than handed back to the block before it -- the preceding intent keeps its own end, and beats in the gap fall into the existing "no committed intent" path, which is the same refusal to absorb that the corpus applies to unlabelled audio. Because an excluded block is invisible to every score, it is counted leading / interior / trailing and those counters are compared exactly by the benchmark: a stop detected *inside* labelled time is either a real fade-out or the sidechain false-stop failure, and without the counter that failure would be a blackout the benchmark cannot see. An interior block is not by itself a fault -- a track that fades out under its last label would produce one honestly -- which is why the number is pinned per track rather than asserted to be zero. On the frozen set today every blackout is trailing and none is interior.
- **Excluding the blackout changes no score; recording one does, and in both directions.** Worth writing down because it was diagnosed the expensive way, twice. The quiet floor always sits past its track's last label and covers no beat, so dropping it from the scored stream is arithmetically a no-op. What moves a score is that the block *before* it stops being the **final** block: `realign_intents` ends a final block at the report duration de-shifted by its own delay, and a non-final one at the next block's start, so appending any block at all re-ends its predecessor. Being last had been cutting that block a full look-ahead early, leaving it with no opinion on seconds of labelled beats it had actually committed to. The stop is a real instant about real audio, so ending the block there is the truthful reading -- and it is truthful rather than flattering precisely because it goes both ways: on the one track where the extended intent was wrong the score fell, and on the three where it was right the score rose. A fix that only ever improved things would be the suspicious one. **A block's end is inherited from its neighbour rather than measured**, which is fine while every block is a classifier commit and is the first thing to suspect the moment they are not; making each block own its end is a real change to consider, and it would move every track, so it belongs to its own baseline rather than to a stop-behaviour commit.
- **A beat with no committed intent is not a class.** Those beats are excluded from every cell and counted instead -- scoring them as errors blames the classifier for the engine's start-up, scoring them as correct flatters it. Their position matters more than their count: an *interior* gap would let the change stream close over a silence and read two commits as one change, so leading / interior / trailing are counted separately as a tripwire.

**The baseline.** The live numbers are in `training/data/raveform/baseline_eval.json` and are deliberately not copied here - the corpus is still growing, so any figure written down goes stale silently. Re-run the script for current values.

**That corpus-wide file has not been re-cut against the neural show.** It describes the retired rule classifier, and re-cutting it costs a full corpus rebuild. The findings it produced are what motivated the replacement, and they are worth keeping as history rather than as a description of what runs today:

- Macro-F1 landed around a fifth of the way to 1.0 in the v1 space, with DROP the one class that worked, BREAKDOWN/GROOVE moderate, and BUILDUP near-random despite firing roughly the right *amount* of time - a placement failure, not a sensitivity one.
- **ATMOSPHERIC was never committed on mastered EDM at all**, putting ~22 % of labelled time permanently out of reach. That capped macro-F1 well below 1.0 before a single classification was made, and it was the single biggest lever available. **This is no longer true**: `intro` and `outro` are decoder classes, so ATMOSPHERIC is committed from evidence rather than from a beat-absence timer that mastered EDM never trips.
- The engine changed intent several times more often than the music changed section, and the large majority of those changes were nowhere near a real boundary. Continuity, not accuracy, was the furthest from acceptable.

The benchmark below **is** cut against the neural show, on ten tracks rather than hundreds, and it is the current reading. Where the two disagree, the benchmark is the newer measurement and the corpus file is the older one.

### The benchmark: the frozen eval set

The simulation used to be judged against one bundled track and a plumbing-only PASS verdict that asked "did anything happen at all". It is now judged against **ten expert-labelled Raveform tracks frozen in `training/eval_set.json`**, with `training/run_eval_set.py` as the gate and `training/eval_set_baseline.json` as the committed answer. The bundled Generate track has been retired; its historical measurements survive in the Stage-1 plan under `docs/superpowers/plans/`. Every test that needs *a real track* rather than *the benchmark* now reads one committed eval-set track through a single fixture, so there is one answer to "which audio does the suite read" instead of two.

- **A benchmark that only runs on one laptop is not a benchmark.** The ten tracks' audio and labels are committed — the audio under names derived from the YouTube id (`training/eval_audio/`, opaque so a directory listing says nothing; derived so code finds a file with no lookup table to go stale), the labels as a verbatim, sha-pinned slice of the corpus annotation. Owner-authorised, and precisely these ten: the rest of the corpus stays gitignored and machine-local. A machine that has the corpus too still reads the committed copies, so every machine benchmarks the same bytes. `training/eval_assets.py` owns the derivation and re-cuts both after a re-freeze.
- **A benchmark that follows the corpus is not a benchmark.** The set is frozen: the selector refuses to overwrite it without `--force`, and the baseline records the eval set's own checksum so a re-freeze fails loudly instead of silently re-scoring a different ten tracks. Re-cutting the set and re-cutting the baseline is one change, never two.
- **That checksum is over the file's bytes, so the checkout must not rewrite them.** Git on Windows defaults to `core.autocrlf=true` and materialises LF as CRLF, which changes the hash while changing nothing about the benchmark — the guard then passed in the worktree the file was written in and failed in every fresh clone, making its verdict a fact about the machine. `.gitattributes` pins the frozen artifacts to `eol=lf` so every checkout agrees with the writers, which already emit LF unconditionally. An older clone made before that pin keeps its CRLF copies until they are re-checked-out; rewriting them to LF is the remedy and leaves the index untouched.
- **The benchmark is never learned from.** Neither the ten ids nor any track sharing an artist with one of them enters a training or validation split. This is stated twice on purpose — once here as policy, once in the split builder as code.
- **Two gates, and they mean different things.** The *report checksum* says the pipeline's behaviour moved: a deterministic run over fixed audio can only change if the code did. The *label-aligned scores* say whether the show got better or worse. A deliberate improvement trips both, and that is the workflow — read the table, decide the change is wanted, re-cut the baseline in the same commit. A regression with no checksum change is impossible and would mean the determinism contract is broken. Beside the scores, the *count facts* each row records (beats detected, beats joined, label boundaries, seconds scored, intent changes committed, intent blocks that landed late, and blackout blocks by where they fell) are compared exactly — a score tolerance wide enough to be useful absorbs a run that measured a different number of things. The last three joined that list after the anchor re-cut moved lateness on four of ten tracks with nothing in the gate able to say so.
- **Crispness@0.5 s is the fifth gated metric, and it is a headline rather than a diagnostic.** Boundary-F1 at the tightest tolerance the scorer computes asks whether the change landed *on* the section change, not merely near it — and it is the axis the shipped decoder was **selected** on. The 2.0 s lens it sits beside hides most of the spread: the post-decoder dwell configs that were rejected score 0.68 at 2 s and 0.01 at 0.5 s. A benchmark that cannot see the axis a model was chosen on cannot defend that choice. It is gated from its first cut with no historical value to compare against — the rule engine's baseline has no such column and one cannot be reconstructed, because the demolition's schema change makes those reports unreproducible — so the aggregate is a starting line.
- **`late` is recorded and deliberately not gated.** On a track slow enough that the chain is older than the playback delay, a decision commits as soon as it can rather than on time. That is accepted lateness and a property of the music, so the benchmark shows it with its denominator (only a block that recorded its own instant can be measured at all) and does not stop a commit for it.
- **The baseline is a neural show's baseline now, and it was cut exactly once.** All ten checksums moved, which a demolition plus a rewire makes certain. Aggregate macro-F1 nearly tripled, boundary-F1@2 s roughly quadrupled, flicker fell about fourfold, and the show changes intent about half as often — a better-scoring show made of fewer decisions, which is the whole argument for the decoder. Every track improved on macro-F1, boundary-F1 and flicker. Two rows carry the qualification and both are the decoder's known shape rather than surprises: one track *loses accuracy* while gaining macro-F1 (two committed runs against the annotator's ten boundaries — a committed classifier does not collect the partial credit a twitchy one does, and its flicker is the best on the set for the same reason), and one scores **zero** crispness at a healthy boundary-F1@2 s, i.e. every hit near and none on. The count facts (beats, rows joined, label boundaries, exposure seconds) are **identical on all ten tracks** across the two baselines, so the comparison is like for like and no score difference is an artifact of measuring a different number of things. Three of the ten checksums equal the determinism-proof and pipeline-digest anchors cut on separate runs from a separate commit, so those artifact families corroborate each other rather than merely coexisting.
- **The ground truth is verified before anything is simulated.** A boundary that moves under a baseline cut before the move leaves every number comparable to nothing while the gate prints "matches". The committed label slice cannot move behind git's back, so what is checked of it is *provenance*: it records the checksum of the annotation file it was cut from, and that must be the one the eval set froze against. A machine falling back to the gitignored corpus annotation gets that file hashed on every run instead. Either way a mismatch is fatal. The manifest that chose *which* tracks are in the set is deliberately not checked: it grows with every download batch and feeds no score.
- **Scores are the corpus's scores, not the benchmark's own.** The runner reuses the training table's beat/label join (and therefore whatever that join does to reach song time -- today, reading each block's recorded instant) and the label-aligned evaluator's metric functions. A benchmark that computed its own numbers would eventually disagree with the corpus evaluation and nobody would know which was right.
- **The integration suite runs a subset, a human runs the set.** Three tracks fit a test-suite wall-time budget; ten do not. A subset run compares only its own tracks and deliberately does not compare the aggregate — an aggregate over three tracks is a different quantity. The full set is a manual command, and its cost now depends on the cell cache rather than on core count alone: the cut that produced the current baseline ran all ten **cold** (every sidecar had missed on a backend-key change) in about sixteen minutes of wall for 68 minutes of audio, at `--workers 1`. That worker count was a GPU constraint, not a determinism one — one simulation process reserves several GB against an 8 GB card — and it is free because parallel and serial produce identical bytes, which is the runner's own contract and is checked by running both. Warm, the encoder does not run at all and the pass is pure CPU.
- **A subset may not overwrite the committed baseline, and the baseline is itself under test.** The two ways this tripwire could be disarmed without anything failing are a baseline cut from a partial run (the gate then compares the tracks it ran against the tracks in the file, so the rest silently stop being checked) and a guarded metric missing from the file (skipped rather than flagged). So a partial `--write-baseline` at the committed path is refused outright — `--allow-partial-baseline` is the deliberate override, an explicit `--baseline PATH` is the experiment — a missing metric is a failure rather than a skip, and a fast unit test reads the committed file and asserts it still covers the whole frozen set, was cut against the current one, and carries every gated number.
- **The madmom migration is the benchmark's first real customer, and the first labelled read of it.** That branch measured what the new beat source did to the beat stream and to the show, and said explicitly that none of the intent-timeline movement was scored against labels — the labelled evaluation lived here. Stacking the two answered it: all ten checksums moved, aggregate macro-F1, accuracy and flicker all improved, boundary-F1 was flat, and the per-track spread is wide in both directions. The committed baseline is the record; the per-track table is in PR #8's merge comment. What it does *not* establish is that the new beat source is better *because* of these numbers — the show changed for many reasons at once, and nothing was retuned.
- **The eval set is exempt from the delete-your-derived-files rule, and there are two of them now.** Everything else in the corpus discards the decoded `.npy` and the extractor cell sidecar beside its mp3 because the corpus is thousands of tracks; the eval set is ten and is re-simulated on every test run, so both persist. They land beside the committed mp3s, where the repository's `*.npy` and `*.mertcells.npz` rules keep them out of git. That is roughly a gigabyte of decode cache plus ~250 MB of cells, and it removes both the decode *and* the GPU from every run after the first — which is most of why the integration suite got about 2.5x faster when the cell cache landed.

### Neural section classifier -- dataset, model, decoder and the offline verdict (`training/nn/`)

The design spec is `docs/superpowers/specs/2026-07-26-nn-section-classifier-design.md`: a CRNN acoustic model over the pipeline's own mel stream, plus a fixed-lag Viterbi decoder that owns stability and latency policy. This package is the offline half. Only code lives in git; every artefact it produces (splits, checkpoints, ONNX graphs, posterior sidecars) lands in the gitignored data directory.

**Read this section as two layers with different lifetimes.** The *decoder* is the live one -- `lib/engine/section_decoder.py` imports `FixedLagViterbi` and `Priors` from here rather than copying them, so every bullet below about the decoder, the priors and the sweep describes what runs tonight. The *acoustic model* described here is the *mel* generation (v1/v2), and it is **not what ships**: the deployed chain is a later generation over MERT features, and its training code lives in a separate phase-B worktree that was never merged into this repository. What is in this tree from that generation is the shipped artifacts (in the gitignored corpus directory), the decoder generation that reads them, and the runtime that runs them. Everything the mel bullets say about dataset construction, masking, augmentation, reproducibility, calibration and export is still the project's stated method; it is simply describing the generation before the one on stage.

- **Splits are a pure function of the track id, so the corpus can keep growing.** The download is still running and the eventual 1,423-track retrain must be comparable with tonight's model, so the 70/15/15 assignment hashes `(seed, youtube_id)` rather than shuffling a list. Adding tracks may only ever *add*; nothing already placed can move. `splits.json` is the frozen record and wins over recomputation -- the hash decides only where new ids land.
- **The benchmark is excluded twice over.** The ten frozen eval-set ids never enter any split, and neither does any track sharing an *artist* with one of them. The second guard is the one that is easy to miss: producers have a sound, so a net that has heard six other tracks by an eval-set artist has partly memorised the benchmark. Artist matching is collaboration-aware -- a credit is split on `Feat.` / `&` / `vs` and any shared participant excludes, so a solo release by half of a featured pair is caught. It deliberately over-excludes (a band name containing `&` splits into two names); a handful of lost training tracks is cheap, a contaminated benchmark is not.
- **Unannotated audio is masked, never labelled.** The same rule the training table applies to beats applies here to frames: audio before the first published section, audio past the last section end, and the time of a dropped `end` sentinel carry no loss. A masked frame teaches nothing; a mislabelled one teaches the wrong thing.
- **The boundary head gets a fourth mask that the label head does not need.** Where two published sections fold to the same `label_v1` class, the join is a statement about section identity, not necessarily an audible event -- so its neighbourhood is *deleted* from the boundary loss rather than taught as a negative. A genuine transition that happens to sit inside a deleted neighbourhood keeps its target: deletion is for ambiguity, not for erasing known events. On the current corpus roughly three quarters of tracks have at least one such join, so this is a live path, not a corner case.
- **The window geometry is derived from the sidecars, not restated.** Frame rate, band count and pooling factor all come from the constants the sidecars were written with, and every sidecar load re-checks its own recorded frame rate against them -- a change to the mel front-end must break loudly rather than silently train the model on a different grid than the runtime produces.
- **Augmentation moves the window, not the truth.** Training draws a fresh window offset and a gain shift per item per epoch, seeded from `(seed, epoch, index)` so a run is reproducible and dataloader workers cannot collide; offsets stay aligned to the label-pooling factor so pooled targets are sliced rather than re-derived. Gain is applied as the additive shift in the log domain that it is, and clamped so the model never sees an input the encoder could not have produced.
- **Torch was an optional extra and no longer is.** It was, for as long as the model was offline: `lib/` and `simulate/` gained no new imports and the dataset's target and mask logic is plain numpy, so it stays testable on a checkout that never installs torch. The runtime integration ended that -- `torch`, `transformers` and `onnxruntime` are base dependencies because they run the show, and `training` is now the smaller group holding only what the *offline* pipeline needs (`onnx`, `tensorboard`). The torch pin has not moved and its reason has not changed: the legacy TorchScript ONNX exporter is the only path that keeps a dynamic time axis for a GRU, and it is deprecated from a version this project has not taken. Two pins for one package would be two answers to which one ships, so the base pin is the only one.
- **The convolutions pool frequency and never time.** The boundary head has to name the frame a section change lands on, to the tolerance the annotations carry; a time-strided front-end throws that away before the recurrence sees it. "Which bands are loud" survives pooling, "exactly when" does not. The recurrence is bidirectional for the reason the whole design exists: the window carries look-ahead the audience has not heard, and telling a buildup from a fake-out means hearing whether the drop actually lands.
- **Three loss terms, one failure each.** Class-weighted focal loss stops the majority class from owning the ambiguous middle *and* stops the easy interior of a long section from drowning out the frames either side of a transition. Boundary BCE takes its positive weight from the corpus's own sparsity rather than a guess. A total-variation penalty on the label posteriors buys the smoothness that flicker -- the metric the runtime is being replaced over -- is made of. That third term is *boundary aware*: a blanket smoothness penalty would teach the model to smear the one step the decoder needs, so it is weighted down where the annotation says a transition is happening and abstains entirely where the boundary target was deleted.
- **Calibration is a metric, not a side effect.** The decoder consumes posteriors, divides by class priors and multiplies along a path, so it reads every column of the softmax. A model that is confidently wrong is worse to it than one that is honestly unsure, and per-class ECE (one-vs-rest, not just on the argmax) is logged every epoch beside macro-F1 as a release gate.
- **Training is bitwise reproducible, and that is enforced rather than hoped for.** Two runs of the same config in fresh processes produce identical weight hashes and identical loss curves. Two ops would silently break it -- `gather` and boolean indexing both route through a nondeterministic `scatter_add` on the backward pass -- so neither appears in the loss path; masking multiplies by a float mask instead. `--resume` restores optimiser, scheduler and RNG state and the per-epoch shuffle is re-seeded from `(seed, epoch)`, so an interrupted run rejoins the *same* trajectory rather than a plausible one. There is a deliberate `--crash-after-epoch` drill for proving exactly that.
- **The schedule is derived from the dataset, never written down.** Steps per epoch and the cosine tail come from `len(dataset)`; the corpus is still growing, and a schedule computed against a stale window count either decays to zero mid-run or never arrives.
- **DataLoader workers are never persistent, and that is a correctness choice.** A persistent worker holds a *pickled copy* of the dataset, so `set_epoch` in the parent never reaches it and the augmentation freezes on whichever epoch the workers spawned at -- every later epoch re-draws identical offsets and gains, which looks exactly like a working run. Respawning per epoch re-pickles the dataset with the new epoch, so correctness costs the documented ~2.2 s Windows spawn per epoch. Measured, not assumed, and pinned by a test. `--num-workers 0` remains the default and is genuinely faster at this corpus size, so the cost is hypothetical.
- **A checkpoint describes its own geometry.** Every save carries the constructor arguments that decide tensor shapes, and resume refuses a mismatch. A `state_dict` alone is not self-describing: change the label pooling factor and the weights load *cleanly* into a model that decodes at the wrong frame rate. `models/v1/<run>/best.pt` is the artifact the whole plan is judged on and it outlives the CLI defaults that produced it. It is also a pickle, so the export path reads it under torch's restricted unpickler — a published checkpoint is a file that gets shared, and nothing the exporter needs from one is worth the right to execute it. Only resume relaxes that, for a file this trainer wrote minutes earlier into the gitignored corpus.
- **Hyperparameter provenance is recorded, not remembered.** The learning rate was specified against one batch size and runs at another; rather than silently "fixing" it, every run config and report carries a note saying so and naming the triage order if the run underperforms. The next person to read a training report is not the person who chose the number.
- **TensorBoard is part of the run, not an afterthought.** The trainer writes per-step losses, per-epoch val metrics, per-class F1 and ECE and a confusion image, and starts a detached server itself if one is not already listening -- the owner watches training live at `http://localhost:6006`.
- **The ONNX graph is the interface, and the exporter that produces it is deliberately the deprecated one.** Torch's new dynamo exporter handles this architecture *without raising* and silently freezes the time axis at whatever length it traced, so the graph then only runs at that length; the legacy TorchScript exporter is the only path that keeps the axis symbolic today, which is why torch is pinned. Because the failure is silent, the exporter reads the declared dimensions back out of the written file and refuses to publish a graph that lost them. The model is rebuilt from the checkpoint's own geometry block and checked back against it before any weight is loaded -- a pooling-factor mismatch changes no tensor shape, so it would load cleanly and decode at the wrong frame rate.
- **Every inference goes through one session definition, and it is single-threaded.** That is a determinism contract rather than a performance choice: a threaded reduction sums in whatever order the pool finishes in, and float addition is not associative. Throughput comes from running *tracks* in parallel over that same pinned session, so no track's numbers depend on how many workers ran -- which is a test, not a claim.
- **The offline pass is the runtime's own sliding window, not a whole-track forward pass.** The model is bidirectional over a fixed window, so a frame's posterior depends on where in the window it was read; a single long pass would be ~150x cheaper and would not be the same model. The window slides at the runtime's callback cadence (snapped to the pooled label grid, since an unaligned hop would land the label head's output between cells of the track-wide grid) and every posterior covering a frame is averaged, which is the standard fix for window flicker.
- **Never read the window's edge -- except where nothing else can.** The outer margin of a window has context on one side only and a cold recurrent state, so it is dropped before aggregation. The head and tail of a *track* are reachable by no window's interior, and there the first and last window donate their margin rather than leave the decoder an undefined posterior. That is arithmetic, not a policy exception, and each sidecar records how many windows voted on each frame so the thin ends are visible instead of looking like the confident middle.
- **The last window re-overlaps its predecessor on purpose, so aggregation averages and never concatenates.** A track length is not a whole number of hops; the final window is clamped back to the end of the track (the same rule the dataset uses in eval mode, so the tail -- almost all outro -- is not silently unreachable). Concatenating window outputs would double-count that tail and shift every frame after it.
- **The boundary score is stored raw, and it is not a probability.** Per-window normalisation was measured to destroy most of its ranking power, so nothing rescales it; and it was trained under a large positive weight against a ~2 % base rate, so the decoder must consume it as evidence, never as P(boundary).
- **A sidecar is keyed on the model *and* the window geometry, and its bytes are a pure function of its contents.** Reusing a cached posterior file after a hop or edge change would silently mix two geometries in one evaluation, so both are part of the cache key. The archive is written with a fixed member order, a fixed timestamp and no compression rather than through `np.savez`, whose reproducibility is an implementation default and whose compressed form folds the zlib build into the bytes -- a determinism claim resting on either is a claim about one machine. Both time bases are recorded in the file (the frame-rate boundary grid and the pooled label grid, which is stamped at the end of each group) so no consumer has to infer one from the other.
- **The comparison runs through the baseline's own scoring functions, with exactly two things changed.** A verdict measured by a second implementation of the metrics is a verdict about the two implementations. So the accumulators, the event matcher, the flicker rule, the time weighting and every derived property are imported from `evaluate_against_labels`, and only two things vary: the *claim map* (which label a prediction asserts) and the *sentinel* for "no prediction here". Both sides then aggregate with the same aggregator, and a divergence in arithmetic is structurally impossible rather than merely unlikely.
- **Undecoded is not the same as wrong, and saying so is not a courtesy to the model.** The decoder runs on a bar grid, so it says nothing before the first downbeat or past the final bar line. Those regions belong almost entirely to `intro` and `outro`, so scoring them as misdecodes would be a systematic charge against two specific classes for where the annotation grid starts. They reuse the evaluator's existing "no committed prediction" sentinel, which already excludes and reports rather than penalises. On the current corpus this is a near no-op -- the beat grid starts on a downbeat -- which is worth knowing precisely because it means the handling cannot be flattering the result.
- **A model that can name the class is held to a stricter standard than the vocabulary it replaces, and both readings are published.** ATMOSPHERIC is credited against `intro` OR `outro` because an intent cannot know its position in the arrangement; a model predicting in the label space can, so its primary score forgives nothing. That is a genuinely higher bar, so the same decode is also scored under the rule's exact ambiguity, and a third number restricts the macro-F1 to the classes both sides can contest. One table, three readings, so "the model only wins because the two sides were scored differently" is a question the artifact answers instead of inviting.
- **Two of the five classes were not contested at all in that verdict, and that was a property of the harness rather than of the classifier.** ATMOSPHERIC -- the only intent covering `intro` and `outro` -- fired from a beat-ABSENCE timer, while the training table carries one row per DETECTED BEAT. The rows that exist are exactly the rows where that trigger cannot be active, so the baseline's zero on those two classes is guaranteed by construction, not measured. They carry roughly two thirds of the headline delta, so the restricted macro-F1 over the contested classes is the model-vs-model number and the one any per-track claim must be read from -- the full five-class reading cannot go negative per track and must never be quoted as evidence that a win is universal. The full reading is still primary and still fair as a description of what the *deployed system* does at beats; it is simply not a like-for-like model comparison.
- **The class stream is the only fair comparand, and for this model it is the only stream.** A label-space model cannot express DROP -> PEAK, so quoting the engine's intent-stream flicker against it would overstate the model. Under identity claims the model's two streams are provably the same stream, which the report asserts rather than assumes -- printing one number twice would look like corroboration.
- **Selection is constrained, not maximised.** The chosen config is the best val macro-F1 *subject to* flickering no more than the shipping classifier and committing inside the look-ahead budget. A model that is right more often while twitching more is not an improvement, and one that needs more future audio than the show has is not runnable. Longer lags are still measured -- the accuracy-versus-lag curve is the evidence for whether the budget should ever move -- but cannot be selected. When nothing clears the constraint the sweep raises rather than quietly returning the best ineligible config.
- **The sweep is joint where the axes interact, and deterministic everywhere.** Prior strength and drop-miss cost both push mass toward the rare expensive classes, and the boundary gain is meaningless without the neutral point it is measured from; a line search over either pair finds a compromise neither axis would pick. Those are full grids, the remaining axes are staged around the running winner, and a joint refinement then re-opens everything at once to catch what a staged search walks past. Nothing samples: every axis is an explicit tuple and the enumeration is a fixed-order product, so a winner is reproducible without replaying the search. It is cheap because bar observations depend only on the two knobs the sweep holds fixed -- so sidecars are read once and a config costs a decode plus a score -- and the cache is keyed on that pair rather than assuming it.
- **Sensitivity is measured by ablating the shipped config, not by reading a curve off the search.** The best result ever seen at a given knob value conflates that knob with whatever the rest of the config happened to be in the stage that produced it, so it cannot say what the knob cost. A final pass moves one axis at a time around the chosen config, reusing configs the search already measured so the anchor appears in every curve. That pass is also a search -- if it finds something better the selection takes it, and the artifact records whether the anchor survived.

**v1 exists, it won, and a successor of it is now driving the show.** v1 was scored once against the held-out test split -- tracks no selection decision had ever seen -- and beat the shipping rule classifier on every metric the plan named, in both the all-classes and the contested-core reading, while committing several times fewer state changes. The verdict artifacts are `models/v1/eval_val.json` (the tuned reading) and `models/v1/eval_test.json` (the selection-clean one); each carries the sha256 of the model, priors, splits and table that produced it, so any figure traces to a chain rather than to a memory. The figures themselves are deliberately not copied into documentation -- the corpus is still growing and a written-down number goes stale in silence. `training/nn/CLAUDE.md` maps the package; the bullets above are the reasoning behind it. What runs live is the phase-B student over MERT features rather than v1 itself, and that model's own offline verdict was taken in the worktree that trained it, not here.

- **The rule-classifier column of the v1/v2 verdict is a fact about the aubio beat stream, and it has not been re-measured.** Those verdicts were scored before the madmom migration landed under this branch. The model side is unaffected — it trains and infers on the mel stream, which the migration held byte-identical, on a fixed frame grid that has nothing to do with beats. The *baseline* side is not: the training table carries one row per detected beat, and the beat stream is now a different stream. Re-scoring it would be a second read of the test split, which the rule above forbids for exactly the reason it exists, so the number stands as dated rather than being quietly refreshed. The eval set is where the rule classifier's post-madmom behaviour *is* measured, and there it got better on three of four metrics — so the recorded margin is more likely an over-statement of today's gap than an under-statement. That direction is the safe one for a claim of "the model wins", but it is a reason to distrust the size of the win, not the sign. **That column is now permanently unrepeatable**: the rule classifier and every feature it read have been deleted, so nothing can re-measure it and the recorded figure is a historical record rather than a comparand. The live comparison that replaces it is the eval-set baseline, where the same ten tracks were scored under the rule engine and then under the neural show.
- **The test split is read once, and that run is the record.** Everything else in this package was chosen on val -- the decoder config, the early-stopping epoch, and which of several training runs to export -- so the test figure is the only number no decision was permitted to see, and the acoustic-layer selection noise alone is comparable to most of what the decoder sweep was tuning. Tuning after reading it spends the one clean measurement the project has. A disappointing test result is therefore a *new versioned model*, never a re-tuned old one, and that model gets its own single read.
- **Stability is not accuracy, and the decoder buys the first with the second.** Committing few, long, confident runs is precisely what produces the large flicker win; on a track whose structure alternates faster than the fitted duration prior expects, the same property lets one run swallow several real sections. The worst tracks in *both* splits share that shape -- a couple of committed runs against an annotator's many -- and it is the dominant reason a track can lose to the rule classifier on the contested core. A twitchy classifier collects partial credit for passing through the right state; a committed one does not. This is priced rather than accidental: the decoder can trade the over-commitment back and pays macro-F1 for it, so removing the cost without paying elsewhere is acoustic-model work, not decoder work.
- **Held-out means held out from selection, not necessarily from the music.** Splits are assigned per track id, so a remix and its original can land on opposite sides -- the artist guard protects the frozen benchmark, and nothing yet groups a corpus track with its own alternate versions. The measured effect on the verdict is negligible (it is a couple of tracks, and the result is unchanged with them removed), but the v2 split should group by song rather than by track id.

### The runtime integration: what the three blockers turned into

The three things this section used to name as gating a live show were resolved,
worked around with a priced fallback, or left standing and said so. In the same
order:

1. **Live downbeat tracking still does not exist, and the show ships without
   it.** The decoder is bar-rate; offline it was handed an expert-annotated grid.
   Live, bars are four beats counted off the beat stream, starting one beat in.
   That fallback was chosen against measurement, not convenience: a
   boundary-logit phase vote was built and **lost to it on every one of 120
   configurations** -- then lost again, monotonically in the weight given to the
   head, on a structurally different continuous-filter design. The deeper
   finding is that the production beat stream does not hold phase at all -- it
   slips a median of twice per track, and an *oracle* frozen phase covers only
   about two thirds of a track. So this is a phase-**tracking** problem, not a
   one-shot phase **decision**, which is a materially larger piece of work than
   the earlier reading suggested. The price is written down (-0.1396
   crispness@0.5 s at the old anchor, +0.0510 of it recovered by the anchor
   fix, the rest all placement) and re-quoting the older -0.0377 figure is a
   mistake: that one was measured on annotated beats, and its caveat was hiding
   most of the cost. 57 of 215 val tracks slip zero times.

   **Slip repair is banked with a measured ceiling and a measured blocker.**
   Perfect live phase would recover the whole remaining gap and no more, so
   about +0.09 crispness is sitting there. Every arm that actually repairs slips
   buys some of it and pays contested macro, because aggressive repair makes bar
   *lengths* irregular and irregular bars fail the decoder's coverage and
   duration expectations; the best arm fires 8,985 false slip detections against
   812 true ones. A precision-tuned version is the obvious next experiment, and
   a third of slips carry no interval evidence at all, so closing the gap needs
   an audio-side cue that is not the boundary head.

   **The offline machinery from the downbeat branch is still in `training/nn/`,
   and it is parked training work rather than a deployment prerequisite.** A
   downbeat head, a bar-phase decoder over a candidate grid, and an evaluation
   harness that scores a predicted grid against the annotator's all live there;
   nothing in `lib/` imports any of it, and the show counts bars instead
   (rulings #157/#158). **That chain's v1 scoring was removed rather than
   carried as a dated record**: it bound to the aubio beat stream madmom has
   since replaced, so its figures and its BLOCKED-at-0.85 verdict are superseded
   (owner decisions #81/#133). `training/nn/CLAUDE.md` carries the removal note;
   the successor measurement is committed in `docs/migration-evidence.md` --
   gate-faithful downbeat F1 0.50 on val against 0.71 on the annotator's own
   grid, the 0.85 gate retired as sitting above published *offline* SOTA on
   general music, and F1 >= 0.55 at a median of two phase flips per track or
   fewer recommended in its place. The expert-grid bound is the figure that
   never depended on the beat source: it is what the head and the decoder are
   worth on a clean grid, and a better beat source moves the live number towards
   it without moving it.
2. **The show's look-ahead grew to the decoder's budget**, and the relation
   between the two systems inverted with it -- `PLAYBACK_DELAY_SEC` is 14.0 s and
   the queue now holds a command for `playback_delay - chain_latency` rather than
   for the whole delay. See "The delay model" above. This closed as a plumbing
   and configuration question, which is what the lag sweep was for.
3. **v1 trained on the subset that had finished downloading**, and that is
   unchanged. The full-corpus retrain is still future work, and the split
   assignment was built to make generations comparable -- new tracks may only be
   *added* to a split, never moved between them.

**What the integration establishes, and what it does not.** It establishes that
the chain runs live inside its budget on this box, that simulation and production
are one code path, that the show is byte-deterministic cold or warm, and that on
the ten-track benchmark the neural show scores substantially better than the rule
engine on every gated metric. It does **not** establish that the offline
verdict's margin transfers: those verdicts were scored on an expert bar grid, and
the live grid costs real crispness. Nor does it establish anything about a second
machine -- every runtime number on this branch is one rig, and the realtime
figures are anchored against the same day's madmom (that component alone measures
2.5x apart between days on identical code, so the committed
`realtime_measurement.json` and a fresh run must not be subtracted).

**Two live measurements are worth carrying, because they are counter-intuitive.**
A 35-minute soak sustained 1x with zero backlog, zero sheds and a peak drift well
inside the watchdog's door -- but its song-boundary result was **vacuous** and
must not be quoted: the machine's default render endpoint was the cable, so the
input was never silent, no sound-stop ever fired, and the three "boundaries" it
counted were the analyser's 15-minute self-reset. Separately, **watching the show
costs the show**: `--ui` serves Dash from the same process as the pipeline, so a
single attached viewer measurably increased sheds (1 -> 5 over one track), and
two stale viewers degraded it enough to reset the decoder repeatedly, and that show never left one intent. A `--ui`
session is therefore not a clean read of what the headless pipeline does.

### Follow-ups the runtime integration deliberately did not do

1. **Sweep the decoder's duration-floor ladder below its current point.**
   Crispness is monotone in it across the whole measured range and the range
   simply stops there. Val-only, zero GPU.
2. **Export MERT to ONNX** and drop `torch`/`transformers` from the live path.
   Would shrink the install substantially and remove the last training-shaped
   dependency from the show; unmeasured, so it ships after the model does.
3. **Continuous bar tracking** (the re-scoped successor to a downbeat head). This
   is the largest single lever on the show's crispness and it is training work.
4. **The look-ahead's *modelling* consequences.** Whether a different model
   geometry would buy a shorter chain is a training question and stays parked.
5. **Corpus report-cache regeneration.** The report schema lost four columns, so
   `pipeline_sha` invalidates the whole corpus cache by construction. Expected,
   and run once after merge.
6. **The corpus-wide `baseline_eval.json` re-cut against the neural show.** It
   still describes the rule engine.

**Data location.** Everything lands in `training/data/raveform/` -- annotations, `manifest.csv`, `audio/`, the download state files, and the build outputs (`reports/`, `features/`, `training_table.csv.gz`, `splits.json`, `models/` and `posteriors/`) -- and is gitignored: the audio is ~13 GiB and is never committed. The committed `.gitignore` covers `training/data/`.

**This directory is now a production dependency, not just a research one.** The shipped encoder, the student's graph and the priors live under `models/`, and the show reads them at startup. `training/corpus_root.py` is the single stdlib-only answer to "where is it" -- `$RAVEFORM_DATA_DIR`, else the repo's copy, else the main worktree's -- and `lib/section_chain.py` asks it rather than reaching through the benchmark harness. That indirection is not cosmetic: resolving the path through `run_eval_set` pulled the table builder, the label evaluator, the acquisition scripts and a git subprocess into production startup, and any failure anywhere in that chain read as "this machine has no model", so the show quietly ran the degradation state on a box that had one.

---

## Known Issues / Gotchas

- **Hardcoded overlay IP** â€” must change per venue in `overlay_client.py`.
- **MusicAnalyser rolls its state** every 15 min to stop the rolling windows growing without bound. It is a statement about memory and says nothing about the music — and that distinction is load-bearing, because a continuous venue feed never trips the 0.3 s silence gate, so this horizon is the *only* boundary the engine would otherwise see. It used to go through the same reset that clears `is_playing`, which made the next buffer of uninterrupted audio announce a new song: the extractor ring, the student, the bar grid, the queued intents and the cold-start floor all restarted, four times an hour, for a track that never changed. The roll now keeps the show's own state; only the rolling windows are rebuilt. **It does still reset the beat tracker, and after the anchor fix that is no longer free.** The roll re-locks madmom's DBN mid-audio while the bar grid keeps counting and nobody tells it, so any beat the re-lock gains or loses rotates the grid permanently. Probed at five instants in one clip: the next beat lands 0.13-0.26 s late every time (less than a beat period -- this is *not* the whole-beat warm-up a cold start pays), and on two of the five the beat count moved (+2, +1). So it is an intermittent rotation, not a guaranteed one. It was invisible before the anchor fix, because the grid was rotated from beat one anyway, and it is invisible to every gate we have: no eval-set or corpus track reaches the horizon and a live set crosses it four times an hour. The candidate fix is to stop resetting the rhythm tracker on a roll at all -- it holds no unbounded state, and re-locking tempo four times an hour was never the point of the horizon -- but that is a change to the beat source and was deliberately out of scope of the anchor change.
- **10 ms delays** between MIDI commands give SoundSwitch hardware time to settle.
- **Os2lSender** runs in a separate thread, and so does the GPU stage; the audio/DSP path is async on the main thread â€” mixing threading models requires care when touching shared state. The design answer is that show state is *not* shared: the consumer of the GPU stage's output is the audio loop, where the command queue, MIDI client and event buffer already live.
- **`PLAYBACK_DELAY_SEC` (14.0 s) must match `playback_delay_seconds` in dmx-enttec-node.** It is defined in `lib/main.py` and mirrored in `simulate/runner.py`, and the engine logs the reconciliation line at startup and every ten seconds: chain latency, its two halves, the derived queue delay, and a reminder to check the other system. Changing `lag_bars` changes this number.
- **Slow tracks commit late, and that is accepted.** The decoder's share of the chain is proportional to bar length, so a track slow enough that the chain exceeds the playback delay commits as soon as it can. The engine logs it once per transition (not per bar) and the benchmark records it per track. It is not gated: it is a property of the music, not a regression.
- **The show's model artifacts are 1.3 GB and are not in git.** The encoder, the student's graph and the priors live in the gitignored corpus directory. `lib/section_chain.artifacts_present()` is the one-line question; without them the app logs a warning at startup and runs the degradation state, and the `nn_artifacts` test fixture skips. The benchmark treats the model as a third input beside the audio and the labels: `run_eval_set.missing_model` refuses before anything is simulated, because a degraded run moves every checksum and zeroes every score, and re-cutting a baseline from *that* would write ten tracks of dark show over the benchmark.
- **`torch`, `transformers` and `onnxruntime` are on the LIVE path**, not in an offline extra, so a base install now pulls the CUDA wheels. `tensorflow`, `tensorflow-hub` and `aubio` left the tree entirely, along with the `tensorflow-io-gcs-filesystem` Windows override and the aubio build variable; a test asserts the live path imports neither of the two removed frameworks.
- **The GPU stage degrades by holding, and there is no second classifier.** `NN_SHED` means: stop consuming posteriors, hold the current intent, keep beats and the silence timer, log loudly on a rate limit, attempt reinit on a backoff that tops out at one attempt per half minute, resume on success. Three of the four named GPU failure modes (a raised CUDA fault, an out-of-memory, a dead context) are the same mechanism reached by different exceptions and are deliberately not told apart — a policy that branched on the message text would be a policy about strings. The fourth, a hung pass, raises nothing at all and is caught by a timeout from the audio thread.
- **A shed keeps feeding the ring**, which looks like waste and is the opposite: the extractor's sample index *is* song time and is what every cell is stamped from, so a stage that stopped taking audio would come back with a clock that disagrees with the beat grid, silently, for the rest of the song. What a shed stops is the encoder pass, not the microseconds of resampling. Both edges of a gap clear state — entering drops the hand-off queue and resets the decoder, leaving resyncs past the gap and starts the student cold — because everything they hold describes audio from before it.
- **VRAM pressure fails silently on Windows.** The WDDM driver spills to host memory under pressure and raises no OOM at all, so a run that has started crawling reads as a healthy one. The gate is therefore the *plateau* in reserved bytes, not the absence of an error: measured on this box it climbs while the 30 s ring fills and then sits flat.
- **Backpressure is monitored, not assumed**: live audio arrives at exactly 1x and the input side DROPS rather than queues, so falling behind costs audio, not latency. The ladder is now one rung with two inputs — drift (which the audio loop measures) and stage health (which only the stage can report, because a CUDA fault costs the loop's pacing nothing). Either input alone holds the door shut. If a log shows sustained shedding, the box is too slow or the GPU is unwell — that is the signal, not a nuisance warning. **It is now also on screen**, because a log nobody is reading is not a signal (see Visualizer smoothness).
- **Watching the show costs the show.** `--ui` serves Dash from the same process as the pipeline, so a viewer's callbacks and the pipeline contend for one GIL. One ordinary viewer measurably increased sheds over a single track (1 -> 5 against a control run); two stale viewers degraded it enough to reset the decoder repeatedly, and that show never left one intent for the whole track. A `--ui` session is not a clean read of headless behaviour, and its checksums are not comparable with a fast-sim baseline either.
- **A soak or live run through a virtual audio cable needs the rig checked first.** A run whose default render endpoint is the cable never sees silence, so the silence gate never trips and no song boundary fires — a "song boundaries exercised" result under that configuration is vacuous. Measured with the machine actually quiet, the cable's idle RMS sits an order of magnitude *below* the gate, so the blocker is configuration (a media player left running was the real cause) rather than routing. The cable is also **not transparent**: measured round trip is about −3.6 dB with a tilted magnitude response and r ≈ 0.94, so the live pipeline analyses audibly different audio from the simulation. Sim/live show agreement is agreement *despite* that channel.
- **`beats_cut` still banks at detection, not at the gate.** `set_playing(False)` counts every beat still travelling as never-heard the instant silence is detected, and the persistence gate deliberately did not move it — the recorder is the one thing the gate is not allowed to touch. When a gap cancels the bypass those beats *do* reach the room, so the displayed beat count under-reports by up to a look-ahead's worth of beats for every song change. Display only: nothing in the report, the training table or the digest reads it. Fixing it means either passing the window into the recorder or deriving the count in the render path, and both are a change to the machinery a ruling froze.
- **Beat dropout false ATMOSPHERIC**: a beat tracker can miss beats during heavy sidechain compression. The beat-absence threshold guards against single-beat dropouts but not sustained artifacts. It matters more than it used to, because the bar grid is counted off the same stream: a gap longer than the decoder's own threshold re-anchors the grid rather than closing one bar across it.
- **Tempo octave choice now moves the bar grid, not just a threshold.** BPM is octave-folded before anyone sees it, which is what the OS2L wire has always carried. The DROP tempo gate that originally motivated the fold died with the rule classifier, but a *tracker* octave flip is now visible in the show: on a drum & bass track a live run emitted a 30-second passage on the half-time grid, which moved the committed DROP by nearly a full decoder bar relative to the simulation. Fast genres remain the exposed case.
- **ATMOSPHERIC is reachable now, and the entry that said otherwise is retired.** It used to be the only intent driven by a beat-absence timer, so on mastered EDM — whose intros and outros have beats — the timer never tripped and `intro`/`outro` were unreachable no matter how thresholds moved. `intro` and `outro` are decoder classes now, so the intent is committed from evidence; the timer survives as a second producer and the cold-start floor is a third. A live soak spent several minutes of a 35-minute session in ATMOSPHERIC.
- **The cold-start floor can put an ATMOSPHERIC block in a report that no class produced.** It fires once when the committer has never spoken at all. Its margin is chosen rather than measured — bounded on one side by not firing on any fixture track, and on the other by the fact that being wrong costs one extra effect change at the top of a set, against a dark stage. Anything scoring a report should treat it as it treats the beat-absence timer's blocks.
- **`trigger` separates the stop bypass from the committer, not "a class was inferred" from "one was not".** The stop path writes `silence` because it goes around the intent committer entirely; everything else writes `classifier` — including the cold-start floor and the beat-absence timer, which are floors rather than readings but do pass through `_commit_intent` and its whole stability pipeline. That is the distinction the field can currently make, and the label-aligned evaluation excludes only `silence`. Giving the two floors their own triggers would be defensible and would move scores on any track where they fire, so it is a deliberate follow-up rather than an oversight — and it is the reason `classifier` must not be read as a claim that a class was actually inferred.
- **`test_the_gpu_thread_runs_under_real_time_pacing` needs a free GPU, and fails honestly without one.** Under a saturated card the show's decoder legitimately never observes a bar -- the gpu stage sheds, the ring overruns, and `observed_bar` stays unset -- so the assertion fails for a true reason that is a fact about the machine rather than about the code. Attribute before debugging: re-run it against a known-green commit on the same box, and check `nvidia-smi` for who holds the card. Same contention class as a suite that dies partway through for no attributable reason.
- **Two caches beside the audio now**: `simulate file` writes `<song>.<samplerate>.npy` (the decode) and `<song>.<decoder>.mertcells.npz` (the extractor's cells). Both are gitignored, both are keyed on the audio's size and mtime, and deleting either forces that stage to re-run. The ten eval-set tracks keep both on purpose (see The benchmark); everything else in the corpus is cleaned up by the batch that created it. A cell sidecar recorded under one decode path is never served to the other — that is in the filename and in the key, because librosa and ffmpeg move 13.2 % of near-boundary decisions.
- **The live view is windowed; a `--report` session is not.** The UI draws 30 s and never reads back past it, so it keeps a rolling window. `--report` promises the whole session and now takes the window off, the way `simulate.cli._session_buffer` always did. Reports written before that fix are truncated to roughly the last two minutes of intent and effect blocks (beats were never windowed) — which includes this branch's own live-match captures and its soak artifact, and is why their block counts are lower than the simulations they are compared against.
- **Intent blocks record `song_t`, and older reports do not.** The delay is per command now (playback delay minus that decision's measured age), so no constant de-shift and no beat-matching rule recovers song time from a block's stamp. The engine records the instant it commits about; `realign_intents` reads it and infers nothing. Reports cut before that stamping keep the old inference — including its one known hole, a timer-fired ATMOSPHERIC that lands a look-ahead late — and the corpus holds thousands of them. A report mixing recorded and inferred blocks was cut across the change, and the counters say so.
- **The eval-set baseline lags a deliberate pipeline change by one command.** Any change to `lib/` or `simulate/` that moves the reports fails `run_eval_set.py` until the baseline is re-cut. That is the gate working, not a flake — but it does mean a pipeline PR is two steps, and the second one must not be skipped. **The gate keys on report content, not on musical behaviour**: the madmom migration moved all ten checksums, and so did a later merge that changed no rhythm at all. Read the printed table before assuming a checksum move means the show moved — `beats` and `changes_intent` sitting still is the signal that it did not. The NN integration is the extreme case: the branch deliberately carried a strict-xfail on this gate through the whole demolition rather than re-cutting an intermediate state as the benchmark, and cut it exactly once at the settled tip.
- **The CPU cost of the front-end, as measured today.** Paced at real time on one eval-set track: the audio loop is about 17 % of one core (mean 0.97 ms, p99 2.80 ms against the 5.805 ms buffer period), of which madmom is ~10 % and the engine's per-buffer audio push ~1.6 %; the GPU pass is ~157 ms mean / ~334 ms p99 against a 1 s hop, with the hand-off queue's p99 depth at zero. **These are anchored against the same day's madmom and cannot be diffed against the committed `realtime_measurement.json`** — madmom alone measures 2.5x apart between days on identical code, so subtracting the two files is meaningless. For history: the earlier aubio front-end cost ~1.4 % of a core end to end, and the madmom migration took that to ~25.7 % before the demolition gave ~11 % of it back by deleting the onset chain.
- **Fast simulation now has a cold cost and a warm one.** A warm run replays the extractor and is about 2.2x faster than a cold one, and needs no GPU at all. A cold run needs a GPU and is what the first pass over any new audio pays. Regenerating the corpus report cache is a cold pass over every track — and the report schema change on this branch invalidates that cache by construction.
- **A cold benchmark run must be serial on a single GPU, and `--workers` above 1 does not merely fail to help.** Each simulation worker holds ~3.3 GB of VRAM, so on an 8 GB card two workers plus the desktop already sit at 7.7 GB and the driver evicts continuously: a 30 s encoder pass that costs ~157 ms warm ran **over ten minutes without completing**, at four workers and again at two, while serial the frozen ten take 16 minutes at 4x realtime. The pathology is invisible from outside — the runner banks no per-track artifact and `pool.map` yields in order, so an empty stdout is what both a wedged run and a healthy one look like for the first several minutes. What separates them is `buffers_fed` in `simulate/runner.py`'s `run_simulation` frame, read out of the live process (`py-spy dump --locals`); times the buffer period it is seconds of audio consumed, and if it does not move in five minutes the run will not finish. A serial pass also leaves the cell sidecars behind, so the next run over the same audio is warm and needs no GPU at all.
- **madmom is CC BY-NC-SA** (models). Fine for a personal project; it forecloses a
  commercial turn without a JKU licence. **MERT's own licence terms are the same
  class of question and are not analysed here** — check them before any
  commercial turn. aubio's GPL exposure is gone with aubio.
- **madmom is pinned to a git SHA**, not a release: the last PyPI release cannot be
  imported on Python 3.10+. An upgrade is deliberate and `tests/test_madmom_contract.py`
  is what makes it a checked one.
