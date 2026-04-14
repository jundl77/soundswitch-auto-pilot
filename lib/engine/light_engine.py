from __future__ import annotations
import time
import logging
from collections import deque
from typing import TYPE_CHECKING
from lib.engine.effect_controller import EffectController
from lib.engine.delayed_command_queue import DelayedCommandQueue
from lib.engine.effect_definitions import LightIntent
from lib.clients.midi_client import MidiClient
from lib.clients.os2l_client import Os2lClient
from lib.clients.overlay_client import OverlayClient, OverlayEffect
from lib.analyser.music_analyser import MusicAnalyser
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler

if TYPE_CHECKING:
    from lib.engine.event_buffer import EventBuffer

# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------
# These thresholds are a starting point — tune against real DJ sets.
# onset_density = onsets/sec over a 1.5-second rolling window (from aubio).
#
# Structure of a typical EDM track and how we detect it:
#   ATMOSPHERIC — no beats detected for >2 s (intro, full breakdown, outro)
#                 Detected via beat-absence in on_100ms_callback, NOT here.
#   BREAKDOWN   — beat present but very sparse onsets (melodic, stripped)
#   GROOVE      — moderate onsets at mid-tempo (main dance-floor loop)
#   BUILDUP     — onset density rising; moderately high BPM (pre-drop tension)
#   DROP        — onset density spikes hard (bass, kick, hat all firing)
#   PEAK        — sustained very high BPM + high density (post-drop peak)

# Hysteresis thresholds (Schmitt trigger) — separate entry and exit values per intent
# prevent threshold-boundary oscillation ("flickering") at the edges of each zone.
_BREAKDOWN_MAX_DENSITY_ENTER = 3.3   # enter BREAKDOWN when density < this
_BREAKDOWN_MAX_DENSITY_EXIT  = 3.8   # exit BREAKDOWN when density exceeds this
_BUILDUP_MIN_TREND           = 1.5   # density trend ratio — rising ≥50% → BUILDUP
_DROP_MIN_DENSITY_ENTER      = 3.247 # enter DROP when density ≥ this
_DROP_MIN_DENSITY_EXIT       = 3.0   # exit DROP when density falls below this
_DROP_MIN_SUB_BASS_RATIO       = 0.250 # enter DROP: windowed sub-bass must reach this
_DROP_MIN_SUB_BASS_RATIO_EXIT  = 0.210 # stay in DROP: sub-bass must exceed this (≤ entry value)
_PEAK_MIN_BPM_ENTER          = 140.0 # enter PEAK when BPM ≥ this
_PEAK_MIN_BPM_EXIT           = 135.0 # exit PEAK when BPM falls below this
# RMS-based PEAK gate: loud + kick + density → PEAK even at constant BPM (e.g. 128 BPM tracks).
# Calibrate _PEAK_MIN_RMS against a feature log: run --report, inspect rms_energy per section.
_PEAK_MIN_RMS                = 999.0 # RMS-based PEAK gate — disabled until calibrated.
# Calibrate by running --report and checking rms_energy per section.  For Generate (128 BPM)
# peak section rms=0.245 vs groove rms=0.235 — only 0.010 margin, too fragile for absolute threshold.
# Needs relative RMS (current vs. rolling track mean) to be gain-invariant.  See Future Work in CLAUDE.md.
_PEAK_MIN_DENSITY_FOR_RMS_PEAK = 3.5 # density gate for RMS-based PEAK (used when RMS gate is enabled)

# Kick detection gate: kick_strength below this means no kick on beats → BREAKDOWN even at
# moderate onset density.  0.533 = calibrated against Eric Prydz "Generate" (128 BPM track).
_KICK_PRESENCE_THRESHOLD          = 0.533
# When kick is absent, clamp BREAKDOWN entry to density below this (prevents misclassifying
# a hi-hat-only pattern with no bass as BREAKDOWN when density is very high).
_BREAKDOWN_NO_KICK_MAX_DENSITY    = 6.0
# Compound BREAKDOWN rule: moderate density + absent sub-bass → BREAKDOWN even when density
# exceeds _BREAKDOWN_MAX_DENSITY_ENTER.  Catches stripped sections (e.g. Generate's breakdown
# at density ~3.7) that are above the sparse threshold but clearly not GROOVE (no sub-bass).
# Applied AFTER BUILDUP check so rising sections are not mis-classified.
_BREAKDOWN_MAX_DENSITY_WITH_LOW_SUBBASS = 4.2   # compound rule only fires below this density
_BREAKDOWN_MAX_SUB_BASS                 = 0.210  # sub-bass ceiling; groove sits at ~0.225
# Spectral centroid trend threshold: centroid rising ≥11% → BUILDUP signal (riser/sweep).
_CENTROID_BUILDUP_TREND           = 1.106

_BEAT_ABSENCE_SEC            = 2.5   # seconds without a beat → ATMOSPHERIC (5+ missed beats at 128 BPM)

# Stability: vote buffer requires this many consecutive identical votes before committing a switch.
_VOTE_BUFFER_SIZE = 8
# Minimum beats spent in current intent before a switch is allowed.
_MIN_DWELL_BEATS  = 2

# Musically impossible transitions: block these regardless of classifier output.
# e.g. you cannot go from dead-silent ATMOSPHERIC straight to a full DROP.
_INVALID_TRANSITIONS: frozenset = frozenset({
    (LightIntent.ATMOSPHERIC, LightIntent.DROP),
    (LightIntent.ATMOSPHERIC, LightIntent.BUILDUP),
    (LightIntent.ATMOSPHERIC, LightIntent.PEAK),
    (LightIntent.PEAK,        LightIntent.BUILDUP),
})


def _classify_intent(
    bpm: float,
    onset_density: float,
    density_trend: float = 1.0,
    current_intent: LightIntent | None = None,
    sub_bass_ratio: float = 0.0,
    kick_strength: float = 2.0,
    centroid_trend: float = 1.0,
    rms_energy: float = 0.0,
    spectral_flux: float = 0.0,
) -> LightIntent:
    """Map audio features → LightIntent using hysteresis thresholds.

    Priority order: DROP → PEAK → BREAKDOWN → BUILDUP → GROOVE.

    Hysteresis (Schmitt trigger): when `current_intent` is provided, the exit
    threshold for the current intent is used instead of the entry threshold.

    kick_strength (default 2.0 = kick assumed present) gates DROP and BREAKDOWN:
      - DROP requires kick on beats — prevents hi-hat-only high-density passages from triggering.
      - BREAKDOWN can fire at moderate density when kick is absent (stripped arrangement).
    centroid_trend fires BUILDUP independently of density_trend — catches riser sweeps
      where the spectral centroid climbs before onset density rises.
    rms_energy gates a second PEAK path: loud + kick + density → PEAK even when BPM is constant.
      Calibrate _PEAK_MIN_RMS from a feature log (--report flag shows rms_energy per beat).

    ATMOSPHERIC is NOT detected here — fired via beat-absence timer in on_100ms_callback.
    """
    currently_drop      = (current_intent == LightIntent.DROP)
    currently_peak      = (current_intent == LightIntent.PEAK)
    currently_breakdown = (current_intent == LightIntent.BREAKDOWN)

    drop_threshold      = _DROP_MIN_DENSITY_EXIT       if currently_drop      else _DROP_MIN_DENSITY_ENTER
    peak_threshold      = _PEAK_MIN_BPM_EXIT           if currently_peak      else _PEAK_MIN_BPM_ENTER
    breakdown_threshold = _BREAKDOWN_MAX_DENSITY_EXIT  if currently_breakdown else _BREAKDOWN_MAX_DENSITY_ENTER
    sub_bass_threshold  = _DROP_MIN_SUB_BASS_RATIO_EXIT if currently_drop      else _DROP_MIN_SUB_BASS_RATIO

    kick_present = kick_strength >= _KICK_PRESENCE_THRESHOLD

    # DROP: density spike + kick confirmed on beats + sub-bass gate (separate entry/exit hysteresis)
    if onset_density >= drop_threshold and bpm >= 100 and kick_present and sub_bass_ratio >= sub_bass_threshold:
        return LightIntent.DROP
    # PEAK: high BPM OR loud + kick + sustained density (energy-based gate for constant-BPM tracks)
    rms_peak = rms_energy >= _PEAK_MIN_RMS and kick_present and onset_density >= _PEAK_MIN_DENSITY_FOR_RMS_PEAK
    if bpm >= peak_threshold or rms_peak:
        return LightIntent.PEAK
    # BREAKDOWN: either very sparse density, or kick absent at moderate density
    # (stripped arrangement with hi-hats only should not read as GROOVE)
    if onset_density < breakdown_threshold:
        return LightIntent.BREAKDOWN
    if not kick_present and onset_density < _BREAKDOWN_NO_KICK_MAX_DENSITY:
        return LightIntent.BREAKDOWN
    # BUILDUP: rising density trend OR rising spectral centroid (riser sweep)
    if density_trend >= _BUILDUP_MIN_TREND or centroid_trend >= _CENTROID_BUILDUP_TREND:
        return LightIntent.BUILDUP
    # Secondary BREAKDOWN: moderate density + absent sub-bass (stripped arrangement).
    # Governs both ENTRY and STAY: if the bass hasn't engaged, we stay in BREAKDOWN
    # even above the normal sparse-density exit threshold.  This prevents oscillation
    # in sections (like Generate's breakdown at density ~3.7, sub_bass ~0.19) where
    # beat-to-beat density fluctuations would otherwise cause GROOVE votes on every
    # other beat.  Only fires strictly above the normal entry threshold to avoid
    # disturbing the hysteresis entry boundary.
    # Must come after BUILDUP so rising sections are not trapped here.
    if (onset_density > _BREAKDOWN_MAX_DENSITY_ENTER
            and onset_density < _BREAKDOWN_MAX_DENSITY_WITH_LOW_SUBBASS
            and sub_bass_ratio < _BREAKDOWN_MAX_SUB_BASS):
        return LightIntent.BREAKDOWN
    return LightIntent.GROOVE


# BeatRecord: (monotonic_time, onset_density, bpm, sub_bass_ratio, rms_energy, kick_strength, centroid_trend, spectral_flux)
BeatRecord = tuple[float, float, float, float, float, float, float, float]


def _classify_windowed(
    window: list[BeatRecord],
    bpm: float,
    current_intent: LightIntent | None = None,
) -> LightIntent:
    """Classify intent using a symmetric look-ahead/look-behind window of beats.

    Because audio playback is delayed by look_ahead_sec (dmx-enttec-node), by
    the time we need to commit a classification for beat T the window contains
    both past and future beats relative to T.  This gives us:

      - Median density      → robust to single-beat transient spikes.
      - Forward density trend → second half vs first half of window.
      - Mean sub-bass       → for DROP sub-bass gate.
      - Mean kick_strength  → averaged over window; kick must be consistently present for DROP.
      - Mean centroid_trend → rising centroid across the window confirms BUILDUP riser.

    Falls back to GROOVE if the window is empty.
    current_intent is forwarded to _classify_intent for hysteresis-aware thresholds.
    """
    if not window:
        return LightIntent.GROOVE

    n = len(window[0])
    densities      = [entry[1] for entry in window]
    sub_bass_vals  = [entry[3] for entry in window] if n >= 5 else [0.0] * len(window)
    rms_vals       = [entry[4] for entry in window] if n >= 6 else []
    kick_vals      = [entry[5] for entry in window] if n >= 7 else []
    centroid_vals  = [entry[6] for entry in window] if n >= 7 else []
    flux_vals      = [entry[7] for entry in window] if n >= 8 else []

    sorted_d = sorted(densities)
    median_density      = sorted_d[len(sorted_d) // 2]
    mean_sub_bass       = sum(sub_bass_vals) / len(sub_bass_vals)
    mean_rms            = sum(rms_vals) / len(rms_vals) if rms_vals else 0.0
    mean_kick           = sum(kick_vals) / len(kick_vals) if kick_vals else 2.0
    mean_centroid_trend = sum(centroid_vals) / len(centroid_vals) if centroid_vals else 1.0
    mean_flux           = sum(flux_vals) / len(flux_vals) if flux_vals else 0.0

    mid = len(densities) // 2
    past        = densities[:mid] if mid > 0 else densities
    future      = densities[mid:] if mid > 0 else densities
    past_mean   = sum(past) / len(past)
    future_mean = sum(future) / len(future)
    window_trend = future_mean / past_mean if past_mean > 0 else 1.0

    return _classify_intent(bpm, median_density, window_trend, current_intent, mean_sub_bass, mean_kick, mean_centroid_trend, mean_rms, mean_flux)


# ---------------------------------------------------------------------------
# LightEngine
# ---------------------------------------------------------------------------

class LightEngine(IMusicAnalyserHandler):
    def __init__(self,
                 midi_client: MidiClient,
                 os2l_client: Os2lClient,
                 overlay_client: OverlayClient,
                 effect_controller: EffectController,
                 command_queue: DelayedCommandQueue | None = None,
                 event_buffer: EventBuffer | None = None,
                 look_ahead_sec: float = 0.0):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.overlay_client: OverlayClient = overlay_client
        self.effect_controller: EffectController = effect_controller
        self.command_queue: DelayedCommandQueue | None = command_queue
        self.event_buffer: EventBuffer | None = event_buffer
        self.analyser: MusicAnalyser = None
        self._look_ahead_sec: float = look_ahead_sec
        self._note_counter: int = 0
        self._needs_initial_effect: bool = False
        self._atmospheric_sent: bool = False  # True while in beat-absence ATMOSPHERIC state
        self._current_intent: LightIntent | None = None  # last committed intent (for change detection)
        # Rolling history of beats for windowed classification: 5-tuple BeatRecord.
        # Kept for 2 × look_ahead_sec so the symmetric window is always available at commit time.
        self._beat_history: deque[BeatRecord] = deque()
        # Stability: vote buffer (rolling deque of last _VOTE_BUFFER_SIZE classified intents).
        # An intent change is only committed when the buffer is full and unanimous.
        self._intent_vote_buffer: deque[LightIntent] = deque(maxlen=_VOTE_BUFFER_SIZE)
        # Count of beats spent in the current intent since last switch.
        # Must reach _MIN_DWELL_BEATS before an outbound switch is allowed.
        self._beats_in_current_intent: int = 0

    def set_analyser(self, analyser: MusicAnalyser):
        self.analyser: MusicAnalyser = analyser

    def on_sound_start(self):
        logging.info('[engine] sound start')
        self.midi_client.on_sound_start()
        self.overlay_client.deactivate_all()
        self.os2l_client.on_sound_start(0, 0, 20000, 120)
        if self.event_buffer:
            self.event_buffer.set_playing(True)
        self._needs_initial_effect = True

    def on_sound_stop(self):
        logging.info('[engine] sound stop')
        self.midi_client.on_sound_stop()
        self.os2l_client.on_sound_stop()
        self.effect_controller.reset_state()
        self.overlay_client.deactivate_all()
        if self.event_buffer:
            self.event_buffer.set_playing(False)
        self._atmospheric_sent = False
        self._current_intent = None
        self._beat_history.clear()
        self._intent_vote_buffer.clear()
        self._beats_in_current_intent = 0

    async def on_cycle(self):
        await self.effect_controller.process_effects()
        self.overlay_client.flush_messages()

    async def on_onset(self):
        pass

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        current_second = self.analyser.get_song_current_duration().total_seconds()
        onset_density = self.analyser.get_onset_density()
        density_trend = self.analyser.get_onset_density_trend()
        sub_bass_ratio = self.analyser.get_sub_bass_ratio()
        rms_energy     = self.analyser.get_rms_energy()
        kick_strength  = self.analyser.get_kick_strength()
        centroid_trend = self.analyser.get_spectral_centroid_trend()
        spectral_flux  = self.analyser.get_spectral_flux()

        # Always record beat to history so _commit_intent has forward context.
        now_mono = time.monotonic()
        self._beat_history.append((now_mono, onset_density, bpm, sub_bass_ratio, rms_energy, kick_strength, centroid_trend, spectral_flux))
        # Prune entries older than 2 × look_ahead_sec (or 5 s minimum for the
        # instantaneous path) so the deque stays bounded.
        history_window = max(self._look_ahead_sec * 2, 5.0)
        while self._beat_history and now_mono - self._beat_history[0][0] > history_window:
            self._beat_history.popleft()

        logging.info(
            f'[engine] [{current_second:.2f}s] beat #{beat_number}  '
            f'bpm={bpm:.1f}  onsets/s={onset_density:.2f}  trend={density_trend:.2f}'
        )
        if self.event_buffer:
            self.event_buffer.add_beat(bpm, onset_density, bpm_changed)

        was_atmospheric = self._atmospheric_sent
        self._atmospheric_sent = False

        if self._needs_initial_effect or was_atmospheric:
            # First beat or returning from beat-absence ATMOSPHERIC: commit immediately
            # with instantaneous classification (no delay, no window — the beat itself
            # is the confirmation we need).
            self._needs_initial_effect = False
            intent = _classify_intent(bpm, onset_density, density_trend, self._current_intent, sub_bass_ratio, kick_strength, centroid_trend, rms_energy, spectral_flux)
            # Apply invalid-transition guard even on immediate classification.
            # Treat None (pre-first-beat) as ATMOSPHERIC — aubio's BPM is unreliable in the
            # first few beats (often reports 2× actual BPM), which can falsely trigger PEAK.
            effective_from = self._current_intent if self._current_intent is not None else LightIntent.ATMOSPHERIC
            if (effective_from, intent) in _INVALID_TRANSITIONS:
                logging.info(f'[engine] [immediate] blocking invalid transition {effective_from.name} → {intent.name}, falling back to GROOVE')
                intent = LightIntent.GROOVE
            logging.info(f'[engine] [immediate] intent={intent.name}')
            if self.event_buffer:
                self.event_buffer.set_intent(intent.value)
            self._intent_vote_buffer.clear()
            self._beats_in_current_intent = 0
            self._current_intent = intent
            await self.effect_controller.change_effect(intent)
        elif self._look_ahead_sec > 0 and self.command_queue:
            # Windowed mode: schedule classification commit after look_ahead_sec.
            # By then the window [T - look_ahead_sec, T + look_ahead_sec] is fully
            # populated in _beat_history and we can classify using both past and
            # future context relative to this beat.
            _enqueue_time = now_mono
            _bpm = bpm
            await self.command_queue.enqueue(
                'intent',
                lambda: self._commit_intent(_enqueue_time, _bpm)
            )
        else:
            # Instantaneous mode (look_ahead_sec == 0): classify and update the
            # event buffer immediately. Effect changes are driven by section
            # changes and atmospheric detection — not every beat.
            intent = _classify_intent(bpm, onset_density, density_trend, self._current_intent, sub_bass_ratio, kick_strength, centroid_trend)
            logging.info(f'[engine] intent={intent.name}')
            if self.event_buffer:
                self.event_buffer.set_intent(intent.value)

        # OS2L beat — always goes through the queue so it fires in sync with audio.
        _change, _pos, _bpm2 = bpm_changed, beat_number, bpm
        if self.command_queue:
            await self.command_queue.enqueue(
                'beat',
                lambda: self.os2l_client.send_beat(change=_change, pos=_pos, bpm=_bpm2, strength=0.5)
            )
        else:
            await self.os2l_client.send_beat(change=bpm_changed, pos=beat_number, bpm=bpm, strength=0.5)

    async def _commit_intent(self, enqueue_time: float, bpm: float) -> None:
        """Fired by DelayedCommandQueue after look_ahead_sec.

        At this point the beat at `enqueue_time` is exactly when the audience
        hears the audio.  _beat_history now contains beats from
        [enqueue_time - look_ahead_sec, enqueue_time + look_ahead_sec], giving
        a symmetric window for a confident classification.

        Stability pipeline (applied in order):
          1. Windowed classification with hysteresis-aware thresholds.
          2. Vote buffer: _VOTE_BUFFER_SIZE consecutive identical votes required.
          3. Minimum dwell: _MIN_DWELL_BEATS beats in current intent before switching.
          4. Invalid-transition guard: musically impossible jumps are blocked.
        """
        if self._atmospheric_sent:
            # Beat absence was detected after this beat was enqueued; ATMOSPHERIC
            # already fired in real-time.  Skip to avoid overriding it.
            logging.debug('[engine] [windowed] skipping commit — currently in ATMOSPHERIC')
            return

        window = [
            entry for entry in self._beat_history
            if abs(entry[0] - enqueue_time) <= self._look_ahead_sec
        ]
        intent = _classify_windowed(window, bpm, self._current_intent)
        self._intent_vote_buffer.append(intent)
        self._beats_in_current_intent += 1

        logging.info(
            f'[engine] [windowed] vote={intent.name}  '
            f'buffer=[{", ".join(v.name for v in self._intent_vote_buffer)}]  '
            f'dwell={self._beats_in_current_intent}  '
            f'window={len(window)} beats  '
            f'densities=[{", ".join(f"{e[1]:.1f}" for e in window)}]'
        )

        # --- 1. Vote consensus required ---
        if len(self._intent_vote_buffer) < _VOTE_BUFFER_SIZE:
            return  # buffer not yet full
        if not all(v == intent for v in self._intent_vote_buffer):
            return  # mixed votes — not confident enough

        # Consensus reached: surface to visualizer regardless of switch outcome.
        if self.event_buffer:
            self.event_buffer.set_intent(intent.value)

        if intent == self._current_intent:
            return  # stable — no effect change needed

        # --- 2. Minimum dwell check ---
        if self._beats_in_current_intent < _MIN_DWELL_BEATS:
            logging.debug(
                f'[engine] [windowed] dwell check: {self._beats_in_current_intent}/'
                f'{_MIN_DWELL_BEATS} beats in {self._current_intent.name if self._current_intent else "None"}'
                f' — holding'
            )
            return

        # --- 3. Invalid-transition guard ---
        if self._current_intent is not None:
            transition = (self._current_intent, intent)
            if transition in _INVALID_TRANSITIONS:
                logging.info(
                    f'[engine] [windowed] blocking invalid transition '
                    f'{self._current_intent.name} → {intent.name}'
                )
                return

        # --- All checks passed: commit the intent change ---
        logging.info(
            f'[engine] [windowed] intent change: '
            f'{self._current_intent.name if self._current_intent else "None"} → {intent.name}'
        )
        self._intent_vote_buffer.clear()
        self._beats_in_current_intent = 0
        self._current_intent = intent
        await self.effect_controller.change_effect(intent)

    async def on_note(self):
        dmx_data = [0] * 24
        self._note_counter = (self._note_counter + 3) % 24
        dmx_data[self._note_counter] = 100
        self.overlay_client.update_overlay_data(OverlayEffect.LIGHT_BAR_24, dmx_data)
        logging.info('[engine] note detected')

    async def on_section_change(self) -> None:
        logging.info('[engine] audio section change detected')
        bpm = self.analyser.get_bpm()
        onset_density = self.analyser.get_onset_density()
        density_trend = self.analyser.get_onset_density_trend()

        if self._look_ahead_sec > 0:
            # Use the windowed classification for section changes too — the window
            # is already populated since beats have been flowing.
            window = list(self._beat_history)
            intent = _classify_windowed(window, bpm, self._current_intent)
        else:
            intent = _classify_intent(bpm, onset_density, density_trend, self._current_intent)

        _intent = intent
        if self.command_queue:
            await self.command_queue.enqueue(
                'section_change',
                lambda: self.effect_controller.change_effect(_intent)
            )
        else:
            await self.effect_controller.change_effect(intent)

    async def on_100ms_callback(self):
        if not self.analyser.is_song_playing():
            return
        if self.analyser.get_seconds_since_last_beat() > _BEAT_ABSENCE_SEC:
            if self.event_buffer:
                self.event_buffer.set_intent(LightIntent.ATMOSPHERIC.value)
            if not self._atmospheric_sent:
                self._atmospheric_sent = True
                self._intent_vote_buffer.clear()   # stale votes no longer relevant
                self._beats_in_current_intent = 0
                self._current_intent = LightIntent.ATMOSPHERIC
                await self.effect_controller.change_effect(LightIntent.ATMOSPHERIC)

    async def on_1sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        # Refresh timing stats in the event buffer so the visualizer can display them.
        if self.event_buffer and self.command_queue:
            self.event_buffer.set_timing_log(self.command_queue.get_timing_log())

    async def on_10sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        bpm = int(self.analyser.get_bpm())
        onset_density = self.analyser.get_onset_density()
        density_trend = self.analyser.get_onset_density_trend()
        current_second = int(self.analyser.get_song_current_duration().total_seconds())
        intent = _classify_intent(float(bpm), onset_density, density_trend, self._current_intent)
        logging.info(f'[engine] == current state ==')
        logging.info(f'[engine]   realtime_bpm:    {bpm}')
        logging.info(f'[engine]   onset_density:   {onset_density:.2f} /s')
        logging.info(f'[engine]   intent:          {intent.name}')
        logging.info(f'[engine]   current_second:  {current_second}')
        logging.info(f'[engine]   last_effect:     {self.effect_controller.last_effect}')
