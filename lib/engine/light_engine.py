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

_BREAKDOWN_MAX_DENSITY  = 3.0   # onsets/sec — sparse = breakdown feel
_BUILDUP_MIN_TREND      = 1.3   # density trend ratio — rising 30% → BUILDUP
_DROP_MIN_DENSITY       = 8.0   # hi-hat alone sits at 4–8 onsets/s; DROP needs more
_PEAK_MIN_BPM           = 138.0

_BEAT_ABSENCE_SEC       = 2.5   # seconds without a beat → ATMOSPHERIC (5+ missed beats at 128 BPM)


def _classify_intent(bpm: float, onset_density: float, density_trend: float = 1.0) -> LightIntent:
    """Map (BPM, onset_density, density_trend) → LightIntent.

    Priority order matters: density spike always wins (DROP), then BPM
    thresholds narrow down the remaining cases.
    ATMOSPHERIC is NOT detected here — it is set in on_100ms_callback via beat absence.
    BUILDUP requires rising onset density (trend ≥ 1.3); steady high-BPM grooves are GROOVE.
    """
    if onset_density >= _DROP_MIN_DENSITY and bpm >= 100:
        return LightIntent.DROP
    if bpm >= _PEAK_MIN_BPM:
        return LightIntent.PEAK
    if onset_density < _BREAKDOWN_MAX_DENSITY:
        return LightIntent.BREAKDOWN
    if density_trend >= _BUILDUP_MIN_TREND:
        return LightIntent.BUILDUP
    return LightIntent.GROOVE


# BeatRecord: (monotonic_time, onset_density, bpm)
BeatRecord = tuple[float, float, float]


def _classify_windowed(window: list[BeatRecord], bpm: float) -> LightIntent:
    """Classify intent using a symmetric look-ahead/look-behind window of beats.

    Because audio playback is delayed by look_ahead_sec (dmx-enttec-node), by
    the time we need to commit a classification for beat T the window contains
    both past and future beats relative to T.  This gives us:

      - Median density  → robust to single-beat transient spikes; a genuine DROP
                          must sustain high density across the window to win.
      - Forward trend   → compare the second half of the window (future beats)
                          against the first half (past beats) to detect whether
                          energy is rising or falling around T.

    Falls back to GROOVE if the window is empty.
    """
    if not window:
        return LightIntent.GROOVE

    densities = [d for _, d, _ in window]

    # Median is more robust than mean against short transient spikes
    sorted_d = sorted(densities)
    median_density = sorted_d[len(sorted_d) // 2]

    # Trend: past half vs future half of the symmetric window
    mid = len(densities) // 2
    past = densities[:mid] if mid > 0 else densities
    future = densities[mid:] if mid > 0 else densities
    past_mean = sum(past) / len(past)
    future_mean = sum(future) / len(future)
    window_trend = future_mean / past_mean if past_mean > 0 else 1.0

    return _classify_intent(bpm, median_density, window_trend)


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
        # Rolling history of beats for windowed classification: (monotonic_time, density, bpm)
        # Kept for 2 × look_ahead_sec so the symmetric window is always available at commit time.
        self._beat_history: deque[BeatRecord] = deque()

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

    async def on_cycle(self):
        await self.effect_controller.process_effects()
        self.overlay_client.flush_messages()

    async def on_onset(self):
        pass

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        current_second = self.analyser.get_song_current_duration().total_seconds()
        onset_density = self.analyser.get_onset_density()
        density_trend = self.analyser.get_onset_density_trend()

        # Always record beat to history so _commit_intent has forward context.
        now_mono = time.monotonic()
        self._beat_history.append((now_mono, onset_density, bpm))
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
            intent = _classify_intent(bpm, onset_density, density_trend)
            logging.info(f'[engine] [immediate] intent={intent.name}')
            if self.event_buffer:
                self.event_buffer.set_intent(intent.value)
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
            intent = _classify_intent(bpm, onset_density, density_trend)
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

        Drives an effect change when the intent has changed — this is safe
        because the window suppresses transients.
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
        intent = _classify_windowed(window, bpm)
        logging.info(
            f'[engine] [windowed] intent={intent.name}  '
            f'window={len(window)} beats  '
            f'densities=[{", ".join(f"{d:.1f}" for _, d, _ in window)}]'
        )

        if self.event_buffer:
            self.event_buffer.set_intent(intent.value)

        if intent != self._current_intent:
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
            intent = _classify_windowed(window, bpm)
        else:
            intent = _classify_intent(bpm, onset_density, density_trend)

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
                self._current_intent = LightIntent.ATMOSPHERIC
                await self.effect_controller.change_effect(LightIntent.ATMOSPHERIC)

    async def on_1sec_callback(self):
        if not self.analyser.is_song_playing():
            return

    async def on_10sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        bpm = int(self.analyser.get_bpm())
        onset_density = self.analyser.get_onset_density()
        density_trend = self.analyser.get_onset_density_trend()
        current_second = int(self.analyser.get_song_current_duration().total_seconds())
        intent = _classify_intent(float(bpm), onset_density, density_trend)
        logging.info(f'[engine] == current state ==')
        logging.info(f'[engine]   realtime_bpm:    {bpm}')
        logging.info(f'[engine]   onset_density:   {onset_density:.2f} /s')
        logging.info(f'[engine]   intent:          {intent.name}')
        logging.info(f'[engine]   current_second:  {current_second}')
        logging.info(f'[engine]   last_effect:     {self.effect_controller.last_effect}')
