from __future__ import annotations
import logging
from collections import deque
from typing import NamedTuple, TYPE_CHECKING
from lib.engine.effect_controller import EffectController
from lib.engine.delayed_command_queue import DelayedCommandQueue
from lib.engine.effect_definitions import LightIntent
from lib.clients.midi_client import MidiClient
from lib.clients.os2l_client import Os2lClient
from lib.clients.overlay_client import OverlayClient, OverlayEffect
from lib.analyser.music_analyser import (MusicAnalyser, KICK_UNKNOWN,
                                         density_is_known)
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.clock import Clock, SYSTEM_CLOCK

if TYPE_CHECKING:
    from lib.engine.event_buffer import EventBuffer

_BREAKDOWN_MAX_DENSITY_ENTER = 3.0
_BREAKDOWN_MAX_DENSITY_EXIT = 3.5
_BUILDUP_MIN_TREND = 1.3
_BUILDUP_MIN_DENSITY = _BREAKDOWN_MAX_DENSITY_ENTER
_DROP_MIN_DENSITY_ENTER = 4.0
_DROP_MIN_DENSITY_EXIT = 3.5
_DROP_MIN_SUB_BASS_RATIO = 0.0   # 0.0 leaves the gate open; kick_strength gates DROP instead

_KICK_PRESENCE_THRESHOLD = 2.4
_BREAKDOWN_NO_KICK_MARGIN = 1.0
_CENTROID_BUILDUP_TREND = 1.1

_BEAT_ABSENCE_SEC = 2.5

_VOTE_BUFFER_SIZE = 3
_MIN_DWELL_BEATS = 4
_PEAK_PROMOTION_BEATS = 32

_INVALID_TRANSITIONS: frozenset = frozenset({
    (LightIntent.ATMOSPHERIC, LightIntent.DROP),
    (LightIntent.ATMOSPHERIC, LightIntent.BUILDUP),
    (LightIntent.ATMOSPHERIC, LightIntent.PEAK),
    (LightIntent.PEAK,        LightIntent.BUILDUP),
})


def _hold(current_intent: LightIntent | None) -> LightIntent:
    """What to show when density is unmeasured: whatever the last real one said.

    ATMOSPHERIC is the exception: only a beat reaches the classifier, so beats
    are flowing and holding would keep the stage dark through live music.
    """
    if current_intent is None or current_intent is LightIntent.ATMOSPHERIC:
        return LightIntent.GROOVE
    return current_intent


def _classify_intent(
    bpm: float,
    onset_density: float,
    density_trend: float = 1.0,
    current_intent: LightIntent | None = None,
    sub_bass_ratio: float = 0.0,
    kick_strength: float = KICK_UNKNOWN,
    centroid_trend: float = 1.0,
) -> LightIntent:
    """Map audio features → LightIntent, using the current intent's exit
    threshold instead of its entry threshold (Schmitt trigger).

    No feature branch yields ATMOSPHERIC (beat-absence timer) or PEAK (engine
    promotion); an unmeasured density echoes back whatever the caller holds.
    """
    if not density_is_known(onset_density):
        return _hold(current_intent)
    currently_drop = current_intent in (LightIntent.DROP, LightIntent.PEAK)
    currently_breakdown = (current_intent == LightIntent.BREAKDOWN)

    drop_threshold = _DROP_MIN_DENSITY_EXIT if currently_drop else _DROP_MIN_DENSITY_ENTER
    breakdown_threshold = (_BREAKDOWN_MAX_DENSITY_EXIT if currently_breakdown
                           else _BREAKDOWN_MAX_DENSITY_ENTER)

    kick_present = kick_strength >= _KICK_PRESENCE_THRESHOLD

    if onset_density >= drop_threshold and bpm >= 100 and kick_present and sub_bass_ratio >= _DROP_MIN_SUB_BASS_RATIO:
        return LightIntent.DROP
    # Ordered ahead of BREAKDOWN: a riser strips the kick by design, so the
    # no-kick branch below would otherwise swallow every buildup.
    if onset_density >= _BUILDUP_MIN_DENSITY and (
            density_trend >= _BUILDUP_MIN_TREND or centroid_trend >= _CENTROID_BUILDUP_TREND):
        return LightIntent.BUILDUP
    if onset_density < breakdown_threshold:
        return LightIntent.BREAKDOWN
    if not kick_present and onset_density < breakdown_threshold + _BREAKDOWN_NO_KICK_MARGIN:
        return LightIntent.BREAKDOWN
    return LightIntent.GROOVE


class BeatRecord(NamedTuple):
    at: float
    onset_density: float
    bpm: float
    sub_bass_ratio: float
    rms_energy: float
    kick_strength: float
    centroid_trend: float


def _classify_windowed(
    window: list[BeatRecord],
    bpm: float,
    current_intent: LightIntent | None = None,
) -> LightIntent | None:
    """Classify from a symmetric window of past and future beats around T.

    A median density outvotes single-beat spikes; the trend is the window's
    second half over its first. No beats at all is not a classification — it
    returns None, and the caller must leave the stage alone.
    """
    if not window:
        return None

    # A sentinel inside a median is just a low number, so unmeasured beats are
    # dropped rather than averaged in.
    window = [beat for beat in window if density_is_known(beat.onset_density)]
    if not window:
        return _hold(current_intent)

    densities = [beat.onset_density for beat in window]
    sub_bass_vals = [beat.sub_bass_ratio for beat in window]
    kick_vals = [beat.kick_strength for beat in window]
    centroid_vals = [beat.centroid_trend for beat in window]

    sorted_d = sorted(densities)
    median_density = sorted_d[len(sorted_d) // 2]
    mean_sub_bass = sum(sub_bass_vals) / len(sub_bass_vals)
    mean_kick = sum(kick_vals) / len(kick_vals)
    mean_centroid_trend = sum(centroid_vals) / len(centroid_vals)

    mid = len(densities) // 2
    past = densities[:mid] if mid > 0 else densities
    future = densities[mid:] if mid > 0 else densities
    past_mean = sum(past) / len(past)
    future_mean = sum(future) / len(future)
    window_trend = future_mean / past_mean if past_mean > 0 else 1.0

    return _classify_intent(bpm, median_density, window_trend, current_intent, mean_sub_bass, mean_kick, mean_centroid_trend)


class LightEngine(IMusicAnalyserHandler):
    def __init__(self,
                 midi_client: MidiClient,
                 os2l_client: Os2lClient,
                 overlay_client: OverlayClient,
                 effect_controller: EffectController,
                 command_queue: DelayedCommandQueue | None = None,
                 event_buffer: EventBuffer | None = None,
                 look_ahead_sec: float = 0.0,
                 clock: Clock = SYSTEM_CLOCK):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.overlay_client: OverlayClient = overlay_client
        self.effect_controller: EffectController = effect_controller
        self.command_queue: DelayedCommandQueue | None = command_queue
        self.event_buffer: EventBuffer | None = event_buffer
        self.analyser: MusicAnalyser = None
        self._look_ahead_sec: float = look_ahead_sec
        self._clock: Clock = clock
        self._note_counter: int = 0
        self._needs_initial_effect: bool = False
        self._atmospheric_sent: bool = False
        self._current_intent: LightIntent | None = None
        self._published_bpm: dict = {}
        self._beat_history: deque[BeatRecord] = deque()
        self._intent_vote_buffer: deque[LightIntent] = deque(maxlen=_VOTE_BUFFER_SIZE)
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

        now_mono = self._clock.monotonic()
        self._beat_history.append(BeatRecord(now_mono, onset_density, bpm, sub_bass_ratio,
                                             rms_energy, kick_strength, centroid_trend))
        history_window = max(self._look_ahead_sec * 2, 5.0)
        while self._beat_history and now_mono - self._beat_history[0].at > history_window:
            self._beat_history.popleft()

        logging.info(
            f'[engine] [{current_second:.2f}s] beat #{beat_number}  '
            f'bpm={bpm:.1f}  onsets/s={onset_density:.2f}  trend={density_trend:.2f}'
        )
        if self.event_buffer:
            self.event_buffer.add_beat(bpm, onset_density, bpm_changed,
                                       kick_strength=kick_strength,
                                       centroid_trend=centroid_trend,
                                       sub_bass_ratio=sub_bass_ratio,
                                       rms=rms_energy)

        was_atmospheric = self._atmospheric_sent
        self._atmospheric_sent = False

        if self._needs_initial_effect or was_atmospheric:
            # The beat itself is the confirmation, so no window — but the change
            # still rides the look-ahead, or it lands before the beat is heard.
            self._needs_initial_effect = False
            intent = _classify_intent(bpm, onset_density, density_trend, self._current_intent, sub_bass_ratio, kick_strength, centroid_trend)
            logging.info(f'[engine] [immediate] intent={intent.name}')
            await self._enqueue_or_apply('intent', intent)
        elif self._look_ahead_sec > 0 and self.command_queue:
            await self.command_queue.enqueue(
                'intent',
                lambda: self._commit_intent(now_mono, bpm)
            )
        else:
            intent = _classify_intent(bpm, onset_density, density_trend, self._current_intent, sub_bass_ratio, kick_strength, centroid_trend)
            logging.info(f'[engine] intent={intent.name}')
            if self.event_buffer:
                self.event_buffer.set_intent(intent.value)

        published_bpm = self._publishable_bpm(self._published_bpm, bpm)
        if self.command_queue:
            await self.command_queue.enqueue(
                'beat',
                lambda: self.os2l_client.send_beat(change=bpm_changed, pos=beat_number,
                                                   bpm=published_bpm, strength=0.5)
            )
        else:
            await self.os2l_client.send_beat(change=bpm_changed, pos=beat_number,
                                             bpm=published_bpm, strength=0.5)

    @staticmethod
    def _publishable_bpm(state: dict, bpm: float) -> float:
        """Hold the last measured tempo rather than publishing a warm-up 0.0,
        which OS2L consumers read as a tempo of zero rather than "not known yet".

        Not cleared between songs: a new track's first second carries the
        previous track's tempo, which a DJ has beat-matched anyway.
        """
        if bpm > 0:
            state['last'] = bpm
            return bpm
        return state.get('last', 0.0)

    async def _apply_intent(self, intent: LightIntent) -> None:
        """The single path that moves the stage: timeline, guard state and MIDI
        together, so nothing can light an intent the guards do not know about."""
        if self.event_buffer:
            self.event_buffer.set_intent(intent.value)
        self._intent_vote_buffer.clear()
        self._beats_in_current_intent = 0
        self._current_intent = intent
        await self.effect_controller.change_effect(intent)

    async def _enqueue_or_apply(self, label: str, intent: LightIntent) -> None:
        if self.command_queue:
            await self.command_queue.enqueue(label, lambda: self._apply_intent(intent))
        else:
            await self._apply_intent(intent)

    async def _commit_intent(self, enqueue_time: float, bpm: float) -> None:
        """Fired by DelayedCommandQueue once the audience hears `enqueue_time`,
        when _beat_history spans a full look-ahead either side of it.
        """
        if self._atmospheric_sent:
            logging.debug('[engine] [windowed] skipping commit — currently in ATMOSPHERIC')
            return

        window = [
            beat for beat in self._beat_history
            if abs(beat.at - enqueue_time) <= self._look_ahead_sec
        ]
        intent = _classify_windowed(window, bpm, self._current_intent)
        if intent is None:
            logging.debug('[engine] [windowed] no beats in the window — nothing to commit')
            return
        self._intent_vote_buffer.append(intent)
        self._beats_in_current_intent += 1

        logging.info(
            f'[engine] [windowed] vote={intent.name}  '
            f'buffer=[{", ".join(v.name for v in self._intent_vote_buffer)}]  '
            f'dwell={self._beats_in_current_intent}  '
            f'window={len(window)} beats  '
            f'densities=[{", ".join(f"{b.onset_density:.1f}" for b in window)}]'
        )

        # A continuation of an already-committed DROP, so it deliberately
        # bypasses the vote and invalid-transition guards below.
        if (self._current_intent == LightIntent.DROP
                and self._beats_in_current_intent >= _PEAK_PROMOTION_BEATS):
            logging.info('[engine] [windowed] sustained DROP — promoting to PEAK')
            await self._apply_intent(LightIntent.PEAK)
            return

        if len(self._intent_vote_buffer) < _VOTE_BUFFER_SIZE:
            return
        if not all(v == intent for v in self._intent_vote_buffer):
            return

        # Absorbed before the surface step below, so the timeline keeps reading
        # the committed PEAK rather than the swallowed DROP vote.
        if self._current_intent == LightIntent.PEAK and intent == LightIntent.DROP:
            return

        if self.event_buffer:
            self.event_buffer.set_intent(intent.value)

        if intent == self._current_intent:
            return

        if self._beats_in_current_intent < _MIN_DWELL_BEATS:
            logging.debug(
                f'[engine] [windowed] dwell check: {self._beats_in_current_intent}/'
                f'{_MIN_DWELL_BEATS} beats in {self._current_intent.name if self._current_intent else "None"}'
                f' — holding'
            )
            return

        if self._current_intent is not None:
            transition = (self._current_intent, intent)
            if transition in _INVALID_TRANSITIONS:
                logging.info(
                    f'[engine] [windowed] blocking invalid transition '
                    f'{self._current_intent.name} → {intent.name}'
                )
                return

        logging.info(
            f'[engine] [windowed] intent change: '
            f'{self._current_intent.name if self._current_intent else "None"} → {intent.name}'
        )
        await self._apply_intent(intent)

    async def on_note(self):
        dmx_data = [0] * 24
        self._note_counter = (self._note_counter + 3) % 24
        dmx_data[self._note_counter] = 100
        self.overlay_client.update_overlay_data(OverlayEffect.LIGHT_BAR_24, dmx_data)
        logging.info('[engine] note detected')

    async def on_section_change(self) -> None:
        logging.info('[engine] audio section change detected')
        bpm = self.analyser.get_bpm()

        if self._look_ahead_sec > 0:
            intent = _classify_windowed(list(self._beat_history), bpm, self._current_intent)
        else:
            intent = _classify_intent(bpm, self.analyser.get_onset_density(),
                                      self.analyser.get_onset_density_trend(),
                                      self._current_intent,
                                      self.analyser.get_sub_bass_ratio(),
                                      self.analyser.get_kick_strength(),
                                      self.analyser.get_spectral_centroid_trend())
        if intent is None:
            return
        await self._enqueue_or_apply('section_change', intent)

    async def on_100ms_callback(self):
        if not self.analyser.is_song_playing():
            return
        if self.analyser.get_seconds_since_last_beat() > _BEAT_ABSENCE_SEC:
            if not self._atmospheric_sent:
                self._atmospheric_sent = True
                await self._enqueue_or_apply('atmospheric', LightIntent.ATMOSPHERIC)

    async def on_1sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        if self.event_buffer and self.command_queue:
            self.event_buffer.set_timing_log(self.command_queue.get_timing_log())

    async def on_10sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        bpm = int(self.analyser.get_bpm())
        onset_density = self.analyser.get_onset_density()
        current_second = int(self.analyser.get_song_current_duration().total_seconds())
        intent_name = self._current_intent.name if self._current_intent else 'None'
        logging.info(f'[engine] == current state ==')
        logging.info(f'[engine]   realtime_bpm:    {bpm}')
        logging.info(f'[engine]   onset_density:   {onset_density:.2f} /s')
        logging.info(f'[engine]   intent:          {intent_name}')
        logging.info(f'[engine]   current_second:  {current_second}')
        logging.info(f'[engine]   last_effect:     {self.effect_controller.last_effect}')
