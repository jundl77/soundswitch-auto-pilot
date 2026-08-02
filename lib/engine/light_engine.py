from __future__ import annotations
import contextlib
import logging
from typing import TYPE_CHECKING
from lib.audio_config import SAMPLE_RATE
from lib.engine.effect_controller import EffectController
from lib.engine.delayed_command_queue import DelayedCommandQueue
from lib.engine.effect_definitions import LightIntent, intent_for_class
from lib.clients.midi_client import SETTLE_SEC, MidiClient
from lib.clients.os2l_client import Os2lClient
from lib.clients.overlay_client import OverlayClient, OverlayEffect
from lib.analyser.music_analyser import MusicAnalyser
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.clock import Clock, SYSTEM_CLOCK

if TYPE_CHECKING:
    from lib.engine.event_buffer import EventBuffer

_BEAT_ABSENCE_SEC = 2.5

# 32 beat-commits on the retired device, converted to bars at 4 beats each.
PEAK_PROMOTION_BARS = 8

_LATENCY_LOG_STEP_SEC = 0.25
COLD_START_FLOOR_MARGIN_SEC = 4.0

REFRESH_COOLDOWN_SEC = 10.0
# Priced by training/nn_boundary_refresh_rate.py at 1.55 refreshes/min.
BOUNDARY_REFRESH_SCORE = 0.5


class LightEngine(IMusicAnalyserHandler):
    """Decoder decisions in, a show out."""

    def __init__(self,
                 midi_client: MidiClient,
                 os2l_client: Os2lClient,
                 overlay_client: OverlayClient,
                 effect_controller: EffectController,
                 command_queue: DelayedCommandQueue | None = None,
                 event_buffer: EventBuffer | None = None,
                 playback_delay_sec: float = 0.0,
                 section_chain=None,
                 section_decoder=None,
                 watchdog=None,
                 clock: Clock = SYSTEM_CLOCK):
        self.midi_client: MidiClient = midi_client
        self.os2l_client: Os2lClient = os2l_client
        self.overlay_client: OverlayClient = overlay_client
        self.effect_controller: EffectController = effect_controller
        self.command_queue: DelayedCommandQueue | None = command_queue
        self.event_buffer: EventBuffer | None = event_buffer
        self.analyser: MusicAnalyser = None
        self.section_chain = section_chain
        self.section_decoder = section_decoder
        self._watchdog = watchdog
        self._playback_delay_sec: float = playback_delay_sec
        self._clock: Clock = clock
        self._note_counter: int = 0
        self._atmospheric_sent: bool = False
        self._current_intent: LightIntent | None = None
        self._pending_intents: list = []
        self._published_bpm: dict = {}
        self._bars_in_current_intent: int = 0
        self._intent_commits: int = 0
        self._audio_sec: float = 0.0
        self._latency_logged_at: float | None = None
        self._committing_late: bool = False
        self._floor_armed: bool = True
        self._last_refresh_sec: float = float('-inf')
        self._committed = None
        self._log_chain_latency()

    def set_analyser(self, analyser: MusicAnalyser):
        self.analyser: MusicAnalyser = analyser

    @property
    def current_intent(self) -> LightIntent | None:
        return self._current_intent

    @property
    def decided_intent(self) -> LightIntent | None:
        return (self._pending_intents[-1][1] if self._pending_intents
                else self._current_intent)

    @property
    def audio_sec(self) -> float:
        return self._audio_sec

    @property
    def intent_commits(self) -> int:
        return self._intent_commits

    def on_sound_start(self):
        logging.info('[engine] sound start')
        self.os2l_client.on_sound_start(0, 0, 20000, 120)
        if self.event_buffer:
            self.event_buffer.set_playing(True)
        self._at_the_room('sound', self._show_sound_start)

    def on_sound_stop(self):
        logging.info('[engine] sound stop')
        self.os2l_client.on_sound_stop()
        self.effect_controller.reset_state()
        if self.event_buffer:
            self.event_buffer.set_playing(False)
        self._at_the_room('sound', self._show_sound_stop)

    def _show_sound_start(self) -> None:
        with self._deliberate_stall():
            self.midi_client.on_sound_start()
            self.overlay_client.deactivate_all()

    def _show_sound_stop(self) -> None:
        with self._deliberate_stall():
            self.midi_client.on_sound_stop()
            self.overlay_client.deactivate_all()

    @contextlib.contextmanager
    def _deliberate_stall(self):
        started = self._clock.monotonic()
        try:
            yield
        finally:
            if self._watchdog is not None:
                self._watchdog.forgive(
                    min(self._clock.monotonic() - started, SETTLE_SEC))

    def _at_the_room(self, label: str, action) -> None:
        if self.command_queue:
            async def command():
                action()

            self.command_queue.schedule(label, command)
        else:
            action()

        self._atmospheric_sent = False
        self._current_intent = None
        self._bars_in_current_intent = 0
        self._floor_armed = True
        if self.command_queue:
            self.command_queue.drop_pending('intent', float('-inf'))
            self.command_queue.drop_pending('refresh', float('-inf'))
        self._pending_intents = []
        self._committed = None
        self._last_refresh_sec = float('-inf')
        self._audio_sec = 0.0
        if self.section_chain is not None:
            self.section_chain.reset()
        if self.section_decoder is not None:
            self.section_decoder.reset()

    async def on_cycle(self):
        await self.effect_controller.process_effects()
        self.overlay_client.flush_messages()

    async def on_audio(self, audio_signal) -> None:
        self._audio_sec += len(audio_signal) / SAMPLE_RATE
        if self.section_chain is None or self.section_decoder is None:
            return
        drained = self.section_chain.push_audio(audio_signal)
        if drained.gap:
            self.section_decoder.reset()
            self._last_refresh_sec = float('-inf')
            self._committed = None
            self._publish_decoder_state(None)
        for posterior in drained.posteriors:
            await self._commit(self.section_decoder.push_posterior(
                posterior.time_sec, posterior.posterior, posterior.boundary))
            await self._refresh_on_boundary(posterior.boundary,
                                            posterior.time_sec)

    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        current_second = self.analyser.get_song_current_duration().total_seconds()
        rms_energy = self.analyser.get_rms_energy()

        logging.info(
            f'[engine] [{current_second:.2f}s] beat #{beat_number}  bpm={bpm:.1f}'
        )
        if self.event_buffer:
            self.event_buffer.add_beat(bpm, bpm_changed, rms=rms_energy)

        self._atmospheric_sent = False

        if self.section_decoder is not None:
            await self._commit(self.section_decoder.push_beat(self._audio_sec))

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
        # OS2L consumers read a warm-up 0.0 as a tempo of zero, not as "unknown".
        if bpm > 0:
            state['last'] = bpm
            return bpm
        return state.get('last', 0.0)

    async def _commit(self, decisions) -> None:
        self._publish_decoder_state(decisions[-1] if decisions else None)
        for decision in decisions:
            intent = intent_for_class(decision.label)
            self._bars_in_current_intent += 1
            decided = self.decided_intent

            if (decided is LightIntent.DROP and intent is LightIntent.DROP
                    and self._bars_in_current_intent >= PEAK_PROMOTION_BARS):
                logging.info(f'[engine] sustained DROP over '
                             f'{self._bars_in_current_intent} bars — promoting to PEAK')
                intent = LightIntent.PEAK
            elif decided is LightIntent.PEAK and intent is LightIntent.DROP:
                continue

            logging.info(
                f'[engine] bar {decision.bar} @ {decision.start_sec:.2f}s  '
                f'{decision.label} → {intent.name}')
            await self._commit_intent(
                intent, max(0.0, self._audio_sec - decision.start_sec))

    def _publish_decoder_state(self, decision) -> None:
        if self.event_buffer is None or self.section_decoder is None:
            return
        decoder = self.section_decoder
        if decision is not None:
            self._committed = decision
        observed = decoder.recent_observations[-1] \
            if decoder.recent_observations else None
        committed = self._committed
        self.event_buffer.set_decoder_state(
            classes=list(decoder.classes),
            posterior=(None if observed is None or observed.posterior is None
                       else [round(float(p), 6) for p in observed.posterior]),
            boundary=(None if observed is None else float(observed.boundary)),
            observed_bar=(None if observed is None else observed.bar),
            committed_bar=(None if committed is None else committed.bar),
            committed_label=(None if committed is None else committed.label),
            lag_bars=(None if observed is None or committed is None
                      else observed.bar - committed.bar),
            chain_latency_sec=decoder.chain_latency_sec,
        )

    def _log_chain_latency(self) -> None:
        decoder = self.section_decoder
        if decoder is None:
            logging.info(f'[engine] no section decoder — holding intent; '
                         f'playback delay {self._playback_delay_sec:.2f}s')
            return
        latency = decoder.chain_latency_sec
        if (self._latency_logged_at is not None
                and abs(latency - self._latency_logged_at) < _LATENCY_LOG_STEP_SEC):
            return
        self._latency_logged_at = latency
        logging.info(
            f'[engine] chain latency {latency:.2f}s = '
            f'{decoder.feature_latency_sec:.2f}s features + '
            f'{latency - decoder.feature_latency_sec:.2f}s decoder '
            f'({decoder.params.lag_bars} + 1 lag × {decoder.bar_sec:.3f}s bars) | '
            f'playback delay {self._playback_delay_sec:.2f}s → queue delay '
            f'{max(0.0, self._playback_delay_sec - latency):.2f}s — ensure '
            f'dmx-enttec-node playback_delay_seconds matches')

    async def _commit_intent(self, intent: LightIntent, age_sec: float) -> None:
        self._floor_armed = False
        now = self._clock.monotonic()
        fire_at = now - age_sec + self._playback_delay_sec
        self._note_lateness(now - fire_at)
        fire_at = max(fire_at, now)

        if self.command_queue:
            self.command_queue.drop_pending('intent', fire_at)
        self._pending_intents = [item for item in self._pending_intents
                                 if item[0] <= fire_at]
        if intent is self.decided_intent:
            return

        self._bars_in_current_intent = 0
        if self.command_queue:
            self.command_queue.drop_pending('refresh', fire_at, inclusive=True)
        song_sec = (None if self.event_buffer is None
                    else self.event_buffer.elapsed() - age_sec)
        if not self.command_queue:
            await self._apply_intent(None, intent, song_sec)
            return
        entry = (fire_at, intent)
        self._pending_intents.append(entry)
        await self.command_queue.enqueue(
            'intent', lambda: self._apply_intent(entry, intent, song_sec),
            delay_sec=fire_at - now)

    def _note_lateness(self, late_sec: float) -> None:
        if late_sec > 0.0:
            if not self._committing_late:
                self._committing_late = True
                logging.warning(
                    f'[engine] the chain is older than the '
                    f'{self._playback_delay_sec:.2f}s playback delay — intents '
                    f'commit {late_sec:.2f}s late (slow tempo, accepted)')
        elif self._committing_late:
            self._committing_late = False
            logging.info('[engine] the chain is back inside the playback delay')

    async def _apply_intent(self, entry, intent: LightIntent,
                            song_sec: float | None) -> None:
        if entry is not None and self._pending_intents \
                and self._pending_intents[0] is entry:
            self._pending_intents.pop(0)
        if self.event_buffer:
            self.event_buffer.set_intent(intent.value, song_sec=song_sec)
        self._current_intent = intent
        self._intent_commits += 1
        await self.effect_controller.change_effect(intent)

    async def on_note(self):
        dmx_data = [0] * 24
        self._note_counter = (self._note_counter + 3) % 24
        dmx_data[self._note_counter] = 100
        if self.command_queue:
            await self.command_queue.enqueue(
                'overlay',
                lambda: self._show_light_bar(dmx_data))
        else:
            self.overlay_client.update_overlay_data(OverlayEffect.LIGHT_BAR_24,
                                                    dmx_data)
        logging.info('[engine] note detected')

    async def _show_light_bar(self, dmx_data: list) -> None:
        self.overlay_client.update_overlay_data(OverlayEffect.LIGHT_BAR_24,
                                                dmx_data)

    async def _refresh_on_boundary(self, boundary: float, song_sec: float) -> None:
        if boundary < BOUNDARY_REFRESH_SCORE:
            return
        if song_sec - self._last_refresh_sec < REFRESH_COOLDOWN_SEC:
            return
        self._last_refresh_sec = song_sec
        if self._current_intent is None and not self._pending_intents:
            return

        now = self._clock.monotonic()
        fire_at = max(now, now - (self._audio_sec - song_sec)
                      + self._playback_delay_sec)
        committed = self._intent_commits
        if not self.command_queue:
            await self._apply_refresh(committed)
            return
        await self.command_queue.enqueue(
            'refresh', lambda: self._apply_refresh(committed),
            delay_sec=fire_at - now)

    async def _apply_refresh(self, committed: int) -> None:
        if self._intent_commits != committed or self._current_intent is None:
            return
        logging.info(f'[engine] boundary inside {self._current_intent.name} — '
                     f'refreshing the effect')
        await self.effect_controller.change_effect(self._current_intent)

    async def _floor_if_nothing_arrived(self) -> None:
        if not self._floor_armed:
            return
        chain = (0.0 if self.section_decoder is None
                 else self.section_decoder.chain_latency_sec)
        if self._audio_sec < chain + COLD_START_FLOOR_MARGIN_SEC:
            return
        self._floor_armed = False
        logging.warning(
            f'[engine] no decision after {self._audio_sec:.1f}s of audio '
            f'(chain {chain:.1f}s) — lighting ATMOSPHERIC as the floor rather '
            f'than leaving the rig dark')
        await self._commit_intent(LightIntent.ATMOSPHERIC,
                                  self._playback_delay_sec)

    async def on_100ms_callback(self):
        if not self.analyser.is_song_playing():
            return
        await self._floor_if_nothing_arrived()
        if self.analyser.get_seconds_since_last_beat() > _BEAT_ABSENCE_SEC:
            if not self._atmospheric_sent:
                self._atmospheric_sent = True
                await self._commit_intent(LightIntent.ATMOSPHERIC, 0.0)

    async def on_1sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        if self.event_buffer and self.command_queue:
            self.event_buffer.set_timing_log(self.command_queue.get_timing_log())

    async def on_10sec_callback(self):
        if not self.analyser.is_song_playing():
            return
        bpm = int(self.analyser.get_bpm())
        current_second = int(self.analyser.get_song_current_duration().total_seconds())
        intent_name = self._current_intent.name if self._current_intent else 'None'
        logging.info(f'[engine] == current state ==')
        logging.info(f'[engine]   realtime_bpm:    {bpm}')
        logging.info(f'[engine]   intent:          {intent_name}')
        logging.info(f'[engine]   current_second:  {current_second}')
        logging.info(f'[engine]   last_effect:     {self.effect_controller.last_effect}')
        self._log_chain_latency()
