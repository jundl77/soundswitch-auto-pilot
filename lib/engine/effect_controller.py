import logging
import random
import datetime
from typing import Optional, Tuple, List
from lib.clients.spotify_client import SpotifyTrackAnalysis, SpotifyAudioSection, LightShowType
from lib.clients.midi_client import MidiClient
from lib.clients.midi_message import MidiChannel
from lib.engine.effect_definitions import EffectSource, OverlayEffect, EffectType, Effect,\
    COLOR_OVERRIDES, SPECIAL_EFFECTS, LOW_INTENSITY_EFFECTS, MEDIUM_INTENSITY_EFFECTS, HIGH_INTENSITY_EFFECTS, HIP_HOP_EFFECTS

# trigger the auto-loop change 1sec before the event because it takes some time for the change to take effect
FIXED_CHANGE_OFFSET_SEC = 1.0
APPLY_COLOR_OVERRIDE_INTERVAL_SEC = 60 * 5


class EffectController:
    def __init__(self, midi_client: MidiClient):
        self.midi_client: MidiClient = midi_client
        self.current_section_index: int = -1
        self.last_audio_section: Optional[SpotifyAudioSection] = None
        self.last_effect: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1A)
        self.last_special_effect: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.SPECIAL_EFFECT_STROBE)
        self.last_color_override: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_1)
        self.last_color_override_time: datetime.datetime = datetime.datetime.now()

    async def change_effect(self, current_second: float, track_analysis: Optional[SpotifyTrackAnalysis]):
        if not track_analysis:
            return

        section_index, audio_section = self._find_current_audio_section_index(current_second, track_analysis)
        if audio_section is None:
            # something has gone wrong in section detection, just reset - likely we switched songs
            self.reset_state()
            return

        self.current_section_index = section_index
        logging.info(f'[effect_controller] audio section change detected,'
                     f' section_start={audio_section.section_start_sec:.2f} sec,'
                     f' duration={audio_section.section_duration_sec:.2f} sec,'
                     f' change_offset={(FIXED_CHANGE_OFFSET_SEC * -1.0):.2f} sec')
        await self._choose_new_effect(track_analysis, audio_section, self.last_audio_section)
        self.last_audio_section = audio_section

    def reset_state(self):
        self.current_section_index: int = -1
        self.last_audio_section: Optional[SpotifyAudioSection] = None
        self.last_effect: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1A)
        self.last_special_effect: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.SPECIAL_EFFECT_STROBE)
        self.last_color_override: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_1)
        self.last_color_override_time: datetime.datetime = datetime.datetime.now()

    def update_audio_section(self, current_second: float, track_analysis: Optional[SpotifyTrackAnalysis]):
        self.reset_state()
        section_index, audio_section = self._find_current_audio_section_index(current_second, track_analysis)
        self.current_section_index: int = section_index
        self.last_audio_section: Optional[SpotifyAudioSection] = audio_section

    async def _choose_new_effect(self,
                                 track_analysis: SpotifyTrackAnalysis,
                                 current_audio_section: SpotifyAudioSection,
                                 last_audio_section: Optional[SpotifyAudioSection]):
        if track_analysis.light_show_type == LightShowType.LOW_INTENSITY:
            new_effect: Effect = self._select_new_random_effect(LOW_INTENSITY_EFFECTS, self.last_effect)
        elif track_analysis.light_show_type == LightShowType.MEDIUM_INTENSITY:
            new_effect: Effect = self._select_new_random_effect(MEDIUM_INTENSITY_EFFECTS, self.last_effect)
        elif track_analysis.light_show_type == LightShowType.HIGH_INTENSITY:
            new_effect: Effect = await self._choose_new_effect_high_intensity(track_analysis, current_audio_section, last_audio_section)
        elif track_analysis.light_show_type == LightShowType.HIP_HOP:
            new_effect: Effect = self._select_new_random_effect(HIP_HOP_EFFECTS, self.last_effect)
        else:
            raise RuntimeError(f"unknown LightShowType {track_analysis.light_show_type}")

        if new_effect.type == EffectType.SPECIAL_EFFECT:
            await self._apply_special_effect(new_effect)
        elif new_effect.type == EffectType.AUTOLOOP:
            await self._apply_autoloop(new_effect)

    async def _choose_new_effect_high_intensity(self,
                                                track_analysis: SpotifyTrackAnalysis,
                                                current_audio_section: SpotifyAudioSection,
                                                last_audio_section: Optional[SpotifyAudioSection]):
        track_loudness: float = track_analysis.loudness
        current_section_loudness: float = current_audio_section.section_loudness
        section_to_track_ratio: float = track_loudness / current_section_loudness

        if last_audio_section:
            last_section_loudness = last_audio_section.section_loudness
            to_last_section_ratio = last_section_loudness / current_section_loudness
            if to_last_section_ratio > 1.25:
                logging.info("[effect_controller] detected high-energy section, selecting high-intensity effect")
                return self._select_new_random_effect(SPECIAL_EFFECTS, self.last_special_effect)
            if to_last_section_ratio < 0.7:
                logging.info("[effect_controller] detected slow section, selecting low-intensity effect")
                return self._select_new_random_effect(LOW_INTENSITY_EFFECTS, self.last_effect)

        if section_to_track_ratio < 0.7:
            return self._select_new_random_effect(LOW_INTENSITY_EFFECTS, self.last_effect)

        return self._select_new_random_effect(HIGH_INTENSITY_EFFECTS, self.last_effect)

    async def _apply_special_effect(self, special_effect: Effect):
        assert special_effect.type == EffectType.SPECIAL_EFFECT
        effect_name = None
        if special_effect.source == EffectSource.MIDI:
            effect_name = special_effect.midi_channel.name
            await self.midi_client.set_special_effect(special_effect.midi_channel, duration_sec=30)
        elif special_effect.source == EffectSource.OVERLAY:
            effect_name = special_effect.overlay_effect.name
        logging.info(f'[effect_controller] applying special effect {effect_name}, type={special_effect.type.name}')
        self.last_special_effect = special_effect

    async def _apply_autoloop(self, autoloop: Effect):
        assert autoloop.type == EffectType.AUTOLOOP
        assert autoloop.source == EffectSource.MIDI
        logging.info(f'[effect_controller] changing autoloop to {autoloop.midi_channel.name}')
        await self.midi_client.set_autoloop(autoloop.midi_channel)
        await self._apply_color_override_if_due()
        self.last_effect = autoloop

    async def _apply_color_override_if_due(self):
        now = datetime.datetime.now()
        time_diff = now - self.last_color_override_time
        if time_diff < datetime.timedelta(seconds=APPLY_COLOR_OVERRIDE_INTERVAL_SEC):
            logging.info(f'[effect_controller] set color override in the last {APPLY_COLOR_OVERRIDE_INTERVAL_SEC} sec,'
                         f' will set it again in {APPLY_COLOR_OVERRIDE_INTERVAL_SEC - time_diff.seconds} sec')
            await self.midi_client.clear_color_overrides()
            return

        new_color: Effect = self._select_new_random_effect(COLOR_OVERRIDES, self.last_color_override)
        logging.info(f'[effect_controller] setting color override to color: {new_color.midi_channel.name}')
        await self.midi_client.set_color_override(new_color.midi_channel)
        self.last_color_override = new_color
        self.last_color_override_time = now

    def _find_current_audio_section_index(self,
                                          current_second: float,
                                          spotify_track_analysis: SpotifyTrackAnalysis) -> Tuple[int, Optional[SpotifyAudioSection]]:
        for i, audio_section in enumerate(spotify_track_analysis.audio_sections):
            section_start_sec = audio_section.section_start_sec - FIXED_CHANGE_OFFSET_SEC
            section_end_sec = section_start_sec + audio_section.section_duration_sec

            if not (section_start_sec <= current_second < section_end_sec):
                continue

            time_to_next_section_sec = section_end_sec - current_second
            time_since_section_start_sec = current_second - section_start_sec
            next_section_exists: bool = i < len(spotify_track_analysis.audio_sections) - 1

            # we are actually much closer to the next section, e.g. section goes from 8sec to 25sec, and we are at 23sec
            # i.e. we detected the section change a bit early
            if next_section_exists and time_to_next_section_sec < time_since_section_start_sec and time_to_next_section_sec < 5:
                return i + 1, spotify_track_analysis.audio_sections[i + 1]

            if section_start_sec <= current_second < section_end_sec:
                return i, audio_section
        return -1, None

    def _select_new_random_effect(self, effects: List[Effect], previous_effect: Effect) -> Effect:
        # make sure we don't select the same channel as last time
        i: int = random.randrange(0, len(effects), 1)
        while effects[i] == previous_effect:
            i = random.randrange(0, len(effects), 1)
        return effects[i]
