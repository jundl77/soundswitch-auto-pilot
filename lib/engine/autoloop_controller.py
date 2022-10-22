import logging
import random
import datetime
from typing import Optional, Tuple, List
from lib.clients.spotify_client import SpotifyTrackAnalysis, SpotifyAudioSection, LightShowType
from lib.clients.midi_client import MidiClient
from lib.clients.midi_message import MidiChannel
from lib.engine.autoloop_actions import COLOR_OVERRIDES, LOW_INTENSITY_AUTOLOOPS, MEDIUM_INTENSITY_AUTOLOOPS, HIGH_INTENSITY_AUTOLOOPS, HIP_HOP_AUTOLOOPS

# trigger the auto-loop change 1sec before the event because it takes some time for the change to take effect
FIXED_CHANGE_OFFSET_SEC = 1.0
APPLY_COLOR_OVERRIDE_INTERVAL_SEC = 60


class AutoloopController:
    def __init__(self, midi_client: MidiClient):
        self.midi_client: MidiClient = midi_client
        self.current_section_index: int = -1
        self.last_audio_section: Optional[SpotifyAudioSection] = None
        self.last_autoloop: MidiChannel = MidiChannel.AUTOLOOP_BANK_1A
        self.last_color_override: MidiChannel = MidiChannel.COLOR_OVERRIDE_1
        self.last_color_override_time: datetime.datetime = datetime.datetime.now()

    async def check_autoloops(self, current_second: float, track_analysis: Optional[SpotifyTrackAnalysis]):
        if not track_analysis:
            return

        section_index, audio_section = self._find_current_audio_section_index(current_second, track_analysis)
        if audio_section is None:
            # something has gone wrong in section detection, just reset - likely we switched songs
            self.reset_state()
            return

        if section_index != self.current_section_index:
            self.current_section_index = section_index
            logging.info(f'[autoloop_controller] audio section change detected,'
                         f' section_start={audio_section.section_start_sec:.2f} sec,'
                         f' duration={audio_section.section_duration_sec:.2f} sec,'
                         f' change_offset={(FIXED_CHANGE_OFFSET_SEC * -1.0):.2f} sec')
            await self._choose_new_autoloop(track_analysis, audio_section, self.last_audio_section)
            self.last_audio_section = audio_section

    def reset_state(self):
        self.current_section_index: int = -1
        self.last_audio_section: Optional[SpotifyAudioSection] = None
        self.last_autoloop: MidiChannel = MidiChannel.AUTOLOOP_BANK_1A
        self.last_color_override: MidiChannel = MidiChannel.COLOR_OVERRIDE_1
        self.last_color_override_time: datetime.datetime = datetime.datetime.now()

    async def _choose_new_autoloop(self,
                                   track_analysis: SpotifyTrackAnalysis,
                                   current_audio_section: SpotifyAudioSection,
                                   last_audio_section: Optional[SpotifyAudioSection]):

        if track_analysis.light_show_type == LightShowType.LOW_INTENSITY:
            new_autoloop: MidiChannel = self._select_new_random_channel(LOW_INTENSITY_AUTOLOOPS, self.last_autoloop)
        elif track_analysis.light_show_type == LightShowType.MEDIUM_INTENSITY:
            new_autoloop: MidiChannel = self._select_new_random_channel(MEDIUM_INTENSITY_AUTOLOOPS, self.last_autoloop)
        elif track_analysis.light_show_type == LightShowType.HIGH_INTENSITY:
            new_autoloop: MidiChannel = await self._choose_new_autoloop_high_itensity(track_analysis, current_audio_section, last_audio_section)
        elif track_analysis.light_show_type == LightShowType.HIP_HOP:
            new_autoloop: MidiChannel = self._select_new_random_channel(HIP_HOP_AUTOLOOPS, self.last_autoloop)
        else:
            raise RuntimeError(f"unknown LightShowType {track_analysis.light_show_type}")

        logging.info(f'[autoloop_controller] changing auto-loop to: {new_autoloop.name}')
        await self.midi_client.set_autoloop(new_autoloop)
        await self._apply_color_override_if_due()
        self.last_autoloop = new_autoloop

    async def _choose_new_autoloop_high_itensity(self,
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
                await self.midi_client.set_special_effect(MidiChannel.SPECIAL_EFFECT_STROBE, duration_sec=10)
                return self._select_new_random_channel(HIGH_INTENSITY_AUTOLOOPS, self.last_autoloop)
            if to_last_section_ratio < 0.7:
                return self._select_new_random_channel(LOW_INTENSITY_AUTOLOOPS, self.last_autoloop)

        if section_to_track_ratio < 0.7:
            return self._select_new_random_channel(LOW_INTENSITY_AUTOLOOPS, self.last_autoloop)

        return self._select_new_random_channel(HIGH_INTENSITY_AUTOLOOPS, self.last_autoloop)

    async def _apply_color_override_if_due(self):
        now = datetime.datetime.now()
        time_diff = now - self.last_color_override_time
        if time_diff < datetime.timedelta(seconds=APPLY_COLOR_OVERRIDE_INTERVAL_SEC):
            logging.info(f'[autoloop_controller] set color override in the last {APPLY_COLOR_OVERRIDE_INTERVAL_SEC} sec,'
                         f' will set it again in {APPLY_COLOR_OVERRIDE_INTERVAL_SEC - time_diff.seconds} sec')
            await self.midi_client.clear_color_overrides()
            return

        new_color: MidiChannel = self._select_new_random_channel(COLOR_OVERRIDES, self.last_color_override)
        logging.info(f'[autoloop_controller] setting color override to color: {new_color.name}')
        await self.midi_client.set_color_override(new_color)
        self.last_color_override = new_color
        self.last_color_override_time = now

    def _find_current_audio_section_index(self,
                                          current_second: float,
                                          spotify_track_analysis: SpotifyTrackAnalysis) -> Tuple[int, Optional[SpotifyAudioSection]]:
        section_index = 0
        for audio_section in spotify_track_analysis.audio_sections:
            section_start_sec = audio_section.section_start_sec - FIXED_CHANGE_OFFSET_SEC
            section_end_sec = section_start_sec + audio_section.section_duration_sec
            if section_start_sec <= current_second < section_end_sec:
                return section_index, audio_section
            else:
                section_index += 1
        return -1, None

    def _select_new_random_channel(self, channels: List[MidiChannel], previous_channel: MidiChannel) -> MidiChannel:
        # make sure we don't select the same channel as last time
        new_channel: int = random.randrange(0, len(channels), 1)
        while channels[new_channel] == previous_channel:
            new_channel = random.randrange(0, len(channels), 1)
        return channels[new_channel]
