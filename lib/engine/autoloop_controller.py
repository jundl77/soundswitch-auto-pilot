import logging
import random
import datetime
from typing import Optional, Tuple
from lib.clients.spotify_client import SpotifyTrackAnalysis, SpotifyAudioSection
from lib.clients.midi_client import MidiClient, AutoloopAction, ColorOverrides

APPLY_COLOR_OVERRIDE_INTERVAL_SEC = 60

# trigger the auto-loop change 1sec before the event because it takes some time for the change to take effect
FIXED_CHANGE_OFFSET_SEC = 1.0


class AutoloopController:
    def __init__(self, midi_client: MidiClient):
        self.midi_client: MidiClient = midi_client
        self.current_section_index: int = -1
        self.last_color_override: int = -1
        self.last_color_override_time: datetime.datetime = datetime.datetime.now()

    async def check_autoloops(self, current_second: float, spotify_track_analysis: Optional[SpotifyTrackAnalysis]):
        if not spotify_track_analysis:
            return

        section_index, audio_section = self._find_current_audio_section_index(current_second, spotify_track_analysis)
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
            await self._choose_new_autoloop()

    def reset_state(self):
        self.current_section_index = -1

    async def _choose_new_autoloop(self):
        await self.midi_client.set_autoloop(AutoloopAction.NEXT_AUTOLOOP)
        await self._apply_color_override_if_due()

    async def _apply_color_override_if_due(self):
        now = datetime.datetime.now()
        time_diff = now - self.last_color_override_time
        if time_diff < datetime.timedelta(seconds=APPLY_COLOR_OVERRIDE_INTERVAL_SEC):
            logging.info(f'[autoloop_controller] set color override in the last {APPLY_COLOR_OVERRIDE_INTERVAL_SEC} sec,'
                         f' will set it again in {APPLY_COLOR_OVERRIDE_INTERVAL_SEC - time_diff.seconds} sec')
            await self.midi_client.clear_color_overrides()
            return

        # make sure we don't set the color override to the same color as last time
        new_color = random.randrange(1, len(ColorOverrides), 1)
        while new_color == self.last_color_override:
            new_color = random.randrange(1, len(ColorOverrides), 1)

        logging.info(f'[autoloop_controller] setting color override to color: {new_color}')
        await self.midi_client.set_color_override(ColorOverrides[new_color])
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

