import logging
from typing import Optional, Tuple
from lib.clients.spotify_client import SpotifyTrackAnalysis, SpotifyAudioSection
from lib.clients.midi_client import MidiClient, AutoloopAction

FIXED_START_OFFSET_SEC = 0.5


class AutoloopController:
    def __init__(self, midi_client: MidiClient):
        self.current_section_index: int = -1
        self.midi_client: MidiClient = midi_client

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
            logging.info(f'[autoloop_controler] audio section change detected, section_start={audio_section.section_start_sec} sec, duration={audio_section.section_duration_sec} sec')
            await self._choose_new_autoloop()

    def reset_state(self):
        self.current_section_index = -1

    async def _choose_new_autoloop(self):
        await self.midi_client.set_autoloop(AutoloopAction.NEXT_AUTOLOOP)

    def _find_current_audio_section_index(self,
                                          current_second: float,
                                          spotify_track_analysis: SpotifyTrackAnalysis) -> Tuple[int, Optional[SpotifyAudioSection]]:
        section_index = 0
        for audio_section in spotify_track_analysis.audio_sections:
            section_start_sec = audio_section.section_start_sec - FIXED_START_OFFSET_SEC
            section_duration_sec = audio_section.section_duration_sec - FIXED_START_OFFSET_SEC
            if section_start_sec <= current_second < section_duration_sec:
                return section_index, audio_section
            else:
                section_index += 1
        return -1, None
