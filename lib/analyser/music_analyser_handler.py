import logging
from abc import ABC, abstractmethod


class IMusicAnalyserHandler(ABC):

    @abstractmethod
    def on_sound_start(self):
        pass

    @abstractmethod
    def on_sound_stop(self):
        pass

    @abstractmethod
    async def on_cycle(self):
        pass

    @abstractmethod
    async def on_onset(self):
        pass

    @abstractmethod
    async def on_beat(self, beat_number: int, bpm: float, bpm_changed: bool) -> None:
        pass

    @abstractmethod
    async def on_note(self):
        pass

    @abstractmethod
    async def on_section_change(self) -> None:
        pass
