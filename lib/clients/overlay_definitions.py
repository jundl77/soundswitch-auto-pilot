from enum import Enum


class OverlayEffect(Enum):
    UV_LIGHT = 1


class OverlayDefinition:
    def __init__(self, start_offset: int, dmx_data: list[int]):
        """ start_offset is 0-indexed """
        assert 0 < start_offset + len(dmx_data) <= 512
        self.start_offset: int = start_offset
        self.dmx_data: list[int] = dmx_data


OVERLAY_EFFECTS = {
    OverlayEffect.UV_LIGHT: OverlayDefinition(64, [0, 0, 0, 0, 255, 0, 79, 0] * 4)
}
