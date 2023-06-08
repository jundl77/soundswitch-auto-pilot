from enum import Enum


class OverlayEffect(Enum):
    EFFECT_1 = 1


class OverlayDefinition:
    def __init__(self, start_offset: int, dmx_data: list[int]):
        assert 0 < start_offset + len(dmx_data) <= 512
        self.start_offset: int = start_offset
        self.dmx_data: list[int] = dmx_data


OVERLAY_EFFECTS = {
    OverlayEffect.EFFECT_1: OverlayDefinition(0, [0] * 10)
}
