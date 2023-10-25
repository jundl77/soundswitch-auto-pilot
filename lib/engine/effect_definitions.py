from typing import List
from enum import Enum
from lib.clients.midi_message import MidiChannel
from lib.clients.overlay_definitions import OverlayEffect


class EffectSource(Enum):
    MIDI = 1
    OVERLAY = 2


class EffectType(Enum):
    SPECIAL_EFFECT = 1
    AUTOLOOP = 2
    COLOR_OVERRIDE = 3


class Effect:
    def __init__(self,
                 type: EffectType,
                 source: EffectSource,
                 midi_channel: MidiChannel = None,
                 overlay_effect: OverlayEffect = None):
        self.type: EffectType = type
        self.source: EffectSource = source
        self.midi_channel: MidiChannel = midi_channel
        self.overlay_effect: OverlayEffect = overlay_effect

    def __str__(self):
        if self.source == EffectSource.MIDI:
            return f"[midi] type={self.type.name} effect={self.midi_channel.name}"
        if self.source == EffectSource.OVERLAY:
            return f"[overlay] type={self.type.name} effect={self.overlay_effect.name}"
        assert False, "unknown effect"


COLOR_OVERRIDES: List[Effect] = [
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_1),
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_2),
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_3),
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_4),
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_5),
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_6),
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_7),
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_8),
    Effect(type=EffectType.COLOR_OVERRIDE, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_9),
]
SPECIAL_EFFECTS: List[Effect] = [
    Effect(type=EffectType.SPECIAL_EFFECT, source=EffectSource.MIDI, midi_channel=MidiChannel.SPECIAL_EFFECT_STROBE),
    Effect(type=EffectType.SPECIAL_EFFECT, source=EffectSource.MIDI, midi_channel=MidiChannel.STATIC_LOOK_1),
    Effect(type=EffectType.SPECIAL_EFFECT, source=EffectSource.MIDI, midi_channel=MidiChannel.STATIC_LOOK_2),
    Effect(type=EffectType.SPECIAL_EFFECT, source=EffectSource.MIDI, midi_channel=MidiChannel.STATIC_LOOK_3),
]
LOW_INTENSITY_EFFECTS: List[Effect] = [
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2A),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2B),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2C),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2D),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2E),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2F),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2G),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2H),
]
MEDIUM_INTENSITY_EFFECTS: List[Effect] = [
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2A),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2B),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2C),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2D),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2E),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2F),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2G),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2H),
]
HIGH_INTENSITY_EFFECTS: List[Effect] = [
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1A),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1B),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1C),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1D),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1E),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1F),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1G),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1H),
]
HIP_HOP_EFFECTS: List[Effect] = [
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1A),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1B),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1C),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1D),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1E),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1F),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1G),
    Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1H),
]
