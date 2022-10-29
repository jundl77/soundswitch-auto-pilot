from typing import List
from lib.clients.midi_message import MidiChannel


COLOR_OVERRIDES: List[MidiChannel] = [
    MidiChannel.COLOR_OVERRIDE_1,
    MidiChannel.COLOR_OVERRIDE_2,
    MidiChannel.COLOR_OVERRIDE_3,
    MidiChannel.COLOR_OVERRIDE_4,
    MidiChannel.COLOR_OVERRIDE_5,
    MidiChannel.COLOR_OVERRIDE_6,
    MidiChannel.COLOR_OVERRIDE_7,
    MidiChannel.COLOR_OVERRIDE_8,
    MidiChannel.COLOR_OVERRIDE_9,
]
SPECIAL_EFFECTS: List[MidiChannel] = [
    MidiChannel.SPECIAL_EFFECT_STROBE,
    MidiChannel.STATIC_LOOK_1,
    MidiChannel.STATIC_LOOK_2,
    MidiChannel.STATIC_LOOK_3,
]
LOW_INTENSITY_AUTOLOOPS: List[MidiChannel] = [
    MidiChannel.AUTOLOOP_BANK_2A,
    MidiChannel.AUTOLOOP_BANK_2B,
    MidiChannel.AUTOLOOP_BANK_2C,
    MidiChannel.AUTOLOOP_BANK_2D,
    MidiChannel.AUTOLOOP_BANK_2E,
    MidiChannel.AUTOLOOP_BANK_2F,
    MidiChannel.AUTOLOOP_BANK_2G,
    MidiChannel.AUTOLOOP_BANK_2H,
]
MEDIUM_INTENSITY_AUTOLOOPS: List[MidiChannel] = [
    MidiChannel.AUTOLOOP_BANK_2A,
    MidiChannel.AUTOLOOP_BANK_2B,
    MidiChannel.AUTOLOOP_BANK_2C,
    MidiChannel.AUTOLOOP_BANK_2D,
    MidiChannel.AUTOLOOP_BANK_2E,
    MidiChannel.AUTOLOOP_BANK_2F,
    MidiChannel.AUTOLOOP_BANK_2G,
    MidiChannel.AUTOLOOP_BANK_2H,
]
HIGH_INTENSITY_AUTOLOOPS: List[MidiChannel] = [
    MidiChannel.AUTOLOOP_BANK_1A,
    MidiChannel.AUTOLOOP_BANK_1B,
    MidiChannel.AUTOLOOP_BANK_1C,
    MidiChannel.AUTOLOOP_BANK_1D,
    MidiChannel.AUTOLOOP_BANK_1E,
    MidiChannel.AUTOLOOP_BANK_1F,
    MidiChannel.AUTOLOOP_BANK_1G,
    MidiChannel.AUTOLOOP_BANK_1H,
]
HIP_HOP_AUTOLOOPS: List[MidiChannel] = [
    MidiChannel.AUTOLOOP_BANK_1A,
    MidiChannel.AUTOLOOP_BANK_1B,
    MidiChannel.AUTOLOOP_BANK_1C,
    MidiChannel.AUTOLOOP_BANK_1D,
    MidiChannel.AUTOLOOP_BANK_1E,
    MidiChannel.AUTOLOOP_BANK_1F,
    MidiChannel.AUTOLOOP_BANK_1G,
    MidiChannel.AUTOLOOP_BANK_1H,
]
