from typing import List, Dict
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


class LightIntent(Enum):
    ATMOSPHERIC = 'atmospheric'
    BREAKDOWN   = 'breakdown'
    BUILDUP     = 'buildup'
    DROP        = 'drop'
    PEAK        = 'peak'


# The decoder's class space -> what the rig does about it (D7).  ATMOSPHERIC
# takes both quiet classes because an intent cannot know where in the track it
# is; PEAK is absent because it is a run length, not a class.
SECTION_CLASS_INTENTS: Dict[str, 'LightIntent'] = {
    'intro':     LightIntent.ATMOSPHERIC,
    'outro':     LightIntent.ATMOSPHERIC,
    'buildup':   LightIntent.BUILDUP,
    'breakdown': LightIntent.BREAKDOWN,
    'drop':      LightIntent.DROP,
}


def intent_for_class(label: str) -> 'LightIntent':
    """Raises on an unmapped class rather than defaulting to a plausible look.

    The class space is whatever the priors file names, so a retrained model with
    a class nobody wired must stop the show's construction, not light breakdown
    for it and let the mistake read as a taste question.
    """
    try:
        return SECTION_CLASS_INTENTS[label]
    except KeyError:
        raise KeyError(
            f"the decoder committed {label!r}, which no LightIntent claims; "
            f"known classes are {', '.join(sorted(SECTION_CLASS_INTENTS))}"
        ) from None


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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Effect):
            return NotImplemented
        return (self.type == other.type
                and self.source == other.source
                and self.midi_channel == other.midi_channel
                and self.overlay_effect == other.overlay_effect)

    def __hash__(self) -> int:
        return hash((self.type, self.source, self.midi_channel, self.overlay_effect))

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

INTENT_EFFECTS: Dict[LightIntent, List[Effect]] = {
    LightIntent.ATMOSPHERIC: [
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2A),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2B),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2C),
    ],
    # Six banks, because GROOVE's three moved here rather than going dark when
    # the class space lost the intent that owned them (D7).
    LightIntent.BREAKDOWN: [
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2C),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2D),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2E),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2F),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2G),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_2H),
    ],
    LightIntent.BUILDUP: [
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1A),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1B),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1C),
    ],
    LightIntent.DROP: [
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1D),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1E),
        Effect(type=EffectType.SPECIAL_EFFECT, source=EffectSource.MIDI, midi_channel=MidiChannel.SPECIAL_EFFECT_STROBE),
    ],
    LightIntent.PEAK: [
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1F),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1G),
        Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1H),
    ],
}
