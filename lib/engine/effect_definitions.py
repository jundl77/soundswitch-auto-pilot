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


# PROVISIONAL for the four classes the vocabulary gained -- owner-pending.  The
# real mapping is a product decision to be taken against measured per-class
# energy and position profiles, not intuition, and this is the one dict that
# changes when it is.  `cooldown` goes to BREAKDOWN because that is where
# GROOVE's banks went under D7, so its documented semantic home still plays
# GROOVE's lights.
SECTION_CLASS_INTENTS: Dict[str, 'LightIntent'] = {
    'intro':     LightIntent.ATMOSPHERIC,
    'altintro':  LightIntent.ATMOSPHERIC,
    'buildup':   LightIntent.BUILDUP,
    'breakdown': LightIntent.BREAKDOWN,
    'bridge':    LightIntent.BREAKDOWN,
    'drop':      LightIntent.DROP,
    'cooldown':  LightIntent.BREAKDOWN,
    'outro':     LightIntent.ATMOSPHERIC,
    'altoutro':  LightIntent.ATMOSPHERIC,
}


def intent_for_class(label: str) -> 'LightIntent':
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
}
