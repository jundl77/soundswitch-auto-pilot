import pytest
from lib.engine.effect_definitions import (INTENT_EFFECTS, SECTION_CLASS_INTENTS,
                                           Effect, EffectSource, EffectType,
                                           LightIntent, intent_for_class)
from lib.clients.midi_message import MidiChannel


def test_groove_is_gone_because_no_path_can_produce_it():
    """D7.  The model's class space is (intro, buildup, breakdown, drop, outro).

    An enum member the show can never enter is a lie the visualizer's legend
    would keep advertising, so it is removed rather than left unreachable.
    """
    assert not hasattr(LightIntent, 'GROOVE')
    assert 'groove' not in {intent.value for intent in LightIntent}


def test_grooves_banks_move_into_breakdown_rather_than_going_dark():
    """The pool is three physical looks, not a label.

    Deleting the intent would retire BANK_2F/G/H from the rig; folding them in
    keeps six banks rotating behind the class the corpus actually labels.
    """
    channels = [effect.midi_channel.name
                for effect in INTENT_EFFECTS[LightIntent.BREAKDOWN]]
    assert channels == ['AUTOLOOP_BANK_2C', 'AUTOLOOP_BANK_2D',
                        'AUTOLOOP_BANK_2E', 'AUTOLOOP_BANK_2F',
                        'AUTOLOOP_BANK_2G', 'AUTOLOOP_BANK_2H']


def test_every_intent_the_show_can_enter_has_a_pool():
    assert set(INTENT_EFFECTS) == set(LightIntent)


def test_the_class_space_maps_onto_intents_and_nothing_is_unmapped():
    """intro/outro -> ATMOSPHERIC: an intent cannot know where in the track it
    is, and quiet at the start and quiet at the end are the same look."""
    assert SECTION_CLASS_INTENTS == {
        'intro': LightIntent.ATMOSPHERIC,
        'outro': LightIntent.ATMOSPHERIC,
        'buildup': LightIntent.BUILDUP,
        'breakdown': LightIntent.BREAKDOWN,
        'drop': LightIntent.DROP,
    }
    assert intent_for_class('drop') is LightIntent.DROP


def test_a_class_the_map_does_not_know_is_refused_rather_than_defaulted():
    """A silent default would light a plausible look for a class nobody wired.

    The decoder's class space comes from the priors file, so a retrained model
    with a sixth class must fail here rather than show breakdown for it.
    """
    with pytest.raises(KeyError, match='cooldown'):
        intent_for_class('cooldown')


def test_peak_is_reachable_only_by_promotion():
    """D8.  No class maps to PEAK -- it is "a committed DROP that has lasted",
    which is a temporal property the class space cannot express."""
    assert LightIntent.PEAK not in SECTION_CLASS_INTENTS.values()


def test_effect_equal_same_values():
    e1 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    e2 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    assert e1 == e2


def test_effect_not_equal_different_channel():
    e1 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    e2 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1B)
    assert e1 != e2


def test_effect_not_equal_different_type():
    e1 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    e2 = Effect(EffectType.SPECIAL_EFFECT, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    assert e1 != e2


def test_effect_hash_equal_for_equal_objects():
    e1 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    e2 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    assert hash(e1) == hash(e2)


def test_effect_usable_in_set():
    e1 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    e2 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    e3 = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1B)
    pool = {e1, e2, e3}
    assert len(pool) == 2


def test_effect_not_equal_to_non_effect():
    e = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    assert e != "not an effect"
    assert e != 42
    assert e != None
