import pytest
from lib.engine.effect_definitions import (INTENT_EFFECTS, SECTION_CLASS_INTENTS,
                                           Effect, EffectSource, EffectType,
                                           LightIntent, intent_for_class)
from lib.clients.midi_message import MidiChannel
from lib.label_space import SECTION_LABELS


def test_groove_is_gone_because_no_path_can_produce_it():
    assert not hasattr(LightIntent, 'GROOVE')
    assert 'groove' not in {intent.value for intent in LightIntent}


def test_grooves_banks_move_into_breakdown_rather_than_going_dark():
    channels = [effect.midi_channel.name
                for effect in INTENT_EFFECTS[LightIntent.BREAKDOWN]]
    assert channels == ['AUTOLOOP_BANK_2C', 'AUTOLOOP_BANK_2D',
                        'AUTOLOOP_BANK_2E', 'AUTOLOOP_BANK_2F',
                        'AUTOLOOP_BANK_2G', 'AUTOLOOP_BANK_2H']


def test_every_intent_the_show_can_enter_has_a_pool():
    assert set(INTENT_EFFECTS) == set(LightIntent)


def test_every_vocabulary_class_has_an_intent_and_nothing_else_does():
    assert tuple(SECTION_CLASS_INTENTS) == SECTION_LABELS


def test_the_five_classes_the_show_already_played_are_untouched():
    assert {label: SECTION_CLASS_INTENTS[label]
            for label in ('intro', 'buildup', 'breakdown', 'drop', 'outro')} == {
        'intro': LightIntent.ATMOSPHERIC,
        'outro': LightIntent.ATMOSPHERIC,
        'buildup': LightIntent.BUILDUP,
        'breakdown': LightIntent.BREAKDOWN,
        'drop': LightIntent.DROP,
    }
    assert intent_for_class('drop') is LightIntent.DROP


def test_the_four_new_classes_carry_the_provisional_owner_pending_mapping():
    assert {label: SECTION_CLASS_INTENTS[label]
            for label in ('altintro', 'bridge', 'cooldown', 'altoutro')} == {
        'altintro': LightIntent.ATMOSPHERIC,
        'bridge': LightIntent.BREAKDOWN,
        'cooldown': LightIntent.BREAKDOWN,
        'altoutro': LightIntent.ATMOSPHERIC,
    }


def test_cooldown_lands_on_the_intent_that_inherited_grooves_banks():
    # `cooldown` is GROOVE's documented semantic home and GROOVE's pool folded
    # into BREAKDOWN under D7, so this is that ruling followed through.
    assert SECTION_CLASS_INTENTS['cooldown'] is LightIntent.BREAKDOWN
    assert MidiChannel.AUTOLOOP_BANK_2F in {
        effect.midi_channel for effect in INTENT_EFFECTS[LightIntent.BREAKDOWN]}


def test_a_class_the_map_does_not_know_is_refused_rather_than_defaulted():
    with pytest.raises(KeyError, match='chorus'):
        intent_for_class('chorus')


def test_the_intent_alphabet_is_exactly_the_image_of_the_class_map():
    assert set(LightIntent) == set(SECTION_CLASS_INTENTS.values())
    assert not hasattr(LightIntent, 'PEAK')


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
