import pytest
from lib.engine.effect_definitions import Effect, EffectType, EffectSource
from lib.clients.midi_message import MidiChannel


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
    assert len(pool) == 2  # e1 and e2 have equal value — deduplicated


def test_effect_not_equal_to_non_effect():
    e = Effect(EffectType.AUTOLOOP, EffectSource.MIDI, MidiChannel.AUTOLOOP_BANK_1A)
    assert e != "not an effect"
    assert e != 42
    assert e != None
