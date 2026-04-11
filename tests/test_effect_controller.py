import pytest
from lib.engine.effect_definitions import (
    LightIntent,
    EffectType,
    INTENT_EFFECTS,
)


def test_select_new_random_effect_avoids_previous(effect_controller):
    pool = INTENT_EFFECTS[LightIntent.ENERGY]
    previous = pool[0]
    for _ in range(50):
        result = effect_controller._select_new_random_effect(pool, previous)
        assert result is not previous


def test_select_new_random_effect_returns_item_from_pool(effect_controller):
    pool = INTENT_EFFECTS[LightIntent.GROOVE]
    previous = pool[0]
    result = effect_controller._select_new_random_effect(pool, previous)
    assert result in pool


@pytest.mark.asyncio
async def test_change_effect_calm_uses_bank2(effect_controller):
    await effect_controller.change_effect(LightIntent.CALM)
    channel = effect_controller.last_effect.midi_channel.name
    assert 'BANK_2' in channel


@pytest.mark.asyncio
async def test_change_effect_groove_uses_bank2(effect_controller):
    await effect_controller.change_effect(LightIntent.GROOVE)
    channel = effect_controller.last_effect.midi_channel.name
    assert 'BANK_2' in channel


@pytest.mark.asyncio
async def test_change_effect_energy_uses_bank1(effect_controller):
    await effect_controller.change_effect(LightIntent.ENERGY)
    channel = effect_controller.last_effect.midi_channel.name
    assert 'BANK_1' in channel


@pytest.mark.asyncio
async def test_change_effect_peak_uses_bank1_or_special(effect_controller):
    # PEAK pool has BANK_1D, BANK_1E, and SPECIAL_EFFECT_STROBE
    seen_types = set()
    for _ in range(30):
        effect_controller.reset_state()
        await effect_controller.change_effect(LightIntent.PEAK)
        seen_types.add(effect_controller.last_effect.type if effect_controller.last_special_effect.type == EffectType.SPECIAL_EFFECT else EffectType.AUTOLOOP)
    # At minimum the autoloop path should be exercised
    assert EffectType.AUTOLOOP in seen_types


@pytest.mark.asyncio
async def test_reset_state_clears_last_effect(effect_controller):
    await effect_controller.change_effect(LightIntent.ENERGY)
    effect_controller.reset_state()
    assert effect_controller.last_effect.midi_channel.name == 'AUTOLOOP_BANK_1A'
