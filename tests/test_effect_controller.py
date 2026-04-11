import pytest
from lib.engine.effect_definitions import (
    LightIntent,
    EffectType,
    INTENT_EFFECTS,
)


def test_select_new_random_effect_avoids_previous(effect_controller):
    pool = INTENT_EFFECTS[LightIntent.DROP]
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
async def test_change_effect_atmospheric_uses_bank2(effect_controller):
    await effect_controller.change_effect(LightIntent.ATMOSPHERIC)
    channel = effect_controller.last_effect.midi_channel.name
    assert 'BANK_2' in channel


@pytest.mark.asyncio
async def test_change_effect_groove_uses_bank2(effect_controller):
    await effect_controller.change_effect(LightIntent.GROOVE)
    channel = effect_controller.last_effect.midi_channel.name
    assert 'BANK_2' in channel


@pytest.mark.asyncio
async def test_change_effect_buildup_uses_bank1(effect_controller):
    await effect_controller.change_effect(LightIntent.BUILDUP)
    channel = effect_controller.last_effect.midi_channel.name
    assert 'BANK_1' in channel


@pytest.mark.asyncio
async def test_change_effect_peak_uses_bank1(effect_controller):
    await effect_controller.change_effect(LightIntent.PEAK)
    channel = effect_controller.last_effect.midi_channel.name
    assert 'BANK_1' in channel


@pytest.mark.asyncio
async def test_change_effect_drop_can_produce_special_or_autoloop(effect_controller):
    # DROP pool has BANK_1D, BANK_1E (autoloop) and STROBE (special effect)
    seen_types = set()
    for _ in range(50):
        effect_controller.reset_state()
        await effect_controller.change_effect(LightIntent.DROP)
        # last_effect is only set for autoloop; last_special_effect for special
        seen_types.add(effect_controller.last_effect.type)
    assert EffectType.AUTOLOOP in seen_types


@pytest.mark.asyncio
async def test_reset_state_clears_last_effect(effect_controller):
    await effect_controller.change_effect(LightIntent.BUILDUP)
    effect_controller.reset_state()
    assert effect_controller.last_effect.midi_channel.name == 'AUTOLOOP_BANK_1A'
