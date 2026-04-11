import pytest
from lib.engine.effect_definitions import (
    EffectType,
    HIGH_INTENSITY_EFFECTS,
    LOW_INTENSITY_EFFECTS,
    SPECIAL_EFFECTS,
)
from tests.conftest import make_section, make_track_analysis


def test_select_new_random_effect_avoids_previous(effect_controller):
    previous = HIGH_INTENSITY_EFFECTS[0]
    for _ in range(50):
        result = effect_controller._select_new_random_effect(HIGH_INTENSITY_EFFECTS, previous)
        assert result is not previous


def test_select_new_random_effect_returns_item_from_list(effect_controller):
    previous = HIGH_INTENSITY_EFFECTS[0]
    result = effect_controller._select_new_random_effect(HIGH_INTENSITY_EFFECTS, previous)
    assert result in HIGH_INTENSITY_EFFECTS


def test_find_section_at_midpoint(effect_controller):
    sections = [
        make_section(0.0, 30.0),
        make_section(30.0, 30.0),
        make_section(60.0, 30.0),
    ]
    analysis = make_track_analysis(audio_sections=sections)
    # With FIXED_CHANGE_OFFSET_SEC=1.0: section 1 effective range = [29, 59)
    idx, section = effect_controller._find_current_audio_section_index(35.0, analysis)
    assert idx == 1
    assert section is sections[1]


def test_find_section_returns_minus_one_outside_range(effect_controller):
    sections = [make_section(0.0, 30.0)]
    analysis = make_track_analysis(audio_sections=sections)
    idx, section = effect_controller._find_current_audio_section_index(200.0, analysis)
    assert idx == -1
    assert section is None


async def test_high_intensity_jump_to_louder_picks_special_effects(effect_controller):
    # ratio = last / current = -8.0 / -4.0 = 2.0 > 1.25 → SPECIAL_EFFECTS
    current = make_section(30.0, 30.0, loudness=-4.0)
    last = make_section(0.0, 30.0, loudness=-8.0)
    analysis = make_track_analysis(loudness=-6.0)
    result = await effect_controller._choose_new_effect_high_intensity(analysis, current, last)
    assert result.type == EffectType.SPECIAL_EFFECT


async def test_high_intensity_drop_to_quieter_picks_low_intensity(effect_controller):
    # ratio = last / current = -4.0 / -8.0 = 0.5 < 0.7 → LOW_INTENSITY_EFFECTS
    current = make_section(30.0, 30.0, loudness=-8.0)
    last = make_section(0.0, 30.0, loudness=-4.0)
    analysis = make_track_analysis(loudness=-6.0)
    result = await effect_controller._choose_new_effect_high_intensity(analysis, current, last)
    assert result in LOW_INTENSITY_EFFECTS


async def test_high_intensity_no_last_section_uses_track_ratio(effect_controller):
    # No last section: section_to_track_ratio = track_loudness / section_loudness
    current = make_section(0.0, 30.0, loudness=-8.0)
    analysis = make_track_analysis(loudness=-6.0)
    # ratio = -6 / -8 = 0.75 — not < 0.7, so picks HIGH_INTENSITY_EFFECTS
    result = await effect_controller._choose_new_effect_high_intensity(analysis, current, None)
    assert result.type == EffectType.AUTOLOOP
