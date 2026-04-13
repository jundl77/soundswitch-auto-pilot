from __future__ import annotations
import logging
import random
import datetime
from typing import TYPE_CHECKING
from lib.clients.midi_client import MidiClient
from lib.clients.midi_message import MidiChannel
from lib.engine.effect_definitions import (
    EffectSource, EffectType, Effect, LightIntent,
    COLOR_OVERRIDES, SPECIAL_EFFECTS, INTENT_EFFECTS,
)

if TYPE_CHECKING:
    from lib.engine.event_buffer import EventBuffer

# Trigger the autoloop change 1 sec before the event so SoundSwitch has time to settle.
APPLY_COLOR_OVERRIDE_INTERVAL_SEC = 60 * 5


class EffectController:
    def __init__(self, midi_client: MidiClient, event_buffer: EventBuffer | None = None):
        self.midi_client: MidiClient = midi_client
        self.event_buffer: EventBuffer | None = event_buffer
        self.last_effect: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1A)
        self.last_special_effect: Effect = Effect(type=EffectType.SPECIAL_EFFECT, source=EffectSource.MIDI, midi_channel=MidiChannel.SPECIAL_EFFECT_STROBE)
        self.last_color_override: Effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_1)
        self.last_color_override_time: datetime.datetime = datetime.datetime.now()

    async def process_effects(self):
        pass

    async def change_effect(self, intent: LightIntent) -> None:
        """Select and apply a MIDI effect for the given intent.

        Picks a random channel from the intent's pool (never the same as last time).
        For PEAK intent the pool includes a strobe special effect.
        """
        logging.info(f'[effect_controller] change_effect called, intent={intent.name}')
        pool = INTENT_EFFECTS[intent]
        new_effect = self._select_new_random_effect(pool, self.last_effect)

        if new_effect.type == EffectType.SPECIAL_EFFECT:
            await self._apply_special_effect(new_effect)
        else:
            await self._apply_autoloop(new_effect)

    def reset_state(self):
        self.last_effect = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.AUTOLOOP_BANK_1A)
        self.last_special_effect = Effect(type=EffectType.SPECIAL_EFFECT, source=EffectSource.MIDI, midi_channel=MidiChannel.SPECIAL_EFFECT_STROBE)
        self.last_color_override = Effect(type=EffectType.AUTOLOOP, source=EffectSource.MIDI, midi_channel=MidiChannel.COLOR_OVERRIDE_1)
        self.last_color_override_time = datetime.datetime.now()

    async def _apply_special_effect(self, special_effect: Effect):
        assert special_effect.type == EffectType.SPECIAL_EFFECT
        effect_name = None
        if special_effect.source == EffectSource.MIDI:
            effect_name = special_effect.midi_channel.name
            await self.midi_client.set_special_effect(special_effect.midi_channel, duration_sec=30)
        elif special_effect.source == EffectSource.OVERLAY:
            effect_name = special_effect.overlay_effect.name
        logging.info(f'[effect_controller] applying special effect {effect_name}, type={special_effect.type.name}')
        if self.event_buffer and effect_name:
            self.event_buffer.add_effect(effect_name, 'SPECIAL_EFFECT')
        self.last_special_effect = special_effect

    async def _apply_autoloop(self, autoloop: Effect):
        assert autoloop.type == EffectType.AUTOLOOP
        assert autoloop.source == EffectSource.MIDI
        logging.info(f'[effect_controller] changing autoloop to {autoloop.midi_channel.name}')
        await self.midi_client.set_autoloop(autoloop.midi_channel)
        if self.event_buffer:
            self.event_buffer.add_effect(autoloop.midi_channel.name, 'AUTOLOOP')
        await self._apply_color_override_if_due()
        self.last_effect = autoloop

    async def _apply_color_override_if_due(self):
        now = datetime.datetime.now()
        time_diff = now - self.last_color_override_time
        if time_diff < datetime.timedelta(seconds=APPLY_COLOR_OVERRIDE_INTERVAL_SEC):
            logging.info(f'[effect_controller] set color override in the last {APPLY_COLOR_OVERRIDE_INTERVAL_SEC} sec,'
                         f' will set it again in {APPLY_COLOR_OVERRIDE_INTERVAL_SEC - time_diff.seconds} sec')
            await self.midi_client.clear_color_overrides()
            return

        new_color: Effect = self._select_new_random_effect(COLOR_OVERRIDES, self.last_color_override)
        logging.info(f'[effect_controller] setting color override to color: {new_color.midi_channel.name}')
        await self.midi_client.set_color_override(new_color.midi_channel)
        self.last_color_override = new_color
        self.last_color_override_time = now

    def _select_new_random_effect(self, effects: list[Effect], previous_effect: Effect) -> Effect:
        assert len(effects) > 1, f'effect pool must have >1 entries to avoid an infinite loop (got {len(effects)})'
        i: int = random.randrange(0, len(effects), 1)
        while effects[i] == previous_effect:
            i = random.randrange(0, len(effects), 1)
        return effects[i]
