import socket
from struct import *
import logging
from lib.clients.overlay_definitions import OverlayEffect, OverlayDefinition, OVERLAY_EFFECTS

OVERLAY_PROTOCOL_ID = 0x7799
UDP_IP = "127.0.0.1"
UDP_PORT = 19001

MAX_NUM_DMX_DEVICES = 20
DMX_UNIVERSE = 0


class DmxFrame:
    def __init__(self):
        self.data: list = [0] * 512

    def pack(self) -> bytes:
        msg = pack('B', self.data[0])
        for i in range(1, 512):
            msg += pack('B', self.data[i])
        return msg


class DmxOverlay:
    def __init__(self, start, length):
        self.start: int = start
        self.length: int = length
        self.active: bool = True
        self.original_length: int = length

    def set_active(self, active: bool):
        self.active = active
        if not self.active:
            self.length = 0
        else:
            self.length = self.original_length

    def is_active(self):
        return self.active

    def pack(self) -> bytes:
        return pack('HH', self.start, self.length)


class DmxOverlays:
    def __init__(self):
        self.overlays: list[DmxOverlay] = [DmxOverlay(0, 0)] * MAX_NUM_DMX_DEVICES
        self.current_index: int = -1
        self.dmx_frame: DmxFrame = DmxFrame()

    def add_overlay(self, start: int, length: int, dmx_data: list[int]):
        assert self.current_index < MAX_NUM_DMX_DEVICES, "full"
        assert start + length <= 512, f"start + length has to be < 512"
        self.current_index += 1
        self.overlays[self.current_index] = DmxOverlay(start, length)
        self.update_overlay_data(self.current_index, dmx_data)
        return self.current_index

    def update_overlay_data(self, index: int, dmx_data: list[int]):
        overlay: DmxOverlay = self.overlays[index]
        start, length = overlay.start, overlay.length
        assert len(dmx_data) == length, "data length and length were not the same"
        for i in range(start, start + length):
            self.dmx_frame.data[i] = dmx_data[i - start]

    def activate_overlay(self, index: int):
        self.overlays[index].set_active(True)

    def deactivate_overlay(self, index: int):
        self.overlays[index].set_active(False)

    def toggle_overlay(self, index: int):
        if self.overlays[index].is_active():
            self.overlays[index].set_active(False)
        else:
            self.overlays[index].set_active(True)

    def get_num_overlays(self):
        return self.current_index + 1

    def pack(self) -> bytes:
        msg = self.overlays[0].pack()
        for i in range(1, MAX_NUM_DMX_DEVICES):
            msg += self.overlays[i].pack()
        msg += self.dmx_frame.pack()
        return msg


class OverlayClient:
    def __init__(self):
        self.socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.overlays: DmxOverlays = DmxOverlays()
        self.effects_to_overlay_index: dict[OverlayEffect, int] = {}

    def start(self):
        logging.info(f'[overlay] starting overlay client, adding all effects')
        for effect, definition in OVERLAY_EFFECTS.items():
            self._add_overlay(effect, definition)
        self.deactivate_all()

    def update_overlay_data(self, effect: OverlayEffect, dmx_data: list[int]):
        assert effect in self.effects_to_overlay_index, f"effect {effect.name} does not exists"
        index: int = self.effects_to_overlay_index[effect]
        self.overlays.update_overlay_data(index, dmx_data)
        self._send_message()

    def toggle_overlay(self, effect: OverlayEffect):
        assert effect in self.effects_to_overlay_index, f"effect {effect.name} does not exists"
        index: int = self.effects_to_overlay_index[effect]
        self.overlays.toggle_overlay(index)
        self._send_message()

    def activate_overlay(self, effect: OverlayEffect):
        assert effect in self.effects_to_overlay_index, f"effect {effect.name} does not exists"
        index: int = self.effects_to_overlay_index[effect]
        self.overlays.activate_overlay(index)
        self._send_message()

    def deactivate_overlay(self, effect: OverlayEffect):
        assert effect in self.effects_to_overlay_index, f"effect {effect.name} does not exists"
        index: int = self.effects_to_overlay_index[effect]
        self.overlays.deactivate_overlay(index)
        self._send_message()

    def deactivate_all(self):
        for effect in OVERLAY_EFFECTS.keys():
            self.deactivate_overlay(effect)

    def _add_overlay(self, effect: OverlayEffect, definition: OverlayDefinition):
        assert effect not in self.effects_to_overlay_index, f"effect {effect.name} already exists"
        index = self.overlays.add_overlay(definition.start_offset, len(definition.dmx_data), definition.dmx_data)
        self.effects_to_overlay_index[effect] = index
        self._send_message()
        logging.info(f'[overlay] added overlay effect: {effect.name}')

    def _send_message(self):
        msg: bytes = self._build_message(DMX_UNIVERSE, self.overlays)
        self.socket.sendto(msg, (UDP_IP, UDP_PORT))

    def _build_message(self, universe: int, overlays: DmxOverlays) -> bytes:
        msg = pack('=IbH', OVERLAY_PROTOCOL_ID, universe, overlays.get_num_overlays())
        msg += overlays.pack()
        return msg
