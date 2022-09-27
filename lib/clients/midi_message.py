from rtmidi.midiconstants import NOTE_ON, NOTE_OFF, CONTROLLER_CHANGE
from enum import Enum


# max is 127 unfortunately, after that SoundSwitch does not respond
class MidiChannel(Enum):
    LINK = 1
    BPM_TAP = 2
    AUTOLOOP_INTENSITY = 3
    SCRIPTED_TRACK_INTENSITY = 4
    GROUP_1_INTENSITY = 5
    GROUP_2_INTENSITY = 6
    GROUP_3_INTENSITY = 7
    GROUP_4_INTENSITY = 8
    PLAY_PAUSE = 9
    NEXT_AUTOLOOP = 10
    
    SPECIAL_EFFECT_MOVEMENT = 11
    SPECIAL_EFFECT_STROBE = 12
    SPECIAL_EFFECT_HUE = 13
    SPECIAL_EFFECT_SMOKE = 14
    SPECIAL_EFFECT_WHITE = 15
    SPECIAL_EFFECT_BLACK_OUT = 16
    SPECIAL_EFFECT_UV = 17

    COLOR_OVERRIDE_1 = 18
    COLOR_OVERRIDE_2 = 19
    COLOR_OVERRIDE_3 = 20
    COLOR_OVERRIDE_4 = 21
    COLOR_OVERRIDE_5 = 22
    COLOR_OVERRIDE_6 = 23
    COLOR_OVERRIDE_7 = 24
    COLOR_OVERRIDE_8 = 25
    COLOR_OVERRIDE_9 = 26
    
    AUTOLOOP_BANK_1A = 27
    AUTOLOOP_BANK_1B = 28
    AUTOLOOP_BANK_1C = 29
    AUTOLOOP_BANK_1D = 30
    AUTOLOOP_BANK_1E = 31
    AUTOLOOP_BANK_1F = 32
    AUTOLOOP_BANK_1G = 33
    AUTOLOOP_BANK_1H = 34
    AUTOLOOP_BANK_2A = 35
    AUTOLOOP_BANK_2B = 36
    AUTOLOOP_BANK_2C = 37
    AUTOLOOP_BANK_2D = 38
    AUTOLOOP_BANK_2E = 39
    AUTOLOOP_BANK_2F = 40
    AUTOLOOP_BANK_2G = 41
    AUTOLOOP_BANK_2H = 42
    AUTOLOOP_BANK_3A = 43
    AUTOLOOP_BANK_3B = 44
    AUTOLOOP_BANK_3C = 45
    AUTOLOOP_BANK_3D = 46
    AUTOLOOP_BANK_3E = 47
    AUTOLOOP_BANK_3F = 48
    AUTOLOOP_BANK_3G = 49
    AUTOLOOP_BANK_3H = 50
    AUTOLOOP_BANK_4A = 51
    AUTOLOOP_BANK_4B = 52
    AUTOLOOP_BANK_4C = 53
    AUTOLOOP_BANK_4D = 54
    AUTOLOOP_BANK_4E = 55
    AUTOLOOP_BANK_4F = 56
    AUTOLOOP_BANK_4G = 57
    AUTOLOOP_BANK_4H = 58

    STATIC_LOOK_1 = 59
    STATIC_LOOK_2 = 60
    STATIC_LOOK_3 = 61
    STATIC_LOOK_4 = 62
    STATIC_LOOK_5 = 63
    STATIC_LOOK_6 = 64
    STATIC_LOOK_7 = 65
    STATIC_LOOK_8 = 66
    STATIC_LOOK_9 = 67
    STATIC_LOOK_10 = 68
    STATIC_LOOK_11 = 69
    STATIC_LOOK_12 = 70
    STATIC_LOOK_13 = 71
    STATIC_LOOK_14 = 72
    STATIC_LOOK_15 = 73
    STATIC_LOOK_16 = 74
    STATIC_LOOK_17 = 75
    STATIC_LOOK_18 = 76
    STATIC_LOOK_19 = 77
    STATIC_LOOK_20 = 78
    STATIC_LOOK_21 = 79
    STATIC_LOOK_22 = 80
    STATIC_LOOK_23 = 81
    STATIC_LOOK_24 = 82
    STATIC_LOOK_25 = 83
    STATIC_LOOK_26 = 84
    STATIC_LOOK_27 = 85
    STATIC_LOOK_28 = 86
    STATIC_LOOK_29 = 87
    STATIC_LOOK_30 = 88
    STATIC_LOOK_31 = 89
    STATIC_LOOK_32 = 90

# use the .__members__ syntax to access all members, otherwise duplicates are coalesced into the first occurrence
ALL_MIDI_CHANNEL_VALUES = [value.value for name, value in MidiChannel.__members__.items()]
assert len(set(ALL_MIDI_CHANNEL_VALUES)) == len(ALL_MIDI_CHANNEL_VALUES), "duplicate values found in MidiChannel enum"


def get_midi_msg_on(channel: MidiChannel):
    """
    get the "press down" action of the midi channel. If the channel controls a button with a toggle function, this can
    also act as a toggle.
    """
    assert isinstance(channel, MidiChannel), f"midi channel '{channel}' is not a valid channel, channel has to be of type MidiChannel"
    return [NOTE_ON, channel.value, 1]


def get_midi_msg_off(channel: MidiChannel):
    """
    get the "press up" action of the midi channel
    """
    assert isinstance(channel, MidiChannel), f"midi channel '{channel}' is not a valid channel, channel has to be of type MidiChannel"
    return [NOTE_OFF, channel.value, 0]


def get_autoloop_intensity_msg(value: float):
    """
    set the autoloop intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MidiChannel.AUTOLOOP_INTENSITY.value, 127 * value]


def get_scripted_track_intensity_msg(value: float):
    """
    set the scripted track intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MidiChannel.SCRIPTED_TRACK_INTENSITY.value, 127 * value]


def get_group_1_intensity_msg(value: float):
    """
    set the group 1 intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MidiChannel.GROUP_1_INTENSITY.value, 127 * value]


def get_group_2_intensity_msg(value: float):
    """
    set the group 2 intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MidiChannel.GROUP_2_INTENSITY.value, 127 * value]


def get_group_3_intensity_msg(value: float):
    """
    set the group 3 intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MidiChannel.GROUP_3_INTENSITY.value, 127 * value]


def get_group_4_intensity_msg(value: float):
    """
    set the group 4 intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MidiChannel.GROUP_4_INTENSITY.value, 127 * value]
