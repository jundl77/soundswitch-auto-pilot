from rtmidi.midiconstants import NOTE_ON, NOTE_OFF, CONTROLLER_CHANGE

# MIDI SOUNDSWITCH CHANNEL DEFINITIONS
MIDI_LINK_CHANNEL = 1
MIDI_BPM_TAP_CHANNEL = 2
MIDI_AUTOLOOP_INTENSITY_CHANNEL = 3
MIDI_SCRIPTED_TRACK_INTENSITY_CHANNEL = 4
MIDI_GROUP_1_INTENSITY_CHANNEL = 5
MIDI_GROUP_2_INTENSITY_CHANNEL = 6
MIDI_GROUP_3_INTENSITY_CHANNEL = 7
MIDI_GROUP_4_INTENSITY_CHANNEL = 8
MIDI_PLAY_PAUSE_CHANNEL = 9


# MIDI SOUNDSWITCH STATIC MESSAGE DEFINITIONS
MIDI_MSG_LINK_TOGGLE = [NOTE_ON, MIDI_LINK_CHANNEL, 100]
MIDI_MSG_BPM_TAP_ON = [NOTE_ON, MIDI_BPM_TAP_CHANNEL, 100]
MIDI_MSG_BPM_TAP_OFF = [NOTE_OFF, MIDI_BPM_TAP_CHANNEL, 0]
MIDI_MSG_PLAY = [NOTE_ON, MIDI_PLAY_PAUSE_CHANNEL, 0]
MIDI_MSG_PAUSE = [NOTE_OFF, MIDI_PLAY_PAUSE_CHANNEL, 0]


# MIDI SOUNDSWITCH DYNAMIC MESSAGE DEFINITIONS

def get_autoloop_intensity_msg(value: float):
    """
    set the autoloop intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MIDI_AUTOLOOP_INTENSITY_CHANNEL, 127 * value]


def get_scripted_track_intensity_msg(value: float):
    """
    set the scripted track intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MIDI_SCRIPTED_TRACK_INTENSITY_CHANNEL, 127 * value]


def get_group_1_intensity_msg(value: float):
    """
    set the group 1 intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MIDI_GROUP_1_INTENSITY_CHANNEL, 127 * value]


def get_group_2_intensity_msg(value: float):
    """
    set the group 2 intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MIDI_GROUP_2_INTENSITY_CHANNEL, 127 * value]


def get_group_3_intensity_msg(value: float):
    """
    set the group 3 intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MIDI_GROUP_3_INTENSITY_CHANNEL, 127 * value]


def get_group_4_intensity_msg(value: float):
    """
    set the group 4 intensity value, the real value is bound to [0, 127] but I remap it to [0, 100] for simplicity
    """
    assert 0 <= value <= 1, "intensity value should be in [0, 1]"
    return [CONTROLLER_CHANGE, MIDI_GROUP_4_INTENSITY_CHANNEL, 127 * value]
