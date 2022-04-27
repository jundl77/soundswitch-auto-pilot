from rtmidi.midiconstants import NOTE_ON, NOTE_OFF

# define midi protocol for soundswitch
MIDI_MSG_LINK_TOGGLE = [NOTE_ON, 1, 100]
MIDI_MSG_LINK_BPM_TAP_ON = [NOTE_ON, 2, 100]
MIDI_MSG_LINK_BPM_TAP_OFF = [NOTE_OFF, 2, 0]