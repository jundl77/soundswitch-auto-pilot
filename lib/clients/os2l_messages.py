
def logon_message() -> str:
    return """
{"evt":"subscribed","trigger":"deck 1 get_text '%SOUNDSWITCH_ID'","value":""}
{"evt":"subscribed","trigger":"deck 2 get_text '%SOUNDSWITCH_ID'","value":""}
{"evt":"subscribed","trigger":"deck 3 get_text '%SOUNDSWITCH_ID'","value":""}
{"evt":"subscribed","trigger":"deck 4 get_text '%SOUNDSWITCH_ID'","value":""}
{"evt":"subscribed","trigger":"deck 1 level","value":1}
{"evt":"subscribed","trigger":"deck 2 level","value":1}
{"evt":"subscribed","trigger":"deck 3 level","value":1}
{"evt":"subscribed","trigger":"deck 4 level","value":1}
{"evt":"subscribed","trigger":"crossfader","value":0.5}
{"evt":"subscribed","trigger":"deck 1 get_bpm","value":120}
{"evt":"subscribed","trigger":"deck 2 get_bpm","value":120}
{"evt":"subscribed","trigger":"deck 3 get_bpm","value":120}
{"evt":"subscribed","trigger":"deck 4 get_bpm","value":120}
{"evt":"subscribed","trigger":"deck 1 play","value":"off"}
{"evt":"subscribed","trigger":"deck 2 play","value":"off"}
{"evt":"subscribed","trigger":"deck 3 play","value":"off"}
{"evt":"subscribed","trigger":"deck 4 play","value":"off"}
{"evt":"subscribed","trigger":"deck 1 loop","value":"off"}
{"evt":"subscribed","trigger":"deck 2 loop","value":"off"}
{"evt":"subscribed","trigger":"deck 3 loop","value":"off"}
{"evt":"subscribed","trigger":"deck 4 loop","value":"off"}
{"evt":"subscribed","trigger":"deck 1 get_loop","value":0.5}
{"evt":"subscribed","trigger":"deck 2 get_loop","value":0.5}
{"evt":"subscribed","trigger":"deck 3 get_loop","value":0.5}
{"evt":"subscribed","trigger":"deck 4 get_loop","value":0.5}
{"evt":"subscribed","trigger":"deck 1 loop_roll 0.03125 ? constant 0.03125 : deck 1 loop_roll 0.0625 ? constant 0.0625 : deck 1 loop_roll 0.125 ? constant 0.125 : deck 1 loop_roll 0.25 ? constant 0.25 : deck 1 loop_roll 0.5 ? constant 0.5 : deck 1 loop_roll 0.75 ? constant 0.75 : deck 1 loop_roll 1 ? constant 1 : deck 1 loop_roll 2 ? constant 2 : deck 1 loop_roll 4 ? constant 4 : constant 0","value":0}
{"evt":"subscribed","trigger":"deck 2 loop_roll 0.03125 ? constant 0.03125 : deck 2 loop_roll 0.0625 ? constant 0.0625 : deck 2 loop_roll 0.125 ? constant 0.125 : deck 2 loop_roll 0.25 ? constant 0.25 : deck 2 loop_roll 0.5 ? constant 0.5 : deck 2 loop_roll 0.75 ? constant 0.75 : deck 2 loop_roll 1 ? constant 1 : deck 2 loop_roll 2 ? constant 2 : deck 2 loop_roll 4 ? constant 4 : constant 0","value":0}
{"evt":"subscribed","trigger":"deck 3 loop_roll 0.03125 ? constant 0.03125 : deck 3 loop_roll 0.0625 ? constant 0.0625 : deck 3 loop_roll 0.125 ? constant 0.125 : deck 3 loop_roll 0.25 ? constant 0.25 : deck 3 loop_roll 0.5 ? constant 0.5 : deck 3 loop_roll 0.75 ? constant 0.75 : deck 3 loop_roll 1 ? constant 1 : deck 3 loop_roll 2 ? constant 2 : deck 3 loop_roll 4 ? constant 4 : constant 0","value":0}
{"evt":"subscribed","trigger":"deck 4 loop_roll 0.03125 ? constant 0.03125 : deck 4 loop_roll 0.0625 ? constant 0.0625 : deck 4 loop_roll 0.125 ? constant 0.125 : deck 4 loop_roll 0.25 ? constant 0.25 : deck 4 loop_roll 0.5 ? constant 0.5 : deck 4 loop_roll 0.75 ? constant 0.75 : deck 4 loop_roll 1 ? constant 1 : deck 4 loop_roll 2 ? constant 2 : deck 4 loop_roll 4 ? constant 4 : constant 0","value":0}
"""


def song_loaded_message(time_elapsed: int, beat_pos: float, first_beat: float, bpm: float) -> str:
    return """
{"evt":"subscribed","trigger":"deck 1 get_text '%SOUNDSWITCH_ID'","value":"{D6516ACF-28B0-4FF2-B553-2D458EE30FF4}"}
""" + """
{"evt":"subscribed","trigger":"deck 1 get_filepath","value":""}
{"evt":"subscribed","trigger":"deck 1 get_time elapsed absolute","value":%d}
{"evt":"subscribed","trigger":"deck 1 get_beatpos","value":%d}
{"evt":"subscribed","trigger":"deck 1 get_firstbeat","value":%d}
{"evt":"subscribed","trigger":"deck 1 get_bpm","value":%d}
""" % (time_elapsed, beat_pos, first_beat, bpm)


def play_start_message():
    return """{"evt":"subscribed","trigger":"deck 1 play","value":"on"}"""


def play_stop_message():
    return """{"evt":"subscribed","trigger":"deck 1 play","value":"off"}"""


def update_message(beat_pos: float, time_elapsed: int):
    return """
{"evt":"subscribed","trigger":"deck 1 get_time elapsed absolute","value":%d}
{"evt":"subscribed","trigger":"deck 1 get_beatpos","value":%d}
""" % (time_elapsed, beat_pos)


def beat_message(change: bool, pos: int, bpm: float, strength: float):
    change_str = 'true' if change else 'false'
    return """{"evt":"beat","change":%s,"pos":%d,"bpm":%d,"strength":%d}""" % (change_str, pos, bpm, strength)