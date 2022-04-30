#! /usr/bin/env python3.9

import argparse
from clients.pyaudio_client import PyAudioClient
from clients.midi_client import MidiClient
from analyser.music_analyser import MusicAnalyser
from analyser.music_analyser_handler import MusicAnalyserHandler

BUFFER_SIZE = 512
SAMPLE_RATE = 44100


class SoundSwitchAutoPilot:
    def __init__(self, input_device_index: int = None, output_device_index: int = None, debug_mode: bool = False):
        self.debug_mode: bool = debug_mode
        self.audio_client: PyAudioClient = PyAudioClient(SAMPLE_RATE, BUFFER_SIZE, input_device_index, output_device_index)
        self.midi_client: MidiClient = MidiClient()
        self.handler: MusicAnalyserHandler = MusicAnalyserHandler(self.midi_client)
        self.music_analyser: MusicAnalyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, self.handler)
        self.handler.set_analyser(self.music_analyser)

    def list_devices(self):
        self.audio_client.list_devices()

    def run(self):
        self.audio_client.start_streams(start_stream_out=self.debug_mode)
        while True:
            audio_signal = self.audio_client.read()
            new_audio_signal = self.music_analyser.analyse(audio_signal)

            if self.audio_client.support_output():
                self.audio_client.play(new_audio_signal)
        self.audio_client.close()


def main(args):
    if args.debug:
        print('starting in debug mode')
        debug_mode = True
    else:
        debug_mode = False
    input_device_index = int(args.input_device) if args.input_device is not None else None
    output_device_index = int(args.output_device) if args.output_device is not None else None
    app = SoundSwitchAutoPilot(input_device_index=input_device_index,
                               output_device_index=output_device_index,
                               debug_mode=debug_mode)

    if args.list_devices:
        app.list_devices()
    else:
        app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--list-devices', help='List all available devices', required=False, action='store_true')
    parser.add_argument('-i', '--input_device', help='Specify the index of the sound INPUT device to use', required=False, default=None)
    parser.add_argument('-o', '--output_device', help='Specify the index of the sound OUTPUT device to use', required=False, default=None)
    parser.add_argument('-d', '--debug', help='Run in debug mode, this will playback audio on the output device with '
                                              'additional auditory information', required=False, action='store_true')

    main(parser.parse_args())
