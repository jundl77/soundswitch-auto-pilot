#! /usr/bin/env python3.9

import argparse
import argcomplete
import logging
import asyncio
import signal

BUFFER_SIZE = 512
SAMPLE_RATE = 44100

logging.basicConfig(format='%(asctime)s [%(levelname)s ] %(message)s', level=logging.INFO)
global_app = None


class SoundSwitchAutoPilot:
    def __init__(self,
                 midi_port_index: int,
                 input_device_index: int = None,
                 output_device_index: int = None,
                 debug_mode: bool = False):
        # import here to avoid loading expensive dependencies during arg parsing
        from lib.clients.pyaudio_client import PyAudioClient
        from lib.clients.midi_client import MidiClient
        from lib.clients.os2l_client import Os2lClient
        from lib.analyser.music_analyser import MusicAnalyser
        from lib.engine.light_engine import LightEngine

        self.debug_mode: bool = debug_mode
        self.is_running: bool = False
        self.loop = asyncio.get_event_loop()
        self.audio_client: PyAudioClient = PyAudioClient(SAMPLE_RATE, BUFFER_SIZE, input_device_index, output_device_index)
        self.midi_client: MidiClient = MidiClient(midi_port_index)
        self.os2l_client: Os2lClient = Os2lClient()
        self.light_engine: LightEngine = LightEngine(self.midi_client, self.os2l_client)
        self.music_analyser: MusicAnalyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, self.light_engine)
        self.light_engine.set_analyser(self.music_analyser)
        self.os2l_client.set_analyser(self.music_analyser)

    def list_devices(self):
        self.audio_client.list_devices()
        self.midi_client.list_devices()

    async def run(self):
        self.audio_client.start_streams(start_stream_out=self.debug_mode)
        self.midi_client.start()
        self.os2l_client.start()
        self.is_running = True

        logging.info("auto pilot is ready, starting")
        while self.is_running:
            audio_signal = self.audio_client.read()
            new_audio_signal = await self.music_analyser.analyse(audio_signal)

            if self.audio_client.support_output():
                self.audio_client.play(new_audio_signal)
        self.audio_client.close()
        self.os2l_client.stop()
        self.midi_client.stop()
        logging.info("auto pilot stopped, clean shutdown")

    def stop(self):
        self.is_running = False
        self.os2l_client.stop()


async def run_cmd(args: argparse.Namespace):
    global global_app
    if args.debug:
        print('starting in debug mode')
        debug_mode = True
    else:
        debug_mode = False

    midi_port_index: int = int(args.midi_port_index)
    input_device_index = int(args.input_device) if args.input_device is not None else None
    output_device_index = int(args.output_device) if args.output_device is not None else None
    global_app = SoundSwitchAutoPilot(midi_port_index=midi_port_index,
                                      input_device_index=input_device_index,
                                      output_device_index=output_device_index,
                                      debug_mode=debug_mode)

    await global_app.run()


async def list_cmd(args: argparse.Namespace):
    app = SoundSwitchAutoPilot(0)
    app.list_devices()


def death_handler(signum, frame):
    if global_app is not None:
        logging.info('[DEATH] caught signal "SIGINT/SIGTERM", stopping')
        global_app.stop()


signal.signal(signal.SIGINT, death_handler)
signal.signal(signal.SIGTERM, death_handler)


async def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Functionality to start')

    subparser = subparsers.add_parser('list', help='List all available sound and midi devices')
    subparser.set_defaults(func=list_cmd)

    subparser = subparsers.add_parser('run', help='Create the specified instance')
    subparser.add_argument('midi_port_index', help='The midi port index of the midi device to use. Available devices are shown by running \'list\'')
    subparser.add_argument('-i', '--input_device', help='Specify the index of the sound INPUT device to use, uses system-default by default', required=False, default=None)
    subparser.add_argument('-o', '--output_device', help='Specify the index of the sound OUTPUT device to use, uses system-default by default', required=False, default=None)
    subparser.add_argument('-d', '--debug', help='Run in debug mode, this will playback audio on the output device with '
                                                 'additional auditory information', required=False, action='store_true')
    subparser.set_defaults(func=run_cmd)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return

    await args.func(args)

