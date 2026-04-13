#! /usr/bin/env python3.9

import argparse
import argcomplete
import logging
import asyncio
import signal
import datetime
from collections import deque

BUFFER_SIZE = 256
SAMPLE_RATE = 44100
# Look-ahead delay: analysis runs LOOK_AHEAD_SEC ahead of what the audience hears.
# Must match playback_delay_seconds in dmx-enttec-node/app_audio_receiver/audio_receiver.json.
# Audio played back locally (debug mode) is held in a FIFO buffer of the same duration
# so that local monitoring stays in sync with the analysis.
LOOK_AHEAD_SEC = 2.5
_AUDIO_DELAY_FRAMES = round(LOOK_AHEAD_SEC * SAMPLE_RATE / BUFFER_SIZE)  # ≈ 431 buffers
_SILENCE = None  # initialised lazily as np.zeros(BUFFER_SIZE, dtype=np.float32)

logging.basicConfig(format='%(asctime)s [%(levelname)s ] %(message)s', level=logging.INFO)
global_app = None


class SoundSwitchAutoPilot:
    def __init__(self,
                 midi_port_index: int,
                 input_device_index: int = None,
                 output_device_index: int = None,
                 debug_mode: bool = False,
                 show_visualizer: bool = False,
                 disable_os2l: bool = False,
                 enable_ui: bool = False,
                 ui_port: int = 8050,
                 report_path: str | None = None):
        # import here to avoid loading expensive dependencies during arg parsing
        from lib.clients.pyaudio_client import PyAudioClient
        from lib.clients.midi_client import MidiClient
        from lib.clients.os2l_client import Os2lClient
        from lib.clients.overlay_client import OverlayClient
        from lib.analyser.music_analyser import MusicAnalyser
        from lib.visualizer.visualizer import Visualizer, VisualizerUpdater
        from lib.engine.light_engine import LightEngine
        from lib.engine.effect_controller import EffectController
        from lib.engine.delayed_command_queue import DelayedCommandQueue

        self.debug_mode: bool = debug_mode
        self.show_visualizer: bool = show_visualizer
        self.disable_os2l: bool = disable_os2l
        self.enable_ui: bool = enable_ui
        self._ui_port: int = ui_port
        self._report_path: str | None = report_path
        self.is_running: bool = False
        self.loop = asyncio.get_event_loop()
        self.command_queue: DelayedCommandQueue = DelayedCommandQueue(LOOK_AHEAD_SEC)
        logging.info(f'[main] look-ahead delay: {LOOK_AHEAD_SEC:.2f}s — ensure dmx-enttec-node playback_delay_seconds matches')

        # construct clients
        self.audio_client: PyAudioClient = PyAudioClient(SAMPLE_RATE, BUFFER_SIZE, input_device_index, output_device_index)
        self.midi_client: MidiClient = MidiClient(midi_port_index)
        self.os2l_client: Os2lClient = Os2lClient()
        self.overlay_client: OverlayClient = OverlayClient()

        # construct visualizer
        if self.show_visualizer:
            self.visualizer: Visualizer = Visualizer(show_freq=False,
                                                     show_freq_gradient=False,
                                                     show_energy=False,
                                                     show_freq_curve=True,
                                                     show_ssm=False)
            self.visualizer_updater: VisualizerUpdater = VisualizerUpdater()
        else:
            self.visualizer: Visualizer = None
            self.visualizer_updater: VisualizerUpdater = None

        # construct event buffer (for --ui and/or --report)
        from lib.engine.event_buffer import EventBuffer
        self.event_buffer: EventBuffer | None = EventBuffer() if (enable_ui or report_path) else None

        # construct engine
        self.effect_controller: EffectController = EffectController(self.midi_client, event_buffer=self.event_buffer)
        self.light_engine: LightEngine = LightEngine(self.midi_client, self.os2l_client, self.overlay_client,
                                                     self.effect_controller,
                                                     self.command_queue, event_buffer=self.event_buffer,
                                                     look_ahead_sec=LOOK_AHEAD_SEC)

        # construct analyser
        self.music_analyser: MusicAnalyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, self.light_engine, self.visualizer_updater)
        self.light_engine.set_analyser(self.music_analyser)
        self.os2l_client.set_analyser(self.music_analyser)

    def list_devices(self):
        self.audio_client.list_devices()
        self.midi_client.list_devices()

    async def run(self):
        logging.info("[main] setting up auto pilot..")
        self.audio_client.start_streams(start_stream_out=self.debug_mode)
        self.midi_client.start()
        self.overlay_client.start()
        self.music_analyser.start()
        if self.disable_os2l:
            logging.info("[main] OS2L is disabled")
        else:
            self.os2l_client.start()
        if self.show_visualizer:
            self.visualizer.show()
            self.visualizer_updater.connect()
        if self.event_buffer is not None:
            self.event_buffer.start()
            import threading
            from simulate.visualizer_app import run_app
            ui_thread = threading.Thread(
                target=run_app,
                args=(self.event_buffer, self._ui_port),
                daemon=True,
            )
            ui_thread.start()
            logging.info(f'[main] visualizer started → http://localhost:{self._ui_port}')
        self.is_running = True

        logging.info("[main] auto pilot is ready, starting")

        last_100ms_callback_execution: datetime.datetime = datetime.datetime.now()
        last_1sec_callback_execution: datetime.datetime = datetime.datetime.now()
        last_10sec_callback_execution: datetime.datetime = datetime.datetime.now()
        # FIFO buffer that delays local audio playback by LOOK_AHEAD_SEC so debug
        # monitoring (headphones) stays in sync with the classified-and-delayed lights.
        import numpy as np
        global _SILENCE
        _SILENCE = np.zeros(BUFFER_SIZE, dtype=np.float32)

        audio_delay_buf: deque = deque()
        _audio_playback_started = False

        while self.is_running:
            now = datetime.datetime.now()
            audio_signal = self.audio_client.read()
            new_audio_signal = await self.music_analyser.analyse(audio_signal)
            await self.command_queue.drain()

            if self.audio_client.support_output():
                audio_delay_buf.append(new_audio_signal)
                if len(audio_delay_buf) > _AUDIO_DELAY_FRAMES:
                    if not _audio_playback_started:
                        logging.info('[main] audio delay buffer full, starting delayed playback')
                        _audio_playback_started = True
                    self.audio_client.play(audio_delay_buf.popleft())
                else:
                    # Feed silence while the delay buffer fills to keep the output
                    # stream alive — a starved PyAudio blocking stream on macOS/CoreAudio
                    # can enter a dead state and never recover.
                    self.audio_client.play(_SILENCE)

            if now - last_100ms_callback_execution > datetime.timedelta(milliseconds=100):
                last_100ms_callback_execution = now
                await self._do_100ms_callback()

            if now - last_1sec_callback_execution > datetime.timedelta(seconds=1):
                last_1sec_callback_execution = now
                await self._do_1s_callback()

            if now - last_10sec_callback_execution > datetime.timedelta(seconds=10):
                last_10sec_callback_execution = now
                await self._do_10s_callback()

        self.audio_client.close()
        self.os2l_client.stop()
        self.midi_client.stop()
        self.overlay_client.stop()
        if self.show_visualizer:
            self.visualizer.stop()
            self.visualizer_updater.stop()
        logging.info("[main] auto pilot stopped, clean shutdown")

    def stop(self):
        self.is_running = False
        self.os2l_client.stop()
        if self.show_visualizer:
            self.visualizer.stop()
            self.visualizer_updater.stop()

    async def _do_100ms_callback(self):
        await self.light_engine.on_100ms_callback()
        await self.midi_client.on_100ms_callback()

    async def _do_1s_callback(self):
        await self.light_engine.on_1sec_callback()

    async def _do_10s_callback(self):
        await self.light_engine.on_10sec_callback()


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
                                      debug_mode=debug_mode,
                                      show_visualizer=args.visualizer,
                                      disable_os2l=args.no_os2l,
                                      enable_ui=args.ui,
                                      ui_port=args.ui_port,
                                      report_path=args.report)

    await global_app.run()

    if args.report and global_app.event_buffer is not None:
        import json
        from simulate.evaluator import evaluate, print_evaluation
        report = global_app.event_buffer.to_report()
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logging.info(f'[main] report written → {args.report}')
        print_evaluation(evaluate(report))


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
    subparser.add_argument('-d', '--debug', help='Run in debug mode, this will playback audio on the output device with additional auditory information', required=False, action='store_true')
    subparser.add_argument('-v', '--visualizer', help='Display the visualizer, which shows auditory information, such as a spectogram', required=False, action='store_true')
    subparser.add_argument('--no-os2l', help='Disable OS2L (connection to SoundSwitch). This can be useful for debugging.', required=False, action='store_true')
    subparser.add_argument('--ui', help='Launch real-time lighting visualizer (requires dash extra)', required=False, action='store_true')
    subparser.add_argument('--ui-port', type=int, default=8050, help='Visualizer Dash server port (default: 8050)', required=False, dest='ui_port')
    subparser.add_argument('--report', default=None, help='Write a JSON session report on exit (e.g. report.json); implies event tracking', required=False)
    subparser.set_defaults(func=run_cmd)

    from simulate.cli import add_simulate_subparser
    add_simulate_subparser(subparsers)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return

    await args.func(args)

