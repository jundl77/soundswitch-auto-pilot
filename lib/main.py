#! /usr/bin/env python3.9

import argparse
import argcomplete
import logging
import asyncio
import signal
import datetime
import time
from collections import deque

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE
# Must match playback_delay_seconds in dmx-enttec-node/app_audio_receiver/audio_receiver.json,
# and simulate/runner.py. 14.0 per #154: the NN chain measures 13.66 s at the
# corpus median bar and 14.08 s at p99, so this is the budget lag_bars=2 needs.
# It is no longer a look-ahead: the engine runs BEHIND the room now, and what a
# command waits is this minus what the chain already spent (B1).
PLAYBACK_DELAY_SEC = 14.0
logging.basicConfig(format='%(asctime)s [%(levelname)s ] %(message)s', level=logging.INFO)
global_app = None


class SoundSwitchAutoPilot:
    def __init__(self,
                 midi_port_index: int,
                 input_device_index: int = None,
                 output_device_index: int = None,
                 debug_mode: bool = False,
                 disable_os2l: bool = False,
                 enable_ui: bool = False,
                 ui_port: int = 8050,
                 report_path: str | None = None):
        # Imported here, not at module scope: hoisting these makes `--help` pay
        # for the whole DSP and model stack.
        from lib.clients.pyaudio_client import PyAudioClient
        from lib.clients.midi_client import MidiClient
        from lib.clients.os2l_client import Os2lClient
        from lib.clients.overlay_client import OverlayClient
        from lib.analyser.music_analyser import MusicAnalyser
        from lib.engine.light_engine import LightEngine
        from lib.engine.effect_controller import EffectController
        from lib.engine.delayed_command_queue import DelayedCommandQueue

        self.debug_mode: bool = debug_mode
        self.disable_os2l: bool = disable_os2l
        self.enable_ui: bool = enable_ui
        self._ui_port: int = ui_port
        self._report_path: str | None = report_path
        self.is_running: bool = False
        self.loop = asyncio.get_event_loop()
        self.command_queue: DelayedCommandQueue = DelayedCommandQueue(PLAYBACK_DELAY_SEC)

        self._enable_playback: bool = debug_mode or output_device_index is not None
        self.audio_client: PyAudioClient = PyAudioClient(SAMPLE_RATE, BUFFER_SIZE, input_device_index, output_device_index)
        self.midi_client: MidiClient = MidiClient(midi_port_index)
        self.os2l_client: Os2lClient = Os2lClient()
        self.overlay_client: OverlayClient = OverlayClient()

        from lib.engine.event_buffer import EventBuffer
        # `--report` promises the whole session, so the rolling window comes off
        # for it the way `simulate.cli._session_buffer` already takes it off:
        # the default 60 s window prunes at 2x itself on every write, so a
        # thirty-minute set reported six intent changes out of sixty and a beat
        # count saturated at the deque cap.  The UI alone keeps the window --
        # it draws 30 s and nothing reads back past it.
        self.event_buffer: EventBuffer | None = (
            EventBuffer(look_ahead_sec=PLAYBACK_DELAY_SEC,
                        window_sec=float('inf') if report_path else 60.0)
            if (enable_ui or report_path) else None
        )

        from lib import section_chain
        from lib.analyser.drift_watchdog import DriftWatchdog

        # One ladder, two inputs: pacing is measured by the analyser and health
        # is reported by the GPU stage, and neither can see what the other does
        # (D3, census 0.4).  Handing it to the chain is also what puts the GPU
        # stage on its own thread.
        self.drift_watchdog: DriftWatchdog = DriftWatchdog(BUFFER_SIZE / SAMPLE_RATE)
        self.section = (section_chain.build_section_chain(watchdog=self.drift_watchdog)
                        if section_chain.artifacts_present() else None)
        if self.section is None:
            logging.warning('[main] no NN artifacts on this machine — the show '
                            'will light the quiet cold-start floor and hold it '
                            '(beats and silence still run)')

        self.effect_controller: EffectController = EffectController(self.midi_client, event_buffer=self.event_buffer)
        self.light_engine: LightEngine = LightEngine(self.midi_client, self.os2l_client, self.overlay_client,
                                                     self.effect_controller,
                                                     self.command_queue, event_buffer=self.event_buffer,
                                                     playback_delay_sec=PLAYBACK_DELAY_SEC,
                                                     section_chain=None if self.section is None else self.section.stream,
                                                     section_decoder=None if self.section is None else self.section.decoder,
                                                     watchdog=self.drift_watchdog)

        self.music_analyser: MusicAnalyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, self.light_engine,
                                                           note_clicks=debug_mode,
                                                           watchdog=self.drift_watchdog)
        self.light_engine.set_analyser(self.music_analyser)
        self.os2l_client.set_analyser(self.music_analyser)

    def list_devices(self):
        self.audio_client.list_devices()
        self.midi_client.list_devices()

    async def run(self):
        """Set up, run the loop, and shut down -- the last one unconditionally.

        Every client is opened before the loop and every one of them holds
        something that outlives this coroutine: a PyAudio stream, a lit rig, a
        CUDA context on a worker thread, and `Os2lSender`'s NON-daemon thread.
        With the teardown sitting after the loop, any exception in it skipped
        all five: the venue rig stayed lit at whatever the last effect was, the
        encoder stayed resident, and the interpreter then blocked at exit
        forever on the OS2L thread while it went on driving VirtualDJ.  A
        traceback that hangs the process with the lights on is the worst
        available failure, and it was the default one.
        """
        logging.info("[main] setting up auto pilot..")
        try:
            await self._run()
        finally:
            self._shut_down()

    async def _run(self):
        self.audio_client.start_streams(start_stream_out=self._enable_playback)
        self.midi_client.start()
        self.overlay_client.start()
        if self.disable_os2l:
            logging.info("[main] OS2L is disabled")
        else:
            self.os2l_client.start()
        if self.event_buffer is not None:
            self.event_buffer.start()
        # The buffer is also what `--report` fills, so it is not the flag that
        # decides whether a web server opens.
        if self.enable_ui:
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
        audio_delay_buf: deque = deque()
        _audio_playback_started = False
        # Monitoring is delayed by the same amount the room is, so headphones
        # and the venue hear the same instant.
        _playback_ready_at: float = time.monotonic() + PLAYBACK_DELAY_SEC

        while self.is_running:
            now = datetime.datetime.now()
            audio_signal = self.audio_client.read()
            # Before `analyse`, which appends the debug click: the feature stage
            # must read the audio the room hears.
            await self.light_engine.on_audio(audio_signal)
            new_audio_signal = await self.music_analyser.analyse(audio_signal)
            await self.command_queue.drain()

            if self.audio_client.support_output():
                audio_delay_buf.append(new_audio_signal)
                if time.monotonic() >= _playback_ready_at:
                    if not _audio_playback_started:
                        logging.info('[main] audio delay buffer ready, starting delayed playback')
                        _audio_playback_started = True
                    self.audio_client.play(audio_delay_buf.popleft())

            if now - last_100ms_callback_execution > datetime.timedelta(milliseconds=100):
                last_100ms_callback_execution = now
                await self._do_100ms_callback()

            if now - last_1sec_callback_execution > datetime.timedelta(seconds=1):
                last_1sec_callback_execution = now
                await self._do_1s_callback()

            if now - last_10sec_callback_execution > datetime.timedelta(seconds=10):
                last_10sec_callback_execution = now
                await self._do_10s_callback()

    def _shut_down(self) -> None:
        """Close everything, and let no one failure stop the rest.

        Ordered by what the room can see: audio in first so nothing new
        arrives, then the GPU thread, then the wires, then the rig -- and the
        rig last because `MidiClient.stop` is what blanks it, so it must not be
        skipped by something upstream of it raising.  The overlay's clear is
        only queued by `stop`, so it is transmitted here rather than waiting
        for an `on_cycle` the loop has already left.
        """
        self.is_running = False
        for what, close in (('audio', self.audio_client.close),
                            ('section chain',
                             None if self.section is None else self.section.stop),
                            ('os2l', self.os2l_client.stop),
                            ('overlay', self.overlay_client.stop),
                            ('overlay flush', self.overlay_client.flush_messages),
                            ('midi', self.midi_client.stop)):
            if close is None:
                continue
            try:
                close()
            except Exception as error:
                logging.exception(f'[main] {what} did not close cleanly ({error!r})')
        logging.info("[main] auto pilot stopped, clean shutdown")

    def stop(self):
        self.is_running = False
        self.os2l_client.stop()

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
    """Device indices, and nothing else built to print them.

    Constructing the whole app for this loaded the 1.3 GB encoder onto the GPU
    on any machine that has it -- and `list` is the documented first step of
    every run, on the venue box, where the GPU may be busy enough to fail.
    """
    from lib.clients.pyaudio_client import PyAudioClient
    from lib.clients.midi_client import MidiClient

    PyAudioClient(SAMPLE_RATE, BUFFER_SIZE, None, None).list_devices()
    MidiClient(0).list_devices()


def death_handler(signum, frame):
    if global_app is not None:
        logging.info('[DEATH] caught signal "SIGINT/SIGTERM", stopping')
        global_app.stop()


signal.signal(signal.SIGINT, death_handler)
signal.signal(signal.SIGTERM, death_handler)


def run_sync():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())


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

