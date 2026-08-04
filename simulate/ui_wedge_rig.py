import argparse
import statistics
import threading
import time

from lib.clock import VirtualClock
from lib.engine.event_buffer import EventBuffer
from lib.ui_bridge import SnapshotServer, snapshot_port
from simulate.visualizer_app import SnapshotPoller, build_app

LOOK_AHEAD_SEC = 14.0
BPM = 128.0
BEAT_SEC = 60.0 / BPM
INTENT_CYCLE = ('groove', 'buildup', 'drop', 'peak', 'breakdown')
INTENT_SEC = 40.0
TRACK_SEC = 300.0
GAP_SEC = 6.0
SEED_STEP_SEC = BEAT_SEC / 4


class RiggedShow:
    def __init__(self, minutes: float):
        self.clock = VirtualClock()
        self.buffer = EventBuffer(window_sec=60.0, clock=self.clock,
                                  look_ahead_sec=LOOK_AHEAD_SEC)
        self.buffer.start()
        self._next_beat = 0.0
        self._next_intent = 0.0
        self._intent = 0
        self._playing = False
        self._advance_to(minutes * 60.0)

    def _advance_to(self, target: float) -> None:
        while self.clock.monotonic() < target:
            self.clock.advance(min(SEED_STEP_SEC, target - self.clock.monotonic()))
            self._emit()

    def _emit(self) -> None:
        now = self.clock.monotonic()
        playing = (now % (TRACK_SEC + GAP_SEC)) < TRACK_SEC
        if playing != self._playing:
            self._playing = playing
            self.buffer.set_playing(playing)
        if not playing:
            return
        if now >= self._next_beat:
            self.buffer.add_beat(bpm=BPM, change=False, rms=0.2)
            self._next_beat = now + BEAT_SEC
        if now >= self._next_intent:
            self.buffer.set_intent(INTENT_CYCLE[self._intent % len(INTENT_CYCLE)])
            self._intent += 1
            self._next_intent = now + INTENT_SEC

    def run_forever(self) -> None:
        last = time.perf_counter()
        while True:
            time.sleep(0.01)
            wall = time.perf_counter()
            self._advance_to(self.clock.monotonic() + (wall - last))
            last = wall


class SlowSource:
    def __init__(self, source, latency_sec: float):
        self._source = source
        self._latency = latency_sec

    def snapshot(self) -> dict:
        snap = self._source.snapshot()
        if self._latency:
            time.sleep(self._latency)
        return snap


def _server_callbacks(app) -> list:
    found = []
    for key, entry in app.callback_map.items():
        wrapped = getattr(entry.get('callback'), '__wrapped__', None)
        if wrapped is not None:
            found.append((key, wrapped))
    return found


def profile(show: RiggedShow, ticks: int) -> None:
    app = build_app(show.buffer)
    snap = show.buffer.snapshot()
    print(f'snapshot: {len(snap["beats"])} beats, '
          f'{len(snap["sound_events"])} sound events, '
          f'{len(snap["intents"])} intent blocks, '
          f'{show.clock.monotonic() / 60:.0f} min into the set')

    from simulate import visualizer_app as V
    for name in ('_build_timeline', '_build_stage', '_build_decoder',
                 '_build_metrics', '_anchor'):
        fn = getattr(V, name)
        samples = []
        for _ in range(ticks):
            at = time.perf_counter()
            fn(snap)
            samples.append((time.perf_counter() - at) * 1000)
        print(f'  {name:<16} median {statistics.median(samples):7.2f} ms   '
              f'max {max(samples):7.2f} ms')

    callbacks = _server_callbacks(app)
    samples = {key: [] for key, _ in callbacks}
    for tick in range(ticks):
        for key, wrapped in callbacks:
            at = time.perf_counter()
            wrapped(tick)
            samples[key].append((time.perf_counter() - at) * 1000)
    for key, taken in samples.items():
        print(f'callback {key}')
        print(f'  median {statistics.median(taken):7.2f} ms   '
              f'mean {statistics.fmean(taken):7.2f} ms   '
              f'max {max(taken):7.2f} ms')


def serve(show: RiggedShow, port: int, latency_ms: float) -> None:
    server = SnapshotServer(show.buffer, snapshot_port(port))
    server.start()
    threading.Thread(target=show.run_forever, name='rigged-show',
                     daemon=True).start()
    source = SlowSource(SnapshotPoller(port=snapshot_port(port)),
                        latency_ms / 1000.0)
    app = build_app(source)
    print(f'\n  rig → http://127.0.0.1:{port}  '
          f'(+{latency_ms:.0f} ms injected, '
          f'{show.clock.monotonic() / 60:.0f} min of history)\n')
    app.run(host='127.0.0.1', port=port, debug=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='The visualizer under a slow callback, with no audio and no '
                    'GPU: a real EventBuffer seeded with an hour of show, the '
                    'real snapshot server, the real Dash app, and a fixed '
                    'latency added to every snapshot read. `serve` is the wedge '
                    'reproduction (drive a browser at it and watch whether the '
                    'sync anchor keeps moving); `profile` times the server '
                    'callbacks on the same payload without a browser.')
    parser.add_argument('mode', choices=('serve', 'profile'))
    parser.add_argument('--minutes', type=float, default=60.0,
                        help='how far into a set the buffer is seeded to')
    parser.add_argument('--latency-ms', type=float, default=250.0,
                        help='latency added to every snapshot read (serve)')
    parser.add_argument('--port', type=int, default=8060)
    parser.add_argument('--ticks', type=int, default=40,
                        help='timed invocations per callback (profile)')
    args = parser.parse_args(argv)

    show = RiggedShow(args.minutes)
    if args.mode == 'profile':
        profile(show, args.ticks)
    else:
        serve(show, args.port, args.latency_ms)


if __name__ == '__main__':
    main()
