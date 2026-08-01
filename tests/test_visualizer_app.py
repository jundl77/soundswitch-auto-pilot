"""The Dash view is acceptance surface (charter 11), so its couplings are pinned.

Three of the four ways it touched the demolished rule engine fail *silently*: a
chart that still renders, a legend that still draws, a health dot that is simply
always the wrong colour.  Each one gets a test here rather than a look at the
screen, because looking at the screen is how all three survived.
"""
import pytest

from lib.engine.effect_definitions import LightIntent
from simulate import visualizer_app as V


def _snapshot(**overrides) -> dict:
    snap = {
        'now': 10.0,
        'is_playing': True,
        'beats': [],
        'effects': [],
        'intents': [],
        'sound_events': [],
        'current_effect': None,
        'bpm': 128.0,
        'beats_detected': 0,
        'intent': 'drop',
        'timing_stats': {'samples': 0, 'mean_delta_sec': None,
                         'mean_error_ms': None, 'max_error_ms': None,
                         'by_label': {}},
        'decoder': {},
    }
    snap.update(overrides)
    return snap


def _texts(node) -> list:
    """Every string in a Dash component tree, in render order."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, (list, tuple)):
        return [text for item in node for text in _texts(item)]
    children = getattr(node, 'children', None)
    return [] if children is None else _texts(children)


def _styles(node) -> list:
    if isinstance(node, (list, tuple)):
        return [style for item in node for style in _styles(item)]
    style = getattr(node, 'style', None)
    out = [] if style is None else [style]
    children = getattr(node, 'children', None)
    return out if children is None else out + _styles(children)


def _colour_of(items, prefix: str) -> str:
    for span in items:
        text = ''.join(_texts(span))
        if text.startswith(prefix):
            return span.style['color']
    raise AssertionError(f'no metric strip item starts with {prefix!r}: '
                         f'{[".".join(_texts(i)) for i in items]}')


# -- coupling 1: the beat marker's size ---------------------------------------- #

def test_beat_marker_size_ignores_the_deleted_strength_channel():
    """Onset density scaled this; the density chain is gone.

    Reading the dead channel is what made the failure silent -- the markers went
    to the clamp floor and the chart carried on looking like a chart.
    """
    quiet = V._build_timeline(_snapshot(
        beats=[{'t': 9.0, 'bpm': 128.0, 'strength': 0.0}]))
    loud = V._build_timeline(_snapshot(
        beats=[{'t': 9.0, 'bpm': 128.0, 'strength': 0.9}]))
    assert quiet.data[0].marker.size == loud.data[0].marker.size
    assert quiet.data[0].marker.size == (V.BEAT_MARKER_SIZE,)


def test_beat_markers_render_when_the_channel_is_gone_from_the_buffer():
    fig = V._build_timeline(_snapshot(
        beats=[{'t': 9.0, 'bpm': 128.0}, {'t': 9.5, 'bpm': 128.0}]))
    assert fig.data[0].marker.size == (V.BEAT_MARKER_SIZE,) * 2


# -- coupling 2: nothing advertises an intent the show cannot enter -------------- #

def test_intent_config_covers_exactly_the_intents_the_show_can_enter():
    assert set(V.INTENT_CONFIG) == {intent.value for intent in LightIntent}


def test_the_legend_advertises_only_reachable_intents():
    assert set(_texts(V._build_legend())) - {V.TITLE} == {
        f'■ {intent.name}' for intent in LightIntent}


def test_the_legend_pin_bites_on_an_intent_no_class_produces(monkeypatch):
    """GROOVE is the one this was written for: `_intent_config` falls back to a
    default for an unknown key, so a stale entry raises nothing at all."""
    monkeypatch.setitem(V.INTENT_CONFIG, 'groove', dict(V._DEFAULT_CONFIG,
                                                        label='GROOVE'))
    assert '■ GROOVE' in _texts(V._build_legend())
    assert set(V.INTENT_CONFIG) != {intent.value for intent in LightIntent}


def test_every_reachable_intent_lights_a_slot():
    """A missing entry is the same silent failure the other way round: the stage
    goes to the default's empty slot set and the rig reads as dark."""
    for intent in LightIntent:
        assert V.INTENT_CONFIG[intent.value]['slots'], intent.name


# -- coupling 3: the timing health dot reads the queue's own targets ------------- #

def _timing_stats(**streams) -> dict:
    """Streams as ``label=(mean delay, mean error ms)`` -- four of them wait
    four different amounts (B1), which is what the old literal could not see."""
    by_label = {label: {'samples': 8, 'mean_delta_sec': delay,
                        'mean_error_ms': error, 'max_error_ms': error * 2}
                for label, (delay, error) in streams.items()}
    return {'samples': 8 * len(streams), 'mean_delta_sec': 0.0,
            'mean_error_ms': max(e for _, e in streams.values()),
            'max_error_ms': max(e for _, e in streams.values()) * 2,
            'by_label': by_label}


def test_timing_health_is_green_when_every_stream_matches_its_own_target():
    """B1 moved the delay off 2.5 s and made it per-stream, so a fixed literal
    would sit amber for the whole show while every command landed on time."""
    items = V._build_metrics(_snapshot(timing_stats=_timing_stats(
        beat=(14.0, 3.0), intent=(0.31, 2.0),
        overlay=(14.0, 3.0), refresh=(6.02, 1.0))))
    assert _colour_of(items, 'cmd timing') == V.OK_COLOR


def test_timing_health_is_amber_when_one_stream_misses_its_own_target():
    items = V._build_metrics(_snapshot(timing_stats=_timing_stats(
        beat=(14.0, 3.0), intent=(0.31, 310.0))))
    assert _colour_of(items, 'cmd timing') == V.WARN_COLOR
    assert 'intent' in ''.join(_texts(items))


def test_timing_health_does_not_depend_on_how_long_a_stream_waits():
    """The whole failure the literal caused: a stream that waits 14 s and one
    that waits 0.3 s are both on time, and neither is 2.5."""
    slow = V._build_metrics(_snapshot(timing_stats=_timing_stats(beat=(14.0, 2.0))))
    fast = V._build_metrics(_snapshot(timing_stats=_timing_stats(intent=(0.31, 2.0))))
    assert (_colour_of(slow, 'cmd timing')
            == _colour_of(fast, 'cmd timing') == V.OK_COLOR)


def test_timing_health_is_muted_before_any_command_has_been_delivered():
    assert _colour_of(V._build_metrics(_snapshot()), 'cmd timing') == V.MUTED


def test_the_event_buffer_splits_timing_by_stream():
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer

    buffer = EventBuffer(clock=VirtualClock())
    buffer.set_timing_log([
        {'label': 'beat', 'target_delta_sec': 14.0, 'actual_delta_sec': 14.004},
        {'label': 'beat', 'target_delta_sec': 14.0, 'actual_delta_sec': 13.998},
        {'label': 'intent', 'target_delta_sec': 0.31, 'actual_delta_sec': 0.316},
    ])
    by_label = buffer.snapshot()['timing_stats']['by_label']
    assert set(by_label) == {'beat', 'intent'}
    assert by_label['beat']['samples'] == 2
    assert by_label['intent']['mean_delta_sec'] == pytest.approx(0.316)
    assert by_label['intent']['max_error_ms'] == pytest.approx(6.0)


# -- D14: the decoder-state row ------------------------------------------------- #

def _decoder_state(**overrides) -> dict:
    state = {
        'classes': ['breakdown', 'buildup', 'drop', 'intro', 'outro'],
        'posterior': [0.05, 0.10, 0.70, 0.10, 0.05],
        'observed_bar': 143,
        'committed_bar': 141,
        'committed_label': 'drop',
        'lag_bars': 2,
        'chain_latency_sec': 13.66,
    }
    state.update(overrides)
    return state


def test_the_decoder_row_shows_one_bar_per_class():
    row = V._build_decoder(_snapshot(decoder=_decoder_state()))
    texts = _texts(row)
    for name in _decoder_state()['classes']:
        assert name in texts
    widths = [style['width'] for style in _styles(row)
              if style.get('background') == V.POSTERIOR_FILL]
    assert widths == ['5.0%', '10.0%', '70.0%', '10.0%', '5.0%']


def test_the_decoder_row_shows_the_committed_cursor_and_its_lag():
    text = ' '.join(_texts(V._build_decoder(_snapshot(decoder=_decoder_state()))))
    assert 'bar 143' in text
    assert 'drop' in text
    assert 'lag 2' in text
    assert '13.7s' in text


def test_the_decoder_row_says_a_stuck_decoder_is_not_a_quiet_passage():
    """The whole reason D14 exists: from the stage view alone the two look the
    same, so 'no posterior for this bar' has to be sayable."""
    text = ' '.join(_texts(V._build_decoder(_snapshot(
        decoder=_decoder_state(posterior=None, committed_bar=None,
                               committed_label=None, lag_bars=None)))))
    assert 'no evidence' in text


def test_the_decoder_row_is_honest_when_there_is_no_decoder_at_all():
    text = ' '.join(_texts(V._build_decoder(_snapshot(decoder={}))))
    assert 'no decoder' in text


def test_the_app_renders_every_panel_from_one_snapshot():
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer

    buffer = EventBuffer(clock=VirtualClock())
    buffer.start()
    buffer.add_beat(bpm=128.0, change=False)
    buffer.set_intent('drop', song_sec=1.0)
    buffer.set_decoder_state(**_decoder_state())
    snap = buffer.snapshot()
    assert V._build_timeline(snap) is not None
    assert len(V._build_stage(snap)) == len(V.SLOT_LABELS)
    assert _texts(V._build_decoder(snap))
    assert _texts(V._build_metrics(snap))


# -- coupling 4: the simulation builds the NN stages, not a monkeypatched YAMNet - #

def test_build_simulation_wires_the_chain_into_the_engine(monkeypatch):
    """The YAMNet detector used to be stubbed out here by hand.  What replaced
    it is not a smaller monkeypatch, it is the real chain."""
    from simulate import runner
    from simulate.fake_audio_client import FileAudioClient

    from tests.test_light_engine import FakeChain, FakeDecoder

    class Chain:
        stream = FakeChain()
        decoder = FakeDecoder()

    monkeypatch.setattr(runner, 'load_section_chain', lambda **kwargs: Chain())
    components, _ = runner.build_simulation(
        FileAudioClient.__new__(FileAudioClient))
    engine = components['light_engine']
    assert engine.section_chain is Chain.stream
    assert engine.section_decoder is Chain.decoder


def test_the_simulation_runner_no_longer_patches_a_change_detector():
    import inspect

    from simulate import runner

    source = inspect.getsource(runner)
    assert 'detect_change' not in source
    assert 'yamnet' not in source.lower()


@pytest.mark.integration
def test_the_gpu_thread_runs_under_real_time_pacing(nn_artifacts, anchor_mp3):
    """The `--ui` paths pace on the wall clock and run the extractor on its own
    thread (D3); every other test of the chain runs it inline under a virtual
    clock, which is the one arrangement the show never uses."""
    import asyncio

    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from lib.engine.event_buffer import EventBuffer
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import PLAYBACK_DELAY_SEC, build_simulation, run_simulation

    audio = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, anchor_mp3)
    buffer = EventBuffer(window_sec=float('inf'),
                         look_ahead_sec=PLAYBACK_DELAY_SEC)
    components, queue = build_simulation(audio, buffer, threaded=True)
    assert components['section'] is not None
    buffer.start()
    asyncio.run(run_simulation(components, 40.0, pace_real_time=True))

    snap = buffer.snapshot()
    assert snap['beats_detected'] > 0
    assert snap['decoder']['observed_bar'] > 0
    assert queue.get_timing_log()
