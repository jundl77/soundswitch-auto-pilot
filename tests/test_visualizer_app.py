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
        'look_ahead_sec': 0.0,
        'timing_stats': {'samples': 0, 'mean_delta_sec': None,
                         'mean_error_ms': None, 'max_error_ms': None,
                         'by_label': {}},
        'decoder': {},
    }
    snap.update(overrides)
    return snap


def _texts(node) -> list:
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


def _classes(node) -> list:
    if isinstance(node, (list, tuple)):
        return [name for item in node for name in _classes(item)]
    name = getattr(node, 'className', None)
    out = [] if name is None else [name]
    children = getattr(node, 'children', None)
    return out if children is None else out + _classes(children)


def _colour_of(items, prefix: str) -> str:
    for span in items:
        text = ''.join(_texts(span))
        if text.startswith(prefix):
            return span.style['color']
    raise AssertionError(f'no metric strip item starts with {prefix!r}: '
                         f'{[".".join(_texts(i)) for i in items]}')


def _metric(items, prefix: str) -> str:
    for span in items:
        text = ''.join(_texts(span))
        if text.startswith(prefix):
            return text
    raise AssertionError(f'no metric strip item starts with {prefix!r}: '
                         f'{[".".join(_texts(i)) for i in items]}')


def test_beat_marker_size_ignores_the_deleted_strength_channel():
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


def test_a_clock_past_a_minute_reads_as_minutes_and_seconds():
    assert V._clock_text(65.0) == '1min 5sec'
    assert V._clock_text(222.0) == '3min 42sec'


def test_a_clock_under_a_minute_never_says_zero_minutes():
    assert V._clock_text(42.0) == '42sec'
    assert V._clock_text(0.0) == '0sec'


def test_the_room_clock_reads_zero_until_the_playback_delay_has_elapsed():
    early = V._build_metrics(_snapshot(
        now=9.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}]))
    assert _metric(early, 'room') == 'room 0sec'
    assert _metric(early, 'song') == 'song 9sec'

    later = V._build_metrics(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}]))
    assert _metric(later, 'room') == 'room 6sec'
    assert _metric(later, 'song') == 'song 20sec'


def test_a_new_sound_start_restarts_both_clocks():
    items = V._build_metrics(_snapshot(
        now=100.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True},
                      {'t': 60.0, 'playing': False},
                      {'t': 80.0, 'playing': True}]))
    assert _metric(items, 'song') == 'song 20sec'
    assert _metric(items, 'room') == 'room 6sec'


def test_both_clocks_are_blank_until_the_first_sound_start():
    items = V._build_metrics(_snapshot(now=30.0, look_ahead_sec=14.0,
                                       sound_events=[]))
    assert _metric(items, 'song') == 'song —'
    assert _metric(items, 'room') == 'room —'


def test_a_beat_is_plotted_at_the_moment_the_room_hears_it():
    fig = V._build_timeline(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        beats=[{'t': 3.0, 'bpm': 128.0}]))
    assert fig.data[0].x == (17.0,)


def test_a_beat_the_room_has_not_heard_yet_is_not_plotted():
    fig = V._build_timeline(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        beats=[{'t': 3.0, 'bpm': 128.0}, {'t': 12.0, 'bpm': 128.0}]))
    assert fig.data[0].x == (17.0,)


def test_the_sound_markers_move_onto_the_room_clock_with_the_beats():
    fig = V._build_timeline(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        sound_events=[{'t': 1.0, 'playing': True}]))
    assert [a.x for a in fig.layout.annotations if 'START' in a.text] == [15.0]


def test_an_intent_block_is_already_room_fired_and_is_not_shifted_again():
    fig = V._build_timeline(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        intents=[{'t': 5.0, 'intent': 'drop', 'end': 10.0}]))
    rect = next(s for s in fig.layout.shapes if s.type == 'rect')
    assert (rect.x0, rect.x1) == (5.0, 10.0)


def test_the_anchor_puts_the_last_beat_on_the_room_clock():
    anchor = V._anchor(_snapshot(now=20.0, look_ahead_sec=14.0,
                                 beats=[{'t': 3.0, 'bpm': 128.0}]))
    assert anchor['beat'] == 17.0


def test_a_beat_the_room_has_not_heard_yet_is_not_in_the_anchor():
    anchor = V._anchor(_snapshot(now=20.0, look_ahead_sec=14.0,
                                 beats=[{'t': 3.0, 'bpm': 128.0},
                                        {'t': 12.0, 'bpm': 128.0}]))
    assert anchor['beat'] == 17.0


def test_the_anchor_carries_both_clocks_and_the_time_it_was_read():
    anchor = V._anchor(_snapshot(now=20.0, look_ahead_sec=14.0,
                                 sound_events=[{'t': 0.0, 'playing': True}]))
    assert (anchor['now'], anchor['song'], anchor['room']) == (20.0, 20.0, 6.0)


def test_the_anchor_leaves_the_clocks_blank_before_the_first_sound_start():
    anchor = V._anchor(_snapshot(now=20.0, sound_events=[]))
    assert anchor['song'] is None and anchor['room'] is None


def _app():
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer

    buffer = EventBuffer(clock=VirtualClock())
    buffer.start()
    return V.build_app(buffer)


def test_the_layout_publishes_the_anchor_the_browser_interpolates():
    ids = {getattr(child, 'id', None) for child in _app().layout.children}
    assert {'sync', 'anim'} <= ids


def test_the_animation_runs_in_the_browser_off_the_anchor():
    app = _app()
    assert 'anim.data' in app.callback_map
    assert app.callback_map['anim.data']['inputs'][0]['id'] == 'sync'
    assert V.ANIMATION_JS in ''.join(app._inline_scripts)


def test_the_browser_drives_the_axis_the_clocks_and_the_glow():
    for driven in ('xaxis.range', 'room-clock', 'song-clock', 'ss-pulse'):
        assert driven in V.ANIMATION_JS, driven


def test_the_scroll_translates_per_frame_and_re_seats_the_axis_rarely():
    # Measured: relayout re-renders every beat marker, 3.2 of 4.7 ms per call.
    # The per-frame path must be the compositor, not Plotly.
    js = V.ANIMATION_JS
    assert 'translate3d' in js
    assert 'RESEAT_PX' in js
    scroll = js[js.index('const scroll'):js.index('const pulse')]
    assert scroll.index('translate3d') > scroll.index('RESEAT_PX')


def test_the_stylesheet_animates_the_beat_glow_instead_of_the_server():
    assert '@keyframes ss-pulse' in _app().index_string


def test_the_lamps_hand_the_stylesheet_the_glow_it_animates():
    lamp = next(style for style in _styles(V._build_stage(_snapshot()))
                if '--ss-peak' in style)
    assert lamp['--ss-decay'] == '0.12s'
    assert lamp['--ss-lamp'] == V.INTENT_CONFIG['drop']['primary']
    assert 'boxShadow' not in lamp


def test_only_the_lamps_the_intent_lights_are_pulse_targets():
    slots = V._build_stage(_snapshot(intent='atmospheric'))
    lit = [name for name in _classes(slots) if 'ss-on' in name]
    assert len(lit) == len(V.INTENT_CONFIG['atmospheric']['slots'])
    assert len([name for name in _classes(slots) if 'ss-lamp' in name]) \
        == len(V.SLOT_LABELS)


def test_the_clock_spans_are_addressable_by_the_browser():
    ids = {getattr(span, 'id', None) for span in V._build_metrics(_snapshot())}
    assert {'room-clock', 'song-clock'} <= ids


def _find_by_id(node, target):
    if isinstance(node, (list, tuple)):
        for item in node:
            found = _find_by_id(item, target)
            if found is not None:
                return found
        return None
    if getattr(node, 'id', None) == target:
        return node
    children = getattr(node, 'children', None)
    return None if children is None else _find_by_id(children, target)


def test_the_now_cursor_is_a_fixed_screen_position_not_a_plotted_shape():
    # The window always ends a fixed lead past now, so the cursor's screen x is
    # a constant -- a shape would ride the scrolling transform and jitter.
    fig = V._build_timeline(_snapshot(now=10.0))
    assert [s for s in fig.layout.shapes if s.xref == 'paper'] == []

    cursor = _find_by_id(_app().layout.children, 'now-cursor')
    assert cursor is not None
    assert cursor.style['left'] == f'{V.NOW_CURSOR_X * 100:.4f}%'
    assert cursor.style['position'] == 'absolute'


def test_the_scrolled_layer_is_outside_the_graph_dash_rerenders():
    scroll = _find_by_id(_app().layout.children, 'timeline-scroll')
    assert scroll is not None
    assert scroll.style.get('willChange') == 'transform'


def test_the_second_grid_is_an_axis_setting_not_thirty_shapes():
    fig = V._build_timeline(_snapshot(now=100.0))
    assert [s for s in fig.layout.shapes if s.xref == 'x'] == []
    assert fig.layout.xaxis.minor.dtick == 1.0


def test_the_running_intent_block_reaches_the_edge_the_browser_scrolls_to():
    # Out to the pad, not just the lead: the pad is the strip the transform
    # translates in, so a block stopping at the lead would open a gap there.
    fig = V._build_timeline(_snapshot(
        now=20.0, intents=[{'t': 5.0, 'intent': 'drop'}]))
    rect = next(s for s in fig.layout.shapes if s.type == 'rect')
    assert rect.x1 == pytest.approx(
        20.0 + V.TIMELINE_LEAD_SEC + V.TIMELINE_PAD_SEC)


def test_the_snapshot_carries_the_delay_the_display_shifts_by():
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer

    buffer = EventBuffer(window_sec=float('inf'), clock=VirtualClock(),
                         look_ahead_sec=14.0)
    buffer.start()
    assert buffer.snapshot()['look_ahead_sec'] == pytest.approx(14.0)


def test_intent_config_covers_exactly_the_intents_the_show_can_enter():
    assert set(V.INTENT_CONFIG) == {intent.value for intent in LightIntent}


def test_the_legend_advertises_only_reachable_intents():
    assert set(_texts(V._build_legend())) - {V.TITLE} == {
        f'■ {intent.name}' for intent in LightIntent}


def test_the_legend_pin_bites_on_an_intent_no_class_produces(monkeypatch):
    monkeypatch.setitem(V.INTENT_CONFIG, 'groove', dict(V._DEFAULT_CONFIG,
                                                        label='GROOVE'))
    assert '■ GROOVE' in _texts(V._build_legend())
    assert set(V.INTENT_CONFIG) != {intent.value for intent in LightIntent}


def test_every_reachable_intent_lights_a_slot():
    for intent in LightIntent:
        assert V.INTENT_CONFIG[intent.value]['slots'], intent.name


def _timing_stats(**streams) -> dict:
    by_label = {label: {'samples': 8, 'mean_delta_sec': delay,
                        'mean_error_ms': error, 'max_error_ms': error * 2}
                for label, (delay, error) in streams.items()}
    return {'samples': 8 * len(streams), 'mean_delta_sec': 0.0,
            'mean_error_ms': max(e for _, e in streams.values()),
            'max_error_ms': max(e for _, e in streams.values()) * 2,
            'by_label': by_label}


def test_timing_health_is_green_when_every_stream_matches_its_own_target():
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


def test_build_simulation_wires_the_chain_into_the_engine(monkeypatch):
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


def test_a_ui_session_leaves_the_report_it_committed(tmp_path, monkeypatch):
    from lib.clock import VirtualClock
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.event_buffer import EventBuffer
    from simulate import cli, runner

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(runner, 'run_simulation', nothing)
    clock = VirtualClock()
    buffer = EventBuffer(window_sec=float('inf'), clock=clock)
    buffer.start()
    buffer.set_intent('drop', song_sec=1.0)
    report = tmp_path / 'session.json'

    cli._run_pipeline({}, 1.0, buffer, DelayedCommandQueue(14.0, clock=clock),
                      False, report_path=str(report))

    import json
    assert [b['intent'] for b in json.loads(report.read_text())['intents']] == ['drop']


def test_a_ui_file_session_keeps_the_whole_track():
    from lib.clock import VirtualClock
    from simulate import cli

    clock = VirtualClock()
    buffer = cli._session_buffer(14.0, clock=clock)
    buffer.start()
    buffer.set_intent('breakdown', song_sec=1.0)
    clock.advance(400.0)
    buffer.set_intent('drop', song_sec=401.0)

    assert [b['intent'] for b in buffer.to_report()['intents']] \
        == ['breakdown', 'drop']


def test_the_pill_leaves_playing_when_the_audio_runs_out():
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer

    clock = VirtualClock()
    buffer = EventBuffer(window_sec=float('inf'), clock=clock)
    buffer.start()
    buffer.set_playing(True)
    clock.advance(240.0)
    buffer.mark_end()
    clock.advance(3300.0)

    pill = ''.join(_texts(V._build_metrics(buffer.snapshot())[0]))
    assert 'PLAYING' not in pill, pill
    assert 'PAUSED' in pill, pill
