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


def _spans(node) -> list:
    if isinstance(node, (list, tuple)):
        return [span for item in node for span in _spans(item)]
    children = getattr(node, 'children', None)
    if isinstance(children, (list, tuple)):
        return _spans(children)
    return [node]


def _span_of(items, prefix: str):
    for span in _spans(items):
        if ''.join(_texts(span)).startswith(prefix):
            return span
    raise AssertionError(f'no metric strip item starts with {prefix!r}: '
                         f'{[".".join(_texts(i)) for i in _spans(items)]}')


def _colour_of(items, prefix: str) -> str:
    return _span_of(items, prefix).style['color']


def _metric(items, prefix: str) -> str:
    return ''.join(_texts(_span_of(items, prefix)))


def test_the_figure_never_draws_beats_because_the_browser_owns_them():
    fig = V._build_timeline(_snapshot(
        beats=[{'t': 9.0, 'bpm': 128.0, 'strength': 0.0}]))
    assert fig.data == ()


def test_the_marker_css_carries_the_size_the_figure_used_to():
    assert '.ss-beat' in V.STYLESHEET
    assert f'height: {V.BEAT_MARKER_SIZE}px' in V.STYLESHEET


def test_the_anchor_ships_beats_when_the_strength_channel_is_gone():
    anchor = V._anchor(_snapshot(
        beats=[{'t': 9.0, 'bpm': 128.0}, {'t': 9.5, 'bpm': 128.0}]))
    assert anchor['beats'] == [9.0, 9.5]


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


def test_a_beat_is_anchored_at_the_moment_the_room_hears_it():
    anchor = V._anchor(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        beats=[{'t': 3.0, 'bpm': 128.0}]))
    assert anchor['beats'] == [17.0]


def test_a_beat_inside_the_drawn_lead_is_shipped_before_the_room_hears_it():
    # The browser withholds nothing: the marker sits past the cursor and
    # crosses it at its room instant, so the future strip must carry it.
    anchor = V._anchor(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        beats=[{'t': 3.0, 'bpm': 128.0}, {'t': 7.0, 'bpm': 128.0}]))
    assert anchor['beats'] == [17.0, 21.0]


def test_a_beat_past_the_drawn_strip_is_not_shipped_yet():
    anchor = V._anchor(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        beats=[{'t': 3.0, 'bpm': 128.0}, {'t': 12.0, 'bpm': 128.0}]))
    assert anchor['beats'] == [17.0]


def test_the_song_zero_is_the_start_on_the_room_clock_not_the_detectors():
    assert V._song_origin(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        sound_events=[{'t': 1.0, 'playing': True}])) == pytest.approx(15.0)


def test_the_start_marker_sits_on_the_zero_it_defines():
    fig = V._build_timeline(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        sound_events=[{'t': 1.0, 'playing': True}]))
    assert [a.x for a in fig.layout.annotations if 'START' in a.text] == [0.0]


def test_an_intent_block_is_already_room_fired_and_is_not_shifted_again():
    fig = V._build_timeline(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        intents=[{'t': 5.0, 'intent': 'drop', 'end': 10.0}]))
    rect = next(s for s in fig.layout.shapes if s.type == 'rect')
    assert (rect.x0, rect.x1) == (5.0, 10.0)


def test_the_anchor_puts_beats_on_the_room_clock():
    anchor = V._anchor(_snapshot(now=20.0, look_ahead_sec=14.0,
                                 beats=[{'t': 3.0, 'bpm': 128.0}]))
    assert anchor['beats'] == [17.0]


def test_the_anchor_carries_both_clocks_and_the_axis_they_are_read_against():
    anchor = V._anchor(_snapshot(now=20.0, look_ahead_sec=14.0,
                                 sound_events=[{'t': 0.0, 'playing': True}]))
    assert (anchor['now'], anchor['song'], anchor['room']) == (6.0, 20.0, 6.0)


def test_the_anchor_leaves_the_clocks_blank_before_the_first_sound_start():
    anchor = V._anchor(_snapshot(now=20.0, sound_events=[]))
    assert anchor['song'] is None and anchor['room'] is None


def _two_songs(**overrides) -> dict:
    return _snapshot(look_ahead_sec=14.0,
                     sound_events=[{'t': 0.0, 'playing': True},
                                   {'t': 60.0, 'playing': False},
                                   {'t': 80.0, 'playing': True}],
                     **overrides)


def _axis(fig) -> tuple:
    return tuple(fig.layout.xaxis.range)


_ZEROED_AXIS = (-V.TIMELINE_WINDOW_SEC,
                V.TIMELINE_LEAD_SEC + V.TIMELINE_PAD_SEC)


def test_a_run_that_never_claimed_a_boundary_plots_on_the_session_clock():
    assert V._song_origin(_snapshot(now=20.0)) == 0.0


def test_the_timeline_counts_from_the_instant_the_room_heard_the_song_start():
    fig = V._build_timeline(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}],
        beats=[{'t': 3.0, 'bpm': 128.0}]))
    assert _axis(fig) == pytest.approx(
        (6.0 - V.TIMELINE_WINDOW_SEC, 6.0 + V.TIMELINE_LEAD_SEC + V.TIMELINE_PAD_SEC))
    assert V._anchor(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}],
        beats=[{'t': 3.0, 'bpm': 128.0}]))['beats'] == [3.0]


def test_the_axis_is_the_room_clock_so_the_two_can_never_disagree():
    snap = _snapshot(now=20.0, look_ahead_sec=14.0,
                     sound_events=[{'t': 0.0, 'playing': True}])
    assert V._anchor(snap)['now'] == pytest.approx(V._song_and_room(snap)[1])


def test_the_first_song_holds_the_axis_at_zero_until_the_room_reaches_it():
    fig = V._build_timeline(_snapshot(
        now=9.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}]))
    assert _axis(fig) == pytest.approx(_ZEROED_AXIS)


def test_a_detected_stop_empties_the_timeline_and_zeroes_it():
    snap = _snapshot(
        now=62.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}, {'t': 60.0, 'playing': False}],
        beats=[{'t': 40.0, 'bpm': 128.0}],
        intents=[{'t': 50.0, 'intent': 'drop'}])
    fig = V._build_timeline(snap)
    assert _axis(fig) == pytest.approx(_ZEROED_AXIS)
    assert fig.layout.shapes == ()
    assert V._anchor(snap)['beats'] == []


def test_both_clocks_go_blank_between_songs():
    items = V._build_metrics(_snapshot(
        now=62.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}, {'t': 60.0, 'playing': False}]))
    assert _metric(items, 'song') == 'song —'
    assert _metric(items, 'room') == 'room —'


def test_the_old_song_ends_at_the_stop_rather_than_playing_out():
    snap = _snapshot(
        now=70.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}, {'t': 60.0, 'playing': False}],
        beats=[{'t': 55.0, 'bpm': 128.0}])
    assert V._anchor(snap)['beats'] == []
    assert _axis(V._build_timeline(snap)) == pytest.approx(_ZEROED_AXIS)


def test_a_start_the_room_has_not_reached_does_not_end_the_gap():
    snap = _two_songs(now=90.0)
    assert V._song_and_room(snap) == (None, None)
    assert _axis(V._build_timeline(snap)) == pytest.approx(_ZEROED_AXIS)


def test_the_new_song_begins_the_moment_the_room_reaches_it():
    assert V._song_and_room(_two_songs(now=94.0)) == (pytest.approx(14.0),
                                                      pytest.approx(0.0))


def test_the_previous_songs_beats_do_not_cross_the_reset():
    anchor = V._anchor(_two_songs(
        now=100.0, beats=[{'t': 40.0, 'bpm': 128.0}, {'t': 85.0, 'bpm': 128.0}]))
    assert anchor['beats'] == [5.0]


def test_a_beat_is_placed_in_the_song_the_room_heard_it_in_not_the_one_it_was_detected_in():
    anchor = V._anchor(_two_songs(now=100.0,
                                  beats=[{'t': 85.0, 'bpm': 128.0}]))
    assert anchor['beats'] == [5.0]


def test_the_previous_songs_intent_block_does_not_reach_back_past_zero():
    fig = V._build_timeline(_two_songs(
        now=100.0, intents=[{'t': 50.0, 'intent': 'drop', 'end': 70.0},
                            {'t': 96.0, 'intent': 'peak'}]))
    rects = [s for s in fig.layout.shapes if s.type == 'rect']
    assert [r.x0 for r in rects] == [2.0]


def test_the_anchor_publishes_the_axis_and_the_beats_in_song_time():
    anchor = V._anchor(_snapshot(
        now=20.0, look_ahead_sec=14.0,
        sound_events=[{'t': 0.0, 'playing': True}],
        beats=[{'t': 3.0, 'bpm': 128.0}]))
    assert anchor['now'] == pytest.approx(6.0)
    assert anchor['beats'] == [pytest.approx(3.0)]


def test_the_anchor_holds_the_axis_still_between_songs():
    anchor = V._anchor(_two_songs(now=90.0, beats=[{'t': 40.0, 'bpm': 128.0}]))
    assert anchor['now'] == 0.0
    assert anchor['beats'] == []


def test_the_browser_re_seats_a_backward_re_base_instead_of_extrapolating_it():
    assert 'sync.now < a.now' in V.ANIMATION_JS
    scroll = V.ANIMATION_JS[V.ANIMATION_JS.index('const scroll'):
                            V.ANIMATION_JS.index('const pulse')]
    assert 'dx < 0' in scroll


def test_the_glow_re_arms_across_a_song_boundary():
    pulse = V.ANIMATION_JS[V.ANIMATION_JS.index('const pulse'):
                           V.ANIMATION_JS.index('const meter')]
    assert 'a.pulsed = null' in pulse


def test_the_glow_fires_when_the_room_clock_crosses_the_beat():
    # Firing on anchor arrival instead cost a measured median 154 ms of lag.
    pulse = V.ANIMATION_JS[V.ANIMATION_JS.index('const pulse'):
                           V.ANIMATION_JS.index('const meter')]
    assert 'a.markerList' in pulse
    assert 'beats[i] <= now' in pulse


def test_the_beat_layer_rides_the_translated_layer():
    scroll = _find_by_id(_app().layout.children, 'timeline-scroll')
    layer = _find_by_id(scroll, 'beat-layer')
    assert layer is not None
    assert layer.style['position'] == 'absolute'
    assert layer.style['pointerEvents'] == 'none'
    assert scroll.style['position'] == 'relative'


def test_the_browser_owns_the_beat_markers_and_seats_them_off_the_live_range():
    js = V.ANIMATION_JS
    assert "getElementById('beat-layer')" in js
    assert 'ss-beat' in js
    seat = js[js.index('const seatBeats'):js.index('const scroll')]
    assert 'xaxis.range' in seat
    assert 'a.seatedAt === stamp' in seat
    frame = js[js.index('const frame'):]
    assert 'syncBeats();' in frame and 'seatBeats(' in frame


class _FakePoller:
    def __init__(self, snapshot):
        self._snapshot = snapshot
        self.reads = 0

    def snapshot(self) -> dict:
        self.reads += 1
        return self._snapshot


def _app(source=None):
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer

    if source is None:
        source = EventBuffer(clock=VirtualClock())
        source.start()
    return V.build_app(source)


_ANCHOR = ('..sync.data...metrics.children...stage.children...'
           'decoder.children..')
_VIEW = '..timeline.figure...drawn.data..'
_GATE = '..gate.data...view-gate.data..'


def test_the_app_reads_its_data_through_a_snapshot_call_and_nothing_else():
    poller = _FakePoller(_snapshot(bpm=131.0, beats_detected=7))
    app = _app(poller)

    anchor, metrics, stage, decoder = app.callback_map[
        _ANCHOR]['callback'].__wrapped__(1)

    assert poller.reads == 1
    assert '131 BPM' in ''.join(_texts(metrics))
    assert '7 beats' in ''.join(_texts(metrics))
    assert anchor['now'] == 10.0
    assert stage and decoder


def test_one_poll_feeds_every_panel_so_they_cannot_disagree():
    poller = _FakePoller(_snapshot())
    app = _app(poller)
    app.callback_map[_ANCHOR]['callback'].__wrapped__(1)
    figure, _ = app.callback_map[_VIEW]['callback'].__wrapped__(1)
    assert figure is not None
    assert poller.reads == 1


def test_the_figure_is_not_rebuilt_from_a_snapshot_it_already_drew():
    import dash

    poller = _FakePoller(_snapshot())
    app = _app(poller)
    app.callback_map[_ANCHOR]['callback'].__wrapped__(1)
    first, first_ack = app.callback_map[_VIEW]['callback'].__wrapped__(1)
    again, again_ack = app.callback_map[_VIEW]['callback'].__wrapped__(2)
    assert first is not None and again is dash.no_update
    assert (first_ack, again_ack) == (1, 2)


def test_no_tick_can_start_a_refresh_while_one_is_still_in_flight():
    app = _app()
    assert app.callback_map[_GATE]['inputs'][0]['id'] == 'tick'
    assert app.callback_map[_ANCHOR]['inputs'][0]['id'] == 'gate'
    assert app.callback_map[_VIEW]['inputs'][0]['id'] == 'view-gate'
    assert 'gate.inflight[stream] = Date.now()' in V.REQUEST_PRE_JS
    assert "free('anchor')" in V.GATE_JS and "free('view')" in V.GATE_JS


def test_a_stream_is_freed_by_its_answer_landing_and_never_by_it_resolving():
    app = _app()
    assert "delete ds.ss_gate.inflight['anchor']" in V.ANIMATION_JS
    assert app.callback_map['anim.data']['inputs'][0]['id'] == 'sync'
    assert "delete gate.inflight['view']" in V.VIEW_LANDED_JS
    assert app.callback_map['taken.data']['inputs'][0]['id'] == 'drawn'

    before, after = V.CALLBACK_RESOLVED_JS.split('if (result.error)')
    assert 'delete gate.inflight[stream]' not in before
    assert 'delete gate.inflight[stream]' in after


def test_the_interval_no_longer_reaches_a_server_callback_directly():
    app = _app()
    for key, entry in app.callback_map.items():
        if getattr(entry.get('callback'), '__wrapped__', None) is None:
            continue
        assert 'tick' not in [i['id'] for i in entry['inputs']], key


def test_the_figure_is_asked_for_on_a_slower_cadence_than_the_anchor():
    assert V.VIEW_EVERY_TICKS >= 2
    assert f'n % {V.VIEW_EVERY_TICKS}' in V.GATE_JS
    assert V.REFRESH_MS == 250


def test_the_renderer_reports_the_answers_dash_itself_would_discard():
    app = _app()
    assert 'new DashRenderer(' in app.renderer
    assert V.REQUEST_PRE_JS in app.renderer
    assert V.CALLBACK_RESOLVED_JS in app.renderer


def test_the_watchdog_warns_when_the_page_stops_taking_what_the_server_sent():
    assert 'landed.now !== gate.sent' in V.CALLBACK_RESOLVED_JS
    assert 'console.warn' in V.CALLBACK_RESOLVED_JS
    assert str(V.STALE_REFRESHES) in V.CALLBACK_RESOLVED_JS


def test_a_refresh_that_never_answers_re_arms_instead_of_wedging_the_gate():
    assert f'waited < {V.STALL_RELEASE_MS}' in V.GATE_JS
    assert 'delete gate.inflight[stream]' in V.GATE_JS


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


def test_the_fps_readout_is_outside_every_callback_output():
    layout = _app().layout.children
    assert _find_by_id(layout, 'fps') is not None
    assert _find_by_id(V._build_metrics(_snapshot()), 'fps') is None


def test_the_fps_readout_is_driven_by_the_frame_loop_once_a_second():
    js = V.ANIMATION_JS
    meter = js[js.index('const meter'):js.index('const frame')]
    assert "getElementById('fps')" in meter
    assert 'span < 1000' in meter
    assert 'meter();' in js[js.index('const frame'):]


def test_the_clock_spans_are_addressable_by_the_browser():
    ids = {getattr(span, 'id', None)
           for span in _spans(V._build_metrics(_snapshot()))}
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
    assert cursor.style['left'] == V.NOW_CURSOR_LEFT
    assert cursor.style['position'] == 'absolute'


def test_the_now_cursor_accounts_for_the_axis_margins():
    # A container-fraction cursor sat ~7 px right of the axis' now -- a
    # measured -146 ms crossing bias against every marker.
    assert V.NOW_CURSOR_LEFT.startswith(f'calc({V.TIMELINE_MARGIN_L_PX}px')
    assert f'{V.NOW_CURSOR_X * 100:.4f}%' in V.NOW_CURSOR_LEFT


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


def _stopping(**overrides) -> dict:
    return _snapshot(look_ahead_sec=14.0, is_playing=False,
                     sound_events=[{'t': 0.0, 'playing': True},
                                   {'t': 60.0, 'playing': False}],
                     **overrides)


def _mid_play(**overrides) -> dict:
    return _snapshot(look_ahead_sec=14.0,
                     sound_events=[{'t': 0.0, 'playing': True}], **overrides)


def test_the_show_stops_once_the_detected_silence_has_persisted():
    assert 'PAUSED' in _metric(V._build_metrics(_stopping(now=62.0)), '◌')


def test_the_room_keeps_playing_the_tail_while_the_bypass_is_still_pending():
    assert 'PLAYING' in _metric(V._build_metrics(_stopping(now=61.99)), '●')


def test_the_timeline_resets_when_the_silence_persists_not_when_it_starts():
    assert V._song_origin(_stopping(now=61.99)) is not None
    assert V._song_origin(_stopping(now=62.0)) is None


def test_a_gap_shorter_than_the_persistence_never_reaches_the_room_at_all():
    seamless = _snapshot(now=200.0, look_ahead_sec=14.0,
                         sound_events=[{'t': 0.0, 'playing': True},
                                       {'t': 100.0, 'playing': False},
                                       {'t': 101.5, 'playing': True}])
    assert [e['t'] for e in V._room_sound_events(seamless)] == [14.0]
    assert V._song_origin(seamless) == pytest.approx(14.0)
    assert 'PLAYING' in _metric(V._build_metrics(seamless), '●')


def test_the_room_holds_its_tail_through_the_window_before_a_resume_lands():
    mid_gap = _snapshot(now=100.5, look_ahead_sec=14.0, is_playing=False,
                        sound_events=[{'t': 0.0, 'playing': True},
                                      {'t': 100.0, 'playing': False}])
    assert V._song_origin(mid_gap) == pytest.approx(14.0)
    assert 'PLAYING' in _metric(V._build_metrics(mid_gap), '●')


def _burst() -> dict:
    return _snapshot(now=130.0, look_ahead_sec=14.0,
                     sound_events=[{'t': 0.0, 'playing': True},
                                   {'t': 60.0, 'playing': False},
                                   {'t': 100.0, 'playing': True},
                                   {'t': 105.0, 'playing': False}])


def test_a_play_burst_shorter_than_the_look_ahead_never_reaches_the_room():
    assert [e['t'] for e in V._room_sound_events(_burst())] == [14.0, 62.0, 107.0]
    assert V._song_origin(_burst()) is None
    assert 'PAUSED' in _metric(V._build_metrics(_burst()), '◌')


def test_a_start_that_lands_before_the_bypass_is_heard_and_then_cut():
    brief = _snapshot(now=30.0, look_ahead_sec=14.0,
                      sound_events=[{'t': 1.0, 'playing': True},
                                    {'t': 15.0, 'playing': False}])
    assert [e['t'] for e in V._room_sound_events(brief)] == [15.0, 17.0]
    assert V._song_origin(dict(brief, now=16.0)) == pytest.approx(15.0)
    assert V._song_origin(brief) is None


def test_room_time_and_list_order_agree_once_one_rule_decides_both():
    resuming = _snapshot(now=94.0, look_ahead_sec=14.0,
                         sound_events=[{'t': 0.0, 'playing': True},
                                       {'t': 60.0, 'playing': False},
                                       {'t': 80.0, 'playing': True}])
    for snapshot in (_burst(), resuming, _stopping(now=60.0),
                     _stopping(now=70.0), _mid_play(now=70.0)):
        heard = V._room_sound_events(snapshot)
        assert V._last_heard(snapshot) == heard[-1]
        assert heard == sorted(heard, key=lambda event: event['t'])


def test_a_start_still_waits_for_the_room_although_a_stop_does_not():
    resuming = _snapshot(look_ahead_sec=14.0,
                         sound_events=[{'t': 0.0, 'playing': True},
                                       {'t': 60.0, 'playing': False},
                                       {'t': 80.0, 'playing': True}])
    assert V._song_origin(dict(resuming, now=93.99)) is None
    assert V._song_origin(dict(resuming, now=94.0)) == pytest.approx(94.0)


def _stop_annotations(fig) -> list:
    return [a for a in fig.layout.annotations if 'STOP' in a.text]


def test_a_stop_still_travelling_is_drawn_where_the_room_will_hear_it():
    # Detected at 60, persisted (heard) at 62; origin is 14, so the marker
    # rides the lead strip at song-t 48 and crosses the cursor at 62.
    fig = V._build_timeline(_stopping(now=61.0))
    stops = _stop_annotations(fig)
    assert [a.x for a in stops] == [pytest.approx(48.0)]
    assert stops[0].xanchor == 'right'
    line = next(s for s in fig.layout.shapes
                if s.type == 'line' and s.line.dash == 'dash')
    assert line.x0 == pytest.approx(48.0)


def test_a_stop_marker_waits_until_it_enters_the_drawn_strip():
    assert _stop_annotations(V._build_timeline(_stopping(now=60.2))) == []


def test_the_stop_marker_crossing_is_the_instant_the_song_ends():
    assert V._room_markers(_stopping(now=61.0)) == [(14.0, True), (62.0, False)]
    assert V._build_timeline(_stopping(now=62.0)).layout.shapes == ()


def test_a_heard_stop_is_not_double_counted_as_pending():
    assert V._room_markers(_stopping(now=62.0)).count((62.0, False)) == 1


def test_a_resume_inside_the_window_leaves_no_stop_marker_at_all():
    pending = _snapshot(now=100.6, look_ahead_sec=14.0, is_playing=False,
                        sound_events=[{'t': 0.0, 'playing': True},
                                      {'t': 100.0, 'playing': False}])
    resumed = _snapshot(now=101.6, look_ahead_sec=14.0, sound_events=[
        {'t': 0.0, 'playing': True}, {'t': 100.0, 'playing': False},
        {'t': 101.5, 'playing': True}])
    assert (102.0, False) in V._room_markers(pending)
    assert _stop_annotations(V._build_timeline(pending)) != []
    assert V._room_markers(resumed) == [(14.0, True)]
    assert _stop_annotations(V._build_timeline(resumed)) == []


def test_a_start_still_travelling_when_the_bypass_fires_gets_no_marker():
    markers = V._room_markers(_burst())
    assert [t for t, is_start in markers if is_start] == [14.0]
    assert V._build_timeline(_burst()).layout.shapes == ()


def test_the_previous_songs_boundaries_stay_out_of_the_new_songs_window():
    fig = V._build_timeline(_two_songs(now=100.0))
    assert [a.x for a in fig.layout.annotations if 'START' in a.text] == [0.0]
    assert _stop_annotations(fig) == []


def test_the_start_marker_scrolls_out_with_the_window():
    fig = V._build_timeline(_mid_play(now=50.0))
    assert [a for a in fig.layout.annotations if 'START' in a.text] == []


def test_the_beats_still_in_the_air_at_a_stop_are_never_heard():
    silenced = _stopping(now=70.0, beats_detected=900, beats_cut=2,
                         beats=[{'t': 50.0, 'bpm': 128.0},
                                {'t': 55.0, 'bpm': 128.0}])
    assert V._heard_beats(silenced) == []
    assert _metric(V._build_metrics(silenced), '898') == '898 beats'

    aged_out = dict(silenced, now=110.0, beats=[])
    assert _metric(V._build_metrics(aged_out), '898') == '898 beats'


def test_the_tempo_is_the_last_beat_the_room_heard():
    items = V._build_metrics(_mid_play(
        now=70.0, bpm=0.0,
        beats=[{'t': 50.0, 'bpm': 128.0}, {'t': 62.0, 'bpm': 174.0}]))
    assert _metric(items, '128') == '128 BPM'


def test_the_beat_count_is_what_the_room_has_heard():
    items = V._build_metrics(_mid_play(
        now=70.0, beats_detected=900,
        beats=[{'t': 50.0, 'bpm': 128.0}, {'t': 62.0, 'bpm': 128.0},
               {'t': 64.0, 'bpm': 128.0}]))
    assert _metric(items, '898') == '898 beats'


def test_the_payload_window_stays_wider_than_the_span_it_has_to_fill():
    from lib.engine.event_buffer import EventBuffer

    assert V.TIMELINE_WINDOW_SEC < EventBuffer.SNAPSHOT_WINDOW_SEC


class _FakeConnection:

    live = []

    def __init__(self, *args, **kwargs):
        self.interfere = None

    def request(self, *args):
        _FakeConnection.live.append(self)
        if self.interfere is not None:
            self.interfere()

    def getresponse(self):
        _FakeConnection.live.remove(self)
        return self

    def read(self):
        return b'{"now": 1.0}'


def _poller(monkeypatch):
    monkeypatch.setattr(V.http.client, 'HTTPConnection', _FakeConnection)
    _FakeConnection.live.clear()
    return V.SnapshotPoller(port=1)


def test_a_poll_survives_a_sibling_dropping_the_connection_underneath_it(
        monkeypatch):
    poller = _poller(monkeypatch)
    poller.snapshot()
    poller._connection.interfere = lambda: setattr(poller, '_connection', None)
    assert poller.snapshot() == {'now': 1.0}


def test_a_poll_already_in_flight_hands_back_the_held_frame(monkeypatch):
    import threading
    import time

    poller = _poller(monkeypatch)
    poller.snapshot()
    result = {}

    def beside():
        started = time.monotonic()
        result['frame'] = poller.snapshot()
        result['waited'] = time.monotonic() - started

    def interfere():
        thread = threading.Thread(target=beside)
        thread.start()
        thread.join(timeout=3.0)

    poller._connection.interfere = interfere
    poller.snapshot()
    assert result['waited'] < 0.5
    assert result['frame'] == {'now': 1.0}


def test_polls_from_several_callback_threads_stay_one_at_a_time(monkeypatch):
    import threading

    poller = _poller(monkeypatch)
    peak, failures = [0], []

    def poll():
        for _ in range(40):
            try:
                poller.snapshot()
            except BaseException as error:
                failures.append(repr(error))
            peak[0] = max(peak[0], len(_FakeConnection.live))

    threads = [threading.Thread(target=poll) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert failures == []
    assert peak[0] <= 1


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
    import asyncio

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

    asyncio.run(cli._run_pipeline(
        {}, 1.0, buffer, DelayedCommandQueue(14.0, clock=clock), False,
        report_path=str(report)))

    import json
    assert [b['intent'] for b in json.loads(report.read_text())['intents']] == ['drop']


class _FakeUi:
    def __init__(self):
        self.waited = 0
        self.stopped = 0

    def wait(self):
        self.waited += 1

    def stop(self):
        self.stopped += 1


def _viewer_session(monkeypatch, pipeline):
    import asyncio

    from lib import ui_bridge
    from simulate import cli

    ui = _FakeUi()
    monkeypatch.setattr(ui_bridge, 'start', lambda buffer, port: ui)

    async def session():
        await cli._with_viewer(object(), 8050, pipeline)
        return asyncio.get_running_loop()

    return ui, session


def test_the_ui_session_runs_the_pipeline_on_the_loop_it_was_called_from(
        monkeypatch):
    # The pipeline has the main thread now, so it must join the loop already
    # running there rather than start one of its own.
    import asyncio

    ran = []

    async def pipeline():
        ran.append(asyncio.get_running_loop())

    ui, session = _viewer_session(monkeypatch, pipeline)
    assert ran == [asyncio.run(session())]
    assert (ui.waited, ui.stopped) == (1, 1)


def test_a_pipeline_that_raises_still_takes_the_viewer_down_with_it(monkeypatch):
    import asyncio

    async def pipeline():
        raise RuntimeError('the audio device went away')

    ui, session = _viewer_session(monkeypatch, pipeline)
    with pytest.raises(RuntimeError):
        asyncio.run(session())
    assert (ui.waited, ui.stopped) == (0, 1)


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


def test_a_run_with_no_watchdog_says_so_rather_than_claiming_health():
    items = V._build_metrics(_snapshot(shed={}))
    assert _metric(items, 'health') == 'health: —'
    assert _colour_of(items, 'health') == V.MUTED
    assert _metric(items, 'sheds') == 'sheds: —'


def test_a_healthy_chain_reads_live():
    items = V._build_metrics(_snapshot(
        shed={'level': 'NONE', 'fault': None, 'sheds': 0, 'sheds_per_min': 0}))
    assert _metric(items, 'health') == 'health: ● LIVE'
    assert _colour_of(items, 'health') == V.OK_COLOR
    assert _metric(items, 'sheds') == 'sheds 0/min'


def test_a_shed_chain_names_the_held_intent_and_the_fault_that_holds_it():
    items = V._build_metrics(_snapshot(
        shed={'level': 'NN_SHED', 'fault': 'hung_pass',
              'sheds': 3, 'sheds_per_min': 2}))
    pill = _metric(items, 'health')
    assert 'DEGRADED' in pill and 'holding intent' in pill
    assert 'hung_pass' in pill
    assert _colour_of(items, 'health') == V.WARN_COLOR


def test_a_shed_with_no_named_fault_still_reads_degraded():
    items = V._build_metrics(_snapshot(
        shed={'level': 'NN_SHED', 'fault': None,
              'sheds': 1, 'sheds_per_min': 1}))
    assert _metric(items, 'health') == 'health: ◆ DEGRADED — holding intent'


def test_the_rate_and_the_run_total_are_both_readable():
    items = V._build_metrics(_snapshot(
        shed={'level': 'NONE', 'fault': None, 'sheds': 9, 'sheds_per_min': 2}))
    assert _metric(items, 'sheds') == 'sheds 2/min  ·  9 this run'
    assert _colour_of(items, 'sheds') == V.WARN_COLOR


def test_a_run_that_settled_hours_ago_stops_shouting_about_it():
    items = V._build_metrics(_snapshot(
        shed={'level': 'NONE', 'fault': None, 'sheds': 9, 'sheds_per_min': 0}))
    assert _metric(items, 'sheds') == 'sheds 0/min  ·  9 this run'
    assert _colour_of(items, 'sheds') == V.MUTED


def _degraded_strip() -> dict:
    return _snapshot(
        now=133.0, look_ahead_sec=14.0, bpm=140.0, intent='atmospheric',
        sound_events=[{'t': 0.0, 'playing': True}],
        shed={'level': 'NN_SHED', 'fault': None,
              'sheds': 10, 'sheds_per_min': 4},
        timing_stats=_timing_stats(beat=(14.01, 9.0), intent=(1.41, 1.0)))


def test_the_strip_reads_as_grouped_rows_rather_than_one_line():
    rows = [' '.join(_texts(row)) for row in V._build_metrics(_degraded_strip())]
    transport, analysis, health, timing = rows

    assert transport == '● PLAYING room 1min 59sec song 2min 13sec'
    assert analysis == '140 BPM 0 beats intent: ATMOSPHERIC'
    assert health == 'health: ◆ DEGRADED — holding intent sheds 4/min  ·  10 this run'
    assert timing.startswith('cmd timing: on target')


def test_every_row_is_a_line_of_its_own_that_wraps_on_its_own():
    for row in V._build_metrics(_degraded_strip()):
        assert row.style['display'] == 'flex'
        assert row.style['flexWrap'] == 'wrap'
        assert row.style['columnGap']


def test_a_ticking_value_reserves_its_width_so_the_row_cannot_jitter():
    rows = V._build_metrics(_degraded_strip())
    assert all(row.style['fontVariantNumeric'] == 'tabular-nums' for row in rows)
    for prefix in ('● PLAYING', 'room', 'song', '140 BPM', '0 beats',
                   'beat 14.01s'):
        assert _span_of(rows, prefix).style['minWidth'], prefix


def test_the_health_verdict_is_the_prominent_reading():
    health = _span_of(V._build_metrics(_degraded_strip()), 'health')
    assert health.style['fontWeight'] == 'bold'
    assert health.style['fontSize'] == V.HEALTH_FONT_SIZE
    assert health.style['border'].endswith(V.WARN_COLOR)
    assert (int(V.HEALTH_FONT_SIZE.rstrip('px'))
            > int(V.TIMING_FONT_SIZE.rstrip('px')))


def test_the_command_timing_block_is_secondary_and_one_span_per_stream():
    timing = V._build_metrics(_degraded_strip())[-1]
    assert timing.style['fontSize'] == V.TIMING_FONT_SIZE
    assert timing.style['borderTop'].endswith(V.BORDER)
    assert _texts(timing) == ['cmd timing: on target',
                              'beat 14.01s ±9ms', 'intent 1.41s ±1ms']


def test_a_run_with_no_delivered_command_still_lays_out_four_rows():
    assert len(V._build_metrics(_snapshot())) == 4
    assert _metric(V._build_metrics(_snapshot()), 'cmd timing') == 'cmd timing: —'


def _gridded(bar_sec=2.0, bars=6, first_bar=0, **overrides):
    edges = [10.0 + index * bar_sec for index in range(bars)]
    snap = _snapshot(now=40.0, look_ahead_sec=14.0,
                     sound_events=[{'t': 0.0, 'playing': True}],
                     decoder={'bar_edges': edges, 'first_bar': first_bar,
                              'bar_sec': bar_sec, 'classes': []})
    snap.update(overrides)
    return snap


def _lines(fig, colour):
    return [s for s in fig.layout.shapes
            if s.type == 'line' and s.line.color == colour]


def test_the_timeline_draws_the_decoders_own_downbeats():
    fig = V._build_timeline(_gridded(bar_sec=2.0, bars=6))
    downbeats = _lines(fig, V.DOWNBEAT_COLOR)
    assert [round(s.x0, 3) for s in downbeats] == [10.0, 12.0, 14.0, 16.0, 18.0]


def test_a_bar_line_is_shifted_into_room_time_like_everything_else():
    fig = V._build_timeline(_gridded(bar_sec=2.0, bars=3))
    # start@0 reaches the room at 14, so song zero is 14 and an edge at 10
    # detected-seconds is heard at 24, which is 10 on the room axis.
    assert round(_lines(fig, V.DOWNBEAT_COLOR)[0].x0, 3) == 10.0


def test_each_bar_carries_its_intra_bar_beats():
    fig = V._build_timeline(_gridded(bar_sec=2.0, bars=3))
    ticks = _lines(fig, V.BEAT_TICK_COLOR)
    assert len(ticks) == 2 * (V.BEATS_PER_BAR - 1)
    assert [round(t.x0, 3) for t in ticks[:3]] == [10.5, 11.0, 11.5]


def test_the_ticks_follow_a_wobbly_grid_instead_of_a_nominal_bar():
    snap = _gridded()
    snap['decoder']['bar_edges'] = [10.0, 12.0, 14.4]
    ticks = _lines(V._build_timeline(snap), V.BEAT_TICK_COLOR)
    assert [round(t.x0, 3) for t in ticks[3:6]] == [12.6, 13.2, 13.8]


def test_a_re_anchor_gap_is_not_padded_with_beats_nobody_played():
    snap = _gridded()
    snap['decoder']['bar_edges'] = [10.0, 12.0, 22.0]
    ticks = _lines(V._build_timeline(snap), V.BEAT_TICK_COLOR)
    assert [round(t.x0, 3) for t in ticks] == [10.5, 11.0, 11.5]


def test_the_downbeats_either_side_of_a_gap_are_real_and_stay_drawn():
    snap = _gridded()
    snap['decoder']['bar_edges'] = [10.0, 12.0, 22.0]
    downbeats = _lines(V._build_timeline(snap), V.DOWNBEAT_COLOR)
    assert [round(s.x0, 3) for s in downbeats] == [10.0, 12.0]


def test_a_grid_with_no_published_bar_period_still_reads_its_own_spacing():
    snap = _gridded()
    snap['decoder']['bar_edges'] = [10.0, 12.0, 14.0, 16.0, 26.0]
    snap['decoder'].pop('bar_sec')
    ticks = _lines(V._build_timeline(snap), V.BEAT_TICK_COLOR)
    assert [round(t.x0, 3) for t in ticks[-3:]] == [14.5, 15.0, 15.5]


def test_the_phrasing_is_scannable_because_bars_are_numbered():
    fig = V._build_timeline(_gridded(bar_sec=2.0, bars=10, first_bar=0))
    numbered = [a for a in fig.layout.annotations if a.text.isdigit()]
    assert [a.text for a in numbered] == ['0', '4', '8']


def test_the_numbers_are_the_decoders_bar_indices_not_screen_positions():
    fig = V._build_timeline(_gridded(bar_sec=2.0, bars=10, first_bar=102))
    numbered = [a for a in fig.layout.annotations if a.text.isdigit()]
    assert [a.text for a in numbered] == ['104', '108']


def test_a_half_tempo_lock_reads_as_bars_twice_as_wide():
    honest = _lines(V._build_timeline(_gridded(bar_sec=1.9)), V.DOWNBEAT_COLOR)
    locked = _lines(V._build_timeline(_gridded(bar_sec=3.8)), V.DOWNBEAT_COLOR)
    honest_span = honest[1].x0 - honest[0].x0
    locked_span = locked[1].x0 - locked[0].x0
    assert round(locked_span / honest_span, 3) == 2.0


def test_a_run_with_no_decoder_draws_no_grid():
    fig = V._build_timeline(_snapshot(now=40.0, decoder={}))
    assert _lines(fig, V.DOWNBEAT_COLOR) == []


def test_a_grid_of_one_edge_cannot_size_a_bar_so_it_draws_none():
    snap = _gridded()
    snap['decoder']['bar_edges'] = [10.0]
    assert _lines(V._build_timeline(snap), V.DOWNBEAT_COLOR) == []


def test_the_grid_scrolls_with_the_transform_rather_than_a_per_frame_relayout():
    assert 'shapes' not in V.ANIMATION_JS
    assert 'translate3d' in V.ANIMATION_JS
