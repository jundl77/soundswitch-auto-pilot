from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from soak_nn import boundary_verdict, buckets, spread  # noqa: E402


def shed(t: float, fault=None) -> dict:
    return {"t": t, "from": "NONE", "to": "NN_SHED", "fault": fault}


def stop(t: float) -> dict:
    return {"t": t, "event": "sound_stop"}


def test_a_shed_at_a_track_change_fails_the_boundary_verdict():
    outcome = boundary_verdict([stop(600.0)], [shed(602.0)])
    assert outcome["passed"] is False
    assert outcome["sheds_near_a_boundary"] == 1
    assert outcome["detail"][0]["gap_sec"] == pytest.approx(2.0)


def test_a_shed_far_from_any_boundary_is_not_attributed_to_one():
    outcome = boundary_verdict([stop(600.0)], [shed(120.0)])
    assert outcome["passed"] is True
    assert outcome["boundaries"] == 1


def test_a_restore_is_not_a_shed():
    recovering = {"t": 601.0, "from": "NN_SHED", "to": "NONE", "fault": None}
    assert boundary_verdict([stop(600.0)], [recovering])["passed"] is True


def test_a_shed_after_a_boundary_counts_for_longer_than_the_stall():
    assert boundary_verdict([stop(600.0)], [shed(629.0)])["passed"] is False
    assert boundary_verdict([stop(600.0)], [shed(631.0)])["passed"] is True


def test_a_run_with_no_boundaries_cannot_pass_by_having_nothing_to_fail():
    outcome = boundary_verdict([], [shed(10.0)])
    assert outcome["boundaries"] == 0
    assert outcome["passed"] is False
    assert "nothing was tested" in outcome["why"]

    quiet = boundary_verdict([], [])
    assert quiet["passed"] is False, 'a capture with no events at all passed'


def test_the_per_minute_tail_splits_where_the_minutes_do():
    stamps = [float(i) for i in range(180)]
    values = [1.0] * 60 + [9.0] * 60 + [1.0] * 60
    rows = buckets(stamps, values)
    assert [row["minute"] for row in rows] == [1, 2, 3]
    assert [row["max_ms"] for row in rows] == [1.0, 9.0, 1.0]


def test_an_empty_capture_summarises_to_nothing_rather_than_raising():
    assert spread([]) == {}
    assert buckets([], []) == []
