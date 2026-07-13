# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import json

from datetime import date, timedelta

import numpy as np
import pandas

from mozdetect import get_timeseries_detectors
from mozdetect.data import TelemetryTimeSeries

from tests.support import get_sample_telemetry_data


def _get_detector(timeseries=None):
    """Build a cdf_squared timeseries detector, defaulting to trivial sample data.

    The pure alert-merging helpers don't touch the timeseries, so trivial data is
    fine for unit-testing them.
    """
    cls = get_timeseries_detectors()["cdf_squared"]
    detector = cls(timeseries or get_sample_telemetry_data())
    detector._multiday_average_days = 7
    return detector


def _dates(n, start=date(2025, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


# ---------------------------------------------------------------------------
# _find_fast_alerts
# ---------------------------------------------------------------------------


def test_find_fast_alerts_flags_large_deviations():
    detector = _get_detector()

    # 14 quiet days plus one large upward and one large downward day. Without
    # confirmation, the band alone should flag exactly the two large days.
    sq_diff = [0.001, -0.0015, 0.002, -0.001, 0.0005, -0.0008, 0.0012] * 2
    sq_diff += [0.9, -0.8]
    diffs = pandas.DataFrame({"sq_diff": sq_diff, "date": _dates(len(sq_diff))})

    alerts = detector._find_fast_alerts(diffs, mad_k=5.0, require_confirmation=False)

    directions = set(alerts["direction"])
    assert directions == {"up", "down"}
    assert (alerts["detection_type"] == "fast").all()
    # Only the two large days should be flagged, not the quiet ones.
    assert len(alerts) == 2


def test_find_fast_alerts_confirmation_keeps_sustained_drops_blips():
    detector = _get_detector()

    # 14 quiet days, then two sustained up days (a real step), a quiet day, then a
    # single large down day that reverts (a blip).
    quiet = [0.001, -0.0015, 0.002, -0.001, 0.0005, -0.0008, 0.0012] * 2
    sq_diff = quiet + [0.9, 0.9, 0.001, -0.8, 0.001]
    dates = _dates(len(sq_diff))
    diffs = pandas.DataFrame({"sq_diff": sq_diff, "date": dates})

    alerts = detector._find_fast_alerts(diffs, mad_k=5.0, require_confirmation=True)

    # The sustained pair is confirmed (reported at its first day); the blip is dropped.
    assert len(alerts) == 1
    assert alerts.iloc[0]["direction"] == "up"
    assert alerts.iloc[0]["date"] == dates[14]


def test_find_fast_alerts_confirmation_rejects_opposite_directions():
    detector = _get_detector()

    # Two consecutive large days, but in opposite directions -- not a sustained shift.
    quiet = [0.001, -0.0015, 0.002, -0.001, 0.0005, -0.0008, 0.0012] * 2
    sq_diff = quiet + [0.9, -0.8, 0.001]
    diffs = pandas.DataFrame({"sq_diff": sq_diff, "date": _dates(len(sq_diff))})

    alerts = detector._find_fast_alerts(diffs, mad_k=5.0, require_confirmation=True)
    assert alerts.empty


def test_find_fast_alerts_ignores_normal_fluctuation():
    detector = _get_detector()

    # Pure noise around zero -- nothing should clear a 6-MAD band.
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, size=30)
    diffs = pandas.DataFrame({"sq_diff": noise, "date": _dates(30)})

    alerts = detector._find_fast_alerts(diffs, mad_k=6.0)
    assert alerts.empty


def test_find_fast_alerts_requires_minimum_days():
    detector = _get_detector()

    # A giant deviation, but too few days to estimate a stable band.
    diffs = pandas.DataFrame({"sq_diff": [0.0, 0.0, 5.0], "date": _dates(3)})
    alerts = detector._find_fast_alerts(diffs, mad_k=5.0, min_days=14)
    assert alerts.empty


# ---------------------------------------------------------------------------
# _collapse_consecutive
# ---------------------------------------------------------------------------


def test_collapse_consecutive_keeps_earliest_of_a_run():
    detector = _get_detector()

    # A step change fires the fast lane for several days running.
    alerts = pandas.DataFrame(
        {
            "date": _dates(5, start=date(2025, 3, 10)),
            "direction": ["up"] * 5,
            "detection_type": ["fast"] * 5,
        }
    )

    collapsed = detector._collapse_consecutive(alerts, window_days=7)
    assert len(collapsed) == 1
    assert collapsed.iloc[0]["date"] == date(2025, 3, 10)


def test_collapse_consecutive_keeps_separate_clusters():
    detector = _get_detector()

    alerts = pandas.DataFrame(
        {
            "date": [date(2025, 3, 1), date(2025, 3, 2), date(2025, 3, 20)],
            "direction": ["up", "up", "up"],
            "detection_type": ["fast"] * 3,
        }
    )

    collapsed = detector._collapse_consecutive(alerts, window_days=7)
    assert list(collapsed["date"]) == [date(2025, 3, 1), date(2025, 3, 20)]


def test_collapse_consecutive_separates_directions():
    detector = _get_detector()

    # Same day, opposite directions should never collapse into each other.
    alerts = pandas.DataFrame(
        {
            "date": [date(2025, 3, 1), date(2025, 3, 2)],
            "direction": ["up", "down"],
            "detection_type": ["fast", "fast"],
        }
    )

    collapsed = detector._collapse_consecutive(alerts, window_days=7)
    assert len(collapsed) == 2


# ---------------------------------------------------------------------------
# _merge_and_dedup
# ---------------------------------------------------------------------------


def _smoothed(dates, directions):
    return pandas.DataFrame(
        {"date": dates, "direction": directions, "detection_type": ["smoothed"] * len(dates)}
    )


def _fast(dates, directions):
    return pandas.DataFrame(
        {"date": dates, "direction": directions, "detection_type": ["fast"] * len(dates)}
    )


def test_merge_prefers_fast_over_overlapping_smoothed():
    detector = _get_detector()

    change = date(2025, 4, 1)
    fast = _fast([change], ["up"])
    smoothed = _smoothed([change + timedelta(days=4)], ["up"])

    merged = detector._merge_and_dedup(smoothed, fast, dedup_window_days=7)
    assert len(merged) == 1
    assert merged.iloc[0]["detection_type"] == "fast"
    assert merged.iloc[0]["date"] == change


def test_merge_keeps_independent_smoothed_alert():
    detector = _get_detector()

    fast = _fast([date(2025, 4, 1)], ["up"])
    # Same direction, but far enough away to be a distinct event.
    smoothed = _smoothed([date(2025, 5, 1)], ["up"])

    merged = detector._merge_and_dedup(smoothed, fast, dedup_window_days=7)
    assert len(merged) == 2
    assert set(merged["detection_type"]) == {"fast", "smoothed"}


def test_merge_keeps_opposite_direction_smoothed_alert():
    detector = _get_detector()

    change = date(2025, 4, 1)
    fast = _fast([change], ["up"])
    smoothed = _smoothed([change + timedelta(days=2)], ["down"])

    merged = detector._merge_and_dedup(smoothed, fast, dedup_window_days=7)
    assert len(merged) == 2


def test_merge_with_no_fast_alerts_returns_smoothed():
    detector = _get_detector()

    smoothed = _smoothed([date(2025, 4, 1), date(2025, 4, 20)], ["up", "down"])
    empty_fast = pandas.DataFrame(columns=["date", "direction", "detection_type"])

    merged = detector._merge_and_dedup(smoothed, empty_fast, dedup_window_days=7)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# Integration: synthetic histogram timeseries with a step change
# ---------------------------------------------------------------------------


def _histogram(center, total=5000, spread=4.0, nbins=40):
    """A discrete bell-shaped histogram centered on `center` as a {bin: count} map."""
    bins = np.arange(nbins)
    weights = np.exp(-0.5 * ((bins - center) / spread) ** 2)
    counts = (weights / weights.sum() * total).round().astype(int)
    return {str(int(b)): int(c) for b, c in zip(bins, counts) if c > 0}


def _build_step_change_timeseries(step_index=28, ndays=40, seed=0):
    """One build per day, with a large right-shift regression at `step_index`."""
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 1)
    rows = []
    for i in range(ndays):
        day = start + timedelta(days=i)
        build_id = day.strftime("%Y%m%d") + "10"  # 10 chars -> parsed as %Y%m%d%H
        base_center = 8.0 if i < step_index else 20.0
        center = base_center + rng.normal(0, 0.05)  # small daily jitter
        rows.append({"build_id": build_id, "non_norm_histogram": json.dumps(_histogram(center))})
    return TelemetryTimeSeries(pandas.DataFrame(rows))


def test_fast_lane_detects_step_change_near_the_change_date():
    step_index = 28
    change_date = date(2025, 1, 1) + timedelta(days=step_index)
    ts = _build_step_change_timeseries(step_index=step_index)

    detector = _get_detector(ts)
    ts.get_cumulative_by_day()
    ts.get_multiday_average(days=7)

    daily = detector._calculate_daily_differences()
    fast = detector._find_fast_alerts(daily, mad_k=5.0)

    assert not fast.empty
    fast = fast.sort_values("date").reset_index(drop=True)
    first = fast.iloc[0]
    # A right shift (values got larger) is an "up" detection.
    assert first["direction"] == "up"
    # The change should be caught within a couple of days of when it landed.
    assert abs((first["date"] - change_date).days) <= 2


def test_detect_changes_reports_fast_detection_for_step_change():
    step_index = 28
    change_date = date(2025, 1, 1) + timedelta(days=step_index)
    ts = _build_step_change_timeseries(step_index=step_index)

    detector = _get_detector(ts)
    detections = detector.detect_changes(fast_detection=True, fast_mad_k=5.0)

    fast_detections = [
        d for d in detections if d.optional_detection_info.get("detection_type") == "fast"
    ]
    assert fast_detections

    earliest_fast = min(fast_detections, key=lambda d: d.location)
    assert earliest_fast.direction == "up"
    assert abs((earliest_fast.location - change_date).days) <= 2


def test_detect_changes_without_fast_lane_has_no_fast_detections():
    ts = _build_step_change_timeseries(step_index=28)

    detector = _get_detector(ts)
    detections = detector.detect_changes(fast_detection=False)

    assert all(d.optional_detection_info.get("detection_type") != "fast" for d in detections)
