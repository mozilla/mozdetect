# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pandas
import pytest

from mozdetect.data import (
    InvalidNumberError,
    TimeSeries,
    TreeherderTimeSeries,
    UnknownDataTypeError,
)

from .support import get_sample_treeherder_data


def test_data():
    ts = TimeSeries([(1, 2, 3), (4, 5, 6)])

    # Check that the first row is returned
    assert (ts.get_current() == (1, 2, 3)).all(axis=1).any()

    # Check second time to make sure it doesn't change
    assert (ts.get_current() == (1, 2, 3)).all(axis=1).any()


def test_data_empty():
    ts = TimeSeries([])
    assert len(ts.data) == 0

    with pytest.raises(IndexError):
        ts.get_current()


def test_data_iteration():
    data = [(1, 2, 3), (4, 5, 6)]
    ts = TimeSeries(data)

    for row in ts:
        assert (row == data[ts._currind]).all().any()


def test_data_complex():
    ts = TimeSeries([(1, 2, "h"), (4, 5, "e", 5)])
    assert (ts.get_current() == (1, 2, "h", None)).all().any()
    assert ts.get_current().isna().loc[0, 3]


def test_data_get_n():
    ts = TimeSeries([(1, 2, 3), (4, 5, 6), (3, 5, 5)])

    subset = ts.get_next_n(2)
    assert all((subset == row).all(axis=1).any() for row in [(4, 5, 6), (3, 5, 5)])
    assert not (subset == (1, 2, 3)).all(axis=1).any()

    # Returns empty as there is nothing before the first element
    subset = ts.get_previous_n(1)
    assert subset.empty

    ts._currind += 2
    subset = ts.get_previous_n(2)
    assert all((subset == row).all(axis=1).any() for row in [(1, 2, 3), (4, 5, 6)])
    assert not (subset == (3, 5, 5)).all(axis=1).any()

    # Returns empty as there is nothing after the last element
    subset = ts.get_next_n(1)
    assert subset.empty


def test_data_get_bad_n():
    ts = TimeSeries([])
    with pytest.raises(InvalidNumberError):
        ts.get_next_n(-1)

    with pytest.raises(InvalidNumberError):
        ts.get_previous_n(-1)


def test_data_get_data_type():
    ts = TimeSeries([(1, 2, "a"), (1, 2, "b"), (1, 2, "c")])

    subset = ts.get_numerical_columns()
    assert all((subset == row).all(axis=1).any() for row in [(1, 2), (1, 2), (1, 2)])

    subset = ts.get_nonnumerical_columns()
    assert all((subset == row).all(axis=1).any() for row in [("a",), ("b",), ("c",)])


def test_data_set_data_type():
    ts = TimeSeries([(1, 2, 3.0, "a"), (1, 2, 3.0, "a"), (1, 2, 3.0, "a")])

    ts.set_data_type("numerical")
    for row in ts:
        assert (row == (1, 2, 3.0)).all().any()

    ts.set_data_type("non-numerical")
    for row in ts:
        assert (row == ("a")).all().any()

    ts.set_data_type([int])
    for row in ts:
        assert (row == (1, 2)).all().any()

    ts.set_data_type("all")
    assert (ts.data == (1, 2, 3.0, "a")).all(axis=1).all()


def test_data_set_data_type_failues():
    ts = TimeSeries([(1, 2, 3.0, "a"), (1, 2, 3.0, "a"), (1, 2, 3.0, "a")])

    with pytest.raises(UnknownDataTypeError):
        ts.set_data_type("bad-type")

    with pytest.raises(UnknownDataTypeError):
        ts.set_data_type(["not a type"])


def test_treeherder_data_metadata():
    ts = get_sample_treeherder_data()

    assert ts.metadata["signature_id"] == 298906
    assert ts.lower_is_better is True
    assert ts.alert_threshold == 2.0
    assert ts.measurement_unit == "ms"
    assert ts.revisions == {
        100: "revision100",
        101: "revision101",
        102: "revision102",
        103: "revision103",
    }

    # The raw data is exposed for anything the helpers don't cover.
    assert len(ts.raw_data) == 4
    assert (ts.data == ts.raw_data).all().all()


def test_treeherder_data_metadata_missing():
    ts = get_sample_treeherder_data(metadata={})

    # A signature that hasn't set a threshold has no floor at all, and
    # lower_is_better is almost always true.
    assert ts.metadata == {}
    assert ts.lower_is_better is True
    assert ts.alert_threshold is None
    assert ts.measurement_unit == ""


def test_treeherder_data_get_trials():
    ts = get_sample_treeherder_data()

    assert ts.get_trials() == [10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 30.0, 32.0]

    # A window of the series, as a detector would ask for it.
    ts._currind = 2
    assert ts.get_trials(ts.get_previous_n(2)) == [10.0, 12.0, 11.0, 13.0]
    assert ts.get_trials(ts.get_next_n(1, inclusive=True)) == [20.0, 22.0]


def test_treeherder_data_get_trials_per_data_point():
    # A per-data-point table holds the measurements in the same column.
    df = pandas.DataFrame(
        [
            {
                "push_id": 100,
                "push_timestamp": pandas.Timestamp("2026-09-01T01:00:00"),
                "revision": "revision100",
                "value": 10.0,
                "trials": [9.0, 11.0],
            },
            {
                "push_id": 100,
                "push_timestamp": pandas.Timestamp("2026-09-01T01:00:00"),
                "revision": "revision100",
                "value": 12.0,
                "trials": [12.0],
            },
        ]
    )
    ts = TreeherderTimeSeries(df)

    assert ts.get_trials() == [9.0, 11.0, 12.0]
    # Both rows sit on the same push, so they pool into one point.
    by_day = ts.get_by_day()
    assert len(by_day) == 1
    assert by_day["trials"].iloc[0] == [9.0, 11.0, 12.0]
    assert by_day["push_count"].iloc[0] == 1


def test_treeherder_data_get_trials_without_replicates():
    # Read with replicates off, the aggregated value stands in for them.
    df = pandas.DataFrame(
        [
            {
                "push_id": 100,
                "push_timestamp": pandas.Timestamp("2026-09-01T01:00:00"),
                "revision": "revision100",
                "value": 10.0,
                "trials": [10.0],
            }
        ]
    )
    ts = TreeherderTimeSeries(df)

    assert ts.get_trials() == [10.0]


def test_treeherder_data_get_by_day():
    ts = get_sample_treeherder_data()

    by_day = ts.get_by_day()

    assert [str(date) for date in by_day["date"]] == [
        "2026-09-01",
        "2026-09-02",
        "2026-09-03",
    ]
    # The two pushes on the first day pool together.
    assert list(by_day["value"]) == [11.5, 21.0, 31.0]
    assert list(by_day["push_count"]) == [2, 1, 1]
    assert list(by_day["trial_count"]) == [4, 2, 2]
    assert by_day["trials"].iloc[0] == [10.0, 12.0, 11.0, 13.0]

    # It's also kept on the timeseries.
    assert (ts.by_day == by_day).all().all()


def test_treeherder_data_get_multiday_average():
    ts = get_sample_treeherder_data()

    multiday = ts.get_multiday_average(days=2)

    # Only full windows, each labelled by the last day it covers.
    assert [str(date) for date in multiday["date"]] == ["2026-09-02", "2026-09-03"]
    assert list(multiday["value"]) == [12.5, 26.0]
    assert list(multiday["push_count"]) == [3, 2]
    assert list(multiday["trial_count"]) == [6, 4]
    assert (ts.multiday == multiday).all().all()

    # The per-day data it builds on is calculated if it wasn't already.
    assert not ts.by_day.empty


def test_treeherder_data_get_multiday_average_not_enough_days():
    ts = get_sample_treeherder_data()

    assert ts.get_multiday_average(days=7).empty
