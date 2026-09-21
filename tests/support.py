# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas

from mozdetect.data import TelemetryTimeSeries, TreeherderTimeSeries

SAMPLE_DATA = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]


def get_sample_telemetry_data():
    """Use this method to get a TelemetryTimeSeries object populated with some sample data."""
    df = pandas.DataFrame(SAMPLE_DATA)
    return TelemetryTimeSeries(df)


# One push per entry, as `get_signature_table` produces: the push it was
# recorded at, and the measurements it recorded.
SAMPLE_PUSHES = [
    (100, "2026-09-01T01:00:00", [10.0, 12.0]),
    (101, "2026-09-01T05:00:00", [11.0, 13.0]),
    (102, "2026-09-02T01:00:00", [20.0, 22.0]),
    (103, "2026-09-03T01:00:00", [30.0, 32.0]),
]

SAMPLE_SIGNATURE_METADATA = {
    "signature_id": 298906,
    "signature_hash": "cf9c0d25b441d19d6073c26505629037fc32c7d2",
    "name": "rasterflood_svg opt e10s",
    "platform": "macosx1470-64-shippable",
    "repository_name": "autoland",
    "framework_id": 1,
    "lower_is_better": True,
    "measurement_unit": "ms",
    "alert_threshold": 2.0,
}


def get_sample_treeherder_data(metadata=SAMPLE_SIGNATURE_METADATA):
    """Use this method to get a TreeherderTimeSeries object populated with some sample data."""
    df = pandas.DataFrame(
        [
            {
                "push_id": push_id,
                "push_timestamp": pandas.Timestamp(push_timestamp),
                "revision": f"revision{push_id}",
                "value": pandas.Series(trials).median(),
                "trials": trials,
                "value_count": 1,
                "trial_count": len(trials),
            }
            for push_id, push_timestamp, trials in SAMPLE_PUSHES
        ]
    )
    df.attrs = dict(metadata)
    return TreeherderTimeSeries(df)


def make_treeherder_timeseries(trials_per_push, metadata=SAMPLE_SIGNATURE_METADATA):
    """Builds a TreeherderTimeSeries out of the measurements of each push.

    :param list trials_per_push: The measurements each push recorded, oldest
        push first.
    :param dict metadata: The signature metadata the series carries.

    :return TreeherderTimeSeries: The series, one row per push.
    """
    start = pandas.Timestamp("2026-09-01T00:00:00")
    df = pandas.DataFrame(
        [
            {
                "push_id": 1000 + index,
                "push_timestamp": start + pandas.Timedelta(hours=index),
                "revision": f"revision{1000 + index}",
                "value": pandas.Series(trials).median(),
                "values": [pandas.Series(trials).median()],
                "trials": list(trials),
                "value_count": 1,
                "trial_count": len(trials),
            }
            for index, trials in enumerate(trials_per_push)
        ]
    )
    df.attrs = dict(metadata)
    return TreeherderTimeSeries(df)
