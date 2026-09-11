# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
from unittest import mock

from mozdetect.treeherder_query import (
    MultipleSignaturesError,
    TreeherderQueryError,
    get_signature_table,
    get_signatures,
)

SIGNATURE_HASH = "cf9c0d25b441d19d6073c26505629037fc32c7d2"

SIGNATURE_METADATA = {
    "signature_id": 298906,
    "framework_id": 1,
    "signature_hash": SIGNATURE_HASH,
    "platform": "macosx1470-64-shippable",
    "suite": "rasterflood_svg",
    "test": "",
    "name": "rasterflood_svg opt e10s fission stylo webrender-sw",
    "repository_name": "autoland",
    "lower_is_better": True,
    "measurement_unit": "ms",
    "alert_threshold": 2.0,
    "alert_change_type": None,
}


def _datum(datum_id, push_id, value, push_timestamp, job_id=1):
    return {
        "job_id": job_id,
        "id": datum_id,
        "value": value,
        "push_timestamp": push_timestamp,
        "push_id": push_id,
        "revision": f"revision{push_id}",
        "submit_time": push_timestamp,
        "machine_name": "machine-1",
    }


def _series(data):
    return [dict(SIGNATURE_METADATA, data=data)]


AGGREGATED_DATA = [
    _datum(1, 100, 10.0, "2026-09-04T13:47:53"),
    _datum(2, 101, 11.0, "2026-09-04T15:00:00"),
    # A second data point on the same push, as a retrigger produces.
    _datum(3, 101, 13.0, "2026-09-04T15:00:00", job_id=2),
]

REPLICATE_DATA = [
    _datum(1, 100, 9.0, "2026-09-04T13:47:53"),
    _datum(1, 100, 11.0, "2026-09-04T13:47:53"),
    _datum(2, 101, 10.5, "2026-09-04T15:00:00"),
    _datum(2, 101, 11.5, "2026-09-04T15:00:00"),
    # Datum 3 reported no replicates.
]


def _mock_client(mocked_client, responses):
    mocked_client.get = mock.MagicMock(side_effect=responses)
    return mocked_client


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_signature_table(mocked_th_client):
    _mock_client(mocked_th_client, [_series(AGGREGATED_DATA), _series(REPLICATE_DATA)])

    result = get_signature_table("autoland", SIGNATURE_HASH, per_push=False)

    # One request for the aggregated values, one for the replicates.
    assert len(mocked_th_client.get.call_args_list) == 2

    path, params = mocked_th_client.get.call_args_list[0][0]
    assert path == "/api/performance/summary/"
    assert params["repository"] == "autoland"
    assert params["signature"] == SIGNATURE_HASH
    assert params["all_data"] == "true"
    assert params["replicates"] == "false"
    assert params["interval"] == 14 * 86400
    assert "framework" not in params

    assert mocked_th_client.get.call_args_list[1][0][1]["replicates"] == "true"

    assert list(result["push_id"]) == [100, 101, 101]
    assert list(result["value"]) == [10.0, 11.0, 13.0]
    assert list(result["datum_id"]) == [1, 2, 3]
    # The data point without replicates falls back to its aggregated value.
    assert list(result["trials"]) == [[9.0, 11.0], [10.5, 11.5], [13.0]]
    assert result.attrs["lower_is_better"] is True
    assert result.attrs["alert_threshold"] == 2.0
    assert "data" not in result.attrs


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_signature_table_no_replicates(mocked_th_client):
    _mock_client(mocked_th_client, [_series(AGGREGATED_DATA)])

    result = get_signature_table("autoland", SIGNATURE_HASH, replicates=False, per_push=False)

    assert len(mocked_th_client.get.call_args_list) == 1
    assert mocked_th_client.get.call_args_list[0][0][1]["replicates"] == "false"
    # The measurements sit in the same column either way; with replicates off
    # they're the aggregated values.
    assert list(result["trials"]) == [[10.0], [11.0], [13.0]]


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_signature_table_options(mocked_th_client):
    _mock_client(mocked_th_client, [_series(AGGREGATED_DATA)])

    get_signature_table(
        "mozilla-central",
        "298906",
        framework=13,
        days=30,
        replicates=False,
        no_retriggers=True,
        per_push=False,
    )

    params = mocked_th_client.get.call_args_list[0][0][1]
    assert params["repository"] == "mozilla-central"
    assert params["signature"] == "298906"
    assert params["framework"] == 13
    assert params["interval"] == 30 * 86400
    assert params["no_retriggers"] == "true"


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_signature_table_date_range(mocked_th_client):
    _mock_client(mocked_th_client, [_series(AGGREGATED_DATA)])

    get_signature_table(
        "autoland",
        SIGNATURE_HASH,
        replicates=False,
        from_date="2026-09-01",
        to_date="2026-09-05",
        per_push=False,
    )

    params = mocked_th_client.get.call_args_list[0][0][1]
    assert params["startday"] == "2026-09-01T00:00:00"
    assert params["endday"] == "2026-09-05T00:00:00"
    assert "interval" not in params


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_signature_table_from_date_only(mocked_th_client):
    _mock_client(mocked_th_client, [_series(AGGREGATED_DATA)])

    get_signature_table(
        "autoland",
        SIGNATURE_HASH,
        replicates=False,
        from_date="2026-09-01",
        per_push=False,
    )

    params = mocked_th_client.get.call_args_list[0][0][1]
    assert params["startday"] == "2026-09-01T00:00:00"
    # An open-ended range runs to now.
    assert params["endday"] > params["startday"]


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_no_signature_matched(mocked_th_client):
    _mock_client(mocked_th_client, [[]])

    with pytest.raises(TreeherderQueryError):
        get_signature_table("autoland", SIGNATURE_HASH)


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_multiple_signatures_matched(mocked_th_client):
    _mock_client(
        mocked_th_client,
        [
            [
                dict(SIGNATURE_METADATA, data=AGGREGATED_DATA),
                dict(SIGNATURE_METADATA, framework_id=13, data=AGGREGATED_DATA),
            ]
        ],
    )

    with pytest.raises(MultipleSignaturesError):
        get_signature_table("autoland", SIGNATURE_HASH)


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_no_data(mocked_th_client):
    _mock_client(mocked_th_client, [_series([]), _series([])])

    with pytest.raises(TreeherderQueryError):
        get_signature_table("autoland", SIGNATURE_HASH)


SIGNATURES_RESPONSE = {
    "2": {
        "id": 2,
        "signature_hash": "hash2",
        "framework_id": 1,
        "suite": "tp5o_scroll",
        "machine_platform": "linux2404-64",
    },
    "1": {
        "id": 1,
        "signature_hash": "hash1",
        "framework_id": 1,
        "suite": "rasterflood_svg",
        "machine_platform": "macosx1470-64",
    },
    "3": {
        "id": 3,
        "signature_hash": "hash3",
        "framework_id": 1,
        "suite": "pdfpaint",
        "test": "xfa_bug1718521_3.pdf",
        "machine_platform": "macosx1470-64",
    },
}


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_signatures(mocked_th_client):
    _mock_client(mocked_th_client, [SIGNATURES_RESPONSE])

    signatures = get_signatures("autoland", framework=1, platform="macosx1470-64")

    path, params = mocked_th_client.get.call_args_list[0][0]
    assert path == "/api/project/autoland/performance/signatures/"
    assert params["framework"] == 1
    assert params["platform"] == "macosx1470-64"
    assert params["subtests"] == 1
    assert params["interval"] == 14 * 86400

    # Sorted by suite, test, then platform, with the response key kept.
    assert [signature["suite"] for signature in signatures] == [
        "pdfpaint",
        "rasterflood_svg",
        "tp5o_scroll",
    ]
    assert [signature["signature_id"] for signature in signatures] == [3, 1, 2]


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_signatures_suite_and_test_filters(mocked_th_client):
    _mock_client(mocked_th_client, [SIGNATURES_RESPONSE, SIGNATURES_RESPONSE])

    assert [
        signature["signature_hash"] for signature in get_signatures("autoland", suite="flood")
    ] == ["hash1"]
    assert [
        signature["signature_hash"] for signature in get_signatures("autoland", test="xfa")
    ] == ["hash3"]


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_signatures_no_subtests(mocked_th_client):
    _mock_client(mocked_th_client, [SIGNATURES_RESPONSE])

    get_signatures("autoland", subtests=False)

    assert mocked_th_client.get.call_args_list[0][0][1]["subtests"] == 0


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_per_push(mocked_th_client):
    _mock_client(mocked_th_client, [_series(AGGREGATED_DATA), _series(REPLICATE_DATA)])

    pushes = get_signature_table("autoland", SIGNATURE_HASH)

    assert list(pushes["push_id"]) == [100, 101]
    assert list(pushes["values"]) == [[10.0], [11.0, 13.0]]
    # The retrigger's replicates join the push's trials, the one without
    # replicates contributes its aggregated value.
    assert list(pushes["trials"]) == [[9.0, 11.0], [10.5, 11.5, 13.0]]
    assert list(pushes["value_count"]) == [1, 2]
    assert list(pushes["trial_count"]) == [2, 3]
    assert list(pushes["value"]) == [10.0, 11.5]
    assert list(pushes["revision"]) == ["revision100", "revision101"]
    assert pushes.attrs["lower_is_better"] is True


@mock.patch("mozdetect.treeherder_query.TreeherderClient")
def test_treeherder_query_per_push_without_replicates(mocked_th_client):
    _mock_client(mocked_th_client, [_series(AGGREGATED_DATA)])

    pushes = get_signature_table("autoland", SIGNATURE_HASH, replicates=False)

    assert list(pushes["values"]) == [[10.0], [11.0, 13.0]]
    assert list(pushes["trials"]) == [[10.0], [11.0, 13.0]]
