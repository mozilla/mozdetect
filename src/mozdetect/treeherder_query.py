# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Queries Treeherder for Perfherder timeseries data.

This is the Perfherder counterpart to `telemetry_query`: where that module
pulls per-build histograms out of BigQuery, this one pulls per-push performance
data out of a Treeherder API server, so a detector can be run over production
data from anywhere. Nothing here writes.

The data comes from `/api/performance/summary/` with `all_data=true`, the
endpoint the Perfherder graphs view uses, which is what makes the replicates
behind each data point reachable. Code running inside Treeherder itself has the
same data in the database rather than over HTTP, and can shape a query's rows
into the same table with `get_signature_table_from_query`.
"""
import logging

from collections.abc import Mapping

import pandas
import requests


DEFAULT_SERVER = "https://treeherder.mozilla.org"

# How much recent data to pull by default, matching the window production
# alert generation works over (Treeherder's PERFHERDER_ALERTS_MAX_AGE).
DEFAULT_DAYS = 14

SUMMARY_ENDPOINT = "/api/performance/summary/"
SIGNATURES_ENDPOINT = "/api/project/{project}/performance/signatures/"

REQUEST_TIMEOUT = 120

USER_AGENT = "mozdetect-treeherder-query"

# The columns a table of data points is built with. Named here so a query
# that selected nothing still gives back a table a detector can read; a table
# read over the API carries whatever else the endpoint reports on top.
DATUM_COLUMNS = [
    "datum_id",
    "job_id",
    "push_id",
    "push_timestamp",
    "revision",
    "value",
    "trials",
]

# The same, once the data points have been collapsed into their pushes.
PUSH_COLUMNS = [
    "push_id",
    "push_timestamp",
    "revision",
    "value",
    "mean",
    "values",
    "trials",
    "value_count",
    "trial_count",
]

# The `values()` key each column is read from. Anything optional a query
# leaves out comes back empty.
DATUM_FIELDS = {
    "datum_id": "id",
    "job_id": "job_id",
    "push_id": "push_id",
    "push_timestamp": "push_timestamp",
    "revision": "push__revision",
    "value": "value",
}

# Selecting this joins the replicates in, one row per replicate.
REPLICATE_FIELD = "performancedatumreplicate__value"

# What a query selects: `.values(*QUERY_FIELDS)`. The replicate field is part
# of it rather than an option, so a series is always compared on the finest
# measurements the jobs recorded.
QUERY_FIELDS = (*DATUM_FIELDS.values(), REPLICATE_FIELD)

# Without these there's no series to build, so a query that missed one is
# refused. `id` is what tells the joined rows of one data point apart from
# the next, so the series collapses into a single point without it.
REQUIRED_DATUM_FIELDS = ("id", "push_id", "push_timestamp", "value", REPLICATE_FIELD)

logger = logging.getLogger("TreeherderQuery")


class TreeherderQueryError(Exception):
    """Raised when a Treeherder server can't be read, or reports no usable data."""

    pass


class MultipleSignaturesError(TreeherderQueryError):
    """Raised when more than one series matched the signature that was asked for."""

    pass


class TreeherderClient:
    session = None
    server = None

    def __init__(self, server=DEFAULT_SERVER):
        if TreeherderClient.session is None or TreeherderClient.server != server.rstrip("/"):
            TreeherderClient.initialize_th_client(server)

    @classmethod
    def initialize_th_client(cls, server=DEFAULT_SERVER):
        TreeherderClient.session = requests.Session()
        TreeherderClient.session.headers.update({"User-Agent": USER_AGENT})
        TreeherderClient.server = server.rstrip("/")

    @classmethod
    def get(cls, path, params):
        """Reads one JSON document off the Treeherder API.

        :param str path: The endpoint path to read, e.g. `/api/performance/summary/`.
        :param dict params: The query parameters to send.

        :return: The decoded JSON response.
        """
        url = f"{TreeherderClient.server}{path}"
        logger.debug(f"Running query against {url} with {params}")

        try:
            response = TreeherderClient.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as error:
            raise TreeherderQueryError(f"Could not read {url}: {error}")

        return response.json()


def _get_interval(days):
    return int(days * 86400)


def _get_framework_filter(framework):
    return {} if framework is None else {"framework": framework}


def _get_time_filter(days, from_date=None, to_date=None):
    """Returns the parameters selecting how much of the series to read.

    A date range is used when one is given, otherwise the last `days` days.
    The endpoint needs both ends of a range, so an open-ended `from_date`
    runs to now.
    """
    if from_date is None and to_date is None:
        return {"interval": _get_interval(days)}

    return {
        "startday": _to_api_datetime(from_date)
        if from_date
        else _to_api_datetime(pandas.Timestamp.utcnow() - pandas.Timedelta(days=days)),
        "endday": _to_api_datetime(to_date)
        if to_date
        else _to_api_datetime(pandas.Timestamp.utcnow()),
    }


def _to_api_datetime(value):
    """Returns a date, datetime, or date string as the API's datetime format."""
    return pandas.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")


def _get_summary_params(
    project,
    signature,
    framework=None,
    days=DEFAULT_DAYS,
    replicates=False,
    all_data=True,
    no_retriggers=False,
    from_date=None,
    to_date=None,
):
    params = {
        "repository": project,
        "signature": signature,
        "all_data": "true" if all_data else "false",
        "replicates": "true" if replicates else "false",
    }
    if no_retriggers:
        params["no_retriggers"] = "true"

    params.update(_get_framework_filter(framework))
    params.update(_get_time_filter(days, from_date=from_date, to_date=to_date))
    return params


def _get_performance_summary(
    project,
    signature,
    framework=None,
    days=DEFAULT_DAYS,
    replicates=False,
    all_data=True,
    no_retriggers=False,
    from_date=None,
    to_date=None,
):
    """Reads one series off the performance summary endpoint.

    :return dict: The series, as the endpoint reports it: the signature's
        metadata, with its data points under the `data` key.
    """
    items = TreeherderClient.get(
        SUMMARY_ENDPOINT,
        _get_summary_params(
            project,
            signature,
            framework=framework,
            days=days,
            replicates=replicates,
            all_data=all_data,
            no_retriggers=no_retriggers,
            from_date=from_date,
            to_date=to_date,
        ),
    )

    if not items:
        raise TreeherderQueryError(
            f"No series matched signature {signature} on {project} " f"in the last {days} days."
        )
    if len(items) > 1:
        matches = ", ".join(f"{item['framework_id']} ({item['platform']})" for item in items[:10])
        raise MultipleSignaturesError(
            f"More than one signature matched; pass a framework to pick one of {matches}"
        )

    return items[0]


def get_signatures(
    project,
    framework=None,
    suite=None,
    test=None,
    platform=None,
    subtests=True,
    days=DEFAULT_DAYS,
    server=DEFAULT_SERVER,
):
    """Returns the signatures a project has data for.

    Use this to find the signature to pass to `get_signature_table`, the way
    `get_metric_labels` is used to find a metric's labels.

    :param str project: The repository to query, e.g. `autoland`.
    :param int framework: Only return signatures from this framework ID.
    :param str suite: Only return signatures whose suite contains this string.
    :param str test: Only return signatures whose test contains this string.
    :param str platform: Only return signatures on this platform, e.g.
        `linux2404-64-shippable`. Matched exactly, as the API filters on it.
    :param bool subtests: If false, only return summary signatures.
    :param int days: Only return signatures updated in this many days.
    :param str server: The Treeherder server to read from.

    :return list: The matching signatures, as a list of dicts sorted by suite,
        test, and platform.
    """
    TreeherderClient(server=server)

    params = {
        "interval": _get_interval(days),
        "subtests": 1 if subtests else 0,
    }
    if platform:
        params["platform"] = platform
    params.update(_get_framework_filter(framework))

    signatures = TreeherderClient.get(SIGNATURES_ENDPOINT.format(project=project), params)

    # The endpoint has no suite or test filter, so those are applied here.
    results = [
        dict(signature, signature_id=int(signature_id))
        for signature_id, signature in signatures.items()
        if (suite is None or suite.lower() in signature.get("suite", "").lower())
        and (test is None or test.lower() in signature.get("test", "").lower())
    ]

    return sorted(
        results,
        key=lambda signature: (
            signature.get("suite", ""),
            signature.get("test", ""),
            signature.get("machine_platform", ""),
        ),
    )


def _check_values_row(row, description):
    """Checks that a row came from a Django `values()` query.

    Rows are taken in one format only, so a query that hands back model
    instances or named rows is caught here rather than quietly reading empty
    columns out of them.

    :param row: The row to check.
    :param str description: What the row was expected to hold, for the error.
    """
    if not isinstance(row, Mapping):
        raise TypeError(
            f"Expecting {description} from a Django `values()` query, "
            f"got {type(row).__name__}. Select the fields to read with "
            "`.values(...)` rather than passing model instances or named rows."
        )


def _read_datum(datum):
    """Reads one row of the query into the table's columns.

    :param dict datum: A row of a `values()` query.

    :return dict: The row's columns, with an empty `trials` to fill in.
    """
    _check_values_row(datum, "a data point")

    missing = [name for name in REQUIRED_DATUM_FIELDS if name not in datum]
    if missing:
        raise ValueError(
            f"The query didn't select {', '.join(missing)}, which the series is "
            "built from. Select the fields with `.values(*QUERY_FIELDS)`."
        )

    row = {column: datum.get(name) for column, name in DATUM_FIELDS.items()}
    # Filled in from the rows this data point was joined across.
    row["trials"] = []

    return row


def _get_signature_metadata(signature):
    """Reads the metadata a signature carries into the table's `attrs`.

    This is what decides how a detected change gets interpreted, so it travels
    with the data rather than being looked up again later. Whatever the
    signature's own query selected is taken as it stands, under the names it
    selected them with.

    :param dict signature: The signature's fields, from a `values()` query.

    :return dict: The signature's metadata.
    """
    if signature is None:
        return {}

    _check_values_row(signature, "a signature")

    return dict(signature)


def _finalize_table(result, metadata, per_push):
    """Puts a table in the order and shape every reader expects.

    :param pandas.DataFrame result: The table to finalize.
    :param dict metadata: The signature metadata to carry on `attrs`.
    :param bool per_push: Whether to collapse the table to one row per push.

    :return pandas.DataFrame: The table, ordered oldest first.
    """
    result["push_timestamp"] = pandas.to_datetime(result["push_timestamp"])
    if "submit_time" in result:
        result["submit_time"] = pandas.to_datetime(result["submit_time"])

    result = result.sort_values(by=["push_timestamp", "push_id"]).reset_index(drop=True)
    result.attrs = metadata

    if per_push:
        return _aggregate_by_push(result)

    return result


def get_signature_table_from_query(datums, signature=None, per_push=True):
    """Returns the timeseries a Django query selected, as `get_signature_table`
    would have read it over HTTP.

    For code running inside Treeherder, where the data is in the database
    rather than behind the API. Both arguments are the rows of a `values()`
    query, and nothing else::

        datums = PerformanceDatum.objects.filter(
            signature=signature, push_timestamp__gte=since
        ).values(*QUERY_FIELDS)

        signature_fields = (
            PerformanceSignature.objects.filter(id=signature.id).values().first()
        )

        table = get_signature_table_from_query(datums, signature_fields)

    `QUERY_FIELDS` always selects the replicates, which joins them in the way
    Perfherder's own endpoint reads them: one row per replicate, with the data
    point's fields repeated across them, and a null replicate value for the
    jobs that didn't report any.

    Everything the signature's own query selected becomes the metadata, under
    the names it selected them with, so a bare `values()` gathers the lot. The
    related rows are the exception: `values()` reports those as the ids they
    are, so a query wanting the platform or repository by name has to ask for
    `platform__platform` or `repository__name`.

    :param datums: The rows to build the table from, as a `values(*QUERY_FIELDS)`
        queryset or any iterable of its rows.
    :param dict signature: The signature's fields, from a `values()` query.
        They travel as they are on the table's `attrs`.
    :param bool per_push: Whether to collapse the series to one row per push.
        On by default, since change detection works over pushes, not jobs.

    :return pandas.DataFrame: The series, ordered oldest first. One row per push,
        or one row per data point when `per_push` is off. A query that selected
        nothing gives an empty table that still carries the usual columns.
    """
    rows = {}
    for datum in datums:
        row = _read_datum(datum)
        # The data point's own fields repeat across the rows its replicates
        # were joined onto, so the first row it appears in is the one kept.
        row = rows.setdefault(row["datum_id"], row)

        replicate = datum.get(REPLICATE_FIELD)
        if replicate is not None:
            row["trials"].append(replicate)

    # The measurements always land in `trials`, so what a detector compares
    # is in the same place either way: a data point the query returned no
    # replicates for falls back to the aggregated value it recorded.
    for row in rows.values():
        if not row["trials"]:
            row["trials"] = [row["value"]]

    result = pandas.DataFrame(
        [row for row in rows.values() if row["value"] is not None], columns=DATUM_COLUMNS
    )
    return _finalize_table(result, _get_signature_metadata(signature), per_push)


def _aggregate_by_push(dataframe):
    """Collapses a signature table down to one row per push.

    A push can hold several data points for the same signature -- retriggers,
    backfills, or a job that simply runs more than once -- and change detection
    works over pushes rather than jobs. The values recorded at each push are
    kept in a `values` column, and the measurements behind them stay in the
    `trials` column they arrived in.

    :param pandas.DataFrame dataframe: A table of data points, as either of
        the entry points above builds it.

    :return pandas.DataFrame: One row per push, ordered oldest first. The
        signature's metadata carries over from the given table's `attrs`.
    """
    rows = []
    for push_id, group in dataframe.groupby("push_id", sort=False):
        values = list(group["value"])
        trials = [value for trials in group["trials"] for value in trials]

        rows.append(
            {
                "push_id": push_id,
                "push_timestamp": group["push_timestamp"].iloc[0],
                "revision": group["revision"].iloc[0],
                "value": pandas.Series(trials).median(),
                "mean": pandas.Series(trials).mean(),
                "values": values,
                "trials": trials,
                "value_count": len(values),
                "trial_count": len(trials),
            }
        )

    result = pandas.DataFrame(rows, columns=PUSH_COLUMNS)
    result = result.sort_values(by=["push_timestamp", "push_id"]).reset_index(drop=True)
    result.attrs = dict(dataframe.attrs)

    return result


def get_signature_table(
    project,
    signature,
    framework=None,
    days=DEFAULT_DAYS,
    replicates=True,
    server=DEFAULT_SERVER,
    no_retriggers=False,
    from_date=None,
    to_date=None,
    per_push=True,
):
    """Returns the timeseries of one performance signature.

    The aggregated value of each data point is always read, since that's what
    production's alerting works on. When `replicates` is on, the replicates
    behind those data points are read as well. Either way, what the detectors
    compare lands in a `trials` column. The signature's metadata is available
    on the returned frame's `attrs`.

    :param str project: The repository the series belongs to, e.g. `autoland`.
    :param str signature: The signature hash, or signature ID, of the series.
    :param int framework: Framework ID, to pick between signatures sharing a hash.
    :param int days: How many days of data to read.
    :param bool replicates: Whether to read the replicates behind each data point.
    :param str server: The Treeherder server to read from.
    :param bool no_retriggers: If true, drop retriggered jobs from the series.
    :param from_date: Read from this date instead of the last `days` days.
        Takes a date, datetime, or `YYYY-MM-DD` string.
    :param to_date: Read up to this date. Defaults to now when `from_date` is given.
    :param bool per_push: Whether to collapse the series to one row per push.
        On by default, since change detection works over pushes, not jobs.

    :return pandas.DataFrame: The series, ordered oldest first. One row per push,
        or one row per data point when `per_push` is off.
    """
    TreeherderClient(server=server)

    logger.debug("Running query...")
    aggregated = _get_performance_summary(
        project,
        signature,
        framework=framework,
        days=days,
        replicates=False,
        no_retriggers=no_retriggers,
        from_date=from_date,
        to_date=to_date,
    )

    trials_by_datum = {}
    if replicates:
        replicate_series = _get_performance_summary(
            project,
            signature,
            framework=framework,
            days=days,
            replicates=True,
            no_retriggers=no_retriggers,
            from_date=from_date,
            to_date=to_date,
        )
        for row in replicate_series["data"]:
            trials_by_datum.setdefault(row["id"], []).append(row["value"])

    result = pandas.DataFrame(aggregated["data"])
    if result.empty:
        raise TreeherderQueryError(
            f"No data for signature {signature} on {project} in the last {days} days."
        )

    result = result.rename(columns={"id": "datum_id"})
    # The measurements always land in `trials`, whether those are the
    # replicates behind each data point or the aggregated value standing in for
    # them, so what a detector compares is in the same place either way.
    result["trials"] = [
        trials_by_datum.get(datum_id) or [value]
        for datum_id, value in zip(result["datum_id"], result["value"])
    ]

    return _finalize_table(
        result,
        {key: value for key, value in aggregated.items() if key != "data"},
        per_push,
    )
