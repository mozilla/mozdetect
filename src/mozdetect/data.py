# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import json
import numpy as np
import pandas
import traceback

from collections import Counter
from datetime import datetime, timedelta


class InvalidNumberError(Exception):
    """Raised when an invalid number of points is requested."""

    pass


class UnknownDataTypeError(Exception):
    """Raised when an unknown data type is requested."""

    pass


class TimeSeries:
    """Represents a time series composed of Datum objects.

    Primarily a wrapper around `pandas.DataFrame`. It provides some additional
    helper functions to make it simpler to iterate over the data, and provide
    a common interface for all detectors to use.

    The pandas.DataFrame was chosen for this so that strings could be included
    in the dataset alongside numerical input. It also easily allows multidimensional
    data to be used.
    """

    def __init__(self, data, data_type="all", **kwargs):
        """Initializes the time series.

        :param list data: A list of tuples representing the time series.
        :param str/list data_type: The data type that should be iterated over. See
            `set_data_type` for more information.
        :param kwargs: A set of options that can be used to finetune the
            options passed into the pandas DataFrame creation.
        """
        self._original_data = self._prepare_data(data, **kwargs)
        self._iteration_data = self._original_data
        self._data_type = "all"

        self.numerical_data = self.get_numerical_columns()
        self.nonnumerical_data = self.get_nonnumerical_columns()

        # Only set data type after getting the numberical, and
        # non-numerical columns from the original data
        self.set_data_type(data_type)

        self._currind = 0
        self._curr = None

    def __iter__(self):
        """Helper method to iterate over the timeseries.

        :return DataFrame: A row in the DataFrame.
        """
        for index, row in self.data.iterrows():
            self._currind = index
            self._curr = pandas.DataFrame(row).T

            yield self._curr

    @property
    def data(self):
        return self._iteration_data

    def _prepare_data(self, data, **kwargs):
        """Formats the data into a pandas DataFrame for detectors.

        Note that since the DataFrame doesn't have restrictions on types,
        multiple types can be combined into a single tuple, e.g. (1, 2, "h")
        is valid. As is the following when data is missing: [(1, 2), (3, 4, 5)]

        :param list data: A list of tuples representing the time series.
        :param kwargs: A set of options that can be used to finetune the
            options passed into the pandas DataFrame creation.

        :return DataFrame: The data converted to a pandas DataFrame.
        """
        return pandas.DataFrame(data=data, **kwargs)

    def set_data_type(self, data_type):
        """Used to set the data type to iterate over.

        By default, all the data will be iterated over. If "numerical" is passed
        here, then only the numerical data will be returned in all methods. If
        "non-numerical" is passed, then only the non-numerical data will be returned.
        This can be reset to all data by passing "all". A custom type may also
        be passed for special returns.

        :param str/list data_type: Either "numerical", "non-numerical", or "all" to
            denote the type of data. Alternatively, pass a list of custom types to get
            alternative data.
        """
        self._data_type = data_type
        if isinstance(self._data_type, str):
            if self._data_type == "numerical":
                self._iteration_data = self.numerical_data
            elif self._data_type == "non-numerical":
                self._iteration_data = self.nonnumerical_data
            elif self._data_type == "all":
                self._iteration_data = self._original_data
            else:
                raise UnknownDataTypeError(
                    f"Unknown data type requested for iteration: {self._data_type}"
                )
        elif isinstance(self._data_type, list):
            try:
                self._iteration_data = self._original_data.select_dtypes(include=self._data_type)
            except Exception:
                raise UnknownDataTypeError(
                    f"Failed to get custom data types for {str(self._data_type)}:"
                    f" {traceback.format_exc()}"
                )
        else:
            raise UnknownDataTypeError("Expecting list or str as type for data type.")

    def get_current(self):
        """Returns the current datum being analyzed.

        :return DataFrame: The row at the current position in the DataFrame.
        """
        if self._curr is None:
            self._curr = self.data.iloc[[self._currind]]
        return self._curr

    def get_next_n(self, n, inclusive=False):
        """Returns the next `n` data points in the time series.

        This methods is exclusive, and doesn't include the current data point
        that is being analyzed.

        :param int n: The number of data points to get.

        :return DataFrame: The number of requested rows if they exist. If the
            current position is at the end of the timeseries, then nothing will
            be returned.
        """
        if n <= 0:
            raise InvalidNumberError("Number of data points must be greater than 0.")

        start_ind = self._currind + 1
        end_ind = self._currind + n + 1
        if inclusive:
            start_ind = start_ind - 1
            end_ind = end_ind - 1

        return self.data.iloc[start_ind:end_ind]

    def get_previous_n(self, n, inclusive=False):
        """Returns the previous `n` data points in the time series.

        This methods is exclusive, and doesn't include the current data point
        that is being analyzed.

        :param int n: The number of data points to get.

        :return DataFrame: The number of requested rows if they exist. If the
            current position is at the beginning of the timeseries, then nothing will
            be returned.
        """
        if n <= 0:
            raise InvalidNumberError("Number of data points must be greater than 0.")

        start_ind = max(self._currind - n, 0)
        end_ind = self._currind
        if inclusive:
            if self._currind == 0:
                start_ind = 0
            else:
                start_ind = start_ind + 1
            end_ind = end_ind + 1

        return self.data.iloc[start_ind:end_ind]

    def get_numerical_columns(self):
        """Returns the data but with only numerical columns.

        :return DataFrame: Returns the data with non-numerical columns removed.
        """
        return self.data.select_dtypes(include=[np.number])

    def get_nonnumerical_columns(self):
        """Returns the data but with only non-numerical columns.

        :return DataFrame: Returns the data with numerical columns removed.
        """
        return self.data.select_dtypes(exclude=[np.number])


class TelemetryTimeSeries(TimeSeries):
    """Provides additional telemetry-specific methods, and exposes the
    raw data in the `raw_data` attribute to provide more customization.
    """

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.raw_data = data
        self.cumulative_by_day_histograms = pandas.DataFrame()
        self.cumulative_multiday_histograms = pandas.DataFrame()

    def _build_id_to_date(self, build_id):
        """Returns a build ID converted into a datetime.date.

        :param str build_id: The build ID to convert.

        :return datetime.date: The date that was parsed from the build ID.
        """
        if len(build_id) == 10:
            return datetime.strptime(build_id, "%Y%m%d%H").date()
        else:
            return datetime.strptime(build_id, "%Y%m%d%H%M%S").date()

    def get_percentiles_over_time(self, histograms=None):
        """Returns the percentiles over time from the telemetry data."""

        def _extract_percentiles(group):
            filtered = group[group["cdf"] > 0.5]
            retval = filtered.iloc[0][["date"]]
            retval["p50"] = filtered.iloc[0]["bin"]
            retval["p75"] = group[group["cdf"] > 0.75].iloc[0]["bin"]
            retval["p95"] = group[group["cdf"] > 0.95].iloc[0]["bin"]
            return retval

        if histograms is None:
            if not self.cumulative_by_day_histograms.empty:
                histograms = self.cumulative_by_day_histograms
            else:
                histograms = self.raw_data

        return (
            histograms.groupby("date", as_index=False)[["date", "bin", "cdf"]]
            .apply(_extract_percentiles)
            .dropna()
            .reset_index(drop=True)
        )

    def get_cumulative_by_day(self, start_date=None, end_date=None):
        """Returns the data downsampled into a per-day granularity.

        :param datetime.date start_date: The start date to downsample from.
        :param datetime.date end_date: The end date to downsample to.

        :return pandas.DataFrame: A pandas.DateFrame of the cumulative histograms at a per-day
            granularity. Can also be obtained from `self.cumulative_histograms`.
        """
        last_start_row = 0
        current_date = start_date or self._build_id_to_date(self.raw_data.iloc[0].build_id)
        end_date = end_date or self._build_id_to_date(self.raw_data.iloc[-1].build_id)

        while current_date <= end_date:
            # Find the start date location
            # TODO: get rid of while True loops
            while True:
                current_row = self.raw_data.iloc[last_start_row]
                current_row_date = self._build_id_to_date(current_row.build_id)
                if current_row_date < current_date:
                    last_start_row += 1
                else:
                    break

            # Gather the data from builds in the current day
            summed_histograms_raw = Counter()
            current_row_iloc = last_start_row
            while current_row_iloc < len(self.raw_data):
                current_row = self.raw_data.iloc[current_row_iloc]
                current_row_date = self._build_id_to_date(current_row.build_id)
                if current_row_date > current_date:
                    break
                if not current_row.non_norm_histogram:
                    current_row_iloc += 1
                    continue

                summed_histograms_raw.update(json.loads(current_row.non_norm_histogram))
                current_row_iloc += 1

            summed_histogram = pandas.DataFrame(
                list(summed_histograms_raw.items()), columns=["bin", "count"]
            )
            if summed_histogram.empty:
                current_date += timedelta(days=1)
                continue

            # Produce the cumulative histogram for this day
            summed_histogram["bin"] = summed_histogram["bin"].astype("Int64")
            summed_histogram["date"] = current_date

            summed_histogram = summed_histogram.sort_values(by="bin").reset_index(drop=True)
            summed_histogram["cumulative"] = summed_histogram["count"].cumsum()
            summed_histogram["cdf"] = (
                summed_histogram["cumulative"] / summed_histogram["count"].sum()
            )

            self.cumulative_by_day_histograms = pandas.concat(
                [summed_histogram.fillna(0), self.cumulative_by_day_histograms], ignore_index=True
            )
            current_date += timedelta(days=1)

        return self.cumulative_by_day_histograms

    def get_multiday_average(self, days=7):
        """Produces a multiday average of the cumulative per-day histograms using
        a rolling window.

        :param int days: The number of days to average together.

        :return pandas.DataFrame: The data with each point being a multiday average instead
            of only a single day.
        """
        if self.cumulative_by_day_histograms is None:
            raise Exception(
                "get_multiday_average expects get_cumulative_by_day to be called first."
            )

        start_date = self.cumulative_by_day_histograms["date"].min()
        end_date = self.cumulative_by_day_histograms["date"].max()

        # Initialize the current sum with the start date
        current_date = start_date
        current_sum = self.cumulative_by_day_histograms[
            self.cumulative_by_day_histograms["date"] == current_date
        ][["bin", "count"]].reset_index(drop=True)

        # Gather the first set of days-1 into a sum
        # TODO use walrus here
        current_date += timedelta(days=1)
        while current_date < start_date + timedelta(days=days - 1):
            hist_to_add = self.cumulative_by_day_histograms[
                self.cumulative_by_day_histograms["date"] == current_date
            ][["bin", "count"]].reset_index(drop=True)["count"]
            if not hist_to_add.empty:
                current_sum["count"] += hist_to_add
            current_date += timedelta(days=1)

        # Add one day at a time to the current_sum, and remove the last day from it
        # after adding the current_sum to the multiday_window data
        # TODO use walrus here
        current_date = start_date
        date_to_add = current_date + timedelta(days=days - 1)
        while date_to_add <= end_date:
            # Add new date data if it has data
            hist_to_add = self.cumulative_by_day_histograms[
                self.cumulative_by_day_histograms["date"] == date_to_add
            ][["bin", "count"]].reset_index(drop=True)["count"]
            if not hist_to_add.empty:
                current_sum["count"] += hist_to_add

            # Calculate new CDF, and add to multiday_window
            current_sum["date"] = current_date
            current_sum["cumulative"] = current_sum["count"].cumsum()
            current_sum["cdf"] = current_sum["cumulative"] / current_sum["count"].sum()
            self.cumulative_multiday_histograms = pandas.concat(
                [current_sum.copy(), self.cumulative_multiday_histograms], ignore_index=True
            )

            # Remove current date from current_sum if it had data
            hist_to_remove = self.cumulative_by_day_histograms[
                self.cumulative_by_day_histograms["date"] == current_date
            ][["bin", "count"]].reset_index(drop=True)["count"]
            if not hist_to_remove.empty:
                current_sum["count"] -= hist_to_remove

            current_date += timedelta(days=1)
            date_to_add += timedelta(days=1)

        return self.cumulative_multiday_histograms


class TreeherderTimeSeries(TimeSeries):
    """Provides additional Perfherder-specific methods, and exposes the
    raw data in the `raw_data` attribute to provide more customization.

    Built with a table from `mozdetect.get_signature_table`. The signature's
    metadata is exposed here as `metadata`.
    """

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.raw_data = data
        self.metadata = dict(getattr(data, "attrs", None) or {})
        self.by_day = pandas.DataFrame()
        self.multiday = pandas.DataFrame()

    @property
    def lower_is_better(self):
        """Whether a drop in this signature's values is an improvement."""
        return self.metadata.get("lower_is_better", True)

    @property
    def alert_threshold(self):
        """How much the signature has to move before the change matters, as set
        by the test itself. None when the test hasn't made that judgement.
        """
        return self.metadata.get("alert_threshold")

    @property
    def measurement_unit(self):
        return self.metadata.get("measurement_unit") or ""

    @property
    def revisions(self):
        """Returns a mapping of push ID to revision.

        A detection is identified by its revision, which is what a sheriff
        looks the change up by.
        """
        if "revision" not in self.raw_data:
            return {}
        return dict(zip(self.raw_data["push_id"], self.raw_data["revision"]))

    def get_trials(self, rows=None):
        """Returns the individual measurements in the given rows, as a flat list.

        :param pandas.DataFrame rows: The rows to pull measurements from, such
            as a window from `get_previous_n`. Defaults to the whole series.

        :return list: The measurements the given rows recorded.
        """
        if rows is None:
            rows = self.raw_data

        return [value for trials in rows["trials"] for value in trials]

    def get_by_day(self):
        """Returns the data downsampled into a per-day granularity.

        :return pandas.DataFrame: A pandas.DataFrame of the measurements pooled
            at a per-day granularity, ordered oldest first. Can also be obtained
            from `self.by_day`.
        """
        rows = []
        for date, group in self.raw_data.groupby(
            self.raw_data["push_timestamp"].dt.date, sort=True
        ):
            trials = self.get_trials(group)
            rows.append(
                {
                    "date": date,
                    "value": np.median(trials),
                    "mean": np.mean(trials),
                    "trials": trials,
                    "push_count": group["push_id"].nunique(),
                    "trial_count": len(trials),
                }
            )

        self.by_day = pandas.DataFrame(rows).reset_index(drop=True)
        return self.by_day

    def get_multiday_average(self, days=7):
        """Produces a multiday average of the per-day measurements using a
        rolling window.

        Each row holds every measurement the `days` days up to and including
        its date recorded.

        :param int days: The number of days to pool together.

        :return pandas.DataFrame: The data with each point covering `days` days
            instead of only a single day. Can also be obtained from
            `self.multiday`.
        """
        if self.by_day.empty:
            self.get_by_day()

        rows = []
        for end in range(days - 1, len(self.by_day)):
            window = self.by_day.iloc[end - days + 1 : end + 1]
            trials = [value for trials in window["trials"] for value in trials]

            rows.append(
                {
                    "date": window["date"].iloc[-1],
                    "value": np.median(trials),
                    "mean": np.mean(trials),
                    "trials": trials,
                    "push_count": int(window["push_count"].sum()),
                    "trial_count": len(trials),
                }
            )

        self.multiday = pandas.DataFrame(rows).reset_index(drop=True)
        return self.multiday
