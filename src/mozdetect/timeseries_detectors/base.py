# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas

from mozdetect.detectors.base import (
    DEFAULT_DETECTOR_TYPE,
    DETECTOR_TYPES,
    check_detector_type,
)


class TimeSeriesDetectorRegistry:
    _timeseries_detectors = {detector_type: {} for detector_type in DETECTOR_TYPES}

    @staticmethod
    def add(
        timeseries_detector_class, timeseries_detector_name, detector_type=DEFAULT_DETECTOR_TYPE
    ):
        """Add a timeseries detector to the registry of timeseries detectors.

        Timeseris Detectors added to the registry will become available through the
        `get_timeseries_detectors` method using the provided `timeseries_detector_name`,
        for the given `detector_type`.

        :param str timeseries_detector_name: Name of the timeseries detector.
        :param str detector_type: The kind of data the timeseries detector is
            for, either "telemetry" or "ci".
        """
        TimeSeriesDetectorRegistry._timeseries_detectors[check_detector_type(detector_type)][
            timeseries_detector_name
        ] = timeseries_detector_class

    @staticmethod
    def get_timeseries_detectors(detector_type=DEFAULT_DETECTOR_TYPE):
        """Return all the timeseries detectors that were gathered for a kind of data.

        :param str detector_type: The kind of data to return the timeseries
            detectors of, either "telemetry" or "ci". Defaults to the telemetry
            timeseries detectors.
        """
        return TimeSeriesDetectorRegistry._timeseries_detectors[check_detector_type(detector_type)]


class BaseTimeSeriesDetector:
    """Base timeseries detector that detectors must inherit from."""

    def __init_subclass__(
        cls, timeseries_detector_name, detector_type=DEFAULT_DETECTOR_TYPE, **kwargs
    ):
        super().__init_subclass__(**kwargs)
        TimeSeriesDetectorRegistry.add(cls, timeseries_detector_name, detector_type)

    def __init__(self, timeseries, **kwargs):
        """Initialize the BaseTimeSeriesDetector.

        :param TimeSeries timeseries: A TimeSeries object that represents
            the timeseries to analyze.
        """
        self.timeseries = timeseries

    def get_sum_of_previous_n(self, n, inclusive=False):
        """Returns the sum of the past n data points.

        :param int n: The number of data points to sum.
        :param bool inclusive: If true, include the current point in the sum.
        :return DataFrame: A single data point as the sum of the previous n data points.
        """
        previous_n = self.timeseries.get_previous_n(n, inclusive=inclusive)
        if previous_n.empty:
            return previous_n
        return pandas.DataFrame(previous_n.sum()).T

    def get_sum_of_next_n(self, n, inclusive=True):
        """Returns the sum of the past n data points.

        :param int n: The number of data points to sum.
        :param bool inclusive: If true, include the current point in the sum.
        :return DataFrame: A single data point as the sum of the next n data points.
        """
        next_n = self.timeseries.get_next_n(n, inclusive=inclusive)
        if next_n.empty:
            return next_n
        return pandas.DataFrame(next_n.sum()).T

    def get_avg_of_previous_n(self, n, inclusive=False):
        """Returns the average of the past n data points.

        :param int n: The number of data points to average.
        :param bool inclusive: If true, include the current point in the average.
        :return DataFrame: A single data point as the average of the previous n data points.
        """
        previous_n = self.timeseries.get_previous_n(n, inclusive=inclusive)
        if previous_n.empty:
            return previous_n
        return pandas.DataFrame(previous_n.mean()).T

    def get_avg_of_next_n(self, n, inclusive=True):
        """Returns the average of the next n data points.

        :param int n: The number of data points to sum.
        :param bool inclusive: If true, include the current point in the average.
        :return DataFrame: A single data point as the average of the next n data points.
        """
        next_n = self.timeseries.get_next_n(n, inclusive=inclusive)
        if next_n.empty:
            return next_n
        return pandas.DataFrame(next_n.mean()).T

    def get_avg_of_surrounding_n(self, n):
        pn = int(n / 2)
        if n % 2 == 1:
            nn = pn + 1
        else:
            nn = pn

        currind = self.timeseries._currind
        data = self.timeseries.data.iloc[currind - pn : currind + nn + 1]
        return pandas.DataFrame(data.mean()).T

    def get_avg_of_next_surrounding_n(self, n):
        pn = int(n / 2)
        if n % 2 == 1:
            nn = pn + 1
        else:
            nn = pn

        currind = self.timeseries._currind
        data = self.timeseries.data.iloc[currind - pn + 1 : currind + nn + 2]
        return pandas.DataFrame(data.mean()).T

    def detect_changes(self, **kwargs):
        """Detect changes in a timeseries.

        :return: A list of Detection objects representing the regressions found.
        """
        pass
