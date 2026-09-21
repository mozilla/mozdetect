# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pandas
import pytest

from pandas.testing import assert_frame_equal

from mozdetect import get_detectors, get_timeseries_detectors
from mozdetect.detectors import BaseDetector
from mozdetect.detectors.base import UnknownDetectorTypeError
from mozdetect.timeseries_detectors import BaseTimeSeriesDetector, Detection

from tests.support import get_sample_telemetry_data


class DetectorTest(BaseDetector, detector_name="test"):
    def detect_changes(self):
        return {"detection": True}


class TimeSeriesDetectorTest(BaseTimeSeriesDetector, timeseries_detector_name="test"):
    def detect_changes(self):
        detector = DetectorTest()

        detections = []
        for i in range(5):
            detection = detector.detect_changes()
            detections.append(
                Detection(
                    previous_value=i,
                    new_value=i + 1,
                    confidence=detection["detection"],
                    location="somewhere",
                    direction="up",
                )
            )

        return detections


class CIDetectorTest(BaseDetector, detector_name="test", detector_type="ci"):
    """A detector registered under the same name, for a different kind of data."""

    def detect_changes(self):
        return {"detection": False}


class CITimeSeriesDetectorTest(
    BaseTimeSeriesDetector, timeseries_detector_name="test", detector_type="ci"
):
    def detect_changes(self):
        return []


def get_ts_detector():
    sample_data = get_sample_telemetry_data()
    return TimeSeriesDetectorTest(sample_data)


def test_get_detectors():
    detectors = get_detectors()
    assert len(detectors) >= 1
    assert "test" in detectors
    # Telemetry is what a caller that doesn't ask gets.
    assert detectors is get_detectors("telemetry")
    assert detectors["test"] is DetectorTest


def test_get_timeseries_detectors():
    ts_detectors = get_timeseries_detectors()
    assert len(ts_detectors) >= 1
    assert "test" in ts_detectors
    assert ts_detectors is get_timeseries_detectors("telemetry")
    assert ts_detectors["test"] is TimeSeriesDetectorTest


def test_detectors_of_each_type_are_kept_apart():
    # The same name in both sets: which detector it means depends on the kind
    # of data being analyzed.
    assert get_detectors("ci")["test"] is CIDetectorTest
    assert get_timeseries_detectors("ci")["test"] is CITimeSeriesDetectorTest
    assert get_detectors("telemetry")["test"] is DetectorTest
    assert get_timeseries_detectors("telemetry")["test"] is TimeSeriesDetectorTest


def test_get_detectors_of_an_unknown_type():
    with pytest.raises(UnknownDetectorTypeError):
        get_detectors("everything")
    with pytest.raises(UnknownDetectorTypeError):
        get_timeseries_detectors("everything")


def test_registering_a_detector_of_an_unknown_type():
    with pytest.raises(UnknownDetectorTypeError):

        class BadDetector(BaseDetector, detector_name="bad", detector_type="everything"):
            pass

    with pytest.raises(UnknownDetectorTypeError):

        class BadTimeSeriesDetector(
            BaseTimeSeriesDetector, timeseries_detector_name="bad", detector_type="everything"
        ):
            pass


def test_detector_basic():
    ts_detector = get_ts_detector()
    detections = ts_detector.detect_changes()
    assert len(detections) == 5
    assert detections[0].previous_value == 0 and detections[0].confidence


def test_detector_get_data():
    ts_detector = get_ts_detector()

    ts_detector.timeseries._currind = 2
    res = ts_detector.get_sum_of_previous_n(2, inclusive=True)
    assert_frame_equal(res, pandas.DataFrame([[0, 2, 4, 6, 8, 10]]))

    ts_detector.timeseries._currind = 0
    res = ts_detector.get_sum_of_next_n(3)
    assert_frame_equal(res, pandas.DataFrame([[0, 3, 6, 9, 12, 15]]))
