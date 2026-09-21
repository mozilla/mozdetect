# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozdetect.detectors import CDFDetector
from mozdetect.timeseries_detectors.base import BaseTimeSeriesDetector
from mozdetect.timeseries_detectors.detection import Detection
from mozdetect.utils import is_dev_mode


class CDFTimeSeriesDetector(
    BaseTimeSeriesDetector, timeseries_detector_name="cdf", detector_type="telemetry"
):
    """Analyzes a time series of histograms using CDFs to detect changes."""

    def detect_changes(self, group_size=7, alert_threshold=0.1, **kwargs):
        detections = []
        detector = CDFDetector()

        prev_detection = None
        cdf_diffs = []
        build_ids = self.timeseries.nonnumerical_data
        max_detection = None
        for i, row in enumerate(self.timeseries):
            if i < group_size or i > (len(build_ids["build_id"]) - group_size):
                # Ensure that we're always comparing groups
                # that have the same size
                continue
            build_id = build_ids["build_id"][i]
            avg_hist_previous = self.get_sum_of_previous_n(group_size, inclusive=False)
            avg_hist_current = self.get_sum_of_next_n(group_size, inclusive=True)

            (cdf_diff, cdf_diff_pct, cdf_diff_total) = detector.detect_changes(
                [avg_hist_previous, avg_hist_current]
            )
            cdf_diffs.append(cdf_diff_pct)

            if cdf_diff_pct > alert_threshold and prev_detection:
                if not max_detection:
                    max_detection = Detection(cdf_diff, prev_detection[0], cdf_diff_pct, build_id)
                elif cdf_diff_pct > max_detection.confidence:
                    max_detection = Detection(cdf_diff, prev_detection[0], cdf_diff_pct, build_id)
                detections.append(Detection(cdf_diff, prev_detection[0], cdf_diff_pct, build_id))
            prev_detection = (cdf_diff, cdf_diff_pct, cdf_diff_total)

        if is_dev_mode():
            from matplotlib import pyplot as plt

            plt.figure()
            plt.plot(cdf_diffs)
            plt.xticks(
                ticks=list(range(len(cdf_diffs))),
                labels=list(build_ids["build_id"])[group_size : -group_size + 1],
            )

            plt.show()

        if max_detection:
            return detections, [max_detection]
        return detections, []
