# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas

from mozdetect.detectors.base import BaseDetector


class CDFSquaredDetector(BaseDetector, detector_name="cdf_squared", detector_type="telemetry"):
    """Uses the CDF of two groups of histograms to detect changes."""

    def detect_changes(self, groups=None, **kwargs):
        """Obtain the difference between two CDFs using a squared diff method.

        Two data points are expected in the groups, with a CDF defined in both of them.
        """
        current_date_hist = groups[0]
        next_date_hist = groups[1]

        merged_hist = pandas.merge(
            current_date_hist, next_date_hist, on="bin", suffixes=("_current", "_next")
        )
        merged_hist["diff"] = merged_hist["cdf_current"] - merged_hist["cdf_next"]
        merged_hist["sq_diff"] = (merged_hist["cdf_next"] - merged_hist["cdf_current"]) ** 2

        def __sign(x):
            return (x > 0) - (x < 0)

        return {
            "diff": [merged_hist["diff"].sum()],
            "sq_diff": [merged_hist["sq_diff"].sum() * __sign(float(merged_hist["diff"].sum()))],
            "maxmin": (abs(merged_hist["diff"].min()) + abs(merged_hist["diff"].max())),
        }
