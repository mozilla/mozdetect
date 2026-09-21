# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np


from mozdetect.detectors.base import BaseDetector


class CDFDetector(BaseDetector, detector_name="cdf", detector_type="telemetry"):
    """Uses the CDF of two groups of histograms to detect changes."""

    def _calculate_cdf(self, group):
        cdf = []
        currsum = 0
        for col, values in group.items():
            currsum += values[0]
            cdf.append(currsum)
        cdf = list((np.asarray(cdf) / cdf[-1]) * 100)
        return cdf

    def _calculate_cdf_difference(self, cdf_a, cdf_b):
        diff = 0
        for val_a, val_b in zip(cdf_a, cdf_b):
            diff += abs(val_a - val_b)
        return diff

    def _calculate_cdf_area_difference(self, cdf_a, cdf_b):
        z = np.asarray(cdf_a) - np.asarray(cdf_b)
        x = np.arange(len(cdf_a))

        # Determine intersection points of the CDFs
        dx = x[1:] - x[:-1]
        cross_test = np.sign(z[:-1] * z[1:])
        dx_intersect = -dx / (z[1:] - z[:-1]) * z[:-1]

        # Find areas of positive, and negative areas
        areas_pos = abs(z[:-1] + z[1:]) * 0.5 * dx
        areas_neg = 0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])

        # Find the total area between the curves
        areas = np.where(cross_test < 0, areas_neg, areas_pos)
        total_area = np.sum(areas)

        return total_area

    def detect_changes(self, groups=None, **kwargs):
        groups = self._coalesce_groups(groups)
        group_a = groups[0]
        group_b = groups[1]

        cdf_a = self._calculate_cdf(group_a)
        cdf_b = self._calculate_cdf(group_b)

        diff = self._calculate_cdf_difference(cdf_a, cdf_b)

        total_diff = len(cdf_a) * 100
        diff_pct = (diff / total_diff) * 100

        return diff, diff_pct, total_diff
