# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Comparison of two groups of measurements, as PerfCompare does it.

PerfCompare decides whether two sets of trials differ by combining three
non-parametric measures rather than a single t-test:

    * a two-sided Mann-Whitney U test, which answers "are these two samples
      drawn from the same distribution?" (statistical significance),
    * Cliff's delta, which answers "how large is the difference?" (effect
      size, so a tiny-but-consistent shift doesn't get flagged),
    * the Common Language Effect Size (CLES) derived from the U statistic,
      which gives the direction: the probability that a base value is
      greater than a new value.

Only when the difference is both significant *and* meaningful does PerfCompare
report an improvement or a regression; everything else is "no change" (noise).
Every statistic comes from `mozdetect.mwu_stats`, a port of PerfCompare's own
implementations, so the numbers here agree with what PerfCompare shows for the
same data. Where the two part company is how much evidence they ask for before
calling a change: see PVALUE_THRESHOLD and CLIFFS_DELTA_THRESHOLD, both
tightened for unattended detection.

This is the comparison alone, registered as the `ci` detector `mwu` and
reached with `mozdetect.get_detectors("ci")["mwu"]`. The technique that walks
a whole series with it is `MWUTimeSeriesDetector`, in
`mozdetect.timeseries_detectors.mwu`.
"""
import numpy as np
import pandas

from dataclasses import asdict, dataclass

from mozdetect import mwu_stats
from mozdetect.detectors.base import BaseDetector

# Mann-Whitney U p-value at or below which the two samples are considered to
# come from different distributions. PerfCompare uses 0.05, which suits a
# comparison a human asked for and is reading; a detector walks every point of
# every series unattended, so it's set far tighter to keep the noise down.
PVALUE_THRESHOLD = 0.001

# Smallest Cliff's delta worth reporting on, regardless of how small the
# p-value is: on a noisy series significance alone is easy to reach, and the
# effect size is what separates a real step from a series that swings about.
# This sits inside PerfCompare's "moderate" band rather than at the bottom of
# its "large" one -- far enough up to drop the noise, low enough to keep real
# changes whose replicates overlap.
CLIFFS_DELTA_THRESHOLD = 0.40

# Resamples behind the confidence interval recorded on a detection. This is
# PerfCompare's own default, and at roughly half a second a call it's only
# affordable because it runs for the points that alert, not every candidate.
BOOTSTRAP_ITERATIONS = 9999


@dataclass
class MWUComparison:
    """The outcome of comparing a base sample against a new sample."""

    base_count: int
    new_count: int
    base_mean: float
    new_mean: float
    base_median: float
    new_median: float
    base_p05: float
    new_p05: float
    base_p95: float
    new_p95: float
    delta_value: float
    delta_percentage: float
    mann_whitney_stat: float
    pvalue: float
    significance: str
    is_significant: bool
    cliffs_delta: float
    cliffs_interpretation: str
    is_effect_meaningful: bool
    cles: float
    cles_explanation: str
    is_base_greater: bool
    direction: str
    # Filled in for the points that go on to be reported; too expensive to
    # compute for every candidate. See `MWUDetector.describe_distributions`.
    median_diff_ci: object = None
    normality: str = None

    @property
    def changed(self):
        """Whether the comparison found a real change in either direction."""
        return self.direction in ("improvement", "regression")

    @property
    def evidence(self):
        """How strongly this comparison argues for a change.

        Used to pick a single point out of a run of adjacent detections. Effect
        size leads because it's what separates a real step from noise that
        happens to be significant; the p-value and the magnitude of the median
        shift break ties.
        """
        return (
            abs(self.cliffs_delta or 0.0),
            -(self.pvalue if self.pvalue is not None else 1.0),
            abs(self.delta_percentage),
        )

    def as_dict(self):
        """Returns the comparison as a dictionary, for reporting."""
        comparison = asdict(self)
        if self.median_diff_ci is not None:
            comparison["median_diff_ci"] = asdict(self.median_diff_ci)
        return comparison


class MWUDetector(BaseDetector, detector_name="mwu", detector_type="ci"):
    """Compares two groups of measurements the way PerfCompare does."""

    def _get_trials(self, group):
        """Returns the measurements in a group, as a flat list.

        A group is either the measurements themselves, or a table holding them
        in a `trials` column (a window of a `TreeherderTimeSeries`), so a
        window can be handed over without unpacking it first.

        :param group: The measurements to compare, or a table holding them.

        :return list: The measurements in the group.
        """
        if isinstance(group, pandas.DataFrame):
            if "trials" in group:
                return [value for trials in group["trials"] for value in trials]
            if "value" in group:
                return list(group["value"])
            raise ValueError(
                "Expecting a `trials` or a `value` column in the group to compare, "
                f"got: {', '.join(str(column) for column in group.columns)}"
            )
        if isinstance(group, pandas.Series):
            return list(group)
        return list(group)

    def detect_changes(
        self,
        groups=None,
        lower_is_better=True,
        pvalue_threshold=PVALUE_THRESHOLD,
        cliffs_threshold=CLIFFS_DELTA_THRESHOLD,
        **kwargs,
    ):
        """Runs PerfCompare's comparison over two groups of measurements.

        :param list groups: The base group, and the new group to compare it
            against. Either the measurements themselves, or tables holding
            them in a `trials` column.
        :param bool lower_is_better: Whether a drop in the values is an
            improvement, which is what decides how a change gets named.
        :param float pvalue_threshold: The p-value at or below which the two
            groups are taken to come from different distributions.
        :param float cliffs_threshold: The smallest Cliff's delta worth
            calling a change.

        :return MWUComparison: The outcome of the comparison, or None when one
            of the groups holds no measurements.
        """
        groups = self._coalesce_groups(groups)
        base_trials = self._get_trials(groups[0])
        new_trials = self._get_trials(groups[1])

        if not base_trials or not new_trials:
            return None

        # Two-sided, since a change in either direction is worth reporting.
        mann_whitney_stat, pvalue = mwu_stats.mann_whitney_u(base_trials, new_trials)
        significance, is_significant = mwu_stats.interpret_significance(pvalue, pvalue_threshold)

        # Cliff's delta is the effect size that pairs with Mann-Whitney U: the
        # degree to which the two samples overlap, independent of their scale.
        cliffs_delta = mwu_stats.cliffs_delta(base_trials, new_trials)
        cliffs_interpretation, _ = mwu_stats.interpret_effect_size(cliffs_delta)
        # PerfCompare treats anything past negligible as meaningful; detection
        # asks for the configured effect size instead.
        is_effect_meaningful = cliffs_delta is not None and abs(cliffs_delta) >= cliffs_threshold

        # CLES = P(base value > new value), which is where the direction of the
        # change comes from.
        cles = mwu_stats.common_language_effect_size(
            mann_whitney_stat, len(base_trials), len(new_trials)
        )
        cles_explanation, is_base_greater = mwu_stats.interpret_cles(cles)

        direction = mwu_stats.direction_of_change(
            is_significant, is_effect_meaningful, cles, lower_is_better
        )

        base_median = mwu_stats.median(base_trials)
        new_median = mwu_stats.median(new_trials)

        return MWUComparison(
            base_count=len(base_trials),
            new_count=len(new_trials),
            base_mean=mwu_stats.mean(base_trials),
            new_mean=mwu_stats.mean(new_trials),
            base_median=base_median,
            new_median=new_median,
            base_p05=float(np.percentile(base_trials, 5)),
            new_p05=float(np.percentile(new_trials, 5)),
            base_p95=float(np.percentile(base_trials, 95)),
            new_p95=float(np.percentile(new_trials, 95)),
            delta_value=new_median - base_median,
            delta_percentage=mwu_stats.median_diff_pct(base_trials, new_trials),
            mann_whitney_stat=mann_whitney_stat,
            pvalue=pvalue,
            significance=significance,
            is_significant=is_significant,
            cliffs_delta=cliffs_delta,
            cliffs_interpretation=cliffs_interpretation,
            is_effect_meaningful=is_effect_meaningful,
            cles=cles,
            cles_explanation=cles_explanation,
            is_base_greater=is_base_greater,
            direction=direction,
        )

    def describe_distributions(self, comparison, groups=None, n_iter=BOOTSTRAP_ITERATIONS):
        """Adds the two readings PerfCompare shows alongside a comparison.

        Neither changes the verdict; they're what a sheriff looks at next. The
        confidence interval says how precisely the median shift is pinned down
        (and warns when it straddles zero), and the normality check flags
        shapes where a median difference is easy to misread.

        :param MWUComparison comparison: The comparison to describe.
        :param list groups: The groups the comparison was made over.
        :param int n_iter: How many resamples the confidence interval takes.

        :return MWUComparison: The comparison, with the two readings filled in.
        """
        groups = self._coalesce_groups(groups)
        base_trials = self._get_trials(groups[0])
        new_trials = self._get_trials(groups[1])

        comparison.median_diff_ci = mwu_stats.bootstrap_median_diff_ci(
            base_trials, new_trials, n_iter=n_iter
        )
        comparison.normality = mwu_stats.check_distribution_normality(base_trials, new_trials)
        return comparison
