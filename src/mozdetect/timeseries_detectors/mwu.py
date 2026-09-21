# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Change detection built on PerfCompare's Mann-Whitney U comparison.

`MWUDetector` compares two groups of measurements and says whether they
differ; see `mozdetect.detectors.mwu` for what that comparison is and why it
calls a change. This module turns that two-sample comparison into a change
detector, in three steps:

    * slide a backward (base) and forward (new) window across the series and
      ask the comparison at every point,
    * collapse each run of adjacent detections to its strongest point, since
      the windows overlap and one step in the data trips the comparison
      several times over (`pick_peaks`),
    * move each surviving detection onto the point that best splits the data
      it was detected over, because a wide forward window only establishes
      that a change happened somewhere inside it, and the point a detection
      names is the one a sheriff bisects around (`localize_change`). The
      recorded numbers are then recomputed for that split, so they describe
      the step rather than the window that found it.

A detected change is dropped when it moves the median less than the series'
own `alert_threshold`; a series that doesn't set one gets no floor at all
(`is_below_magnitude_threshold`).

The windows are counted in points of the series rather than in measurements,
because a point is what a detection names: a push for Perfherder data, a day
for data pooled per day. Everything the point recorded goes into the
comparison, so a short window still gives Mann-Whitney U a large sample once
the replicates behind those data points are included. The point counts are a
floor rather than a cap: a window that still holds fewer than `MIN_TRIALS`
trials keeps taking points until it has enough.

It's registered as a `ci` detector, since it works on the per-push runs of a
CI test rather than on telemetry histograms, so it's reached with::

    detector = mozdetect.get_timeseries_detectors("ci")["mwu"](timeseries)
    detections = detector.detect_changes()

where `timeseries` is a `TreeherderTimeSeries`. The thresholds and window
sizes below are all parameters of `detect_changes`. See
`examples/mwu_detection_run.py` for a script that runs it over a series read
from a Treeherder API server.
"""
import logging
import pandas

from dataclasses import dataclass

from mozdetect.detectors.mwu import (
    BOOTSTRAP_ITERATIONS,
    CLIFFS_DELTA_THRESHOLD,
    PVALUE_THRESHOLD,
    MWUDetector,
)
from mozdetect.mwu_stats import mann_whitney_u
from mozdetect.timeseries_detectors.base import BaseTimeSeriesDetector
from mozdetect.timeseries_detectors.detection import Detection

logger = logging.getLogger("MWUTimeSeries")

# How many points the base (backward) window covers.
BACK_WINDOW = 4

# How many points the new (forward) window covers, starting at the point
# being examined.
FORE_WINDOW = 4

# How many trials each side has to hold before the comparison is worth
# running.
# The point counts above are a minimum, not a cap: a window that comes up
# short here keeps taking points until it has this many trials. Both tests
# need a reasonable sample to say anything (a two-sided Mann-Whitney U on 3 vs
# 3 values can't reach even p <= 0.05, let alone the threshold the detector
# asks for, no matter how separated the samples are), and how many points it
# takes to get there depends on how many measurements each one carries.
MIN_TRIALS = 30

# Fewest points either side of a change point when locating it. A step pinned
# to the last point of a window has only that one point arguing for it, which
# is too little to attribute a regression to a revision.
MIN_SPLIT_POINTS = 2

# How far apart two detections must be to count as separate changes. Windows
# overlap heavily, so a single step in the data usually trips the comparison
# at a few adjacent points; only the strongest of each run of detections is
# kept.
DETECTION_SPACING = 4

# The `alert_change_type` a signature carries when its `alert_threshold` is an
# absolute amount rather than a percentage (Treeherder's
# PerformanceSignature.ALERT_ABS).
ALERT_ABS = 1


@dataclass
class Sample:
    """What the series recorded at one point in time.

    `values` holds one aggregated value per data point. `trials` holds what
    gets compared: the replicates behind those data points when the series
    carries them, otherwise the same aggregated values.
    """

    location: object
    timestamp: object
    values: list
    trials: list


@dataclass
class Window:
    """One side of a comparison: the trials to compare, and how many points of
    the series they came from.
    """

    trials: list
    points: int


class MWUTimeSeriesDetector(
    BaseTimeSeriesDetector, timeseries_detector_name="mwu", detector_type="ci"
):
    """Analyzes a timeseries of measurements with the MWU comparison."""

    def __init__(self, timeseries, **kwargs):
        super().__init__(timeseries, **kwargs)
        self.detector = MWUDetector()

    def _get_data(self):
        """Returns the table the samples are built from.

        The raw data is preferred over the iteration data, since that's where
        the measurements behind each point are: a `TimeSeries` iterating over
        its numerical columns alone leaves the `trials` behind.
        """
        data = getattr(self.timeseries, "raw_data", None)
        if data is None:
            data = self.timeseries.data
        return data

    def gather_samples(self):
        """Collects the series into one sample per point in time, oldest first.

        A point is a push when the series has one (Perfherder data), which is
        what a detection names. The data points of one push -- retriggers and
        backfills -- are pooled into that push's sample, so a push that ran
        the test several times weighs more than one that ran it once, rather
        than counting as several points in the series.

        :return list: The `Sample` objects the series is made of.
        """
        data = self._get_data()
        if data is None or (hasattr(data, "empty") and data.empty):
            return []

        if "push_id" in data:
            grouped = [(push_id, rows) for push_id, rows in data.groupby("push_id", sort=False)]
        else:
            grouped = [(index, data.iloc[[index]]) for index in range(len(data))]

        samples = []
        for group_id, rows in grouped:
            first = rows.iloc[0]

            if "values" in rows:
                values = [value for values in rows["values"] for value in values]
            else:
                values = list(rows["value"])

            if "trials" in rows:
                trials = [value for trials in rows["trials"] for value in trials]
            else:
                trials = list(values)

            if not trials:
                continue

            location = first["revision"] if "revision" in rows else group_id
            timestamp = first["push_timestamp"] if "push_timestamp" in rows else None
            if timestamp is None and "date" in rows:
                timestamp = first["date"]

            samples.append(
                Sample(
                    location=location,
                    timestamp=timestamp,
                    values=values,
                    trials=trials,
                )
            )

        if samples and samples[0].timestamp is not None:
            samples.sort(key=lambda sample: sample.timestamp)

        return samples

    def get_windows(self, samples, index, back_window, fore_window, min_trials=MIN_TRIALS):
        """Collects the two sides compared at `index`.

        The base window covers the `back_window` points before `index`, the
        new window `index` and the `fore_window - 1` points after it. A window
        that still holds fewer than `min_trials` trials at that point keeps
        taking points until it has enough, since how much data a point carries
        depends on how many jobs ran and how many replicates they reported.

        Either window can come back short at the edges of the series, so the
        caller can decide whether there's enough data yet.

        :param list samples: The series, as `Sample` objects.
        :param int index: The position in the series to compare around.
        :param int back_window: How many points the base window covers.
        :param int fore_window: How many points the new window covers.
        :param int min_trials: How many trials each side has to hold.

        :return tuple: The base `Window`, and the new `Window`.
        """
        base = Window(trials=[], points=0)
        prev_index = index - 1
        while prev_index >= 0 and (base.points < back_window or len(base.trials) < min_trials):
            sample = samples[prev_index]
            base.trials = sample.trials + base.trials
            base.points += 1
            prev_index -= 1

        new = Window(trials=[], points=0)
        next_index = index
        while next_index < len(samples) and (
            new.points < fore_window or len(new.trials) < min_trials
        ):
            sample = samples[next_index]
            new.trials.extend(sample.trials)
            new.points += 1
            next_index += 1

        return base, new

    def is_below_magnitude_threshold(self, comparison):
        """Whether the median moved less than the series says is worth
        reporting.

        Only series that set their own `alert_threshold` get a floor. That
        number is a judgement about how much movement matters for that test,
        which no amount of evidence can supply -- Cliff's delta measures
        separation, not size, so a 0.1% shift on a very quiet test can be both
        significant and large. Where a test hasn't made that judgement there's
        no floor at all, rather than borrowing a global default, which says
        nothing about this test in particular.

        :param MWUComparison comparison: The comparison to check.

        :return bool: Whether the change is too small to report.
        """
        alert_threshold = getattr(self.timeseries, "alert_threshold", None)
        if alert_threshold is None:
            return False

        metadata = getattr(self.timeseries, "metadata", None) or {}
        if metadata.get("alert_change_type") in (ALERT_ABS, "absolute"):
            return abs(comparison.delta_value) < alert_threshold
        return abs(comparison.delta_percentage) < alert_threshold

    def pick_peaks(self, candidates, detection_spacing=DETECTION_SPACING):
        """Reduces each run of adjacent detections to its strongest point.

        The windows overlap, so one step in the data is normally detected at
        several consecutive points. Candidates no more than
        `detection_spacing` apart are treated as the same change event.

        :param list candidates: The (index, MWUComparison) detections found.
        :param int detection_spacing: How far apart two detections have to be
            to count as separate changes.

        :return list: The strongest (index, MWUComparison) of each run.
        """
        peaks = []
        cluster = []

        for candidate in candidates:
            if cluster and (candidate[0] - cluster[-1][0]) > detection_spacing:
                peaks.append(max(cluster, key=lambda entry: entry[1].evidence))
                cluster = []
            cluster.append(candidate)

        if cluster:
            peaks.append(max(cluster, key=lambda entry: entry[1].evidence))

        return peaks

    def localize_change(self, samples, index, span_start, span_end, min_split=MIN_SPLIT_POINTS):
        """Finds the point in the forward window that the change belongs to.

        A detection says the base and new windows differ. With a wide forward
        window that only means "a change happened somewhere in it": the
        comparison trips at the first point whose windows are both full, which
        can be many points before the step itself. That matters because the
        point a detection names is the one sheriffs bisect around.

        So walk every split point from `index` to the end of the forward
        window and keep the one that separates the data most sharply, measured
        by how far the Mann-Whitney U statistic sits from what equal samples
        would give. Taking the strongest split as the change point is
        Pettitt's estimator (1979), which is the same rank statistic the
        comparison itself is built on.

        :param list samples: The series, as `Sample` objects.
        :param int index: The position the detection fired at.
        :param int span_start: The first point the detection looked at.
        :param int span_end: The last point the detection looked at.
        :param int min_split: Fewest points either side of the split.

        :return int: The chosen index, which is `index` itself when nothing
            later splits the data better.
        """
        values_by_point = [sample.values for sample in samples]

        best_index = index
        best_score = None
        for split in range(max(index, span_start + min_split), span_end - min_split + 2):
            base = [value for values in values_by_point[span_start:split] for value in values]
            new = [value for values in values_by_point[split : span_end + 1] for value in values]
            if not base or not new:
                continue

            statistic, _ = mann_whitney_u(base, new)
            if statistic is None:
                continue

            # |U - n*m/2| scaled by the sample sizes, so splits with different
            # amounts of data either side stay comparable.
            score = abs(statistic / (len(base) * len(new)) - 0.5)
            if best_score is None or score > best_score:
                best_index, best_score = split, score

        return best_index

    def compare_split(
        self,
        samples,
        span_start,
        split,
        span_end,
        lower_is_better=True,
        pvalue_threshold=PVALUE_THRESHOLD,
        cliffs_threshold=CLIFFS_DELTA_THRESHOLD,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
    ):
        """Compares the two sides of a located change point.

        Everything the detection looked at, cut at `split` rather than at the
        point the comparison happened to fire on, so the recorded medians and
        effect size describe the step itself.

        :return MWUComparison: The comparison, or None when the split doesn't
            hold up as a change on its own, which leaves the original
            detection in place.
        """
        base_trials, new_trials = [], []
        for sample in samples[span_start:split]:
            base_trials.extend(sample.trials)
        for sample in samples[split : span_end + 1]:
            new_trials.extend(sample.trials)

        comparison = self.detector.detect_changes(
            [base_trials, new_trials],
            lower_is_better=lower_is_better,
            pvalue_threshold=pvalue_threshold,
            cliffs_threshold=cliffs_threshold,
        )
        if comparison is None or not comparison.changed:
            return None

        self.detector.describe_distributions(
            comparison, [base_trials, new_trials], n_iter=bootstrap_iterations
        )
        return comparison

    def find_changes(
        self,
        samples,
        lower_is_better=True,
        pvalue_threshold=PVALUE_THRESHOLD,
        cliffs_threshold=CLIFFS_DELTA_THRESHOLD,
        back_window=BACK_WINDOW,
        fore_window=FORE_WINDOW,
        min_trials=MIN_TRIALS,
        detection_spacing=DETECTION_SPACING,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
    ):
        """Slides the comparison across the series and returns the change points.

        Each detection is then localized: the comparison that fires may sit
        well before the step when the forward window is wide, so the change
        point is moved to the point that best splits the data it was detected
        over.

        A detected change is dropped when it moves the median less than the
        series' own `alert_threshold`; series that don't set one get no floor.

        :param list samples: The series, as `Sample` objects.

        :return list: The (index, MWUComparison) tuples, one per detected
            change, ordered oldest first.
        """
        candidates = []
        for index in range(1, len(samples)):
            base, new = self.get_windows(
                samples, index, back_window, fore_window, min_trials=min_trials
            )

            # Not enough data on one of the sides yet: either the window ran
            # out of points, or the points it has don't carry enough trials
            # between them. More may arrive later, in which case a subsequent
            # run picks this point up.
            if base.points < back_window or new.points < fore_window:
                continue
            if len(base.trials) < min_trials or len(new.trials) < min_trials:
                continue

            comparison = self.detector.detect_changes(
                [base.trials, new.trials],
                lower_is_better=lower_is_better,
                pvalue_threshold=pvalue_threshold,
                cliffs_threshold=cliffs_threshold,
            )
            if comparison is None or not comparison.changed:
                continue

            if self.is_below_magnitude_threshold(comparison):
                continue

            candidates.append((index, comparison, base, new))

        peaks = self.pick_peaks(
            [(index, comparison) for index, comparison, _, _ in candidates],
            detection_spacing=detection_spacing,
        )

        windows = {index: (base, new) for index, _, base, new in candidates}
        localized = {}
        for index, comparison in peaks:
            base, new = windows[index]
            span_start = index - base.points
            span_end = index + new.points - 1

            split = self.localize_change(samples, index, span_start, span_end)
            split_comparison = None
            if split != index:
                split_comparison = self.compare_split(
                    samples,
                    span_start,
                    split,
                    span_end,
                    lower_is_better=lower_is_better,
                    pvalue_threshold=pvalue_threshold,
                    cliffs_threshold=cliffs_threshold,
                    bootstrap_iterations=bootstrap_iterations,
                )
            if split_comparison is None:
                # Either the detection was already on the best split, or
                # splitting there doesn't hold up on its own; keep what was
                # detected.
                split, split_comparison = index, comparison
                self.detector.describe_distributions(
                    split_comparison, [base.trials, new.trials], n_iter=bootstrap_iterations
                )

            # Two detections can localize onto the same point; keep the stronger.
            if split not in localized or split_comparison.evidence > localized[split].evidence:
                localized[split] = split_comparison

        return sorted(localized.items())

    def detect_changes(
        self,
        pvalue_threshold=PVALUE_THRESHOLD,
        cliffs_threshold=CLIFFS_DELTA_THRESHOLD,
        back_window=BACK_WINDOW,
        fore_window=FORE_WINDOW,
        min_trials=MIN_TRIALS,
        detection_spacing=DETECTION_SPACING,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        lower_is_better=None,
        **kwargs,
    ):
        """Detects the changes in the series with the MWU comparison.

        :param float pvalue_threshold: The p-value at or below which two
            windows are taken to come from different distributions.
        :param float cliffs_threshold: The smallest Cliff's delta worth
            calling a change.
        :param int back_window: How many points the base window covers.
        :param int fore_window: How many points the new window covers.
        :param int min_trials: How many trials each window has to hold before
            the comparison is worth running. Windows grow past their point
            counts until they hold this many.
        :param int detection_spacing: How far apart two detections have to be
            to count as separate changes.
        :param int bootstrap_iterations: How many resamples the confidence
            interval recorded on a detection takes.
        :param bool lower_is_better: Whether a drop in the values is an
            improvement. Read from the series' own metadata when not given.

        :return list: A list of Detection objects representing where a change
            was detected.
        """
        if lower_is_better is None:
            lower_is_better = getattr(self.timeseries, "lower_is_better", True)

        samples = self.gather_samples()
        changes = self.find_changes(
            samples,
            lower_is_better=lower_is_better,
            pvalue_threshold=pvalue_threshold,
            cliffs_threshold=cliffs_threshold,
            back_window=back_window,
            fore_window=fore_window,
            min_trials=min_trials,
            detection_spacing=detection_spacing,
            bootstrap_iterations=bootstrap_iterations,
        )

        detections = []
        for index, comparison in changes:
            sample = samples[index]
            previous = samples[index - 1] if index else None

            # `direction` is the movement of the values, which is what the
            # other detectors report there; which way the comparison called it
            # travels alongside as `change`.
            comparison_info = comparison.as_dict()
            comparison_info.pop("direction")

            detections.append(
                Detection(
                    comparison.base_median,
                    comparison.new_median,
                    comparison.pvalue,
                    sample.location,
                    "up" if comparison.delta_value > 0 else "down",
                    index=index,
                    timestamp=sample.timestamp,
                    previous_location=previous.location if previous else None,
                    change=comparison.direction,
                    **comparison_info,
                )
            )

        return detections

    def summarize(self, detections):
        """Returns the detections as a table, for reporting.

        :param list detections: The Detection objects to summarize.

        :return pandas.DataFrame: One row per detection.
        """
        return pandas.DataFrame(
            [
                {
                    "location": detection.location,
                    "timestamp": detection.optional_detection_info.get("timestamp"),
                    "change": detection.optional_detection_info.get("change"),
                    "previous_value": detection.previous_value,
                    "new_value": detection.new_value,
                    "delta_percentage": detection.optional_detection_info.get("delta_percentage"),
                    "pvalue": detection.confidence,
                    "cliffs_delta": detection.optional_detection_info.get("cliffs_delta"),
                    "effect": detection.optional_detection_info.get("cliffs_interpretation"),
                    "base_count": detection.optional_detection_info.get("base_count"),
                    "new_count": detection.optional_detection_info.get("new_count"),
                }
                for detection in detections
            ]
        )
