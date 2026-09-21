# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Checks on the MWU detector, and on the change detection built on it."""
import numpy as np
import pandas
import pytest

from mozdetect import get_detectors, get_timeseries_detectors
from mozdetect.detectors.mwu import (
    CLIFFS_DELTA_THRESHOLD,
    PVALUE_THRESHOLD,
    MWUDetector,
)
from mozdetect.timeseries_detectors.mwu import (
    BACK_WINDOW,
    DETECTION_SPACING,
    FORE_WINDOW,
    MIN_TRIALS,
    MWUTimeSeriesDetector,
    Sample,
)

from tests.support import SAMPLE_SIGNATURE_METADATA, make_treeherder_timeseries

# Enough replicates per push that a window of BACK_WINDOW pushes carries a
# sample Mann-Whitney U can say something about.
REPLICATES_PER_PUSH = 5


def make_step_series(before, after, count=15, replicates=REPLICATES_PER_PUSH, metadata=None):
    """A series that sits at `before` and steps to `after` halfway through."""
    trials_per_push = [[before] * replicates for _ in range(count)]
    trials_per_push += [[after] * replicates for _ in range(count)]
    if metadata is None:
        metadata = SAMPLE_SIGNATURE_METADATA
    return make_treeherder_timeseries(trials_per_push, metadata=metadata)


def make_samples(values_per_point):
    """`Sample` objects out of the values each point of a series recorded."""
    return [
        Sample(location=f"revision{index}", timestamp=index, values=values, trials=values)
        for index, values in enumerate(values_per_point)
    ]


def test_detector_is_registered_as_a_ci_detector():
    assert get_detectors("ci")["mwu"] is MWUDetector
    assert get_timeseries_detectors("ci")["mwu"] is MWUTimeSeriesDetector
    # It works on the runs of a CI test, not on telemetry histograms, so it's
    # not in the set a telemetry caller gets.
    assert "mwu" not in get_detectors()
    assert "mwu" not in get_timeseries_detectors()


def test_default_thresholds():
    assert CLIFFS_DELTA_THRESHOLD == 0.40
    assert PVALUE_THRESHOLD == 0.001


def test_compare_detects_regression():
    comparison = MWUDetector().detect_changes([[100.0] * 12, [120.0] * 12])

    assert comparison.direction == "regression"
    assert comparison.changed is True
    assert comparison.is_significant
    assert comparison.is_effect_meaningful
    assert comparison.cliffs_interpretation == "large"
    assert comparison.base_median == 100.0
    assert comparison.new_median == 120.0
    assert comparison.delta_value == 20.0
    assert comparison.delta_percentage == 20.0


def test_compare_direction_follows_lower_is_better():
    detector = MWUDetector()
    base = [100.0] * 12
    new = [120.0] * 12

    assert detector.detect_changes([base, new], lower_is_better=False).direction == "improvement"
    assert detector.detect_changes([new, base], lower_is_better=True).direction == "improvement"
    assert detector.detect_changes([new, base], lower_is_better=False).direction == "regression"


def test_compare_reports_no_change_on_same_distribution():
    rng = np.random.default_rng(7)
    base = list(rng.normal(100, 2, 12))
    new = list(rng.normal(100, 2, 12))

    comparison = MWUDetector().detect_changes([base, new])

    assert comparison.direction == "no change"
    assert comparison.changed is False


def test_compare_ignores_negligible_effect():
    # The samples are shifted, but they overlap so heavily that a value drawn
    # from one is nearly as likely to be above as below one from the other,
    # which is what Cliff's delta measures.
    base = [float(value) for value in range(40)]
    new = [float(value) + 2 for value in range(40)]

    comparison = MWUDetector().detect_changes([base, new])

    assert comparison.cliffs_interpretation == "negligible"
    assert comparison.is_effect_meaningful is False
    assert comparison.direction == "no change"


def test_compare_without_data():
    assert MWUDetector().detect_changes([[], [1.0, 2.0]]) is None
    assert MWUDetector().detect_changes([[1.0, 2.0], []]) is None


def test_compare_takes_the_groups_it_was_built_with():
    detector = MWUDetector(groups=[[100.0] * 12, [120.0] * 12])

    assert detector.detect_changes().direction == "regression"
    # The groups passed in win over the ones it was built with.
    assert detector.detect_changes([[120.0] * 12, [100.0] * 12]).direction == "improvement"

    with pytest.raises(ValueError):
        MWUDetector().detect_changes()


def test_compare_reads_the_trials_out_of_a_table():
    """A window of a series can be compared without unpacking it first."""
    series = make_step_series(100.0, 120.0, count=3).raw_data

    comparison = MWUDetector().detect_changes([series.iloc[:3], series.iloc[3:]])

    assert comparison.base_median == 100.0
    assert comparison.new_median == 120.0
    assert comparison.base_count == 3 * REPLICATES_PER_PUSH
    assert comparison.direction == "regression"


def test_compare_rejects_a_table_without_measurements():
    with pytest.raises(ValueError):
        MWUDetector().detect_changes([pandas.DataFrame({"other": [1.0]}), [1.0, 2.0]])


def test_pvalue_threshold_is_configurable():
    # Fully separated, but only 7 values a side: significant by PerfCompare's
    # 0.05, nowhere near the 0.001 this technique asks for.
    base = [float(value) for value in range(1, 8)]
    new = [float(value) for value in range(11, 18)]
    detector = MWUDetector()

    comparison = detector.detect_changes([base, new])
    assert 0.001 < comparison.pvalue < 0.05
    assert comparison.is_significant is False
    assert comparison.direction == "no change"

    relaxed = detector.detect_changes([base, new], pvalue_threshold=0.05)
    assert relaxed.is_significant is True
    assert relaxed.direction == "regression"


def test_cliffs_delta_threshold_is_configurable():
    # Heavily overlapping samples, but enough of them for a tiny p-value: only
    # the effect size stands between this and a detection.
    base = [float(value) for value in range(100)]
    new = [float(value) + 20 for value in range(100)]
    detector = MWUDetector()

    comparison = detector.detect_changes([base, new])
    assert comparison.pvalue < 0.001
    assert comparison.cliffs_interpretation == "moderate"
    assert comparison.is_effect_meaningful is False
    assert comparison.direction == "no change"

    relaxed = detector.detect_changes([base, new], cliffs_threshold=0.33)
    assert relaxed.is_effect_meaningful is True
    assert relaxed.direction == "regression"


def test_describe_distributions_adds_the_readings():
    base = [10.2, 11.4, 9.8, 10.9, 12.1, 10.0, 11.7, 9.5, 10.4, 11.1]
    new = [12.9, 13.4, 11.8, 14.1, 12.2, 13.9, 12.5, 13.1, 14.6, 12.0]
    detector = MWUDetector()

    comparison = detector.detect_changes([base, new])
    assert comparison.median_diff_ci is None
    assert comparison.normality is None

    detector.describe_distributions(comparison, [base, new], n_iter=999)
    assert comparison.median_diff_ci.ci_low > 0
    assert comparison.normality == "both"
    # The readings travel with the rest of the comparison.
    assert comparison.as_dict()["median_diff_ci"]["ci_low"] == comparison.median_diff_ci.ci_low


def test_gather_samples_pools_the_data_points_of_a_push():
    series = make_treeherder_timeseries([[1.0, 2.0], [3.0, 4.0]])
    # A retrigger: a second data point recorded at the first push.
    retrigger = series.raw_data.iloc[[0]].copy()
    retrigger["trials"] = [[5.0, 6.0]]
    retrigger["values"] = [[5.5]]
    series.raw_data = pandas.concat([series.raw_data, retrigger], ignore_index=True)

    samples = MWUTimeSeriesDetector(series).gather_samples()

    assert len(samples) == 2
    assert samples[0].location == "revision1000"
    assert samples[0].trials == [1.0, 2.0, 5.0, 6.0]
    assert samples[1].trials == [3.0, 4.0]


def test_windows_gather_every_trial_a_point_carries():
    samples = [
        Sample(location=index, timestamp=index, values=[1.0], trials=[0.5, 1.5])
        for index in range(1, 11)
    ]
    detector = MWUTimeSeriesDetector(make_treeherder_timeseries([[1.0]]))

    base, new = detector.get_windows(samples, 5, back_window=2, fore_window=2, min_trials=4)

    assert base.points == 2
    assert base.trials == [0.5, 1.5, 0.5, 1.5]
    assert new.points == 2
    assert new.trials == [0.5, 1.5, 0.5, 1.5]


def test_windows_grow_until_they_hold_enough_trials():
    # One trial per point, so the window has to reach MIN_TRIALS points back
    # (and forward) rather than the BACK_WINDOW it asked for.
    samples = make_samples([[1.0]] * 80)
    detector = MWUTimeSeriesDetector(make_treeherder_timeseries([[1.0]]))

    base, new = detector.get_windows(samples, 40, BACK_WINDOW, FORE_WINDOW)

    assert base.points == MIN_TRIALS
    assert new.points == MIN_TRIALS
    assert len(base.trials) == MIN_TRIALS
    assert len(new.trials) == MIN_TRIALS


def test_windows_come_back_short_at_the_edges():
    samples = make_samples([[1.0] * 10] * 10)
    detector = MWUTimeSeriesDetector(make_treeherder_timeseries([[1.0]]))

    base, new = detector.get_windows(samples, 1, BACK_WINDOW, FORE_WINDOW, min_trials=10)

    assert base.points == 1
    assert new.points == FORE_WINDOW


def test_pick_peaks_keeps_strongest_of_adjacent_detections():
    detector = MWUDetector()
    weak = detector.detect_changes([[100.0] * 12, [100.0] * 6 + [120.0] * 6])
    strong = detector.detect_changes([[100.0] * 12, [120.0] * 12])
    timeseries_detector = MWUTimeSeriesDetector(make_treeherder_timeseries([[1.0]]))

    peaks = timeseries_detector.pick_peaks([(10, weak), (11, strong), (12, weak), (20, strong)])

    assert [index for index, _ in peaks] == [11, 20]
    # The runs are DETECTION_SPACING apart, so a tighter spacing splits them.
    assert 20 - 12 > DETECTION_SPACING


def test_localize_change_moves_a_detection_onto_the_step():
    samples = make_samples([[0.5] if index < 40 else [1.0] for index in range(61)])
    detector = MWUTimeSeriesDetector(make_treeherder_timeseries([[1.0]]))

    # A detection forced to fire well before the step, because that's the last
    # point with a full forward window.
    assert detector.localize_change(samples, 31, 1, 60) == 40


def test_localize_change_leaves_a_correct_detection_alone():
    samples = make_samples([[0.5] if index < 40 else [1.0] for index in range(61)])
    detector = MWUTimeSeriesDetector(make_treeherder_timeseries([[1.0]]))

    assert detector.localize_change(samples, 40, 10, 50) == 40


def test_detects_a_step_in_a_series():
    detector = MWUTimeSeriesDetector(make_step_series(0.5, 1.0))

    detections = detector.detect_changes()

    assert len(detections) == 1
    detection = detections[0]
    # The push the step lands on, which is the one a sheriff bisects around.
    assert detection.location == "revision1015"
    assert detection.direction == "up"
    assert detection.previous_value == 0.5
    assert detection.new_value == 1.0
    assert detection.confidence <= PVALUE_THRESHOLD

    info = detection.optional_detection_info
    assert info["change"] == "regression"
    assert info["index"] == 15
    assert info["previous_location"] == "revision1014"
    assert info["delta_percentage"] == 100.0
    assert info["delta_value"] == 0.5
    assert info["cliffs_interpretation"] == "large"
    assert info["significance"] == "significant"
    # Three pushes only carry 15 trials, so both windows grew until they held
    # MIN_TRIALS.
    assert info["base_count"] == MIN_TRIALS
    assert info["new_count"] == MIN_TRIALS
    # Perfectly constant data: the median shift is pinned to exactly 0.5, and
    # a sample with no spread at all can't be called normal.
    assert info["median_diff_ci"]["ci_low"] == 0.5
    assert info["median_diff_ci"]["ci_high"] == 0.5
    assert info["normality"] == "neither"


def test_detects_an_improvement_in_a_series():
    detections = MWUTimeSeriesDetector(make_step_series(1.0, 0.5)).detect_changes()

    assert len(detections) == 1
    assert detections[0].direction == "down"
    assert detections[0].optional_detection_info["change"] == "improvement"

    # The same drop on a series where more is better is a regression.
    metadata = dict(SAMPLE_SIGNATURE_METADATA, lower_is_better=False)
    series = make_step_series(1.0, 0.5, metadata=metadata)
    detections = MWUTimeSeriesDetector(series).detect_changes()

    assert len(detections) == 1
    assert detections[0].direction == "down"
    assert detections[0].optional_detection_info["change"] == "regression"


def test_no_detections_on_noise():
    rng = np.random.default_rng(13)
    trials_per_push = [list(rng.normal(100, 5, REPLICATES_PER_PUSH)) for _ in range(40)]

    detections = MWUTimeSeriesDetector(make_treeherder_timeseries(trials_per_push)).detect_changes()

    assert detections == []


def test_no_detections_with_too_few_pushes():
    # Both windows have to be full, and neither can be here.
    series = make_step_series(0.5, 1.0, count=BACK_WINDOW - 1)

    assert MWUTimeSeriesDetector(series).detect_changes() == []


def test_no_detections_below_a_threshold_the_test_set():
    # A 1% step, on a signature that says only a 2% move matters.
    series = make_step_series(100.0, 101.0)

    assert MWUTimeSeriesDetector(series).detect_changes() == []


def test_detects_a_small_step_when_the_test_sets_no_threshold():
    metadata = dict(SAMPLE_SIGNATURE_METADATA, alert_threshold=None)
    series = make_step_series(100.0, 101.0, metadata=metadata)

    detections = MWUTimeSeriesDetector(series).detect_changes()

    assert len(detections) == 1
    assert detections[0].optional_detection_info["delta_percentage"] == 1.0


def test_magnitude_threshold_applies_only_when_the_test_sets_one():
    # A clean +1% step: delta_percentage 1.0, delta_value 1.0.
    comparison = MWUDetector().detect_changes([[100.0] * 12, [101.0] * 12])

    def is_below(**metadata):
        series = make_treeherder_timeseries(
            [[1.0]], metadata=dict(SAMPLE_SIGNATURE_METADATA, **metadata)
        )
        return MWUTimeSeriesDetector(series).is_below_magnitude_threshold(comparison)

    assert is_below(alert_threshold=None) is False
    assert is_below(alert_threshold=2.0) is True
    assert is_below(alert_threshold=0.5) is False
    # An absolute threshold is compared against the amount, not the percentage.
    assert is_below(alert_threshold=2.0, alert_change_type=1) is True
    assert is_below(alert_threshold=0.5, alert_change_type=1) is False


def test_detection_lands_on_the_step_when_the_forward_window_runs_out():
    # One trial per push, so each window needs MIN_TRIALS pushes: the last
    # push with a full forward window is #31, but the step is at #40. Without
    # localization the detection would name a revision 9 pushes earlier.
    trials_per_push = [[0.5] if index < 40 else [1.0] for index in range(61)]
    series = make_treeherder_timeseries(trials_per_push)

    detections = MWUTimeSeriesDetector(series).detect_changes()

    assert len(detections) == 1
    assert detections[0].optional_detection_info["index"] == 40
    assert detections[0].location == "revision1040"


def test_detects_both_steps_in_a_series():
    trials_per_push = [[0.5] * REPLICATES_PER_PUSH for _ in range(15)]
    trials_per_push += [[1.0] * REPLICATES_PER_PUSH for _ in range(15)]
    trials_per_push += [[2.0] * REPLICATES_PER_PUSH for _ in range(15)]

    detections = MWUTimeSeriesDetector(make_treeherder_timeseries(trials_per_push)).detect_changes()

    assert [detection.location for detection in detections] == ["revision1015", "revision1030"]
    assert [detection.direction for detection in detections] == ["up", "up"]


def test_thresholds_can_be_passed_to_a_run():
    # A clean step the defaults call a change, which either gate turns away
    # once it's asked for evidence the data can't reach.
    series = make_step_series(0.5, 1.0)
    detector = MWUTimeSeriesDetector(series)

    assert len(detector.detect_changes()) == 1
    assert detector.detect_changes(pvalue_threshold=1e-30) == []
    assert detector.detect_changes(cliffs_threshold=1.5) == []


def test_summarize():
    detector = MWUTimeSeriesDetector(make_step_series(0.5, 1.0))
    detections = detector.detect_changes()

    summary = detector.summarize(detections)

    assert len(summary) == 1
    assert summary["location"].iloc[0] == "revision1015"
    assert summary["change"].iloc[0] == "regression"
    assert summary["delta_percentage"].iloc[0] == 100.0
