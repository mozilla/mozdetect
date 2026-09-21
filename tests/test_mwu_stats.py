# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Checks on the PerfCompare port in mozdetect.mwu_stats.

The expected values for the ported routines were produced by running
PerfCompare's own JavaScript (src/utils/bootstrap-ci.ts and
src/utils/shapiroWilk.ts) on the samples below, so these tests fail if the
port ever drifts from it. The routines with no JavaScript counterpart are
checked against scipy, and against a definition-following implementation of
Cliff's delta, instead.
"""
import pytest

from scipy.stats import mannwhitneyu

from mozdetect.mwu_stats import (
    SW_NORMALITY_THRESHOLD,
    bootstrap_median_diff_ci,
    check_distribution_normality,
    cliffs_delta,
    common_language_effect_size,
    direction_of_change,
    interpret_cles,
    interpret_effect_size,
    interpret_significance,
    mann_whitney_u,
    median,
    median_diff_pct,
    shapiro_wilk_test,
)

BASE = [
    10.2, 11.4, 9.8, 10.9, 12.1, 10.0, 11.7, 9.5, 10.4, 11.1,
    10.8, 9.9, 12.4, 10.3, 11.9, 10.6, 9.7, 11.2, 10.1, 12.8,
]  # fmt: skip
NEW = [
    12.9, 13.4, 11.8, 14.1, 12.2, 13.9, 12.5, 13.1, 14.6, 12.0,
    13.7, 12.3, 13.2, 14.0, 12.7, 13.5, 11.9, 14.3, 12.6, 13.8,
]  # fmt: skip


def reference_cliffs_delta(base, new):
    """Cliff's delta straight from its definition, pair by pair."""
    wins = sum(1 for x in base for y in new if x > y)
    losses = sum(1 for x in base for y in new if x < y)
    return (wins - losses) / (len(base) * len(new))


def test_median():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_median_diff_pct():
    assert median_diff_pct([10.0, 10.0], [11.0, 13.0]) == 20.0
    assert median_diff_pct([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert median_diff_pct([], [1.0]) is None
    assert median_diff_pct([1.0], []) is None


@pytest.mark.parametrize(
    "base, new",
    [
        ([10.2, 11.4, 9.8, 10.9, 12.1], [12.9, 13.4, 11.8, 14.1, 12.2]),
        ([1.0] * 15, [2.0] * 15),
        ([1.0, 1.0, 2.0, 3.0, 3.0, 3.0], [1.0, 2.0, 2.0, 3.0, 4.0, 5.0]),
        (BASE, NEW),
    ],
)
def test_mann_whitney_u_matches_scipy(base, new):
    reference = mannwhitneyu(base, new, alternative="two-sided", method="asymptotic")
    statistic, pvalue = mann_whitney_u(base, new)

    assert statistic == pytest.approx(reference.statistic)
    # The p-value goes through an erf approximation good to ~1.5e-7.
    assert pvalue == pytest.approx(reference.pvalue, abs=1e-6)


def test_mann_whitney_u_without_data():
    assert mann_whitney_u([], [1.0]) == (None, None)
    assert mann_whitney_u([1.0], []) == (None, None)


def test_mann_whitney_u_with_no_spread():
    # Every value tied: nothing tells the samples apart.
    assert mann_whitney_u([1.0] * 5, [1.0] * 5) == (12.5, 1.0)


def test_interpret_significance():
    assert interpret_significance(0.01) == ("significant", True)
    assert interpret_significance(0.05) == ("significant", True)
    assert interpret_significance(0.2) == ("not significant", False)
    assert interpret_significance(None) == ("", False)


@pytest.mark.parametrize(
    "base, new",
    [
        (BASE, NEW),
        ([1.0] * 15, [2.0] * 15),
        ([1.0, 1.0, 2.0, 3.0, 3.0, 3.0], [1.0, 2.0, 2.0, 3.0, 4.0, 5.0]),
        ([float(value) for value in range(40)], [float(value) + 2 for value in range(40)]),
    ],
)
def test_cliffs_delta_matches_reference(base, new):
    assert cliffs_delta(base, new) == pytest.approx(reference_cliffs_delta(base, new))


def test_cliffs_delta_without_data():
    assert cliffs_delta([], [1.0]) is None
    assert cliffs_delta([1.0], []) is None


def test_interpret_effect_size():
    assert interpret_effect_size(0.1) == ("negligible", False)
    assert interpret_effect_size(-0.2) == ("small", True)
    assert interpret_effect_size(0.4) == ("moderate", True)
    assert interpret_effect_size(-0.9) == ("large", True)
    assert interpret_effect_size(None)[1] is False


def test_common_language_effect_size():
    # Base always wins, so a base value is certain to be the greater one.
    statistic, _ = mann_whitney_u([2.0] * 5, [1.0] * 4)
    assert common_language_effect_size(statistic, 5, 4) == 1.0
    assert common_language_effect_size(None, 5, 4) is None
    assert common_language_effect_size(10, 0, 4) is None


def test_interpret_cles():
    assert interpret_cles(0.75)[1] is True
    assert interpret_cles(0.25)[1] is False
    assert interpret_cles(0.5)[1] is None
    assert interpret_cles(None)[1] is None


@pytest.mark.parametrize(
    "cles, lower_is_better, expected",
    [
        # cles > 0.5: base values run higher, so new is the lower one.
        (0.9, True, "improvement"),
        (0.9, False, "regression"),
        (0.1, True, "regression"),
        (0.1, False, "improvement"),
        (0.5, True, "no change"),
        (None, True, "no change"),
    ],
)
def test_direction_of_change(cles, lower_is_better, expected):
    assert direction_of_change(True, True, cles, lower_is_better) == expected


def test_direction_of_change_needs_both_gates():
    assert direction_of_change(False, True, 0.9, True) == "no change"
    assert direction_of_change(True, False, 0.9, True) == "no change"


def test_shapiro_wilk_test_matches_perfcompare():
    # Values from PerfCompare's shapiroWilkTest on the same samples.
    base_result = shapiro_wilk_test(BASE)
    assert base_result["w"] == pytest.approx(0.9499796683122572)
    assert base_result["pvalue"] == pytest.approx(0.2664268339081124)

    new_result = shapiro_wilk_test(NEW)
    assert new_result["w"] == pytest.approx(0.9596959849258533)
    assert new_result["pvalue"] == pytest.approx(0.642031805749638)

    # An outlier the IQR filter drops before testing.
    small_result = shapiro_wilk_test([1, 2, 3, 4, 5, 90])
    assert small_result["w"] == pytest.approx(0.9867621555675287)
    assert small_result["pvalue"] == pytest.approx(0.9799505304565954)

    # Under 12 values, the p-value takes the other Royston branch.
    ten_result = shapiro_wilk_test([3.1, 1.2, 4.4, 2.8, 5.6, 2.2, 3.9, 4.8, 1.9, 3.3])
    assert ten_result["w"] == pytest.approx(0.986849454985644)
    assert ten_result["pvalue"] == pytest.approx(0.9948056443260693)


def test_shapiro_wilk_test_without_enough_data():
    assert shapiro_wilk_test([1.0, 2.0]) is None
    assert shapiro_wilk_test([1.0] * 10) is None  # no spread
    assert shapiro_wilk_test(list(range(5001))) is None


def test_check_distribution_normality():
    normal = BASE
    # A hard step between two levels: nothing like a normal shape.
    lumpy = [1.0] * 10 + [100.0] * 10

    assert check_distribution_normality(normal, NEW) == "both"
    assert check_distribution_normality(normal, lumpy) == "one"
    assert check_distribution_normality(lumpy, lumpy) == "neither"
    # The threshold is loose on purpose: it drives a warning, not a verdict.
    assert SW_NORMALITY_THRESHOLD == 0.2


def test_bootstrap_median_diff_ci_matches_perfcompare():
    # Values from PerfCompare's bootstrapMedianDiffCI on the same samples;
    # the seeded PRNG is ported too, so these are reproducible.
    interval = bootstrap_median_diff_ci(BASE, NEW)
    assert interval.median_diff == pytest.approx(2.4499999999999993)
    assert interval.ci_low == pytest.approx(1.5)
    assert interval.ci_high == pytest.approx(3.2500000000000018)
    assert interval.significant is True

    fewer_iterations = bootstrap_median_diff_ci(BASE, NEW, n_iter=999)
    assert fewer_iterations.ci_low == pytest.approx(1.450000000000001)
    assert fewer_iterations.ci_high == pytest.approx(3.299999999999999)


def test_bootstrap_median_diff_ci_spanning_zero():
    interval = bootstrap_median_diff_ci(BASE, BASE[::-1], n_iter=999)

    assert interval.median_diff == 0
    assert interval.ci_low <= 0 <= interval.ci_high
    assert interval.significant is False


def test_bootstrap_median_diff_ci_without_enough_data():
    assert bootstrap_median_diff_ci([1.0], [1.0, 2.0]) is None
    assert bootstrap_median_diff_ci([1.0, 2.0], [1.0]) is None
