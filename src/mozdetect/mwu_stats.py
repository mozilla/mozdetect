# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
Python port of the statistics PerfCompare computes for a comparison.

PerfCompare splits this work in two: the significance tests come back from the
Treeherder API, while the front end computes the rest itself in JavaScript.
This module brings both halves together in one place so the `mwu` detector
matches what a sheriff sees in PerfCompare.

Ported from PerfCompare (https://github.com/mozilla/perfcompare):

    * shapiro_wilk_test, and the IQR filter it applies first
      -- src/utils/shapiroWilk.ts
    * bootstrap_median_diff_ci, and the seeded PRNG behind it
      -- src/utils/bootstrap-ci.ts
    * median_diff_pct, check_distribution_normality, SW_NORMALITY_THRESHOLD
      -- src/common/testVersions/mannWhitney.tsx

Mann-Whitney U, Cliff's delta and the CLES have no JavaScript counterpart --
PerfCompare reads them off the API rather than computing them -- so they're
implemented here to keep the whole comparison in one module.

Everything is plain Python: these are ports of specific numerical
approximations, and swapping in a library routine would quietly change the
numbers they produce.
"""
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

# p-value at or below which the two samples are taken to come from different
# distributions.
PVALUE_THRESHOLD = 0.05

# Cliff's delta bands. PerfCompare's documentation describes a score past
# +/-0.47 as a large difference and one near 0 as no real difference; anything
# below the negligible bound isn't treated as a change at all, however small
# the p-value.
# https://openpublishing.library.umass.edu/pare/article/1977/galley/1980/view/
CLIFFS_NEGLIGIBLE = 0.20
CLIFFS_SMALL = 0.33
CLIFFS_MODERATE = 0.47

# Shapiro-Wilk p-value above which a sample is called normal. PerfCompare uses
# a much looser bound here than the 0.05 used for significance: this only
# drives a "distribution shapes aren't normal" warning, so it errs towards
# flagging.
SW_NORMALITY_THRESHOLD = 0.2


def mean(values):
    return sum(values) / len(values)


def median(values):
    """Median of `values`, matching the JS implementations' definition."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


def median_diff_pct(base, new):
    """Difference of medians as a signed percentage of the base median.

    Ported from medianDiffPct in mannWhitney.tsx: None when there's nothing to
    compute from, and 0 for a zero base median rather than an undefined
    division.
    """
    if not base or not new:
        return None

    base_median = median(base)
    if base_median == 0:
        return 0.0
    return (median(new) - base_median) / base_median * 100


# ---------------------------------------------------------------------------
# Normal distribution helpers
#
# PerfCompare carries two different normal CDF approximations, one in each of
# the two JS files ported here. They agree to about 1e-7, but each is kept with
# the routine that uses it so the ported numbers match PerfCompare's exactly.
# ---------------------------------------------------------------------------


def _erf(x):
    """Abramowitz & Stegun 7.1.26 -- max error 1.5e-7. From bootstrap-ci.ts."""
    t = 1 / (1 + 0.3275911 * abs(x))
    p = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return (1 if x >= 0 else -1) * (1 - p * math.exp(-x * x))


def _normal_cdf(x):
    """Normal CDF via the erf approximation above. From bootstrap-ci.ts."""
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def _normal_cdf_royston(x):
    """Normal CDF, Horner-form erfc approximation. From shapiroWilk.ts."""
    t = 1 / (1 + 0.3275911 * abs(x))
    poly = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    p = 1 - poly * math.exp(-x * x / 2) * 0.3989422804
    return p if x >= 0 else 1 - p


# Acklam rational approximation -- max absolute error 1.15e-9. Both JS files
# carry a copy of this (normalPPF in bootstrap-ci.ts, normalQuantile in
# shapiroWilk.ts); they differ only in trailing digits of one coefficient.
_ACKLAM_A = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
]
_ACKLAM_B = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
]
_ACKLAM_C = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
]
_ACKLAM_D = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
]
_ACKLAM_P_LOW = 0.02425


def _normal_ppf(p):
    """Inverse normal CDF (Acklam 2003)."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf

    c, d = _ACKLAM_C, _ACKLAM_D
    if p < _ACKLAM_P_LOW:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )

    if p <= 1 - _ACKLAM_P_LOW:
        a, b = _ACKLAM_A, _ACKLAM_B
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )

    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


# ---------------------------------------------------------------------------
# Mann-Whitney U, Cliff's delta and the CLES
# ---------------------------------------------------------------------------


def _rank_sum(base, new):
    """Sum of `base`'s midranks in the two samples pooled together, and the
    tie correction term for the variance of U.
    """
    pooled = sorted([(value, 0) for value in base] + [(value, 1) for value in new])

    rank_sum = 0.0
    tie_term = 0.0
    start = 0
    while start < len(pooled):
        end = start
        while end + 1 < len(pooled) and pooled[end + 1][0] == pooled[start][0]:
            end += 1

        # Tied values all take the average of the ranks they span. Ranks are
        # 1-based, so the group spans ranks start+1 .. end+1.
        size = end - start + 1
        midrank = (start + 1 + end + 1) / 2
        rank_sum += midrank * sum(1 for _, group in pooled[start : end + 1] if group == 0)
        tie_term += size**3 - size
        start = end + 1

    return rank_sum, tie_term


def mann_whitney_u(base, new):
    """Two-sided Mann-Whitney U test.

    Returns (U, p-value), where U counts the pairs in which the base value
    wins, ties counting half. The p-value comes from the normal approximation
    with the usual tie and continuity corrections, which is what the sample
    sizes this runs on call for.

    Returns (None, None) when either sample is empty.
    """
    n_base, n_new = len(base), len(new)
    if n_base == 0 or n_new == 0:
        return None, None

    rank_sum, tie_term = _rank_sum(base, new)
    u = rank_sum - n_base * (n_base + 1) / 2

    total = n_base + n_new
    mu = n_base * n_new / 2
    variance = (n_base * n_new / 12) * ((total + 1) - tie_term / (total * (total - 1)))
    if variance <= 0:
        # Every value is tied, so there's nothing to tell the samples apart.
        return u, 1.0

    # Continuity correction, applied towards the mean from whichever side U is
    # on, then doubled for the two-sided alternative.
    z = (abs(u - mu) - 0.5) / math.sqrt(variance)
    pvalue = min(1.0, 2 * (1 - _normal_cdf(max(z, 0))))
    return u, pvalue


def interpret_significance(pvalue, pvalue_threshold=PVALUE_THRESHOLD):
    """PerfCompare's "Real" / "Noise" verdict, in the API's wording."""
    if pvalue is None:
        return "", False
    if pvalue <= pvalue_threshold:
        return "significant", True
    return "not significant", False


def cliffs_delta(base, new):
    """Cliff's delta: how far the two samples are from overlapping.

    Positive when base values tend to be the larger ones, negative when new
    values do, and near zero when a value from one is about as likely to be
    above as below a value from the other.

    Returns None when either sample is empty.
    """
    if not base or not new:
        return None

    ordered_new = sorted(new)
    greater = 0
    less = 0
    for value in base:
        # Values below `value` are ones base wins against, values above are
        # ones it loses to; the gap between the two bisects is the ties.
        greater += bisect_left(ordered_new, value)
        less += len(ordered_new) - bisect_right(ordered_new, value)

    return (greater - less) / (len(base) * len(ordered_new))


def interpret_effect_size(delta):
    """Plain-language magnitude of a Cliff's delta, and whether it's a
    difference worth calling a change.
    """
    if delta is None:
        return "Effect cannot be interpreted", False
    if abs(delta) < CLIFFS_NEGLIGIBLE:
        return "negligible", False
    if abs(delta) < CLIFFS_SMALL:
        return "small", True
    if abs(delta) < CLIFFS_MODERATE:
        return "moderate", True
    return "large", True


def common_language_effect_size(u, n_base, n_new):
    """The chance that a value picked from base is greater than one picked
    from new, which is where the direction of a change comes from.

    Returns None when it can't be computed.
    """
    if u is None or not n_base or not n_new:
        return None
    return u / (n_base * n_new)


def interpret_cles(cles):
    """PerfCompare's wording for a CLES, and which side it favours."""
    if cles is None:
        return "CLES cannot be interpreted", None
    if cles > 0.5:
        return f"{cles:.0%} chance a base value > a new value", True
    if cles < 0.5:
        return f"{1 - cles:.0%} chance a new value > base value", False
    return "CLES cannot be interpreted", None


def direction_of_change(is_significant, is_effect_meaningful, cles, lower_is_better):
    """Whether the change is an improvement, a regression, or neither.

    A change has to clear both gates -- significant *and* a meaningful effect
    -- before it's called in either direction. The CLES says which sample runs
    higher, and `lower_is_better` says what that means for the metric.
    """
    if not (is_significant and is_effect_meaningful) or cles is None or cles == 0.5:
        return "no change"

    is_base_greater = cles > 0.5
    new_is_lower = is_base_greater
    is_improvement = new_is_lower if lower_is_better else not new_is_lower
    return "improvement" if is_improvement else "regression"


# ---------------------------------------------------------------------------
# Shapiro-Wilk normality test -- src/utils/shapiroWilk.ts
# ---------------------------------------------------------------------------


def _poly5(coefficients, u):
    return (
        (((coefficients[0] * u + coefficients[1]) * u + coefficients[2]) * u + coefficients[3]) * u
        + coefficients[4]
    ) * u + coefficients[5]


def _iqr_filter(data):
    """Drops values more than 1.5 IQR outside the quartiles.

    PerfCompare applies this before testing for normality so a couple of
    stragglers don't make an otherwise normal sample look skewed. The quartiles
    are taken by position rather than interpolated, as in the JS.
    """
    if len(data) < 4:
        return list(data)

    ordered = sorted(data)
    n = len(ordered)
    q1 = ordered[int(n * 0.25)]
    q3 = ordered[int(n * 0.75)]
    iqr = q3 - q1
    return [x for x in ordered if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]


def shapiro_wilk_test(data):
    """Shapiro-Wilk test for normality.

    W statistic: Shapiro & Wilk (1965). Coefficients: Royston (1992), AS R94.
    p-value: Royston (1995).

    Returns {"w", "pvalue"}, or None when the sample is too small, too large,
    or has no spread at all.
    """
    x = sorted(_iqr_filter(data))
    n = len(x)
    if n < 3 or n > 5000:
        return None

    # Expected normal order statistics
    m = [_normal_ppf((i + 1 - 0.375) / (n + 0.25)) for i in range(n)]
    md = sum(value * value for value in m)
    sqrt_md = math.sqrt(md)

    # Royston (1992) polynomial corrections for the first two a coefficients
    c1 = [-2.706056, 4.434685, -2.07119, -0.147981, 0.221157, m[n - 1] / sqrt_md]
    c2 = [-3.582633, 5.682633, -1.752461, -0.293762, 0.042981, m[n - 2] / sqrt_md]
    u = 1 / math.sqrt(n)
    an = _poly5(c1, u)  # corrected a_n (largest coeff)
    ann = _poly5(c2, u)  # corrected a_{n-1}

    # phi normalizes the remaining middle coefficients
    half = n // 2
    if n > 5:
        phi = (md - 2 * m[n - 1] ** 2 - 2 * m[n - 2] ** 2) / (1 - 2 * an**2 - 2 * ann**2)
    else:
        phi = (md - 2 * m[n - 1] ** 2) / (1 - 2 * an**2)
    sqrt_phi = math.sqrt(phi)

    # Half-length a array: a[j] is the coefficient for (x[n-1-j] - x[j])
    a = [0.0] * half
    a[0] = an
    if n > 5 and half > 1:
        a[1] = ann
    for j in range(2 if n > 5 else 1, half):
        a[j] = m[n - 1 - j] / sqrt_phi

    xbar = sum(x) / n
    ss = sum((value - xbar) ** 2 for value in x)
    if ss == 0:
        return None

    num = sum(a[j] * (x[n - 1 - j] - x[j]) for j in range(half))
    w = min(num**2 / ss, 1)

    # p-value via Royston (1995) log-normal approximation
    logn = math.log(n)
    if n < 12:
        gamma = 0.459 * n - 2.273
        g = -math.log(gamma - math.log(1 - w))
        mu = -0.0006714 * n**3 + 0.025054 * n**2 - 0.39978 * n + 0.544
        sigma = math.exp(-0.0020322 * n**3 + 0.062767 * n**2 - 0.77857 * n + 1.3822)
    else:
        g = math.log(1 - w)
        mu = 0.0038915 * logn**3 - 0.083751 * logn**2 - 0.31082 * logn - 1.5861
        sigma = math.exp(0.0030302 * logn**2 - 0.082676 * logn - 0.4803)

    return {"w": w, "pvalue": 1 - _normal_cdf_royston((g - mu) / sigma)}


def check_distribution_normality(base, new, threshold=SW_NORMALITY_THRESHOLD):
    """Whether "both", "one" or "neither" of the two samples look normal.

    PerfCompare shows a warning on anything other than "both": the comparison
    itself doesn't assume normality, but a non-normal shape is worth knowing
    about when reading a median difference.
    """
    base_result = shapiro_wilk_test(base)
    new_result = shapiro_wilk_test(new)
    base_normal = base_result is not None and base_result["pvalue"] > threshold
    new_normal = new_result is not None and new_result["pvalue"] > threshold

    if base_normal and new_normal:
        return "both"
    if base_normal or new_normal:
        return "one"
    return "neither"


# ---------------------------------------------------------------------------
# BCa bootstrap confidence interval -- src/utils/bootstrap-ci.ts
# ---------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    median_diff: float
    ci_low: float
    ci_high: float
    significant: bool  # CI does not contain 0


def _mulberry32(seed):
    """Fast seedable PRNG, so a comparison always gives the same interval.

    A direct port: JS coerces these operations to 32 bits, so every step is
    masked to 32 bits here to produce the identical sequence.
    """
    state = seed & 0xFFFFFFFF

    def imul(a, b):
        return (a * b) & 0xFFFFFFFF

    def rand():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(state ^ (state >> 15), 1 | state)
        t ^= (t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF
        return (t ^ (t >> 14)) / 0x100000000

    return rand


def _resample(values, rand):
    count = len(values)
    return [values[int(rand() * count)] for _ in range(count)]


def _jackknife_diffs(base, new):
    """Leave-one-out estimates of median(new) - median(base): leave out each
    base observation in turn, then each new observation in turn.
    """
    median_new = median(new)
    median_base = median(base)

    jack = [median_new - median(base[:i] + base[i + 1 :]) for i in range(len(base))]
    jack += [median(new[:i] + new[i + 1 :]) - median_base for i in range(len(new))]
    return jack


def bootstrap_median_diff_ci(base, new, n_iter=9999, alpha=0.05, seed=42):
    """BCa confidence interval for median(new) - median(base).

    BCa improves on the plain percentile interval by correcting for bias (z0,
    how far the bootstrap distribution sits from the observed difference) and
    for skewness (the acceleration term, estimated by leave-one-out
    jackknife), which matters because performance samples are rarely
    symmetric.

    Returns None when either sample has fewer than 2 values: the jackknife is
    undefined on a single observation, and a one-point sample carries no
    resampling variability to build an interval from.
    """
    base = list(base)
    new = list(new)
    if len(base) < 2 or len(new) < 2:
        return None

    rand = _mulberry32(seed)
    observed = median(new) - median(base)

    diffs = [median(_resample(new, rand)) - median(_resample(base, rand)) for _ in range(n_iter)]

    # Bias-correction: the share of resamples below the observed difference,
    # clamped away from 0 and 1 to keep the quantile finite.
    below = sum(1 for diff in diffs if diff < observed)
    prop = max(0.5 / n_iter, min(1 - 0.5 / n_iter, below / n_iter))
    z0 = _normal_ppf(prop)

    jack = _jackknife_diffs(base, new)
    jack_mean = sum(jack) / len(jack)
    numerator = sum((jack_mean - value) ** 3 for value in jack)
    denominator = 6 * sum((jack_mean - value) ** 2 for value in jack) ** 1.5
    acceleration = 0 if denominator == 0 else numerator / denominator

    def adjust(z):
        denom = 1 - acceleration * (z0 + z)
        if denom == 0:
            return 0 if z < 0 else 1
        return _normal_cdf(z0 + (z0 + z) / denom)

    alpha1 = adjust(_normal_ppf(alpha / 2))
    alpha2 = adjust(_normal_ppf(1 - alpha / 2))

    diffs.sort()
    low_index = max(0, min(int(alpha1 * n_iter), n_iter - 1))
    high_index = max(0, min(int(alpha2 * n_iter), n_iter - 1))

    ci_low = diffs[low_index]
    ci_high = diffs[high_index]
    return BootstrapCI(
        median_diff=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=ci_low > 0 or ci_high < 0,
    )
