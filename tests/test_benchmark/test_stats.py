import warnings

import numpy as np
import pytest
import xarray as xr

from mushin.benchmark._aggregate import to_dataset
from mushin.benchmark._result import BenchmarkResult
from mushin.benchmark._stats import (
    cohens_d,
    compare_methods,
    confidence_interval,
    holm_correction,
    paired_item_bootstrap,
    warn_if_underpowered,
)


def test_confidence_interval_brackets_mean():
    mean, lo, hi = confidence_interval([0.8, 0.82, 0.79, 0.81, 0.80])
    assert lo < mean < hi
    assert abs(mean - 0.804) < 1e-6


def test_confidence_interval_degenerate_for_constant_values():
    """A method constant across seeds gets the degenerate point, never NaN bounds.

    `confidence_interval` delegates to `scipy.stats.t.interval`, which returns
    `(nan, nan)` when the standard error is 0 (`inf * 0`). Without the explicit
    zero-sem guard this surfaces as NaN `ci_low`/`ci_high` in `.summary()` for a
    deterministic method — see `test_summary_bounds_finite_for_constant_method`.
    """
    assert confidence_interval([0.9, 0.9, 0.9]) == (0.9, 0.9, 0.9)
    # a single observation has no spread either
    assert confidence_interval([2.5]) == (2.5, 2.5, 2.5)
    # and the guard must not swallow a genuine interval
    mean, lo, hi = confidence_interval([0.9, 0.9, 0.9, 0.91])
    assert lo < mean < hi


def test_summary_bounds_finite_for_constant_method():
    """`.summary()` never reports NaN bounds for a method constant across seeds."""
    ds = xr.Dataset(
        {
            "accuracy": (
                ("method", "seed"),
                np.array([[0.90, 0.90, 0.90], [0.70, 0.71, 0.72]]),
            )
        },
        coords={"method": ["deterministic", "base"], "seed": [0, 1, 2]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comparisons = compare_methods(ds, test="welch")
        summary = BenchmarkResult(
            data=ds, comparisons=comparisons, alpha=0.05
        ).summary()
    row = summary[summary["method"] == "deterministic"].iloc[0]
    assert not np.isnan(row["ci_low"]) and not np.isnan(row["ci_high"])
    assert row["ci_low"] == row["ci_high"] == row["mean"] == 0.90


def test_cohens_d_zero_for_identical():
    assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_cohens_d_zero_variance_same_means_is_zero():
    assert cohens_d([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0


def test_cohens_d_zero_variance_different_means_is_infinite():
    # perfectly separated constant groups -> effect is infinite, not zero
    assert cohens_d([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) == float("inf")
    assert cohens_d([0.0, 0.0], [1.0, 1.0]) == float("-inf")


def test_cohens_d_zero_variance_means_equal_within_roundoff_is_zero():
    # constant groups whose means differ only by float roundoff -> no effect
    assert cohens_d([1.0, 1.0], [1.0 + 1e-13, 1.0 + 1e-13]) == 0.0


def test_holm_is_monotone_and_capped():
    corrected = holm_correction([0.01, 0.04, 0.03])
    assert all(0.0 <= c <= 1.0 for c in corrected)
    # Exact step-down values: sorted p = [0.01, 0.03, 0.04] scaled by [3, 2, 1]
    # = [0.03, 0.06, 0.04], then a running max -> [0.03, 0.06, 0.06], mapped
    # back to input order [0.01, 0.04, 0.03].
    assert np.allclose(corrected, [0.03, 0.06, 0.06])
    # ...and the step-down enforces monotonicity in ascending-p order.
    ascending = [corrected[i] for i in np.argsort([0.01, 0.04, 0.03])]
    assert all(a <= b + 1e-12 for a, b in zip(ascending, ascending[1:]))


def test_compare_flags_clear_difference_as_significant():
    rng = np.random.default_rng(0)
    results = {
        "ours": [{"accuracy": float(v)} for v in 0.90 + 0.01 * rng.standard_normal(8)],
        "base": [{"accuracy": float(v)} for v in 0.70 + 0.01 * rng.standard_normal(8)],
    }
    ds = to_dataset(results)
    df = compare_methods(ds, test="wilcoxon", alpha=0.05)

    row = df[df["metric"] == "accuracy"].iloc[0]
    assert row["significant"]
    assert row["mean_diff"] > 0


def test_welch_test_runs():
    results = {
        "a": [{"accuracy": 0.9}, {"accuracy": 0.92}, {"accuracy": 0.88}],
        "b": [{"accuracy": 0.7}, {"accuracy": 0.72}, {"accuracy": 0.68}],
    }
    ds = to_dataset(results)
    df = compare_methods(ds, test="welch", alpha=0.05)
    assert "p_value" in df.columns and len(df) == 1


def test_compare_default_wilcoxon_survives_identical_methods():
    # identical per-seed values must not crash. Both methods are constant across
    # seeds (zero within-group variance), so the comparison is masked (NaN p, not
    # significant): there is no sampling distribution to test, regardless of the
    # means matching.
    import math

    results = {
        "a": [{"accuracy": 1.0}, {"accuracy": 1.0}, {"accuracy": 1.0}],
        "b": [{"accuracy": 1.0}, {"accuracy": 1.0}, {"accuracy": 1.0}],
    }
    ds = to_dataset(results)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = compare_methods(ds, test="wilcoxon", alpha=0.05)
    row = df[df["metric"] == "accuracy"].iloc[0]
    assert math.isnan(row["p_value"])
    assert not row["significant"]


def test_holm_correction_is_nan_safe():
    # a NaN p-value must stay NaN and not corrupt the other corrected values
    corrected = holm_correction([0.01, float("nan"), 0.04])
    assert np.isnan(corrected[1])
    assert not np.isnan(corrected[0])
    assert not np.isnan(corrected[2])
    assert all(0.0 <= c <= 1.0 for c in corrected if not np.isnan(c))


def test_compare_single_seed_parametric_is_nan_not_significant():
    # one seed + a parametric test -> scipy returns a NaN p-value; it must be
    # surfaced as NaN and flagged not-significant rather than crashing or lying
    results = {"a": [{"accuracy": 0.9}], "b": [{"accuracy": 0.7}]}
    ds = to_dataset(results)
    df = compare_methods(ds, test="welch", alpha=0.05)
    row = df.iloc[0]
    assert np.isnan(row["p_value"])
    assert not row["significant"]


def test_warn_if_underpowered_fires_for_small_n_wilcoxon():
    with pytest.warns(UserWarning, match="cannot reach"):
        warn_if_underpowered("wilcoxon", n_seeds=3, alpha=0.05)


def test_warn_if_underpowered_silent_for_welch_small_n():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> failure
        warn_if_underpowered("welch", n_seeds=3, alpha=0.05)


def test_compare_methods_warns_when_test_underpowered():
    # default wilcoxon over 3 seeds can never reach alpha=0.05 -> compare warns
    results = {
        "a": [{"accuracy": v} for v in (0.90, 0.91, 0.92)],
        "b": [{"accuracy": v} for v in (0.70, 0.71, 0.72)],
    }
    ds = to_dataset(results)
    with pytest.warns(UserWarning, match="cannot reach"):
        compare_methods(ds, test="wilcoxon", alpha=0.05)


def test_compare_methods_masks_zero_variance_methods():
    """Two methods each constant-across-seeds at different means have no sampling
    distribution -> masked (NaN p/effect, not significant), not a catastrophic-
    cancellation false positive. Matches the llm compare path."""
    import math

    import xarray as xr

    ds = xr.Dataset(
        {"acc": (("method", "seed"), np.array([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]]))},
        coords={"method": ["a", "b"], "seed": [0, 1, 2]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = compare_methods(ds, test="welch").iloc[0]
    assert math.isnan(row["p_value"])
    assert math.isnan(row["effect_size"])
    assert not row["significant"]


def test_compare_methods_warns_on_method_constant_in_every_metric():
    import xarray as xr

    ds = xr.Dataset(
        {"acc": (("method", "seed"), np.array([[1.0, 1.0, 1.0], [0.6, 0.7, 0.65]]))},
        coords={"method": ["det", "stoch"], "seed": [0, 1, 2]},
    )
    with pytest.warns(UserWarning, match="identical scores across all"):
        compare_methods(ds, test="welch")


def test_paired_test_reports_paired_effect_size():
    # For paired tests the effect size must be Cohen's d_z = mean(diff)/std(diff),
    # not the pooled independent-samples d (which ignores the pairing).
    a = np.array([1.0, 2.0, 3.0, 4.0])
    diffs = np.array([0.5, 0.6, 0.4, 0.5])
    b = a - diffs
    results = {
        "A": [{"m": float(v)} for v in a],
        "B": [{"m": float(v)} for v in b],
    }
    df = compare_methods(to_dataset(results), test="ttest_rel", alpha=0.05)
    expected = diffs.mean() / diffs.std(ddof=1)
    assert np.isclose(df["effect_size"].iloc[0], expected)


def test_unpaired_test_keeps_pooled_effect_size():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.5, 1.4, 2.6, 3.5]
    results = {
        "A": [{"m": v} for v in a],
        "B": [{"m": v} for v in b],
    }
    df = compare_methods(to_dataset(results), test="welch", alpha=0.05)
    assert np.isclose(df["effect_size"].iloc[0], cohens_d(a, b))


def test_bh_correction_known_values():
    from mushin.benchmark._stats import bh_correction

    # classic BH: adjusted_i = min over j>=i of p_(j) * n / rank_j
    assert np.allclose(bh_correction([0.01, 0.02, 0.03, 0.04]), [0.04] * 4)
    # monotone from the top: rank-2's 0.03*3/2=0.045 is capped by rank-3's 0.04
    assert np.allclose(bh_correction([0.005, 0.04, 0.03]), [0.015, 0.04, 0.04])
    # NaNs stay NaN and are excluded from the family
    out = bh_correction([0.01, np.nan, 0.02])
    assert np.isnan(out[1]) and np.allclose([out[0], out[2]], [0.02, 0.02])


def test_bonferroni_correction_caps_at_one():
    from mushin.benchmark._stats import bonferroni_correction

    assert bonferroni_correction([0.01, 0.4, 0.6]) == [0.03, 1.0, 1.0]


def _three_method_ds():
    rng = np.random.default_rng(1)
    results = {
        m: [{"acc": float(v)} for v in base + 0.01 * rng.standard_normal(6)]
        for m, base in [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    }
    return to_dataset(results)


def test_compare_methods_correction_options():
    ds = _three_method_ds()
    holm = compare_methods(ds, test="welch", correction="holm")
    none = compare_methods(ds, test="welch", correction="none")
    bh = compare_methods(ds, test="welch", correction="fdr_bh")
    assert np.allclose(none["p_corrected"], none["p_value"])
    assert (holm["p_corrected"] >= holm["p_value"] - 1e-15).all()
    assert (bh["p_corrected"] <= holm["p_corrected"] + 1e-15).all()  # BH less strict


def test_compare_methods_unknown_correction_raises():
    with pytest.raises(ValueError, match="correction"):
        compare_methods(_three_method_ds(), correction="fdr_by")


def test_incomplete_path_masks_constant_methods_too():
    """Under allow_incomplete=True the zero-variance mask must be computed over
    the COMPLETED seeds — a method constant across every completed seed has no
    sampling distribution, and NaN-contaminated rows must not sneak it past the
    mask into a reported significance."""
    ds = xr.Dataset(
        {
            "acc": (
                ("method", "seed"),
                [[1.0, 1.0, 1.0, np.nan, 1.0], [2.0, 3.0, 4.0, np.nan, 5.0]],
            )
        },
        coords={"method": ["det", "stoch"], "seed": [0, 1, 2, 3, 4]},
        attrs={"mushin_failures": ["seed=3"]},
    )
    with pytest.warns(UserWarning):
        row = compare_methods(ds, test="wilcoxon", allow_incomplete=True).iloc[0]
    assert np.isnan(row["p_value"])
    assert np.isnan(row["effect_size"])
    assert not bool(row["significant"])


def test_incomplete_path_keeps_infinite_cells():
    """±Inf is a real completed value (e.g. diverged loss), not a missing cell:
    allow_incomplete must only drop NaN (failed/skipped) cells, so an Inf cell
    stays in the pairing instead of being silently excluded."""
    ds = xr.Dataset(
        {
            "loss": (
                ("method", "seed"),
                [[1.0, 2.0, np.inf, 4.0], [2.0, 3.0, 4.0, 5.0]],
            )
        },
        coords={"method": ["a", "b"], "seed": [0, 1, 2, 3]},
        attrs={"mushin_failures": ["elsewhere"]},
    )
    with pytest.warns(UserWarning):
        row = compare_methods(ds, test="wilcoxon", allow_incomplete=True).iloc[0]
    # the Inf pair participates: the mean difference is -Inf, not a finite
    # value computed over a silently reduced sample
    assert np.isinf(row["mean_diff"])


def test_paired_item_bootstrap_detects_consistent_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(size=150) + 0.5
    b = a - 0.5 + rng.normal(scale=0.2, size=150)
    diff, lo, hi, p = paired_item_bootstrap(a, b, n_resamples=2000, seed=1)
    assert diff > 0 and lo > 0 and p < 0.05  # consistent per-item win


def test_paired_item_bootstrap_straddles_zero_when_items_disagree():
    """An aggregate edge built on inconsistent per-item wins is not robust."""
    rng = np.random.default_rng(7)
    n = 200
    adv = rng.choice([1.0, -1.0], size=n, p=[0.55, 0.45])
    base = rng.random(n)
    diff, lo, hi, p = paired_item_bootstrap(
        base + 0.5 * adv, base - 0.5 * adv, n_resamples=4000, seed=0
    )
    assert diff > 0  # A does lead on this eval set...
    assert lo < 0 < hi and p > 0.05  # ...but it would not survive other items


def test_paired_item_bootstrap_is_deterministic():
    a, b = [1.0, 2.0, 3.0, 4.0], [1.1, 1.8, 3.3, 3.6]
    assert paired_item_bootstrap(a, b, n_resamples=500, seed=3) == (
        paired_item_bootstrap(a, b, n_resamples=500, seed=3)
    )


def test_paired_item_bootstrap_rejects_ragged_and_handles_tiny_input():
    with pytest.raises(ValueError, match="equal-length"):
        paired_item_bootstrap([1.0, 2.0], [1.0])
    diff, lo, hi, p = paired_item_bootstrap([1.0], [0.0])
    assert diff == 1.0 and np.isnan(lo) and np.isnan(hi) and np.isnan(p)


def test_item_bootstrap_masks_constant_nonzero_difference():
    """A constant per-item difference has no sampling distribution.

    Every d_i identical makes the resampled distribution a point mass, which
    naively yields a zero-width CI and the floor p-value at ANY n — total
    domination on a small binary eval set is ordinary, not contrived. Mirror the
    seed axis: mask it instead of asserting certainty.
    """
    with pytest.warns(UserWarning, match="identical difference"):
        diff, lo, hi, p = paired_item_bootstrap(
            [1.0] * 4, [0.0] * 4, n_resamples=1000, seed=0
        )
    assert diff == 1.0
    assert np.isnan(lo) and np.isnan(hi) and np.isnan(p)


def test_item_bootstrap_identical_systems_still_report_no_difference():
    """d == 0 everywhere is informative (they match), not a masked unknown."""
    a = np.arange(30) * 0.1
    diff, lo, hi, p = paired_item_bootstrap(a, a, n_resamples=500, seed=0)
    assert (diff, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)


def test_item_bootstrap_warns_on_too_few_items():
    """The percentile bootstrap is ~49% false-positive at n=2, ~16% at n=5."""
    with pytest.warns(UserWarning, match="only 5 items"):
        paired_item_bootstrap(
            [1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0], n_resamples=500
        )


def test_item_bootstrap_chunking_is_bit_identical_and_bounded():
    """Chunked resampling must consume the same rng stream as one big draw."""
    import mushin.benchmark._stats as st

    r = np.random.default_rng(0)
    a, b = r.normal(size=500), r.normal(size=500)
    small_budget = st._BOOTSTRAP_CELL_BUDGET
    try:
        st._BOOTSTRAP_CELL_BUDGET = 10**9  # effectively one chunk
        one = paired_item_bootstrap(a, b, n_resamples=2000, seed=1)
        st._BOOTSTRAP_CELL_BUDGET = 5000  # many small chunks
        many = paired_item_bootstrap(a, b, n_resamples=2000, seed=1)
    finally:
        st._BOOTSTRAP_CELL_BUDGET = small_budget
    assert one == many


def test_cluster_bootstrap_fixes_undercoverage_on_grouped_items():
    """Grouped items break independence; resampling items alone under-covers."""
    labels = np.repeat(np.arange(25), 8)
    cov_naive = cov_clustered = 0
    trials = 120
    for k in range(trials):
        r = np.random.default_rng(k)
        # items in a group share a group effect; the TRUE difference is 0
        d = r.normal(scale=1.0, size=25)[labels] + r.normal(scale=0.3, size=labels.size)
        a, b = d, np.zeros_like(d)
        _, lo, hi, _ = paired_item_bootstrap(a, b, n_resamples=600, seed=k)
        cov_naive += lo <= 0 <= hi
        _, lo2, hi2, _ = paired_item_bootstrap(
            a, b, clusters=labels, n_resamples=600, seed=k
        )
        cov_clustered += lo2 <= 0 <= hi2
    # naive badly under-covers; clustering recovers most of the nominal 95%
    assert cov_naive / trials < 0.75
    assert cov_clustered / trials > 0.85


def test_cluster_bootstrap_matches_naive_for_singleton_clusters():
    """One item per cluster is the un-clustered problem — the two must agree."""
    r = np.random.default_rng(0)
    a, b = r.normal(size=120), r.normal(size=120)
    naive = paired_item_bootstrap(a, b, n_resamples=3000, seed=1)
    singles = paired_item_bootstrap(
        a, b, clusters=np.arange(120), n_resamples=3000, seed=1
    )
    assert np.allclose(naive, singles)


def test_cluster_bootstrap_guards():
    with pytest.raises(ValueError, match="one label per item"):
        paired_item_bootstrap([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], clusters=[1, 1])
    # a single cluster cannot be resampled -> NaN interval rather than a fake one
    diff, lo, hi, p = paired_item_bootstrap([1.0, 2.0], [0.0, 0.0], clusters=[7, 7])
    assert diff == 1.5 and np.isnan(lo) and np.isnan(hi) and np.isnan(p)
