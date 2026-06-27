import warnings

import numpy as np
import pytest

from mushin.benchmark._aggregate import to_dataset
from mushin.benchmark._stats import (
    cohens_d,
    compare_methods,
    confidence_interval,
    holm_correction,
    warn_if_underpowered,
)


def test_confidence_interval_brackets_mean():
    mean, lo, hi = confidence_interval([0.8, 0.82, 0.79, 0.81, 0.80])
    assert lo < mean < hi
    assert abs(mean - 0.804) < 1e-6


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
    assert corrected[0] >= 0.01 * 3 - 1e-9


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
