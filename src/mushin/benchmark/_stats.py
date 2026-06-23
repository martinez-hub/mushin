# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Statistics for benchmark comparison, delegated to scipy.stats."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# name -> (callable returning (statistic, pvalue), is_paired)
_TESTS = {
    "wilcoxon": (lambda a, b: stats.wilcoxon(a, b), True),
    "ttest_rel": (lambda a, b: stats.ttest_rel(a, b), True),
    "welch": (lambda a, b: stats.ttest_ind(a, b, equal_var=False), False),
    "ttest_ind": (lambda a, b: stats.ttest_ind(a, b, equal_var=True), False),
    "mannwhitney": (lambda a, b: stats.mannwhitneyu(a, b), False),
}


def available_tests() -> list[str]:
    return list(_TESTS)


def confidence_interval(values, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` using a Student-t interval."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, mean, mean
    half = float(stats.sem(values) * stats.t.ppf(1 - alpha / 2, n - 1))
    return mean, mean - half, mean + half


def cohens_d(a, b) -> float:
    """Pooled-variance Cohen's d for two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_sd = float(np.sqrt(pooled_var))
    diff = float(a.mean() - b.mean())
    if pooled_sd == 0.0:
        # zero within-group variance: the effect is 0 only if the means also
        # match (up to floating-point roundoff); otherwise the groups are
        # perfectly separated (d is undefined / infinite) and reporting 0.0
        # would hide a real difference.
        if np.isclose(a.mean(), b.mean()):
            return 0.0
        return float("inf") if diff > 0 else float("-inf")
    return diff / pooled_sd


def holm_correction(pvalues) -> list[float]:
    """Holm-Bonferroni step-down correction, returned in original order.

    NaN p-values (e.g. a scipy test on a single seed) stay NaN and are excluded
    from the family so they cannot corrupt the correction of the valid ones."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)  # NaNs sort to the end
    n_valid = int(np.count_nonzero(~np.isnan(pvalues)))
    corrected = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        p = pvalues[idx]
        if np.isnan(p):
            corrected[idx] = np.nan
            continue
        running = max(running, (n_valid - rank) * p)
        corrected[idx] = min(running, 1.0)
    return [float(c) for c in corrected]


def compare_methods(
    ds: xr.Dataset, test: str = "wilcoxon", alpha: float = 0.05
) -> pd.DataFrame:
    """Pairwise comparison of methods for every metric in ``ds``.

    Holm correction is applied per metric across the method pairs."""
    if test not in _TESTS:
        raise ValueError(f"unknown test {test!r}; choose from {available_tests()}")
    func, _ = _TESTS[test]
    methods = [str(m) for m in ds["method"].values]

    rows = []
    for metric in ds.data_vars:
        recs, pvals = [], []
        for a, b in itertools.combinations(methods, 2):
            va = ds[metric].sel({"method": a}).values
            vb = ds[metric].sel({"method": b}).values
            if np.allclose(va, vb):
                p = 1.0
            else:
                _, p = func(va, vb)
            recs.append(
                {
                    "metric": str(metric),
                    "method_a": a,
                    "method_b": b,
                    "mean_diff": float(np.mean(va) - np.mean(vb)),
                    "effect_size": cohens_d(va, vb),
                    "p_value": float(p),
                }
            )
            pvals.append(float(p))
        corrected = holm_correction(pvals) if len(pvals) > 1 else pvals
        for rec, pc in zip(recs, corrected):
            rec["p_corrected"] = float(pc)
            rec["significant"] = False if np.isnan(pc) else bool(pc < alpha)
            rows.append(rec)

    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "method_a",
            "method_b",
            "mean_diff",
            "effect_size",
            "p_value",
            "p_corrected",
            "significant",
        ],
    )
