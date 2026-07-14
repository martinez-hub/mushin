# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Statistics for benchmark comparison, delegated to scipy.stats."""

from __future__ import annotations

import itertools
import warnings

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


def warn_if_underpowered(test: str, n_seeds: int, alpha: float) -> None:
    """Warn if ``test`` cannot reach ``alpha`` at ``n_seeds`` seeds.

    Determined empirically: run the test on maximally-separated samples of size
    ``n_seeds`` and check whether the best-case p-value clears ``alpha``. (A
    paired Wilcoxon over 3 seeds, for example, can never go below p=0.25.)"""
    if test not in _TESTS:
        return
    func, _ = _TESTS[test]
    a = np.arange(n_seeds, dtype=float) + 1000.0
    b = np.arange(n_seeds, dtype=float)
    try:
        _, p = func(a, b)
    except ValueError:
        return  # test could not even run at this n; nothing useful to say
    if float(p) > alpha:
        warnings.warn(
            f"test={test!r} cannot reach alpha={alpha} with {n_seeds} seeds "
            f"(best-case p={float(p):.4g}); use more seeds or a parametric test "
            f"such as test='welch'.",
            UserWarning,
            stacklevel=3,
        )


class IncompleteSweepError(RuntimeError):
    """Raised when statistics are requested on a sweep that has failed/missing runs."""


def _is_constant(values) -> bool:
    """True if values have no meaningful within-group variance (``np.allclose`` to
    the first), so they form no sampling distribution. Mirrors the between-group
    ``np.allclose`` short-circuit below; sub-epsilon float jitter counts as
    constant rather than leaking into a catastrophic-cancellation p-value."""
    arr = np.asarray(values, dtype=float)
    return bool(np.allclose(arr, arr[0]))


def compare_methods(
    ds: xr.Dataset, test: str = "wilcoxon", alpha: float = 0.05
) -> pd.DataFrame:
    """Pairwise comparison of methods for every metric in ``ds``.

    Holm correction is applied per metric across the method pairs. A method whose
    scores are constant across all seeds (for a metric) has no sampling
    distribution, so comparisons involving it are masked — ``p_value``,
    ``p_corrected`` and ``effect_size`` become ``NaN`` and ``significant`` is
    ``False`` — rather than reporting a duplicated-point p-value of ~0 and a
    meaningless ±huge effect size; Holm is then applied over the surviving pairs.
    Emits a warning when ``test`` cannot reach ``alpha`` at the dataset's seed
    count, and when a method is constant across seeds in *every* metric.

    Raises
    ------
    IncompleteSweepError
        If ``ds.attrs["mushin_failures"]`` is a non-empty list, meaning the sweep
        that produced ``ds`` had failed/missing runs (recorded under
        ``on_error="nan"``). A dataset without the attr — a plain user dataset or
        a clean (failure-free) sweep — is unaffected. This is keyed purely on the
        completeness signal, never on raw NaN values in the data, so a metric that
        is legitimately NaN for other reasons does not trigger it.
    """
    failures = ds.attrs.get("mushin_failures")
    if failures:
        raise IncompleteSweepError(
            f"{len(failures)} run(s) failed ({', '.join(map(str, failures))}); "
            "fix the cause and re-run with resume=True to complete the sweep "
            "before comparing."
        )
    if test not in _TESTS:
        raise ValueError(f"unknown test {test!r}; choose from {available_tests()}")
    n_seeds = int(ds.sizes["seed"])
    warn_if_underpowered(test, n_seeds, alpha)
    func, _ = _TESTS[test]
    methods = [str(m) for m in ds["method"].values]

    rows = []
    constant_metric_count: dict[str, int] = {}  # method -> #metrics it is constant in
    n_metrics = 0
    for metric in ds.data_vars:
        n_metrics += 1
        vals = {m: ds[metric].sel({"method": m}).values for m in methods}
        # A method with no within-group variance has no sampling distribution.
        constant = {m for m in methods if n_seeds > 1 and _is_constant(vals[m])}
        for m in constant:
            constant_metric_count[m] = constant_metric_count.get(m, 0) + 1
        recs, pvals = [], []
        for a, b in itertools.combinations(methods, 2):
            va, vb = vals[a], vals[b]
            eff = cohens_d(va, vb)
            if a in constant or b in constant:
                # A method with zero within-group variance has no valid sampling
                # distribution, so any comparison involving it is masked (regardless
                # of whether the means happen to match). NaN p excludes this pair
                # from the Holm family (see holm_correction), so survivors are
                # corrected over the reduced set; the meaningless ±huge effect size
                # is NaN'd too.
                p = float("nan")
                eff = float("nan")
            elif np.allclose(va, vb):
                # Indistinguishable methods that DO have within-group variance:
                # no difference -> p=1.0.
                p = 1.0
            else:
                _, p = func(va, vb)
                p = float(p)
            recs.append(
                {
                    "metric": str(metric),
                    "method_a": a,
                    "method_b": b,
                    "mean_diff": float(np.mean(va) - np.mean(vb)),
                    "effect_size": eff,
                    "p_value": p,
                }
            )
            pvals.append(p)
        corrected = holm_correction(pvals) if len(pvals) > 1 else pvals
        for rec, pc in zip(recs, corrected, strict=True):
            rec["p_corrected"] = float(pc)
            rec["significant"] = False if np.isnan(pc) else bool(pc < alpha)
            rows.append(rec)

    # Warn for methods constant across seeds in *every* metric (they ignore the
    # seed): the seeds are duplicated points, not independent samples.
    for m in sorted(methods):
        if n_metrics and constant_metric_count.get(m, 0) == n_metrics:
            warnings.warn(
                f"method {m!r} produced identical scores across all {n_seeds} "
                "seeds — it is deterministic or ignores the seed, so seed-based "
                "significance involving it is not meaningful (the seeds are "
                "duplicated points, not independent samples).",
                UserWarning,
                stacklevel=2,
            )

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
