# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Statistics for benchmark comparison, delegated to scipy.stats."""

from __future__ import annotations

import itertools
import json
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
    """Return ``(mean, ci_low, ci_high)`` using scipy's Student-t interval."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, mean, mean
    sem = float(stats.sem(values))
    if sem == 0.0:
        # Identical values every seed: the honest interval is the degenerate
        # point. scipy's t.interval would return (nan, nan) here (inf * 0), which
        # would show up as NaN bounds in `.summary()` for a deterministic method.
        return mean, mean, mean
    low, high = stats.t.interval(1 - alpha, n - 1, loc=mean, scale=sem)
    return mean, float(low), float(high)


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


def cohens_dz(a, b) -> float:
    """Paired-samples Cohen's :math:`d_z`: ``mean(a - b) / std(a - b)``.

    The effect size matching the paired tests (``wilcoxon``, ``ttest_rel``);
    the pooled-variance :func:`cohens_d` ignores the pairing and understates
    the effect when the per-seed differences are consistent.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    if sd == 0.0:
        # identical differences every seed: zero effect only if that constant
        # difference is itself ~0; otherwise the shift is perfectly consistent
        # (d_z is undefined/infinite) and 0.0 would hide a real difference.
        if np.isclose(mean, 0.0):
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    return mean / sd


def _normalize_failures(val) -> list[str]:
    """``attrs['mushin_failures']`` as a list of combo strings, any vintage.

    Current writers store a JSON string. Legacy in-memory datasets used a raw
    list, a legacy 1-element list round-trips netCDF4 as a bare string, and a
    multi-element one comes back as an ndarray.
    """
    if val is None:
        return []
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except ValueError:
            return [val]  # legacy scalar: a single combo string
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
        return [str(parsed)]
    if isinstance(val, np.ndarray):
        return [str(c) for c in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [str(c) for c in val]
    return [str(val)]


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


def bonferroni_correction(pvalues) -> list[float]:
    """Bonferroni correction (``p * n_valid``, capped at 1), original order.
    NaNs stay NaN and are excluded from the family size."""
    pvalues = np.asarray(pvalues, dtype=float)
    n_valid = int(np.count_nonzero(~np.isnan(pvalues)))
    return [
        float("nan") if np.isnan(p) else float(min(p * n_valid, 1.0)) for p in pvalues
    ]


def bh_correction(pvalues) -> list[float]:
    """Benjamini-Hochberg (FDR) adjusted p-values, returned in original order.

    Delegates to :func:`scipy.stats.false_discovery_control` (scipy >= 1.11;
    our floor is 1.13) rather than reimplementing the step-up. scipy has no NaN
    handling, so NaN p-values are held out of the family and restored as NaN —
    matching :func:`holm_correction`."""
    pvalues = np.asarray(pvalues, dtype=float)
    corrected = np.full(pvalues.shape, np.nan)
    valid = ~np.isnan(pvalues)
    if valid.any():
        corrected[valid] = stats.false_discovery_control(pvalues[valid], method="bh")
    return [float(c) for c in corrected]


def _no_correction(pvalues) -> list[float]:
    return [float(p) for p in pvalues]


_CORRECTIONS = {
    "holm": holm_correction,
    "bonferroni": bonferroni_correction,
    "fdr_bh": bh_correction,
    "none": _no_correction,
}


def available_corrections() -> list[str]:
    return list(_CORRECTIONS)


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
    ds: xr.Dataset,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    correction: str = "holm",
    allow_incomplete: bool = False,
) -> pd.DataFrame:
    """Pairwise comparison of methods for every metric in ``ds``.

    ``correction`` — one of :func:`available_corrections` (``"holm"`` default,
    ``"bonferroni"``, ``"fdr_bh"`` for Benjamini-Hochberg FDR, ``"none"`` for
    raw p-values) — is applied **per metric** across the method pairs; the
    family is one metric's pairs, not the whole battery, so scanning for
    "significant on any metric" across a large battery still inflates
    family-wise error. A method whose
    scores are constant across all seeds (for a metric) has no sampling
    distribution, so comparisons involving it are masked — ``p_value``,
    ``p_corrected`` and ``effect_size`` become ``NaN`` and ``significant`` is
    ``False`` — rather than reporting a duplicated-point p-value of ~0 and a
    meaningless ±huge effect size; Holm is then applied over the surviving pairs.
    Emits a warning when ``test`` cannot reach ``alpha`` at the dataset's seed
    count, and when a method is constant across seeds in *every* metric.

    ``allow_incomplete`` (default ``False``) — when ``True``, an incomplete sweep
    is compared anyway (with a warning) instead of raising, computing stats over
    only the completed cells. Use it for exploratory analysis of a ``sample=`` or
    budget-limited sweep; the result may be under-powered or biased.

    Raises
    ------
    IncompleteSweepError
        If ``ds.attrs["mushin_failures"]`` or ``ds.attrs["mushin_skipped"]`` is a
        non-empty list — the sweep that produced ``ds`` had failed runs (recorded
        under ``on_error="nan"``) or cells that were never run (skipped by a
        ``sample=`` subset or an exhausted ``max_total_seconds`` budget) — unless
        ``allow_incomplete=True``. A dataset without those attrs — a plain user
        dataset or a clean, fully-completed sweep — is unaffected. This is keyed
        purely on the completeness signals, never on raw NaN values in the data,
        so a metric that is legitimately NaN for other reasons does not trigger
        it.
    """
    # Refuse an incomplete sweep — failed cells (on_error="nan") AND cells that
    # were never run (skipped by a `sample=` subset or an exhausted
    # `max_total_seconds` budget). Both leave NaN in the grid, so computing stats
    # over them silently would under-power or bias the result. Keyed on the
    # completeness *signals* (attrs), never on raw NaN values, so a metric that is
    # legitimately NaN for other reasons does not trigger it.
    failures = _normalize_failures(ds.attrs.get("mushin_failures"))
    skipped = _normalize_failures(ds.attrs.get("mushin_skipped"))
    if failures or skipped:
        if not allow_incomplete:
            reasons = []
            if failures:
                reasons.append(
                    f"{len(failures)} run(s) failed ({', '.join(map(str, failures))})"
                )
            if skipped:
                reasons.append(
                    f"{len(skipped)} cell(s) not run/skipped "
                    f"({', '.join(map(str, skipped))})"
                )
            raise IncompleteSweepError(
                "; ".join(reasons) + ". Complete the sweep (fix failures and "
                "resume, or resume without `sample`/with more time) before "
                "comparing, or pass allow_incomplete=True to compute stats on the "
                "partial grid."
            )
        warnings.warn(
            f"comparing an incomplete sweep ({len(failures) + len(skipped)} "
            "cell(s) failed/skipped, NaN in the grid); statistics use only the "
            "completed cells and may be under-powered or biased.",
            UserWarning,
            stacklevel=2,
        )
    if test not in _TESTS:
        raise ValueError(f"unknown test {test!r}; choose from {available_tests()}")
    if correction not in _CORRECTIONS:
        raise ValueError(
            f"unknown correction {correction!r}; choose from {available_corrections()}"
        )
    correct = _CORRECTIONS[correction]
    n_seeds = int(ds.sizes["seed"])
    warn_if_underpowered(test, n_seeds, alpha)
    func, is_paired = _TESTS[test]
    methods = [str(m) for m in ds["method"].values]

    rows = []
    constant_metric_count: dict[str, int] = {}  # method -> #metrics it is constant in
    n_metrics = 0
    for metric in ds.data_vars:
        n_metrics += 1
        vals = {m: ds[metric].sel({"method": m}).values for m in methods}
        # A method with no within-group variance has no sampling distribution.
        # Constancy is judged over the COMPLETED (non-NaN) seeds, so the mask
        # holds on the allow_incomplete path too — NaN would defeat allclose
        # and let a deterministic method leak into a reported significance.
        completed = {
            m: (lambda f: f[~np.isnan(f)])(np.asarray(vals[m], dtype=float))
            for m in methods
        }
        constant = {
            m for m in methods if completed[m].size > 1 and _is_constant(completed[m])
        }
        for m in constant:
            constant_metric_count[m] = constant_metric_count.get(m, 0) + 1
        recs, pvals = [], []
        for a, b in itertools.combinations(methods, 2):
            va, vb = vals[a], vals[b]
            if allow_incomplete:
                # Compute over the seeds completed for BOTH methods: drop pairs
                # where either cell is missing (NaN). ±Inf is a real completed
                # value (e.g. diverged loss) and stays. No-op for a complete
                # grid (nothing is NaN), so the normal path is unchanged.
                _fa = np.asarray(va, dtype=float)
                _fb = np.asarray(vb, dtype=float)
                _ok = ~np.isnan(_fa) & ~np.isnan(_fb)
                va, vb = _fa[_ok], _fb[_ok]
                if va.size < 2:
                    # too few completed pairs to form a test
                    recs.append(
                        {
                            "metric": str(metric),
                            "method_a": a,
                            "method_b": b,
                            "mean_diff": float("nan"),
                            "effect_size": float("nan"),
                            "p_value": float("nan"),
                        }
                    )
                    pvals.append(float("nan"))
                    continue
            # Match the effect size to the test: paired tests get d_z over the
            # per-seed differences; unpaired tests get the pooled-variance d.
            eff = cohens_dz(va, vb) if is_paired else cohens_d(va, vb)
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
        corrected = correct(pvals) if len(pvals) > 1 else pvals
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


def paired_item_bootstrap(
    a_items,
    b_items,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired bootstrap over EVAL ITEMS -> ``(mean_diff, ci_low, ci_high, p_value)``.

    ``a_items``/``b_items`` are per-item scores for two systems on the *same* eval
    set (already reduced over seeds), so ``d_i = a_i - b_i`` is a paired
    per-item difference. Resampling items with replacement answers the question
    seed-based testing cannot: *would this difference survive a different sample
    of eval items?* — the standard paired bootstrap of Koehn (2004).

    This is complementary to, not a replacement for, the seed-based test:
    seeds capture decoding/judge noise, items capture eval-set uncertainty. The
    latter is usually far larger, so a seed-significant difference whose item
    interval straddles 0 is not a result you should report.

    ``p_value`` is the two-sided proportion of resamples whose mean difference
    falls on the opposite side of 0 from the observed one (doubled, capped at 1),
    with the standard +1 smoothing so it is never exactly 0.
    """
    a = np.asarray(a_items, dtype=float)
    b = np.asarray(b_items, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired bootstrap needs equal-length per-item scores, got "
            f"{a.shape} and {b.shape}"
        )
    d = a - b
    n = d.size
    observed = float(d.mean())
    if n < 2 or not np.isfinite(d).all():
        return observed, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = d[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided: how often does a resample land on the other side of 0?
    opposite = (means <= 0).sum() if observed > 0 else (means >= 0).sum()
    p = min(1.0, 2.0 * (opposite + 1) / (n_resamples + 1))
    return observed, float(low), float(high), float(p)
