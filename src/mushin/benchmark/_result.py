# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The object returned by ``compare``: dataset + comparisons + summary table."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import xarray as xr

from ._stats import confidence_interval


@dataclass
class BenchmarkResult:
    """Holds benchmark results.

    Attributes
    ----------
    data : xarray.Dataset
        Dims ``(method, seed)``, one data variable per metric.
    comparisons : pandas.DataFrame
        Pairwise significance results (see ``_stats.compare_methods``).
    alpha : float
        Significance level used.
    """

    data: xr.Dataset
    comparisons: pd.DataFrame
    alpha: float = 0.05

    def summary(self, reference: str | None = None) -> pd.DataFrame:
        """Publication-ready table: per method/metric ``mean`` and CI, with a
        ``"*"`` marker when the method differs significantly from ``reference``
        (default: the first method in ``data``)."""
        methods = [str(m) for m in self.data["method"].values]
        ref = reference if reference is not None else methods[0]

        rows = []
        for method in methods:
            for metric in self.data.data_vars:
                vals = self.data[metric].sel({"method": method}).values
                mean, lo, hi = confidence_interval(vals, self.alpha)
                marker = ""
                if method != ref:
                    match = self.comparisons[
                        (self.comparisons["metric"] == str(metric))
                        & (
                            (
                                (self.comparisons["method_a"] == method)
                                & (self.comparisons["method_b"] == ref)
                            )
                            | (
                                (self.comparisons["method_a"] == ref)
                                & (self.comparisons["method_b"] == method)
                            )
                        )
                    ]
                    if len(match) and bool(match.iloc[0]["significant"]):
                        marker = "*"
                rows.append(
                    {
                        "method": method,
                        "metric": str(metric),
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "significant_vs_ref": marker,
                    }
                )
        return pd.DataFrame(rows)
