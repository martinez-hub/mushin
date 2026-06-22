# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Stack per-(method, seed) metric dicts into a labeled xarray Dataset."""

from __future__ import annotations

import numpy as np
import xarray as xr


class _MethodAwareDataArray(xr.DataArray):
    """DataArray subclass that treats 'method' kwarg in sel() as a coordinate selector.

    xarray's DataArray.sel() has a built-in ``method`` parameter for interpolation,
    which shadows any coordinate named ``method``. This subclass routes
    ``sel(method=<value>)`` through the indexers dict when a ``method`` dimension
    is present so that label-based selection works as expected.
    """

    __slots__ = ()

    def sel(self, indexers=None, **kwargs):  # type: ignore[override]
        if "method" in self.dims and "method" in kwargs:
            method_val = kwargs.pop("method")
            if indexers is None:
                indexers = {"method": method_val}
            else:
                indexers = dict(indexers)
                indexers["method"] = method_val
        result = super().sel(indexers, **kwargs)
        if isinstance(result, xr.DataArray) and not isinstance(
            result, _MethodAwareDataArray
        ):
            result = _MethodAwareDataArray(result)
        return result


class _BenchmarkDataset(xr.Dataset):
    """Dataset subclass that returns _MethodAwareDataArray on item access."""

    __slots__ = ()

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, xr.DataArray) and not isinstance(
            result, _MethodAwareDataArray
        ):
            result = _MethodAwareDataArray(result)
        return result


def to_dataset(results: dict[str, list[dict[str, float]]]) -> xr.Dataset:
    """``results`` maps method name -> list (over seeds) of metric dicts.

    Returns a Dataset with dims ``(method, seed)`` and one data variable per
    metric. All methods must have the same seeds and metric keys."""
    methods = list(results)
    if not methods:
        raise ValueError("`results` is empty")

    n_seeds = len(results[methods[0]])
    metric_names = list(results[methods[0]][0])

    data_vars = {}
    for metric in metric_names:
        arr = np.array(
            [[results[m][s][metric] for s in range(n_seeds)] for m in methods],
            dtype=float,
        )
        data_vars[metric] = (("method", "seed"), arr)

    return _BenchmarkDataset(
        data_vars,
        coords={"method": methods, "seed": np.arange(n_seeds)},
    )
