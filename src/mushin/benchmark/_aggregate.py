# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Stack per-(method, seed) metric dicts into a labeled xarray Dataset."""

from __future__ import annotations

import numpy as np
import xarray as xr


def to_dataset(results: dict[str, list[dict[str, float]]]) -> xr.Dataset:
    """``results`` maps method name -> list (over seeds) of metric dicts.

    Returns a Dataset with dims ``(method, seed)`` and one data variable per
    metric. All methods must have the same seeds and metric keys.

    Note: the ``method`` dimension shadows xarray's reserved ``.sel(method=...)``
    keyword, so select along it with the dict form: ``ds.sel({"method": name})``.
    """
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

    return xr.Dataset(
        data_vars,
        coords={"method": methods, "seed": np.arange(n_seeds)},
    )
