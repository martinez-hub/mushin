# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Datasets of perfectly separated methods, and a warning catcher, shared by the
statistics tests and the docs-pinning tests so both judge the same seed floors.
Importable as ``_separated`` because pytest puts ``tests/`` (the directory of
``conftest.py``) on ``sys.path``."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import xarray as xr


def separated(
    n_methods: int, n_seeds: int, metrics: Sequence[str] = ("score",)
) -> xr.Dataset:
    """Methods separated perfectly. Each method's scores vary across seeds (a
    method constant across seeds gets NaN), and each pair's differences are
    distinct and nonzero (tied differences make the Wilcoxon p approximate)."""
    scores = np.array([(np.arange(n_seeds) + 1.0) * (i + 1) for i in range(n_methods)])
    return xr.Dataset(
        # One buffer per metric, so masking a method in one leaves the others.
        {m: (("method", "seed"), scores.copy()) for m in metrics},
        coords={
            "method": [f"m{i}" for i in range(n_methods)],
            "seed": np.arange(n_seeds),
        },
    )


def underpowered_warnings(fn: Callable, *args, **kwargs) -> list[str]:
    """The "cannot reach alpha" warnings one call of ``fn`` emits."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args, **kwargs)
    return [str(c.message) for c in caught if "cannot reach alpha" in str(c.message)]
