# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Shared core: load checkpoints, regroup by method, and compare."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mushin.benchmark import BenchmarkResult, compare
from mushin.benchmark._stats import _TESTS


def warn_if_underpowered(test: str, n_seeds: int, alpha: float) -> None:
    """Warn if ``test`` cannot reach ``alpha`` at ``n_seeds`` seeds.

    Determined empirically: run the test on maximally-separated samples of size
    ``n_seeds`` and check whether the best-case p-value clears ``alpha``."""
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
            stacklevel=2,
        )


def evaluate_checkpoints(
    checkpoints: dict[str, Sequence[str]],
    load_fn: Callable[[str], Any],
    data,
    task: str,
    num_classes: int,
    test: str = "wilcoxon",
    alpha: float = 0.05,
) -> BenchmarkResult:
    """Load each checkpoint via ``load_fn``, regroup into ``{method: [models]}``,
    and run ``compare``."""
    if not checkpoints:
        raise ValueError("`checkpoints` must not be empty")
    models = {
        method: [load_fn(p) for p in paths] for method, paths in checkpoints.items()
    }
    n_seeds = len(next(iter(checkpoints.values())))
    warn_if_underpowered(test, n_seeds, alpha)
    return compare(
        methods=models,
        data=data,
        task=task,
        num_classes=num_classes,
        test=test,
        alpha=alpha,
    )
