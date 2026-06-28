# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Shared core: load checkpoints, regroup by method, and compare."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mushin.benchmark import BenchmarkResult, compare


def evaluate_checkpoints(
    checkpoints: dict[str, Sequence[str]],
    load_fn: Callable[[str], Any],
    data,
    task,  # str | Task — resolved inside compare()
    num_classes: int,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    ignore_index: int | None = None,
) -> BenchmarkResult:
    """Load each checkpoint via ``load_fn``, regroup into ``{method: [models]}``,
    and run ``compare`` (which warns if the test is underpowered for the seed
    count)."""
    if not checkpoints:
        raise ValueError("`checkpoints` must not be empty")
    models = {
        method: [load_fn(p) for p in paths] for method, paths in checkpoints.items()
    }
    return compare(
        methods=models,
        data=data,
        task=task,
        num_classes=num_classes,
        test=test,
        alpha=alpha,
        ignore_index=ignore_index,
    )
