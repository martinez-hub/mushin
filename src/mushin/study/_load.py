# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Shared core: load checkpoints, regroup by method, and compare."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Deferred: keep `import mushin` from pulling in the benchmark subsystem.
    # Safe here because `from __future__ import annotations` (above) means this
    # name is only ever needed at type-check time, never at runtime.
    from mushin.benchmark import BenchmarkResult


def evaluate_checkpoints(
    checkpoints: dict[str, Sequence[str]],
    load_fn: Callable[[str], Any],
    data,
    task,  # str | Task — resolved inside compare()
    num_classes: int | None = None,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    ignore_index: int | None = None,
) -> BenchmarkResult:
    """Load each checkpoint via ``load_fn``, regroup into ``{method: [models]}``,
    and run ``compare`` (which warns if the test is underpowered for the seed
    count)."""
    from mushin.benchmark import compare  # local import: see module-level note above

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
