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


class _LazyModels(Sequence):
    """Sequence view over checkpoint paths that loads each model on access and
    keeps no reference afterward. ``compare`` consumes models strictly one at a
    time, so peak memory is one model — not method × seed models resident at
    once (a 10-method × 20-seed study would otherwise hold 200 models)."""

    def __init__(self, paths: Sequence[str], load_fn: Callable[[str], Any]):
        self._paths = list(paths)
        self._load_fn = load_fn

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, i):
        return self._load_fn(self._paths[i])


def evaluate_checkpoints(
    checkpoints: dict[str, Sequence[str]],
    load_fn: Callable[[str], Any],
    data,
    task,  # str | Task — resolved inside compare()
    num_classes: int | None = None,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    ignore_index: int | None = None,
    *,
    predict_fn: Callable | None = None,
    metrics: dict | None = None,
    prob_metrics: frozenset[str] | None = None,
    correction: str = "holm",
) -> BenchmarkResult:
    """Regroup checkpoints into ``{method: models}`` (loaded lazily, one at a
    time, via ``load_fn``) and run ``compare`` (which warns if the test is
    underpowered for the seed count). ``predict_fn``/``metrics``/
    ``prob_metrics``/``correction`` forward to :func:`mushin.benchmark.compare`."""
    from mushin.benchmark import compare  # local import: see module-level note above

    if not checkpoints:
        raise ValueError("`checkpoints` must not be empty")
    models = {
        method: _LazyModels(paths, load_fn)
        for method, paths in checkpoints.items()
    }
    return compare(
        methods=models,
        data=data,
        task=task,
        num_classes=num_classes,
        predict_fn=predict_fn,
        metrics=metrics,
        prob_metrics=prob_metrics,
        test=test,
        alpha=alpha,
        correction=correction,
        ignore_index=ignore_index,
    )
