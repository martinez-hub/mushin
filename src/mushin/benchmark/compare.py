# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The ``compare`` facade: evaluate methods on a task battery and report significance."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import torch

from ._aggregate import to_dataset
from ._inference import PredictFn, evaluate
from ._result import BenchmarkResult
from ._stats import compare_methods
from ._tasks import Task, get_task


def compare(
    methods: dict[str, Sequence[torch.nn.Module]],
    data: Iterable,
    task: str | Task = "classification",
    *,
    num_classes: int | None = None,
    predict_fn: PredictFn | None = None,
    metrics: dict | None = None,
    prob_metrics: frozenset[str] | None = None,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    ignore_index: int | None = None,
    device: torch.device | None = None,
) -> BenchmarkResult:
    """Compare methods on a standard battery and report significance.

    Parameters
    ----------
    task : str
        ``"classification"``, ``"segmentation"``, or ``"detection"``.
    num_classes : int or None
        Required when ``metrics`` is not provided (not for ``"detection"``).
    ignore_index : int or None
        Label to exclude from segmentation metrics (e.g. a void/boundary class).
    prob_metrics : frozenset[str] or None
        Metrics whose names need probabilities; defaults to the task's set.
    """
    spec = task if isinstance(task, Task) else get_task(task)

    if isinstance(data, Iterator):
        raise TypeError(
            "`data` must be re-iterable (e.g. a DataLoader), not a one-shot "
            "iterator/generator — it is iterated once per model."
        )

    if metrics is not None:
        battery = metrics
    elif spec.requires_num_classes and num_classes is None:
        raise ValueError("`num_classes` is required when `metrics` is not provided")
    else:
        battery = spec.battery(num_classes, ignore_index=ignore_index)

    fn = predict_fn or spec.predict_fn
    pm = spec.prob_metrics if prob_metrics is None else prob_metrics

    results: dict[str, list[dict[str, float]]] = {}
    for name, models in methods.items():
        results[name] = [
            evaluate(model, data, battery, fn, pm, device) for model in models
        ]

    ds = to_dataset(results)
    comparisons = compare_methods(ds, test=test, alpha=alpha)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
