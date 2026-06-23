# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The ``compare`` facade: evaluate methods on a task battery and report significance."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from ._aggregate import to_dataset
from ._inference import PredictFn, evaluate
from ._result import BenchmarkResult
from ._stats import compare_methods
from ._tasks import get_task_spec


def compare(
    methods: dict[str, Sequence[torch.nn.Module]],
    data: Iterable,
    task: str = "classification",
    *,
    num_classes: int | None = None,
    predict_fn: PredictFn | None = None,
    metrics: dict | None = None,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    ignore_index: int | None = None,
    device: torch.device | None = None,
) -> BenchmarkResult:
    """Compare methods on a standard battery and report significance.

    Parameters
    ----------
    task : str
        ``"classification"`` or ``"segmentation"``.
    num_classes : int or None
        Required when ``metrics`` is not provided.
    ignore_index : int or None
        Label to exclude from segmentation metrics (e.g. a void/boundary class).
    """
    spec = get_task_spec(task)

    if metrics is not None:
        battery = metrics
    else:
        if num_classes is None:
            raise ValueError("`num_classes` is required when `metrics` is not provided")
        battery = spec.battery(num_classes, ignore_index=ignore_index)

    fn = predict_fn or spec.predict_fn

    results: dict[str, list[dict[str, float]]] = {}
    for name, models in methods.items():
        results[name] = [
            evaluate(model, data, battery, fn, spec.prob_metrics, device)
            for model in models
        ]

    ds = to_dataset(results)
    comparisons = compare_methods(ds, test=test, alpha=alpha)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
