# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The ``compare`` facade: run a benchmark across methods and seeds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from ._aggregate import to_dataset
from ._inference import PredictFn, run_inference
from ._metrics import classification_battery, compute_metrics
from ._result import BenchmarkResult
from ._stats import compare_methods


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
    device: torch.device | None = None,
) -> BenchmarkResult:
    """Compare methods on a standard benchmark and report significance.

    Parameters
    ----------
    methods : dict[str, Sequence[Module]]
        Method name -> one trained model per seed.
    data : Iterable
        A dataloader yielding ``(x, y)`` batches.
    task : str
        Only ``"classification"`` is supported in this version.
    num_classes : int or None
        Number of classes (keyword-only). Required when ``metrics`` is not
        provided; ignored otherwise.
    test : str
        Significance test key (default ``"wilcoxon"``). See
        ``mushin.benchmark._stats.available_tests``.

    Returns
    -------
    BenchmarkResult
    """
    if task != "classification":
        raise NotImplementedError(
            f"task={task!r} is not supported; only 'classification' in this version"
        )

    if metrics is not None:
        battery = metrics
    else:
        if num_classes is None:
            raise ValueError("`num_classes` is required when `metrics` is not provided")
        battery = classification_battery(num_classes)

    results: dict[str, list[dict[str, float]]] = {}
    for name, models in methods.items():
        per_seed = []
        for model in models:
            preds, probs, targets = run_inference(model, data, predict_fn, device)
            per_seed.append(compute_metrics(preds, probs, targets, battery))
        results[name] = per_seed

    ds = to_dataset(results)
    comparisons = compare_methods(ds, test=test, alpha=alpha)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
