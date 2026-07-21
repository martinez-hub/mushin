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
    methods : dict[str, Sequence[torch.nn.Module]]
        ``{method_name: [model_per_seed, ...]}``; every model is evaluated on
        ``data``, giving one score per (method, seed).
    data : Iterable
        A re-iterable batch source (e.g. a ``DataLoader``); it is streamed once
        per model.
    task : str or Task
        A registered task name (``"classification"``, ``"segmentation"``,
        ``"detection"``, ``"regression"``, ``"retrieval"``, ``"image_quality"``,
        ``"audio"``, or a custom one — see ``list_tasks()``) or a ``Task`` object.
    num_classes : int or None
        Required (when ``metrics`` is not provided) only for tasks whose battery
        needs it — ``"classification"`` and ``"segmentation"``; ignored for the
        others.
    predict_fn : callable or None
        ``(model, x) -> (preds, probs)``; defaults to the task's.
    metrics : dict[str, torchmetrics.Metric] or None
        Replaces the task's battery entirely. A custom battery gets no implicit
        probability routing — name the probability-consuming entries in
        ``prob_metrics``.
    prob_metrics : frozenset[str] or None
        Battery entries fed probabilities instead of hard predictions. ``None``
        means the task's own set when the task's battery is used, and the empty
        set when a custom ``metrics=`` battery is given. Every name must be a
        key of the battery in use.
    test : str
        Statistical test for pairwise method comparison; one of
        ``mushin.benchmark.available_tests()`` (default paired Wilcoxon).
    alpha : float
        Significance level for the (Holm-corrected) comparisons.
    ignore_index : int or None
        Label to exclude from segmentation metrics (e.g. a void/boundary class).
    device : torch.device or None
        Evaluation device; defaults to the device of each model's parameters.
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
    if prob_metrics is None:
        # The task's prob routing applies only to the task's own battery. A
        # custom `metrics=` battery gets NO implicit routing: silently feeding
        # argmax hard predictions to a custom probability metric (or probs to a
        # custom hard metric that shares a task metric's name) computes a wrong
        # value with no error. Pass prob_metrics= explicitly for custom batteries.
        pm = spec.prob_metrics if metrics is None else frozenset()
    else:
        pm = frozenset(prob_metrics)
    missing = pm - battery.keys()
    if missing:
        raise ValueError(
            f"prob_metrics {sorted(missing)} are not in the metric battery "
            f"{sorted(battery)}; prob_metrics must name entries of the battery "
            "being used."
        )

    results: dict[str, list[dict[str, float]]] = {}
    for name, models in methods.items():
        # spec.update_fn is forwarded even when metrics= overrides the battery, so
        # a task's update_fn must stay compatible with the substituted battery.
        results[name] = [
            evaluate(model, data, battery, fn, pm, device, update_fn=spec.update_fn)
            for model in models
        ]

    ds = to_dataset(results)
    comparisons = compare_methods(ds, test=test, alpha=alpha)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
