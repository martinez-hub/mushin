# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Registry mapping a task name to its battery, predict_fn, and prob-metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torchmetrics import Metric

from ._inference import PredictFn
from ._metrics import classification_battery, detection_battery, segmentation_battery
from ._predict import (
    default_classification_predict_fn,
    default_detection_predict_fn,
    default_segmentation_predict_fn,
)


@dataclass(frozen=True)
class Task:
    """A reusable evaluation task: a metric ``battery`` factory, a ``predict_fn``
    that extracts ``(predictions, probabilities)`` from a model, the subset of
    metric names that consume probabilities, and whether the battery needs
    ``num_classes``. ``description`` is shown by :func:`list_tasks`."""

    battery: Callable[..., dict[str, Metric]]
    predict_fn: PredictFn
    prob_metrics: frozenset[str] = frozenset()
    requires_num_classes: bool = True
    description: str = ""


# Backward-compat alias (deprecated; removed in a future release).
TaskSpec = Task


_TASKS: dict[str, Task] = {
    "classification": Task(
        classification_battery,
        default_classification_predict_fn,
        frozenset({"auroc", "ece"}),
        description="Multiclass classification (accuracy, f1, precision, "
        "recall, auroc, ece).",
    ),
    "segmentation": Task(
        segmentation_battery,
        default_segmentation_predict_fn,
        frozenset(),
        description="Semantic segmentation (miou, dice, pixel_acc, precision, recall).",
    ),
    "detection": Task(
        detection_battery,
        default_detection_predict_fn,
        frozenset(),
        requires_num_classes=False,
        description="Object detection (mAP/mAR family + IoU variants).",
    ),
}


def register_task(name: str, task: Task, *, overwrite: bool = False) -> None:
    """Register ``task`` under ``name`` so ``compare(task=name)`` and
    ``Study(task=name)`` can look it up. Set ``overwrite=True`` to replace an
    existing entry."""
    if not isinstance(name, str) or not name:
        raise ValueError("`name` must be a non-empty string")
    if not isinstance(task, Task):
        raise TypeError(f"`task` must be a Task, got {type(task).__name__}")
    if name in _TASKS and not overwrite:
        raise ValueError(
            f"task {name!r} is already registered; pass overwrite=True to replace it"
        )
    _TASKS[name] = task


def get_task(task: str) -> Task:
    """Look up a registered task by name."""
    if task not in _TASKS:
        raise NotImplementedError(
            f"task={task!r} is not supported; choose from {sorted(_TASKS)}"
        )
    return _TASKS[task]


def list_tasks() -> dict[str, str]:
    """Return ``{name: description}`` for every registered task, name-sorted."""
    return {name: _TASKS[name].description for name in sorted(_TASKS)}


# Backward-compat alias (deprecated; use get_task).
get_task_spec = get_task
