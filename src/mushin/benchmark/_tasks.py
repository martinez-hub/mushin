# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Registry mapping a task name to its battery, predict_fn, and prob-metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torchmetrics import Metric

from ._inference import PredictFn
from ._metrics import classification_battery, segmentation_battery
from ._predict import (
    default_classification_predict_fn,
    default_segmentation_predict_fn,
)


@dataclass(frozen=True)
class TaskSpec:
    battery: Callable[..., dict[str, Metric]]
    predict_fn: PredictFn
    prob_metrics: frozenset[str]


_TASKS: dict[str, TaskSpec] = {
    "classification": TaskSpec(
        classification_battery,
        default_classification_predict_fn,
        frozenset({"auroc", "ece"}),
    ),
    "segmentation": TaskSpec(
        segmentation_battery,
        default_segmentation_predict_fn,
        frozenset(),
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    if task not in _TASKS:
        raise NotImplementedError(
            f"task={task!r} is not supported; choose from {sorted(_TASKS)}"
        )
    return _TASKS[task]
