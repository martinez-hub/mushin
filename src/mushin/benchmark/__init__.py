# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard evaluation protocols: run a benchmark, get a labeled dataset back."""

from ._metrics import (
    classification_battery,
    detection_battery,
    image_quality_battery,
    regression_battery,
    retrieval_battery,
    segmentation_battery,
)
from ._result import BenchmarkResult
from ._tasks import Task, get_task, list_tasks, register_task
from .compare import compare

__all__ = [
    "compare",
    "BenchmarkResult",
    "Task",
    "register_task",
    "get_task",
    "list_tasks",
    "classification_battery",
    "segmentation_battery",
    "detection_battery",
    "regression_battery",
    "retrieval_battery",
    "image_quality_battery",
]
