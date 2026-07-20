# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard evaluation protocols: run a benchmark, get a labeled dataset back."""

# The evaluation layer (compare, batteries, LLM eval, Study) is an optional extra
# so the core sweep -> dataset install stays lean. Surface a clear install hint
# instead of a raw torchmetrics/scipy ImportError when the extra is missing.
try:
    import scipy as _scipy  # noqa: F401
    import torchmetrics as _torchmetrics  # noqa: F401
except ImportError as _exc:  # pragma: no cover - exercised by the core-only CI job
    raise ImportError(
        "mushin's evaluation features (compare, the metric batteries, LLM "
        "evaluation, and Study) require the optional 'eval' extra:\n\n"
        "    pip install mushin-py[eval]\n"
    ) from _exc

from ._metrics import (
    audio_battery,
    classification_battery,
    detection_battery,
    image_quality_battery,
    regression_battery,
    retrieval_battery,
    segmentation_battery,
)
from ._result import BenchmarkResult
from ._stats import IncompleteSweepError, compare_methods
from ._tasks import Task, get_task, list_tasks, register_task
from .compare import compare

__all__ = [
    "compare",
    "compare_methods",
    "IncompleteSweepError",
    "BenchmarkResult",
    "Task",
    "register_task",
    "get_task",
    "list_tasks",
    "audio_battery",
    "classification_battery",
    "segmentation_battery",
    "detection_battery",
    "regression_battery",
    "retrieval_battery",
    "image_quality_battery",
]
