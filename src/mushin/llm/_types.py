"""Type aliases for the LLM-eval API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torchmetrics import Metric as TorchMetric

System = Callable[[Sequence[Any], int], Sequence[Any]]
"""A system under evaluation: a callable ``(inputs, seed) -> outputs`` that maps
a batch of inputs and an integer seed to a batch of outputs in the same order."""

Metric = TorchMetric | Callable[[Any, Any], float]
"""A metric: a ``torchmetrics.Metric``, or a per-example scorer
``(output, reference) -> float``."""

Example = Any
"""One evaluation example: a ``{"input": ..., "reference": ...}`` mapping, or a
bare input when the metric needs no reference. To attach a reference, use the
mapping form: a bare tuple is treated as the input itself, not as an
``(input, reference)`` pair, so a tuple-valued prompt passes through intact."""
