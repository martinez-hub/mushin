"""Type aliases for the LLM-eval API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Union

from torchmetrics import Metric as TorchMetric

# A system maps a batch of inputs + a seed to a batch of outputs (same order).
System = Callable[[Sequence[Any], int], Sequence[Any]]

# A metric is a torchmetrics Metric, or a per-example scorer (output, reference)->float.
Metric = Union[TorchMetric, Callable[[Any, Any], float]]

# An example is a {"input": ..., "reference": ...} mapping, or a bare input (when
# the metric needs no reference). To attach a reference, use the mapping form — a
# bare tuple is treated as the input itself, not an (input, reference) pair, so a
# tuple-valued prompt is passed through intact.
Example = Any
