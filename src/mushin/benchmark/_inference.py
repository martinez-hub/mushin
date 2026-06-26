# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Run a model over a dataloader and collect predictions and targets."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable

import torch
from torchmetrics import Metric

PredictFn = Callable[[torch.nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _to_device(obj, device: torch.device):
    """Recursively move tensors to ``device`` through tensors, lists/tuples, and
    dicts; anything else passes through. Lets one streaming loop serve tensor tasks
    and detection's ``list[dict]`` batches alike."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _as_float(v) -> float:
    """Coerce one metric value to a float. The COCO ``-1.0`` 'not applicable'
    sentinel (a bucket with no matching ground truth) becomes ``NaN`` so the
    significance machinery treats it as missing rather than a real score."""
    if isinstance(v, torch.Tensor) and v.numel() != 1:
        raise TypeError(
            f"metric produced a non-scalar value of shape {tuple(v.shape)}; "
            "battery metrics must return scalar values per key"
        )
    f = float(v)
    return float("nan") if f == -1.0 else f


def expand_metric_value(name: str, value) -> dict[str, float]:
    """Flatten one metric's ``compute()`` output into ``{data_var: float}``. A dict
    expands to one entry per key (using the metric's own key names); a scalar is
    kept under ``name``."""
    if isinstance(value, dict):
        return {k: _as_float(v) for k, v in value.items()}
    return {name: _as_float(value)}


def evaluate(
    model: torch.nn.Module,
    data: Iterable,
    battery: dict[str, Metric],
    predict_fn: PredictFn,
    prob_metrics: Collection[str],
    device: torch.device | None = None,
) -> dict[str, float]:
    """Stream ``data`` through ``model``, updating each metric in ``battery`` per
    batch, and return ``{name: value}``. Metrics named in ``prob_metrics`` are fed
    probabilities; the rest hard predictions. O(C^2) memory for confusion-matrix
    metrics."""
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")

    model = model.to(device)
    model.eval()
    for metric in battery.values():
        metric.reset()
        metric.to(device)

    with torch.no_grad():
        for x, y in data:
            x = _to_device(x, device)
            y = _to_device(y, device)
            preds, probs = predict_fn(model, x)
            for name, metric in battery.items():
                metric.update(probs if name in prob_metrics else preds, y)

    out: dict[str, float] = {}
    for name, metric in battery.items():
        out.update(expand_metric_value(name, metric.compute()))
    return out
