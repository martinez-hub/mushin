# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Run a model over a dataloader and collect predictions and targets."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable

import torch
from torchmetrics import Metric

PredictFn = Callable[[torch.nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


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
            x = x.to(device)
            y = y.to(device)
            preds, probs = predict_fn(model, x)
            for name, metric in battery.items():
                metric.update(probs if name in prob_metrics else preds, y)

    return {name: float(metric.compute()) for name, metric in battery.items()}
