# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Run a model over a dataloader and collect predictions and targets."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from typing import Optional

import torch
from torchmetrics import Metric

PredictFn = Callable[[torch.nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

# Owns all metric.update() calls for one batch: (battery, preds, probs, target).
# `Optional[...]` (not `... | None`) because this alias is evaluated at runtime,
# and `X | None` is a TypeError under Python 3.9.
UpdateFn = Callable[
    [dict[str, Metric], torch.Tensor, Optional[torch.Tensor], object], None
]


def _to_device(obj, device: torch.device):
    """Recursively move tensors to ``device`` through tensors, lists/tuples, and
    dicts; anything else passes through. Lets one streaming loop serve tensor tasks
    and detection's ``list[dict]`` batches alike."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(_to_device(v, device) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _as_float(v) -> float:
    """Coerce one scalar metric value to a float. (The COCO ``-1.0`` 'not
    applicable' sentinel is normalized to ``NaN`` upstream, inside the detection
    mAP battery — not here — so it is not applied to metrics like the IoU variants
    whose range legitimately includes ``-1``.)"""
    if isinstance(v, torch.Tensor) and v.numel() != 1:
        raise TypeError(
            f"metric produced a non-scalar value of shape {tuple(v.shape)}; "
            "battery metrics must return scalar values per key"
        )
    return float(v)


def expand_metric_value(name: str, value) -> dict[str, float]:
    """Flatten one metric's ``compute()`` output into ``{data_var: float}``. A dict
    expands to one entry per key (using the metric's own key names); a scalar is
    kept under ``name``."""
    if isinstance(value, dict):
        return {k: _as_float(v) for k, v in value.items()}
    return {name: _as_float(value)}


def accumulate_metric(out: dict[str, float], name: str, value) -> None:
    """Expand ``value`` into ``out``, raising on a data-variable name collision (two
    metrics producing the same key — e.g. a scalar ``score`` metric alongside one
    returning ``{"score": ...}``) rather than silently overwriting the earlier one."""
    scored = expand_metric_value(name, value)
    clash = out.keys() & scored.keys()
    if clash:
        raise ValueError(
            f"metric battery produced colliding data-variable name(s) "
            f"{sorted(clash)}; a dict-valued metric expands to its own keys — "
            "rename the battery entry to avoid the clash."
        )
    out.update(scored)


def evaluate(
    model: torch.nn.Module,
    data: Iterable,
    battery: dict[str, Metric],
    predict_fn: PredictFn,
    prob_metrics: Collection[str],
    device: torch.device | None = None,
    update_fn: UpdateFn | None = None,
) -> dict[str, float]:
    """Stream ``data`` through ``model``, updating each metric in ``battery`` per
    batch, and return ``{name: value}``. Metrics named in ``prob_metrics`` are fed
    probabilities; the rest hard predictions. O(C^2) memory for confusion-matrix
    metrics. If ``update_fn`` is given it owns every ``metric.update(...)`` call
    for a batch (receiving ``(battery, preds, probs, target)``); when ``None`` the
    default ``(probs|preds, target)`` dispatch is used."""
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")

    model = model.to(device)
    model.eval()
    for metric in battery.values():
        metric.reset()
        metric.to(device)

    if update_fn is None:

        def update_fn(battery, preds, probs, target):
            for name, metric in battery.items():
                metric.update(probs if name in prob_metrics else preds, target)

    with torch.no_grad():
        for x, y in data:
            x = _to_device(x, device)
            y = _to_device(y, device)
            preds, probs = predict_fn(model, x)
            update_fn(battery, preds, probs, y)

    out: dict[str, float] = {}
    for name, metric in battery.items():
        accumulate_metric(out, name, metric.compute())
    return out
