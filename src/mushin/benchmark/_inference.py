# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Run a model over a dataloader and collect predictions and targets."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from ._predict import default_classification_predict_fn

PredictFn = Callable[[torch.nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def run_inference(
    model: torch.nn.Module,
    data: Iterable,
    predict_fn: PredictFn | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate ``model`` over ``data`` (yielding ``(x, y)``) and return
    ``(preds, probs, targets)`` concatenated across all batches (on CPU)."""
    if predict_fn is None:
        predict_fn = default_classification_predict_fn
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")

    model = model.to(device)
    model.eval()

    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            preds, probs = predict_fn(model, x)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

    return torch.cat(all_preds), torch.cat(all_probs), torch.cat(all_targets)
