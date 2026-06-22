# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Default predict step for classification models."""

from __future__ import annotations

import torch


def default_classification_predict_fn(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a classification model on ``x`` and return ``(preds, probs)``.

    Assumes ``model(x)`` returns class logits of shape ``(N, num_classes)``.
    ``probs`` is the softmax over the last dim; ``preds`` is its argmax.
    """
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return preds, probs
