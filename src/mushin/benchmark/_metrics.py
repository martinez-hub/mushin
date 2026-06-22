# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard classification metric battery, delegated to torchmetrics."""

from __future__ import annotations

import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

# Metrics that require probabilities rather than hard class predictions.
_PROB_METRICS = frozenset({"auroc", "ece"})


def classification_battery(num_classes: int) -> dict[str, Metric]:
    """The standard multiclass battery. F1/precision/recall use macro averaging;
    accuracy uses micro (overall) averaging."""
    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "auroc": MulticlassAUROC(num_classes=num_classes),
        "ece": MulticlassCalibrationError(num_classes=num_classes),
    }


def compute_metrics(
    preds: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor,
    battery: dict[str, Metric],
) -> dict[str, float]:
    """Compute each metric in ``battery``. Resets every metric first so a shared
    battery cannot accumulate state across evaluations."""
    out: dict[str, float] = {}
    for name, metric in battery.items():
        metric.reset()
        inp = probs if name in _PROB_METRICS else preds
        out[name] = float(metric(inp, targets))
    return out
