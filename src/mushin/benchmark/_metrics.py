# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Metric batteries (classification + segmentation), delegated to torchmetrics."""

from __future__ import annotations

import warnings
from collections.abc import Collection

import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)


def classification_battery(
    num_classes: int, ignore_index: int | None = None
) -> dict[str, Metric]:
    """The standard multiclass classification battery. ``ignore_index`` is
    accepted for a uniform task interface but is not applied here (the battery's
    AUROC/ECE do not support it)."""
    if ignore_index is not None:
        warnings.warn(
            "ignore_index is not applied to the classification battery "
            "(its AUROC/ECE do not support it); it is ignored.",
            UserWarning,
            stacklevel=2,
        )
    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "auroc": MulticlassAUROC(num_classes=num_classes),
        "ece": MulticlassCalibrationError(num_classes=num_classes),
    }


def segmentation_battery(
    num_classes: int, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Semantic-segmentation battery over per-pixel class labels. ``dice`` is the
    macro F1 (the Dice coefficient); all metrics are confusion-matrix based, so
    streaming evaluation uses O(C^2) memory."""
    return {
        "miou": MulticlassJaccardIndex(num_classes, ignore_index=ignore_index),
        "dice": MulticlassF1Score(
            num_classes, average="macro", ignore_index=ignore_index
        ),
        "pixel_acc": MulticlassAccuracy(
            num_classes, average="micro", ignore_index=ignore_index
        ),
        "precision": MulticlassPrecision(
            num_classes, average="macro", ignore_index=ignore_index
        ),
        "recall": MulticlassRecall(
            num_classes, average="macro", ignore_index=ignore_index
        ),
    }


def compute_battery(
    battery: dict[str, Metric],
    preds: torch.Tensor,
    targets: torch.Tensor,
    prob_metrics: Collection[str],
    probs: torch.Tensor | None = None,
) -> dict[str, float]:
    """One-shot metric computation: reset, update once, compute. Metrics named in
    ``prob_metrics`` are fed ``probs`` (required if non-empty); the rest ``preds``."""
    from ._inference import expand_metric_value

    out: dict[str, float] = {}
    for name, metric in battery.items():
        metric.reset()
        inp = probs if name in prob_metrics else preds
        out.update(expand_metric_value(name, metric(inp, targets)))
    return out
