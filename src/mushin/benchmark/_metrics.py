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
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
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


def regression_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Scalar-regression battery. ``num_classes``/``ignore_index`` are accepted for
    the uniform task interface but unused. Predictions and targets are continuous
    tensors of matching shape (e.g. ``(N,)``)."""
    return {
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError(),
        "rmse": MeanSquaredError(squared=False),
        "r2": R2Score(),
        "pearson": PearsonCorrCoef(),
        "spearman": SpearmanCorrCoef(),
    }


_MAP_DROP = frozenset({"classes", "map_per_class", "mar_100_per_class"})


def detection_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """The bounding-box detection battery: mean-average-precision plus the IoU
    variants, every scalar output surfaced as its own metric. ``num_classes`` and
    ``ignore_index`` are accepted for the uniform task interface but unused (mAP
    infers classes from the labels). Requires the optional ``detection`` extra."""
    try:
        from torchmetrics.detection import (
            CompleteIntersectionOverUnion,
            DistanceIntersectionOverUnion,
            GeneralizedIntersectionOverUnion,
            IntersectionOverUnion,
            MeanAveragePrecision,
        )

        class _DetectionMAP(MeanAveragePrecision):
            """``MeanAveragePrecision`` minus its non-scalar bookkeeping keys
            (``classes``/``*_per_class``), which are not single comparable scores.

            Also normalizes the COCO ``-1.0`` 'not applicable' sentinel (a size
            bucket with no matching ground truth) to ``NaN`` so it is excluded from
            significance. Scoped to mAP/mAR here on purpose: the IoU-variant metrics
            legitimately range into ``-1`` and must not be sentinel-converted."""

            def compute(self):
                out = {}
                for k, v in super().compute().items():
                    if k in _MAP_DROP:
                        continue
                    if (
                        isinstance(v, torch.Tensor)
                        and v.numel() == 1
                        and v.item() == -1.0
                    ):
                        v = torch.tensor(float("nan"))
                    out[k] = v
                return out

        # Build the metrics inside the try: torchvision may be importable while
        # pycocotools is not, in which case MeanAveragePrecision raises a
        # ModuleNotFoundError at *construction* — caught here and reported as the
        # same clear missing-extra error rather than leaking from torchmetrics.
        return {
            "map": _DetectionMAP(box_format="xyxy"),
            "iou": IntersectionOverUnion(),
            "giou": GeneralizedIntersectionOverUnion(),
            "ciou": CompleteIntersectionOverUnion(),
            "diou": DistanceIntersectionOverUnion(),
        }
    except ImportError as e:
        raise ImportError(
            "the detection battery requires the optional detection extra; install "
            "it with `pip install mushin-py[detection]` (torchvision + pycocotools)."
        ) from e


def compute_battery(
    battery: dict[str, Metric],
    preds: torch.Tensor,
    targets: torch.Tensor,
    prob_metrics: Collection[str],
    probs: torch.Tensor | None = None,
) -> dict[str, float]:
    """One-shot metric computation: reset, update once, compute. Metrics named in
    ``prob_metrics`` are fed ``probs`` (required if non-empty); the rest ``preds``."""
    from ._inference import accumulate_metric

    out: dict[str, float] = {}
    for name, metric in battery.items():
        metric.reset()
        inp = probs if name in prob_metrics else preds
        accumulate_metric(out, name, metric(inp, targets))
    return out
