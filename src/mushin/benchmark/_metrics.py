# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Metric batteries (classification, segmentation, detection, regression, image
quality, audio, retrieval), delegated to torchmetrics."""

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
from torchmetrics.retrieval import (
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
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
    the uniform task interface but unused. This is a SINGLE-TARGET battery:
    predictions and targets are continuous tensors of shape ``(N,)`` or ``(N, 1)``.
    Multi-output targets ``(N, D>1)`` are not supported here — ``pearson``/
    ``spearman`` are built with ``num_outputs=1`` and raise on ``D>1``; use a custom
    Task with ``num_outputs=D`` for multi-output regression."""
    return {
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError(),
        "rmse": MeanSquaredError(squared=False),
        "r2": R2Score(),
        "pearson": PearsonCorrCoef(),
        "spearman": SpearmanCorrCoef(),
    }


def retrieval_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Information-retrieval battery over grouped (per-query) predictions.
    ``num_classes``/``ignore_index`` are accepted for the uniform interface but
    unused. Batches must yield ``y = (relevance, indexes)``; see
    ``retrieval_update``. ``relevance`` must be BINARY (0/1) for ``retrieval_map``/
    ``mrr``/``precision``/``recall`` (they raise on graded values); only ``ndcg``
    accepts graded relevance. For graded judgments, use a custom Task with just the
    graded-capable metrics (e.g. ``RetrievalNormalizedDCG``)."""
    return {
        "retrieval_map": RetrievalMAP(),
        "ndcg": RetrievalNormalizedDCG(),
        "mrr": RetrievalMRR(),
        "precision": RetrievalPrecision(),
        "recall": RetrievalRecall(),
    }


def retrieval_update(battery, preds, probs, target):
    """update_fn for the retrieval task: ``target`` is a ``(relevance, indexes)``
    tuple, and every retrieval metric takes ``(preds, relevance, indexes=...)``."""
    relevance, indexes = target
    for metric in battery.values():
        metric.update(preds, relevance, indexes=indexes)


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


def image_quality_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Paired image-quality battery (generated vs reference image). Requires the
    optional ``image`` extra (LPIPS pulls in torchvision + lpips). ``num_classes``/
    ``ignore_index`` are accepted for the uniform interface but unused. Images are
    ``(N, C, H, W)``; ``data_range=1.0`` assumes inputs in ``[0, 1]`` and
    ``LearnedPerceptualImagePatchSimilarity(normalize=True)`` accepts that range.
    Note ``ms_ssim`` requires ``H, W > 160`` (torchmetrics default 5 scales / kernel
    11); for smaller images drop ``ms_ssim`` or pass a custom
    ``MultiScaleStructuralSimilarityIndexMeasure`` with fewer ``betas``."""
    try:
        from torchmetrics.image import (
            MultiScaleStructuralSimilarityIndexMeasure,
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )

        # LPIPS is imported from its submodule, not the top-level ``torchmetrics.image``
        # namespace: torchmetrics only re-exports it at top level when its deps
        # (torchvision + lpips) are present, so a clean env without the ``image``
        # extra would AttributeError rather than reach the construction-time check.
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        return {
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
            "psnr": PeakSignalNoiseRatio(data_range=1.0),
            "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0),
            "lpips": LearnedPerceptualImagePatchSimilarity(normalize=True),
        }
    except ImportError as e:
        raise ImportError(
            "the image_quality battery requires the optional image extra; install "
            "it with `pip install mushin-py[image]` (torchvision + lpips)."
        ) from e


def audio_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Speech/audio battery (estimated vs reference waveform). Requires the optional
    ``audio`` extra (PESQ needs ``pesq``; STOI needs ``pystoi``). SI-SDR/SI-SNR are
    core, but the battery is all-or-nothing. ``num_classes``/``ignore_index`` are
    accepted for the uniform interface but unused. Waveforms are ``(N, T)``; PESQ/
    STOI assume a 16 kHz sample rate (override via a custom Task for other rates)."""
    try:
        from torchmetrics.audio import (
            ScaleInvariantSignalDistortionRatio,
            ScaleInvariantSignalNoiseRatio,
        )
        from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
        from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

        return {
            "si_sdr": ScaleInvariantSignalDistortionRatio(),
            "si_snr": ScaleInvariantSignalNoiseRatio(),
            "pesq": PerceptualEvaluationSpeechQuality(fs=16000, mode="wb"),
            "stoi": ShortTimeObjectiveIntelligibility(fs=16000),
        }
    except ImportError as e:
        raise ImportError(
            "the audio battery requires the optional audio extra; install it with "
            "`pip install mushin-py[audio]` (pesq + pystoi)."
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
