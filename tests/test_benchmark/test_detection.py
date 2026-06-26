"""Hermetic detection-battery tests (no real dataset; needs the detection extra)."""

import math

import pytest
import torch

pytest.importorskip("torchvision")
pytest.importorskip("pycocotools")

from mushin.benchmark import BenchmarkResult, compare  # noqa: E402
from mushin.benchmark._inference import evaluate  # noqa: E402
from mushin.benchmark._metrics import detection_battery  # noqa: E402
from mushin.benchmark._predict import default_detection_predict_fn  # noqa: E402


def _box(x0, y0, x1, y1):
    return torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])


class _FixedDetector(torch.nn.Module):
    """Ignores the input image and emits fixed predictions per batch."""

    def __init__(self, preds):
        super().__init__()
        self._preds = preds

    def forward(self, x):
        return self._preds


def test_perfect_predictions_score_one():
    preds = [
        {
            "boxes": _box(0, 0, 10, 10),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    tgts = [{"boxes": _box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    out = evaluate(
        _FixedDetector(preds),
        data,
        detection_battery(),
        default_detection_predict_fn,
        prob_metrics=frozenset(),
    )
    assert out["map"] == pytest.approx(1.0)
    assert out["iou"] == pytest.approx(1.0)
    assert out["giou"] == pytest.approx(1.0)


def test_disjoint_predictions_score_low():
    preds = [
        {
            "boxes": _box(100, 100, 110, 110),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    tgts = [{"boxes": _box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    out = evaluate(
        _FixedDetector(preds),
        data,
        detection_battery(),
        default_detection_predict_fn,
        prob_metrics=frozenset(),
    )
    assert out["map"] == pytest.approx(0.0, abs=1e-6)


def test_battery_matches_torchmetrics_reference():
    """Our streaming/expansion path reproduces torchmetrics' own numbers."""
    from torchmetrics.detection import MeanAveragePrecision

    preds = [
        {
            "boxes": _box(0, 0, 10, 10),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([1]),
        }
    ]
    tgts = [{"boxes": _box(1, 1, 11, 11), "labels": torch.tensor([1])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    out = evaluate(
        _FixedDetector(preds),
        data,
        detection_battery(),
        default_detection_predict_fn,
        prob_metrics=frozenset(),
    )

    ref = MeanAveragePrecision(box_format="xyxy")
    ref.update(preds, tgts)
    ref_out = ref.compute()
    for key in ("map", "map_50", "map_75", "mar_100"):
        ours = out[key]
        gold = float(ref_out[key])
        if gold == -1.0:
            assert math.isnan(ours)
        else:
            assert ours == pytest.approx(gold)


def test_size_bucket_sentinel_becomes_nan():
    """A 10x10 box (area 100) is 'small' under COCO, so the large bucket has no GT
    -> torchmetrics returns -1 -> we expose NaN; map_small keeps its real score."""
    preds = [
        {
            "boxes": _box(0, 0, 10, 10),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    tgts = [{"boxes": _box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    out = evaluate(
        _FixedDetector(preds),
        data,
        detection_battery(),
        default_detection_predict_fn,
        prob_metrics=frozenset(),
    )
    assert math.isnan(out["map_large"])  # empty bucket -> NaN, not -1.0
    assert out["map_small"] == pytest.approx(1.0)


def test_compare_detection_end_to_end():
    good = [
        {
            "boxes": _box(0, 0, 10, 10),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    bad = [
        {
            "boxes": _box(50, 50, 60, 60),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    tgts = [{"boxes": _box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    result = compare(
        {
            "good": [_FixedDetector(good), _FixedDetector(good)],
            "bad": [_FixedDetector(bad), _FixedDetector(bad)],
        },
        data,
        task="detection",
        test="welch",
    )
    assert isinstance(result, BenchmarkResult)
    for key in ("map", "map_50", "map_75", "mar_100", "iou", "giou", "ciou", "diou"):
        assert key in result.data.data_vars
    assert float(result.data["map"].sel({"method": "good"}).mean()) == pytest.approx(
        1.0
    )


@pytest.mark.real_data
def test_real_coco_sample_end_to_end():
    """Manual validation: a pretrained torchvision detector runs end-to-end through
    compare(task="detection") and yields a plausible mAP. Run with: pytest -m real_data."""
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).eval()

    img = torch.rand(3, 320, 320)
    tgts = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 200.0, 200.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    data = [([img], tgts)]

    result = compare({"frcnn": [model]}, data, task="detection", test="welch")
    m = float(result.data["map"].mean())
    assert -1.0 <= m <= 1.0  # ran end-to-end on a real detector without error
