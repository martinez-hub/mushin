# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Tests for the runnable per-battery toys in ``examples/batteries.py``.

Each test calls the corresponding ``run_<task>()`` toy and asserts the returned
object is a ``BenchmarkResult`` carrying the expected metric keys. Toys whose
battery needs an optional extra (detection, image_quality, audio) are guarded
with ``pytest.importorskip`` so they skip cleanly when the extra is absent.

``pythonpath = ["examples"]`` (see pyproject ``[tool.pytest.ini_options]``) makes
``import batteries`` resolve.
"""

import batteries
import pytest

# The metric batteries are part of the optional 'eval' extra.
pytest.importorskip("torchmetrics")

from mushin.benchmark import BenchmarkResult  # noqa: E402


def test_run_walkthrough():
    result = batteries.run_walkthrough()
    assert isinstance(result, BenchmarkResult)
    for name in ["accuracy", "f1", "precision", "recall", "auroc", "ece"]:
        assert name in result.data.data_vars
    # Lock the "reliably significant" property the doc's pasted output depends on:
    # at least one metric must be a real (non-NaN p-value) significant verdict.
    comps = result.comparisons
    sig = comps[comps["significant"]]
    assert not sig.empty, "walkthrough produced no significant comparison"
    assert sig["p_value"].notna().any(), "significant rows have NaN p_value"


@pytest.mark.parametrize(
    ("toy", "metrics", "extras"),
    [
        pytest.param(
            "run_classification",
            ["accuracy", "f1", "precision", "recall", "auroc", "ece"],
            [],
            id="classification",
        ),
        pytest.param(
            "run_segmentation",
            ["miou", "dice", "pixel_acc", "precision", "recall"],
            [],
            id="segmentation",
        ),
        pytest.param(
            "run_detection",
            ["map", "map_50", "map_75", "mar_100", "iou", "giou", "ciou", "diou"],
            ["torchvision", "pycocotools"],
            id="detection",
        ),
        pytest.param(
            "run_regression",
            ["mse", "mae", "rmse", "r2", "pearson", "spearman"],
            [],
            id="regression",
        ),
        pytest.param(
            "run_retrieval",
            ["retrieval_map", "ndcg", "mrr", "precision", "recall"],
            [],
            id="retrieval",
        ),
        pytest.param(
            "run_image_quality",
            ["ssim", "psnr", "ms_ssim", "lpips"],
            ["torchvision", "lpips"],
            id="image_quality",
        ),
        pytest.param("run_audio", ["si_sdr", "si_snr", "stoi"], ["pystoi"], id="audio"),
    ],
)
def test_run_battery_toy(toy, metrics, extras):
    # In the body, not at module level: a module-level importorskip would skip
    # every toy when one optional extra is absent.
    for module in extras:
        pytest.importorskip(module)
    result = getattr(batteries, toy)()
    assert isinstance(result, BenchmarkResult)
    for name in metrics:
        assert name in result.data.data_vars
