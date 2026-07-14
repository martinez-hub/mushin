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

from mushin.benchmark import BenchmarkResult


def test_run_classification():
    result = batteries.run_classification()
    assert isinstance(result, BenchmarkResult)
    for name in ["accuracy", "f1", "precision", "recall", "auroc", "ece"]:
        assert name in result.data.data_vars


def test_run_segmentation():
    result = batteries.run_segmentation()
    assert isinstance(result, BenchmarkResult)
    for name in ["miou", "dice", "pixel_acc", "precision", "recall"]:
        assert name in result.data.data_vars


def test_run_detection():
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    result = batteries.run_detection()
    assert isinstance(result, BenchmarkResult)
    for name in ["map", "map_50", "map_75", "mar_100", "iou", "giou", "ciou", "diou"]:
        assert name in result.data.data_vars


def test_run_regression():
    result = batteries.run_regression()
    assert isinstance(result, BenchmarkResult)
    for name in ["mse", "mae", "rmse", "r2", "pearson", "spearman"]:
        assert name in result.data.data_vars


def test_run_retrieval():
    result = batteries.run_retrieval()
    assert isinstance(result, BenchmarkResult)
    for name in ["retrieval_map", "ndcg", "mrr", "precision", "recall"]:
        assert name in result.data.data_vars


def test_run_image_quality():
    pytest.importorskip("torchvision")
    pytest.importorskip("lpips")
    result = batteries.run_image_quality()
    assert isinstance(result, BenchmarkResult)
    for name in ["ssim", "psnr", "ms_ssim", "lpips"]:
        assert name in result.data.data_vars


def test_run_audio():
    pytest.importorskip("pystoi")
    result = batteries.run_audio()
    assert isinstance(result, BenchmarkResult)
    for name in ["si_sdr", "si_snr", "stoi"]:
        assert name in result.data.data_vars
