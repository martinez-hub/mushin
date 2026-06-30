# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest
import torch


def test_image_quality_missing_extra_raises(monkeypatch):
    # Force the optional LPIPS construction to fail and assert the clear missing-
    # extra error, regardless of whether the extra is installed. Patch the submodule
    # the battery imports from (torchmetrics.image.lpip) — the top-level
    # torchmetrics.image namespace lacks the attribute when lpips is absent.
    import torchmetrics.image.lpip as tmi_lpip

    class _Boom:
        def __init__(self, *a, **k):
            raise ImportError("simulated missing lpips")

    monkeypatch.setattr(tmi_lpip, "LearnedPerceptualImagePatchSimilarity", _Boom)

    from mushin.benchmark._metrics import image_quality_battery

    with pytest.raises(ImportError, match=r"mushin-py\[image\]"):
        image_quality_battery()


def test_image_quality_battery_end_to_end():
    pytest.importorskip("torchvision")
    pytest.importorskip("lpips")
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    g = torch.Generator().manual_seed(0)
    # MS-SSIM needs images large enough for its 5 scales; 256x256 is safe. Use a
    # near (not exact) reconstruction so PSNR stays finite (identical -> inf).
    ref = torch.rand(2, 3, 256, 256, generator=g)
    gen = (ref + 0.01 * torch.randn(2, 3, 256, 256, generator=g)).clamp(0, 1)
    loader = DataLoader(TensorDataset(gen, ref), batch_size=2)

    class _Recon(torch.nn.Module):
        def forward(self, x):
            return x  # returns the "generated" image; target is the reference

    result = compare(
        methods={"m": [_Recon() for _ in range(3)]},
        data=loader,
        task="image_quality",
    )
    assert isinstance(result, BenchmarkResult)
    for name in ["ssim", "psnr", "ms_ssim", "lpips"]:
        assert name in result.data

    import math

    def val(name):
        return float(result.data[name].sel(method="m").values.ravel()[0])

    # All finite (a near, not exact, reconstruction keeps psnr finite), and the
    # near-perfect reconstruction scores high on similarity / low on lpips.
    for name in ["ssim", "psnr", "ms_ssim", "lpips"]:
        assert math.isfinite(val(name))
    assert val("ssim") > 0.9
    assert val("ms_ssim") > 0.9
    assert val("lpips") < 0.05
