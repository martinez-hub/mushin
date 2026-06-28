# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest
import torch


def test_audio_missing_extra_raises(monkeypatch):
    import torchmetrics.audio.pesq as tma_pesq

    class _Boom:
        def __init__(self, *a, **k):
            raise ImportError("simulated missing pesq")

    monkeypatch.setattr(tma_pesq, "PerceptualEvaluationSpeechQuality", _Boom)

    from mushin.benchmark._metrics import audio_battery

    with pytest.raises(ImportError, match=r"mushin-py\[audio\]"):
        audio_battery()


def test_audio_battery_end_to_end():
    pytest.importorskip("pesq")
    pytest.importorskip("pystoi")
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    g = torch.Generator().manual_seed(0)
    # PESQ wideband expects 16 kHz; give ~1 s of audio so it is long enough. Use a
    # near (not exact) reconstruction so SI-SDR/SI-SNR stay finite (identical -> inf).
    ref = torch.randn(2, 16000, generator=g)
    est = ref + 0.01 * torch.randn(2, 16000, generator=g)
    loader = DataLoader(TensorDataset(est, ref), batch_size=2)

    class _Enh(torch.nn.Module):
        def forward(self, x):
            return x

    result = compare(
        methods={"m": [_Enh() for _ in range(3)]},
        data=loader,
        task="audio",
    )
    assert isinstance(result, BenchmarkResult)
    for name in ["si_sdr", "si_snr", "pesq", "stoi"]:
        assert name in result.data

    import math

    def val(name):
        return float(result.data[name].sel(method="m").values.ravel()[0])

    # All finite (a near, not exact, reconstruction keeps si_sdr/si_snr finite), and
    # the near-perfect estimate scores a high SI-SDR/SI-SNR.
    for name in ["si_sdr", "si_snr", "pesq", "stoi"]:
        assert math.isfinite(val(name))
    assert val("si_sdr") > 20.0
    assert val("si_snr") > 20.0
