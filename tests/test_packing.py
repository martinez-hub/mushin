# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import os

import pytest


def test_round_robin_mapping(monkeypatch):
    from mushin._packing import pin_gpu_round_robin

    for job_index, expected in [(0, "0"), (1, "1"), (4, "0"), (5, "1")]:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        gpu = pin_gpu_round_robin(num_gpus=4, job_index=job_index)
        assert gpu == int(expected)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


def test_num_gpus_must_be_positive():
    from mushin._packing import pin_gpu_round_robin

    with pytest.raises(ValueError, match="num_gpus"):
        pin_gpu_round_robin(num_gpus=0)
    with pytest.raises(ValueError, match="num_gpus"):
        pin_gpu_round_robin(num_gpus=-1)


def test_job_index_defaults_to_hydra_job_num(monkeypatch):
    from types import SimpleNamespace

    import hydra.core.hydra_config as hc

    from mushin._packing import pin_gpu_round_robin

    cfg = SimpleNamespace(hydra=SimpleNamespace(job=SimpleNamespace(num=3)))

    class _FakeHydraConfig:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            return cfg

    # the helper does `from hydra.core.hydra_config import HydraConfig` at call
    # time, so patching the attribute on the module is seen by the function.
    monkeypatch.setattr(hc, "HydraConfig", _FakeHydraConfig)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    gpu = pin_gpu_round_robin(num_gpus=2)  # 3 % 2 == 1
    assert gpu == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_no_active_hydra_raises(monkeypatch):
    import hydra.core.hydra_config as hc

    from mushin._packing import pin_gpu_round_robin

    class _NotInit:
        @staticmethod
        def initialized():
            return False

    monkeypatch.setattr(hc, "HydraConfig", _NotInit)
    with pytest.raises(RuntimeError, match="job_index"):
        pin_gpu_round_robin(num_gpus=2)


def test_warns_when_cuda_already_initialized(monkeypatch):
    import torch

    from mushin._packing import pin_gpu_round_robin

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.warns(UserWarning, match="already initialized"):
        gpu = pin_gpu_round_robin(num_gpus=2, job_index=1)
    assert gpu == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"  # still set
