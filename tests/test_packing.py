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
    import hydra.core.hydra_config as hc
    from omegaconf import OmegaConf

    from mushin._packing import pin_gpu_round_robin

    # HydraConfig.get() returns the HydraConf node itself, so the job is at the
    # top level (`.job.num`), not under a `.hydra` wrapper.
    cfg = OmegaConf.create({"job": {"num": 3}})

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


def test_single_run_missing_job_num_raises(monkeypatch):
    # single-run (plain @hydra.main): HydraConfig is initialized but job.num is
    # MISSING. Must raise the friendly RuntimeError, not an OmegaConf error.
    import hydra.core.hydra_config as hc
    from omegaconf import OmegaConf

    from mushin._packing import pin_gpu_round_robin

    cfg = OmegaConf.create({"job": {"num": "???"}})  # ??? == MISSING

    class _FakeHydraConfig:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            return cfg

    monkeypatch.setattr(hc, "HydraConfig", _FakeHydraConfig)
    with pytest.raises(RuntimeError, match="job_index"):
        pin_gpu_round_robin(num_gpus=2)


def test_raises_when_cuda_already_initialized(monkeypatch):
    # A reused worker whose first job already touched CUDA cannot be re-pinned;
    # fail loud instead of silently leaving the job on the previous GPU.
    import torch

    from mushin._packing import pin_gpu_round_robin

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match="already initialized"):
        pin_gpu_round_robin(num_gpus=2, job_index=1)
    # nothing was changed before the raise
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_indexes_into_existing_allocation(monkeypatch):
    # SLURM/containers may restrict the process to a device subset; the helper
    # must select from that pool, not overwrite it with a bare ordinal.
    from mushin._packing import pin_gpu_round_robin

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    slot = pin_gpu_round_robin(num_gpus=2, job_index=3)  # 3 % 2 == 1 -> "5"
    assert slot == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")  # restore the allocation
    slot = pin_gpu_round_robin(num_gpus=2, job_index=2)  # 2 % 2 == 0 -> "4"
    assert slot == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"


def test_num_gpus_exceeding_allocation_raises(monkeypatch):
    from mushin._packing import pin_gpu_round_robin

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    with pytest.raises(ValueError, match="exceeds"):
        pin_gpu_round_robin(num_gpus=4, job_index=0)


def test_pin_gpu_round_robin_exported():
    import mushin
    from mushin import pin_gpu_round_robin

    assert "pin_gpu_round_robin" in mushin.__all__
    assert callable(pin_gpu_round_robin)
