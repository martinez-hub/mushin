# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Tests for mushin._utils.load_experiment."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import mushin
from mushin._utils import Experiment, load_experiment, original_cwd


def _write_hydra_cfg(hydra_dir, lr=0.1):
    """Write a minimal .hydra/config.yaml using OmegaConf (alias of load_from_yaml)."""
    hydra_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({"lr": lr})
    OmegaConf.save(cfg, hydra_dir / "config.yaml")


def _write_metrics(job_dir, acc=0.9):
    """Write a metrics.pt file into job_dir."""
    torch.save({"acc": [acc]}, job_dir / "metrics.pt")


# BUG 2: a single run's working_dir is its own dir, not the parent.
def test_single_run_layout(tmp_path):
    """A plain single run: the Experiment carries its cfg, its metrics, and
    the run's own directory."""
    _write_hydra_cfg(tmp_path / ".hydra", lr=0.1)
    _write_metrics(tmp_path, acc=0.9)

    result = load_experiment(tmp_path)

    assert isinstance(result, Experiment)
    assert float(result.cfg.lr) == pytest.approx(0.1)
    assert "metrics" in result.metrics
    assert result.working_dir == str(tmp_path)


# BUG 1 (P1): a DDP rank config beside .hydra/config.yaml made cfg None.
def test_ddp_layout_reads_the_job_config_not_a_rank_config(tmp_path):
    """HydraDDP writes a rank config beside the job's own .hydra/config.yaml;
    the job's config is the one loaded, and working_dir is the job dir."""
    _write_hydra_cfg(tmp_path / ".hydra", lr=0.42)
    _write_metrics(tmp_path)
    # A DIFFERENT lr in the rank config tells which one was loaded.
    _write_hydra_cfg(tmp_path / ".pl_hydra_rank_1", lr=0.99)

    result = load_experiment(tmp_path)

    assert isinstance(result, Experiment)
    assert float(result.cfg.lr) == pytest.approx(0.42)
    assert result.working_dir == str(tmp_path)


# BUG 2: each multirun job's working_dir is its own subdirectory.
def test_multirun_layout(tmp_path):
    """A multirun root gives one Experiment per job, each with the lr from its
    own config and its own directory (not the shared root)."""
    for i, lr in enumerate([0.01, 0.001]):
        job_dir = tmp_path / str(i)
        job_dir.mkdir()
        _write_hydra_cfg(job_dir / ".hydra", lr=lr)
        _write_metrics(job_dir)

    results = load_experiment(tmp_path)

    assert isinstance(results, list) and len(results) == 2
    assert {e.working_dir: float(e.cfg.lr) for e in results} == pytest.approx(
        {str(tmp_path / "0"): 0.01, str(tmp_path / "1"): 0.001}
    )


def test_original_cwd_outside_hydra_returns_process_cwd():
    # No active Hydra run -> falls back to the process working directory.
    assert original_cwd() == Path.cwd()


def test_original_cwd_is_exported():
    assert mushin.original_cwd is original_cwd


def test_original_cwd_uses_hydra_when_available(monkeypatch):
    import mushin._utils as utils

    monkeypatch.setattr(utils, "_hydra_original_cwd", lambda: "/tmp/launch-dir")
    assert original_cwd() == Path("/tmp/launch-dir")


def test_load_from_checkpoint_missing_weights_key_names_available(tmp_path):
    """A wrong weights_key must raise a real error listing the available keys,
    not a bare AssertionError (stripped entirely under python -O)."""
    import torch

    from mushin._utils import load_from_checkpoint

    ckpt = tmp_path / "m.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(1)}}, ckpt)
    model = torch.nn.Linear(1, 1, bias=False)
    with pytest.raises(KeyError, match="wrong_key.*state_dict"):
        load_from_checkpoint(model, ckpt=ckpt, weights_key="wrong_key")


def test_load_from_checkpoint_missing_model_attr_raises(tmp_path):
    import torch

    from mushin._utils import load_from_checkpoint

    ckpt = tmp_path / "m.ckpt"
    torch.save({"w": torch.zeros(1, 1)}, ckpt)
    model = torch.nn.Linear(1, 1, bias=False)
    with pytest.raises(AttributeError, match="no_such_submodule"):
        load_from_checkpoint(model, ckpt=ckpt, model_attr="no_such_submodule")


def test_load_experiment_missing_path_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        load_experiment(tmp_path / "nope")
