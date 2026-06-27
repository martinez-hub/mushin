# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Tests for mushin._utils.load_experiment."""

import pytest
import torch
from omegaconf import OmegaConf

from mushin._utils import Experiment, load_experiment


def _write_hydra_cfg(hydra_dir, lr=0.1):
    """Write a minimal .hydra/config.yaml using OmegaConf (alias of load_from_yaml)."""
    hydra_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({"lr": lr})
    OmegaConf.save(cfg, hydra_dir / "config.yaml")


def _write_metrics(job_dir, acc=0.9):
    """Write a metrics.pt file into job_dir."""
    torch.save({"acc": [acc]}, job_dir / "metrics.pt")


# ---------------------------------------------------------------------------
# BUG 2: single-run layout — working_dir should be the per-job dir, not parent
# ---------------------------------------------------------------------------
class TestSingleRunLayout:
    def test_cfg_loaded(self, tmp_path):
        """cfg must not be None for a plain single-run experiment."""
        hydra_dir = tmp_path / ".hydra"
        _write_hydra_cfg(hydra_dir, lr=0.1)
        _write_metrics(tmp_path)

        result = load_experiment(tmp_path)

        assert isinstance(result, Experiment)
        assert result.cfg is not None, "cfg should not be None for single-run layout"

    def test_cfg_values(self, tmp_path):
        """cfg.lr must equal the value written into config.yaml."""
        hydra_dir = tmp_path / ".hydra"
        _write_hydra_cfg(hydra_dir, lr=0.1)
        _write_metrics(tmp_path)

        result = load_experiment(tmp_path)

        assert float(result.cfg.lr) == pytest.approx(0.1)

    def test_metrics_loaded(self, tmp_path):
        """metrics dict must contain the saved 'metrics' key."""
        hydra_dir = tmp_path / ".hydra"
        _write_hydra_cfg(hydra_dir, lr=0.1)
        _write_metrics(tmp_path, acc=0.9)

        result = load_experiment(tmp_path)

        assert "metrics" in result.metrics

    def test_working_dir_is_job_dir(self, tmp_path):
        """working_dir must be the per-job directory, NOT its parent."""
        hydra_dir = tmp_path / ".hydra"
        _write_hydra_cfg(hydra_dir, lr=0.1)
        _write_metrics(tmp_path)

        result = load_experiment(tmp_path)

        # BUG 2: current code does path.parent.parent (one level too high)
        assert result.working_dir == str(tmp_path), (
            f"working_dir should be {tmp_path}, got {result.working_dir}"
        )


# ---------------------------------------------------------------------------
# BUG 1 (P1): DDP layout — second config must NOT cause cfg to become None
# ---------------------------------------------------------------------------
class TestDDPLayout:
    def test_cfg_not_none_with_extra_rank_config(self, tmp_path):
        """DDP rank configs must not shadow the canonical .hydra/config.yaml."""
        hydra_dir = tmp_path / ".hydra"
        _write_hydra_cfg(hydra_dir, lr=0.2)
        _write_metrics(tmp_path)

        # Simulate the extra config written by mushin's HydraDDP launcher
        rank1_dir = tmp_path / ".pl_hydra_rank_1"
        _write_hydra_cfg(rank1_dir, lr=0.2)  # same lr, different path

        result = load_experiment(tmp_path)

        # BUG 1: current code globs **/config.yaml, finds 2 files, == 1 fails → None
        assert isinstance(result, Experiment)
        assert result.cfg is not None, (
            "cfg must not be None when a DDP rank config exists alongside .hydra/config.yaml"
        )

    def test_cfg_loaded_from_hydra_dir(self, tmp_path):
        """The cfg loaded must be from .hydra/config.yaml (canonical source)."""
        hydra_dir = tmp_path / ".hydra"
        _write_hydra_cfg(hydra_dir, lr=0.42)
        _write_metrics(tmp_path)

        # DDP rank config with a DIFFERENT lr to distinguish which was loaded
        rank1_dir = tmp_path / ".pl_hydra_rank_1"
        _write_hydra_cfg(rank1_dir, lr=0.99)

        result = load_experiment(tmp_path)

        assert result.cfg is not None
        assert float(result.cfg.lr) == pytest.approx(0.42), (
            "Should load from .hydra/config.yaml (lr=0.42), not rank config (lr=0.99)"
        )

    def test_working_dir_is_job_dir_ddp(self, tmp_path):
        """working_dir must equal the job dir even in DDP layout."""
        hydra_dir = tmp_path / ".hydra"
        _write_hydra_cfg(hydra_dir, lr=0.1)
        _write_metrics(tmp_path)
        rank1_dir = tmp_path / ".pl_hydra_rank_1"
        _write_hydra_cfg(rank1_dir, lr=0.1)

        result = load_experiment(tmp_path)

        assert result.working_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# BUG 2: multirun layout — each job working_dir must be its own subdirectory
# ---------------------------------------------------------------------------
class TestMultirunLayout:
    def test_returns_list_of_two_experiments(self, tmp_path):
        """load_experiment on a multirun root must return a list of Experiments."""
        for i, lr in enumerate([0.01, 0.001]):
            job_dir = tmp_path / str(i)
            job_dir.mkdir()
            _write_hydra_cfg(job_dir / ".hydra", lr=lr)
            _write_metrics(job_dir)

        result = load_experiment(tmp_path)

        assert isinstance(result, list), "multirun should return a list"
        assert len(result) == 2

    def test_each_experiment_has_cfg(self, tmp_path):
        """Each Experiment in the multirun list must have cfg loaded."""
        lrs = [0.01, 0.001]
        for i, lr in enumerate(lrs):
            job_dir = tmp_path / str(i)
            job_dir.mkdir()
            _write_hydra_cfg(job_dir / ".hydra", lr=lr)
            _write_metrics(job_dir)

        results = load_experiment(tmp_path)

        for exp in results:
            assert exp.cfg is not None, "Each experiment must have its cfg loaded"

    def test_each_experiment_has_correct_lr(self, tmp_path):
        """Each Experiment must carry the lr from its own config.yaml."""
        lrs = [0.01, 0.001]
        for i, lr in enumerate(lrs):
            job_dir = tmp_path / str(i)
            job_dir.mkdir()
            _write_hydra_cfg(job_dir / ".hydra", lr=lr)
            _write_metrics(job_dir)

        results = load_experiment(tmp_path)
        # Sort by lr to make the assertion order-independent
        results_sorted = sorted(results, key=lambda e: float(e.cfg.lr))

        assert float(results_sorted[0].cfg.lr) == pytest.approx(0.001)
        assert float(results_sorted[1].cfg.lr) == pytest.approx(0.01)

    def test_working_dirs_are_per_job(self, tmp_path):
        """working_dir for each Experiment must point to its own job dir.

        BUG 2: current code uses path.parent.parent which collapses every job's
        working_dir to the shared multirun root instead of its own subdirectory.
        """
        for i, lr in enumerate([0.01, 0.001]):
            job_dir = tmp_path / str(i)
            job_dir.mkdir()
            _write_hydra_cfg(job_dir / ".hydra", lr=lr)
            _write_metrics(job_dir)

        results = load_experiment(tmp_path)
        working_dirs = {exp.working_dir for exp in results}

        expected = {str(tmp_path / "0"), str(tmp_path / "1")}
        assert working_dirs == expected, (
            f"working_dirs should be per-job {expected}, got {working_dirs}"
        )
