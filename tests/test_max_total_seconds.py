# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`max_total_seconds` is a graceful wall-clock budget: once it is exhausted the
remaining cells are skipped (marked 'skipped', NaN in the dataset, no compute)
rather than the sweep running to the end. The clock starts at the first computed
cell, so at least one cell always runs and cache hits don't consume the budget.
Skipped cells are not 'completed', so a later resume with more time finishes them.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def test_budget_skips_remaining_cells(tmp_path):
    calls = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            calls.append(seed)
            if seed == 0:
                time.sleep(0.3)  # first cell blows the tiny budget
            return dict(v=float(seed))

    wf = W()
    with pytest.warns(UserWarning, match="skip"):
        wf.run(
            seed=multirun([0, 1, 2, 3]),
            max_total_seconds=0.1,
            working_dir=str(tmp_path / "s"),
        )

    assert calls == [0]  # only the first cell ran; the rest were skipped
    assert wf.is_complete is False
    assert len(wf.skipped) == 3

    ds = wf.to_xarray()
    assert float(ds["v"].sel(seed=0)) == 0.0
    assert bool(np.isnan(float(ds["v"].sel(seed=1))))


def test_generous_budget_completes_all_cells(tmp_path):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(v=float(seed))

    wf = W()
    wf.run(
        seed=multirun([0, 1, 2]),
        max_total_seconds=60,
        working_dir=str(tmp_path / "s"),
    )
    assert wf.is_complete
    assert wf.skipped == []


def test_nonpositive_budget_rejected(tmp_path):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(v=float(seed))

    with pytest.raises(ValueError, match="max_total_seconds"):
        W().run(
            seed=multirun([0, 1]),
            max_total_seconds=0,
            working_dir=str(tmp_path / "s"),
        )


def test_skipped_cells_are_resumable(tmp_path):
    calls = []

    class W(MultiRunMetricsWorkflow):
        SLOW = True

        @staticmethod
        def task(seed):
            calls.append(seed)
            if seed == 0 and W.SLOW:
                time.sleep(0.3)
            return dict(v=float(seed))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="skip"):
        W().run(
            seed=multirun([0, 1, 2, 3]),
            max_total_seconds=0.1,
            working_dir=wd,
        )

    W.SLOW = False
    calls.clear()
    wf2 = W()
    wf2.run(seed=multirun([0, 1, 2, 3]), resume=True, working_dir=wd)  # no budget

    assert set(calls) == {1, 2, 3}  # only the skipped cells re-ran; seed 0 reused
    assert wf2.is_complete


def test_budget_immune_to_wall_clock_jumps(tmp_path, monkeypatch):
    """The budget must run on a monotonic clock: a wall-clock step (NTP/DST)
    mid-sweep must not extend it. Freezing time.time simulates the clock never
    advancing — the budget must still expire and skip the remaining cells."""
    import time as _time

    frozen = _time.time()
    monkeypatch.setattr(_time, "time", lambda: frozen)

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            import time

            time.sleep(0.05)
            return dict(m=float(a))

    wf = W()
    with pytest.warns(UserWarning, match="skipped"):
        wf.run(
            a=multirun([1, 2, 3]),
            working_dir=str(tmp_path / "s"),
            max_total_seconds=0.01,
        )
    assert len(wf.skipped) >= 1


def test_budget_disabled_under_multi_rank_launch(tmp_path, monkeypatch):
    """Under an external multi-rank launch (submitit DDP: every rank runs the
    task with its own clock), a per-process budget could expire on one rank but
    not its siblings — the skipped rank would leave the others hanging at NCCL
    rendezvous. The budget must be disabled with a warning instead."""
    import time as _time

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    # a real external rank has both the world size and a per-rank marker
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            _time.sleep(0.05)
            return dict(m=float(a))

    wf = W()
    with pytest.warns(UserWarning, match="multi-rank"):
        wf.run(
            a=multirun([1, 2]),
            working_dir=str(tmp_path / "s"),
            max_total_seconds=0.01,  # would expire after cell 1 if enforced
        )
    assert wf.skipped == []  # every cell ran; no rank-divergence hazard
    assert wf.is_complete


def test_multi_rank_detection_branches(monkeypatch):
    """The multi-rank signal must be a real per-rank launch, not just an
    allocation-wide variable: SLURM_NTASKS>1 needs SLURM_PROCID (srun sets it
    per task; a sequential driver inside an salloc shell has no PROCID), and
    WORLD_SIZE>1 needs RANK/SLURM_PROCID (mushin's own single-node launcher
    exports WORLD_SIZE but never a per-rank var)."""
    from mushin.workflows import _TaskRunner

    for var in ("WORLD_SIZE", "RANK", "SLURM_NTASKS", "SLURM_PROCID"):
        monkeypatch.delenv(var, raising=False)
    assert not _TaskRunner._multi_rank_world()

    # SLURM positive: a submitit/srun-launched rank
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "1")
    assert _TaskRunner._multi_rank_world()

    # SLURM negative: sequential driver inside a multi-task allocation
    monkeypatch.delenv("SLURM_PROCID")
    assert not _TaskRunner._multi_rank_world()
    monkeypatch.delenv("SLURM_NTASKS")

    # torchrun positive: WORLD_SIZE + RANK
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    assert _TaskRunner._multi_rank_world()

    # negative: WORLD_SIZE leaked without a per-rank marker
    monkeypatch.delenv("RANK")
    assert not _TaskRunner._multi_rank_world()
