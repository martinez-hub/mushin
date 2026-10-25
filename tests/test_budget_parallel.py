# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`max_total_seconds` is one deadline for the whole sweep, however many
processes run its cells. A parallel launcher (joblib, submitit) pickles the
task runner to its workers, once per job under joblib, so a deadline kept on
the runner restarted with every job and the budget never fired. The sweep's
first computed cell now records the start in a file in the sweep root, every
copy of the runner reads it, and the next run() (a resume) starts afresh.
"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pytest
from hydra_zen import zen
from omegaconf import OmegaConf

import mushin.workflows as wf_mod
from mushin import multirun
from mushin._resume import read_cell_status
from mushin.workflows import MultiRunMetricsWorkflow, _SkippedRun, _TaskRunner

# Each cell outlasts the budget, so a worker's second cell always starts after
# the deadline, however slow the machine: at most one cell per worker computes.
_CELL_SECONDS = 0.5
_BUDGET = 0.25


class _SleepyWF(MultiRunMetricsWorkflow):
    # Module-level, so the joblib (loky) workers can unpickle the task.
    @staticmethod
    def task(seed):
        import time

        started = time.time()
        time.sleep(_CELL_SECONDS)
        return dict(v=float(seed), started=started)


def _start_files(root):
    # The name the `run` docstring documents: `.mushin_budget_start.<id>`.
    return sorted(p.name for p in root.glob(".mushin_budget_start.*"))


def _computed(ds):
    return [
        int(s) for s in ds["seed"].values if not np.isnan(float(ds["v"].sel(seed=s)))
    ]


def test_budget_holds_across_joblib_workers(tmp_path):
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    n_jobs, seeds = 2, list(range(8))
    root = tmp_path / "s"

    wf = _SleepyWF()
    with pytest.warns(UserWarning, match="not run"):
        wf.run(
            seed=multirun(seeds),
            max_total_seconds=_BUDGET,
            working_dir=str(root),
            launcher="joblib",
            launcher_config={"n_jobs": n_jobs},
        )
    ds = wf.to_xarray()
    computed = _computed(ds)
    # One budget for the sweep, not one per job: before the shared start, every
    # job unpickled a runner with no deadline and all 8 cells ran.
    assert 1 <= len(computed) <= n_jobs, computed
    assert len(wf.skipped) == len(seeds) - len(computed)
    for cell in root.iterdir():
        if cell.is_dir() and int(cell.name) not in computed:
            assert read_cell_status(cell)["status"] == "skipped"
    # run() removed this run's start file once its launch returned.
    assert _start_files(root) == []

    # A resume without the budget finishes the skipped cells and reuses the
    # computed ones (their start time is the first run's, not recomputed).
    first_starts = {s: float(ds["started"].sel(seed=s)) for s in computed}
    wf2 = _SleepyWF()
    wf2.run(
        seed=multirun(seeds),
        resume=True,
        working_dir=str(root),
        launcher="joblib",
        launcher_config={"n_jobs": n_jobs},
    )
    assert wf2.is_complete
    ds2 = wf2.to_xarray()
    assert _computed(ds2) == seeds
    for s, started in first_starts.items():
        assert float(ds2["started"].sel(seed=s)) == started


def test_resume_with_a_budget_gets_a_fresh_budget(tmp_path, monkeypatch):
    """A start file a killed run left behind must not spend the next run's
    budget: the resume names its own start file, so its first computed cell
    starts a fresh clock."""
    calls = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            calls.append(seed)
            time.sleep(0.3)  # each cell outlasts the 0.2 s budget
            return dict(v=float(seed))

    root = tmp_path / "s"
    seeds = multirun([0, 1, 2, 3])
    # Simulate a driver killed before it could clean up: the start stays.
    with monkeypatch.context() as m:
        m.setattr(wf_mod, "_remove_budget_start", lambda *a, **k: None)
        with pytest.warns(UserWarning, match="not run"):
            W().run(seed=seeds, max_total_seconds=0.2, working_dir=str(root))
    assert calls == [0]
    stale = _start_files(root)
    assert len(stale) == 1  # the first run's start, now older than the budget

    calls.clear()
    wf = W()
    with pytest.warns(UserWarning, match="not run"):
        wf.run(seed=seeds, max_total_seconds=0.2, resume=True, working_dir=str(root))
    # seed 0 is reused; seed 1 starts the resume's own clock and runs; the
    # budget it starts is spent by the time seeds 2 and 3 come up.
    assert calls == [1]
    assert len(wf.skipped) == 2
    # The resume removed its own start file and left the stale one alone.
    assert _start_files(root) == stale


def _compute(seed):
    return dict(v=float(seed))


def _no_pre_task():
    pass


def _runner(max_total_seconds):
    return _TaskRunner(
        task=zen(_compute),
        pre_task=zen(_no_pre_task),
        swept_names=(),
        base_provenance={},
        on_error="raise",
        inject_resume=False,
        prior_manifest=None,
        max_total_seconds=max_total_seconds,
    )


def _cell(root, num, monkeypatch):
    cell = root / str(num)
    cell.mkdir(parents=True)
    monkeypatch.chdir(cell)  # a cell runs in its own dir, under the sweep root
    return OmegaConf.create({"seed": num})


def test_pickled_runner_copies_share_one_start(tmp_path, monkeypatch):
    """Two copies of one runner, as a parallel launcher ships to two workers,
    share the budget that the first of them to compute starts."""
    root = tmp_path / "s"
    runner = _runner(max_total_seconds=0.2)
    a, b = (pickle.loads(pickle.dumps(runner)) for _ in range(2))

    assert a(_cell(root, 0, monkeypatch)) == {"v": 0.0}  # starts the clock
    time.sleep(0.3)
    # `b` has computed nothing, yet the sweep's budget is spent.
    assert isinstance(b(_cell(root, 1, monkeypatch)), _SkippedRun)
    assert isinstance(a(_cell(root, 2, monkeypatch)), _SkippedRun)
    assert _start_files(root) == [runner.budget_start_path(root).name]

    # A new runner is a new run() (a resume): its budget starts afresh, even
    # though the spent run's start file is still in the root.
    assert _runner(max_total_seconds=0.2)(_cell(root, 3, monkeypatch)) == {"v": 3.0}


def test_an_empty_start_file_is_dated_by_its_mtime(tmp_path, monkeypatch):
    """A cell can read the start file between its creator's create and write
    (or after a failed write): the file's mtime stands in for the start."""
    root = tmp_path / "s"
    runner = _runner(max_total_seconds=60)
    cfg = _cell(root, 0, monkeypatch)
    path = runner.budget_start_path(root)
    path.touch()
    old = time.time() - 120  # two budgets ago
    os.utime(path, (old, old))
    assert isinstance(runner(cfg), _SkippedRun)


def test_unshareable_start_falls_back_to_this_process(tmp_path):
    """When the start file cannot be created the budget is timed from this
    process's first cell, with a warning that it may not hold across workers."""
    runner = _runner(max_total_seconds=60)
    missing = tmp_path / "no-such-root"
    with pytest.warns(UserWarning, match="may not hold across worker"):
        assert runner._budget_expired(missing) is False
    assert runner._budget_expired(missing) is False  # the deadline is 60 s out
