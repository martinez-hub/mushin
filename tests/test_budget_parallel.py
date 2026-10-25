# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`max_total_seconds` is one deadline for the whole sweep, however many
processes run its cells. A parallel launcher (joblib, submitit) pickles the
task runner to its workers, once per job under joblib, so a deadline kept on
the runner restarted with every job and the budget never fired. An unpickled
copy now takes the sweep's start from a file in the sweep root, which the
first computed cell creates, and keeps the deadline per process; the runner
the sequential launcher calls directly keeps it in memory and writes no file.
The next run() (a resume) starts afresh.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from hydra_zen import zen
from omegaconf import OmegaConf

import mushin.workflows as wf_mod
from mushin import multirun
from mushin._resume import read_cell_status, write_cell_status
from mushin.workflows import MultiRunMetricsWorkflow, _SkippedRun, _TaskRunner

# Each cell outlasts the budget, so a worker's second cell always starts after
# the deadline, however slow the machine: at most one cell per worker computes.
_CELL_SECONDS = 0.5
_BUDGET = 0.25
_N_JOBS = 2
_JOBLIB = {"launcher": "joblib", "launcher_config": {"n_jobs": _N_JOBS}}


class _SleepyWF(MultiRunMetricsWorkflow):
    # Module-level, so the joblib (loky) workers can unpickle the task.
    @staticmethod
    def task(seed):
        import time

        started = time.time()
        time.sleep(_CELL_SECONDS)
        return dict(v=float(seed), started=started)


class _FailsOnceWF(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed, marker):
        import time
        from pathlib import Path

        if seed == 1 and not Path(marker).exists():
            Path(marker).touch()
            raise RuntimeError("boom")
        time.sleep(_CELL_SECONDS)
        return dict(v=float(seed))


def _start_files(root):
    # The name the `run` docstring documents: `.mushin_budget_start.<id>`.
    return sorted(p.name for p in root.glob(".mushin_budget_start.*"))


def _computed(ds):
    return [
        int(s) for s in ds["seed"].values if not np.isnan(float(ds["v"].sel(seed=s)))
    ]


def _completed_cells(root):
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (read_cell_status(d) or {}).get("status") == "completed"
    )


def test_budget_holds_across_joblib_workers(tmp_path):
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    seeds = list(range(8))
    root = tmp_path / "s"

    wf = _SleepyWF()
    with pytest.warns(UserWarning, match="not run"):
        wf.run(
            seed=multirun(seeds),
            max_total_seconds=_BUDGET,
            working_dir=str(root),
            **_JOBLIB,
        )
    ds = wf.to_xarray()
    computed = _computed(ds)
    # One budget for the sweep, not one per job: before the shared start, every
    # job unpickled a runner with no deadline and all 8 cells ran.
    assert 1 <= len(computed) <= _N_JOBS, computed
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
    wf2.run(seed=multirun(seeds), resume=True, working_dir=str(root), **_JOBLIB)
    assert wf2.is_complete
    ds2 = wf2.to_xarray()
    assert _computed(ds2) == seeds
    for s, started in first_starts.items():
        assert float(ds2["started"].sel(seed=s)) == started


def test_budget_holds_under_a_nested_sweep_subdir(tmp_path):
    """With `hydra.sweep.subdir` two levels deep, a cell dir's parent is not
    the sweep root: every cell must still find the one start in the root."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    root = tmp_path / "s"

    wf = _SleepyWF()
    with pytest.warns(UserWarning, match="not run"):
        wf.run(
            seed=multirun(list(range(8))),
            max_total_seconds=_BUDGET,
            working_dir=str(root),
            overrides=["hydra.sweep.subdir=${hydra.job.num}/cell"],
            **_JOBLIB,
        )
    assert 1 <= len(_computed(wf.to_xarray())) <= _N_JOBS
    assert wf.working_dir == root  # the sweep root, not the first job's group
    assert (root / "multirun.yaml").exists()
    assert sorted(root.rglob(".mushin_budget_start.*")) == []


def test_a_run_without_working_dir_removes_its_start(tmp_path):
    """Without `working_dir` the cleanup has only the launched jobs to find the
    sweep root from, and Hydra's default sweep dir holds a `${now:...}` that
    names a later second when read again after the launch."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    # Hydra's default sweep dir, made absolute: joblib's workers may have been
    # started from another cwd, against which a relative dir would resolve.
    sweep_dir = f"{tmp_path.as_posix()}/multirun/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}"

    wf = _SleepyWF()
    with pytest.warns(UserWarning, match="not run"):
        wf.run(
            seed=multirun(list(range(8))),
            max_total_seconds=_BUDGET,
            overrides=[f"hydra.sweep.dir='{sweep_dir}'"],
            **_JOBLIB,
        )
    assert 1 <= len(_computed(wf.to_xarray())) <= _N_JOBS
    assert wf.working_dir.parent.parent == tmp_path / "multirun"
    assert (wf.working_dir / "multirun.yaml").exists()
    assert sorted(tmp_path.rglob(".mushin_budget_start.*")) == []


def test_a_failed_run_leaves_its_start_and_a_resume_starts_afresh(tmp_path):
    """A run whose launch raises keeps its start file: under submitit, jobs the
    scheduler still holds can start after the driver is gone and must find the
    same deadline. A resume names its own start, so the stale one cannot spend
    its budget."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    root = tmp_path / "s"
    kwargs = dict(
        seed=multirun(list(range(8))),
        marker=str(tmp_path / "failed-once"),
        max_total_seconds=_BUDGET,
        working_dir=str(root),
        **_JOBLIB,
    )

    with pytest.raises(RuntimeError, match="boom"):
        _FailsOnceWF().run(**kwargs)  # seed 1 fails under on_error="raise"
    stale = _start_files(root)
    assert len(stale) == 1
    done = _completed_cells(root)

    with pytest.warns(UserWarning, match="not run"):
        _FailsOnceWF().run(resume=True, **kwargs)
    # The stale start is older than the budget; had the resume read it, every
    # cell it did not reuse would have been skipped.
    assert len(_completed_cells(root)) > len(done)
    # The resume returned normally, so it removed its own start, not the stale.
    assert _start_files(root) == stale


def test_the_sequential_launcher_writes_no_start_file(tmp_path):
    """The default launcher calls the runner run() built: its in-memory
    deadline is the whole budget, so no file is written or leaked."""
    root = tmp_path / "s"
    seen = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed, fail=-1):
            seen.extend(p.name for p in root.glob(".mushin_budget_start.*"))
            if seed == fail:
                raise RuntimeError("boom")
            time.sleep(0.3 if seed == 0 else 0)
            return dict(v=float(seed))

    with pytest.warns(UserWarning, match="not run"):
        W().run(seed=multirun([0, 1, 2]), max_total_seconds=0.1, working_dir=str(root))
    # Nor does a failed sequential run, which keeps a parallel run's file.
    with pytest.raises(RuntimeError, match="boom"):
        W().run(
            seed=multirun([0, 1, 2]),
            fail=0,
            max_total_seconds=60,
            working_dir=str(tmp_path / "f"),
        )
    assert seen == []
    assert sorted(tmp_path.rglob(".mushin_budget_start.*")) == []


def _compute(seed):
    return dict(v=float(seed))


def _no_pre_task():
    pass


class _Preempted(BaseException):
    """Stands in for SLURM's preemption. Not an Exception, so the runner does
    not record a failure: the cell's status stays 'running', as it does when
    the process is killed."""


def _preempted_once(seed):
    marker = Path("preempted")  # in the cell's dir
    if not marker.exists():
        marker.touch()
        raise _Preempted
    return dict(v=float(seed))


def _runner(max_total_seconds, task=_compute):
    return _TaskRunner(
        task=zen(task),
        pre_task=zen(_no_pre_task),
        swept_names=(),
        base_provenance={},
        on_error="raise",
        inject_resume=False,
        prior_manifest=None,
        max_total_seconds=max_total_seconds,
    )


def _copy(runner):
    """The runner as a parallel launcher's worker receives it."""
    return pickle.loads(pickle.dumps(runner))


def _in_another_process(runner):
    """Forget this process's deadline for the run, as a new worker process
    has none."""
    wf_mod._BUDGET_DEADLINES.pop(runner.budget_id, None)


def _cell(root, num, monkeypatch):
    from hydra.core.hydra_config import HydraConfig

    # A runner called directly runs outside a Hydra job, but a launch earlier
    # in this process can leave its config set: clear it, so the cell finds
    # the sweep root as its dir's parent.
    monkeypatch.setattr(HydraConfig.instance(), "cfg", None)
    cell = root / str(num)
    cell.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cell)  # a cell runs in its own dir, under the sweep root
    return OmegaConf.create({"seed": num})


def test_pickled_runner_copies_share_one_start(tmp_path, monkeypatch):
    """Two copies of one runner, as a parallel launcher ships to two worker
    processes, share the budget that the first of them to compute starts."""
    root = tmp_path / "s"
    runner = _runner(max_total_seconds=0.2)
    a, b = _copy(runner), _copy(runner)

    assert a(_cell(root, 0, monkeypatch)) == {"v": 0.0}  # starts the clock
    _in_another_process(runner)
    time.sleep(0.3)
    # `b` has computed nothing, yet the sweep's budget is spent.
    assert isinstance(b(_cell(root, 1, monkeypatch)), _SkippedRun)
    assert _start_files(root) == [runner.budget_start_path(root).name]

    # A new runner is a new run() (a resume): its budget starts afresh, even
    # though the spent run's start file is still in the root.
    assert _copy(_runner(0.2))(_cell(root, 2, monkeypatch)) == {"v": 2.0}


def test_copies_in_one_process_share_its_deadline(tmp_path, monkeypatch):
    """joblib unpickles a fresh copy for every job inside the same worker
    process. When the start file cannot be created, that process times the
    budget from its own first cell, as the warning says, and its later copies
    keep that deadline instead of each restarting the clock."""
    real_open = os.open

    def open_refusing_start_files(path, *args, **kwargs):
        if ".mushin_budget_start." in os.fspath(path):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_refusing_start_files)
    root = tmp_path / "s"
    runner = _runner(max_total_seconds=0.2)
    with pytest.warns(UserWarning, match="may not hold across worker"):
        assert _copy(runner)(_cell(root, 0, monkeypatch)) == {"v": 0.0}
    time.sleep(0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no second attempt, so no second warning
        for num in (1, 2, 3):
            assert isinstance(_copy(runner)(_cell(root, num, monkeypatch)), _SkippedRun)


def test_a_requeued_cell_runs_after_the_deadline(tmp_path, monkeypatch):
    """A cell that started before the deadline and was preempted and requeued
    (SLURM `requeue`) enters again in a new process after the deadline: it is
    never cut short, so it runs. A cell that never started is skipped."""
    root = tmp_path / "s"
    runner = _runner(max_total_seconds=0.2, task=_preempted_once)
    cfg = _cell(root, 0, monkeypatch)
    with pytest.raises(_Preempted):
        _copy(runner)(cfg)  # starts the clock, then is preempted mid-task
    assert read_cell_status(Path.cwd())["status"] == "running"
    time.sleep(0.3)
    _in_another_process(runner)  # the requeued job is a new process

    assert _copy(runner)(cfg) == {"v": 0.0}
    assert isinstance(_copy(runner)(_cell(root, 1, monkeypatch)), _SkippedRun)
    # A 'running' record of another run is no exemption.
    other = _cell(root, 2, monkeypatch)
    write_cell_status(Path.cwd(), status="running", combo={}, attempt=1, budget_id="x")
    assert isinstance(_copy(runner)(other), _SkippedRun)


def test_budget_expired_never_starts_the_clock(tmp_path, monkeypatch):
    """The expiry check is pure: only `_start_budget`, for a cell about to
    compute, starts the clock, so a stray check cannot start it early."""
    root = tmp_path / "s"
    _cell(root, 0, monkeypatch)
    copy = _copy(_runner(max_total_seconds=0.2))
    assert copy._budget_expired() is False
    assert _start_files(root) == []
    assert copy.budget_id not in wf_mod._BUDGET_DEADLINES
    time.sleep(0.3)
    assert copy._start_budget() is True  # the clock starts here, not above
    assert _start_files(root) == [copy.budget_start_path(root).name]
    assert copy._budget_expired() is False


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
    assert isinstance(_copy(runner)(cfg), _SkippedRun)


def test_a_start_far_from_the_files_mtime_is_not_believed(tmp_path):
    path = tmp_path / ".mushin_budget_start.x"
    path.write_text(repr(1.0))  # finite, but not a start this run wrote
    assert wf_mod._read_budget_start(path) == path.stat().st_mtime

    mtime = time.time() - 10
    os.utime(path, (mtime, mtime))
    path.write_text(repr(mtime - 1.5))  # within the tolerance: believed
    os.utime(path, (mtime, mtime))
    assert wf_mod._read_budget_start(path) == mtime - 1.5


def test_an_empty_read_is_retried_before_the_mtime(tmp_path, monkeypatch):
    """Caught between the creator's create and write, a reader retries for a
    few milliseconds: the mtime may be truncated to the second, which would
    shorten a sub-second budget."""
    path = tmp_path / ".mushin_budget_start.x"
    path.touch()
    start = path.stat().st_mtime - 2.0  # within the tolerance of the mtime
    reads = []
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == path:
            reads.append(self)
            return "" if len(reads) == 1 else repr(start)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert wf_mod._read_budget_start(path) == start
    assert len(reads) == 2
