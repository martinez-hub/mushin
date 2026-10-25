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


class _GridWF(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a, b):
        import time

        time.sleep(_CELL_SECONDS)
        return dict(v=float(10 * a + b))


def test_budget_holds_under_a_nested_sweep_subdir(tmp_path):
    """With `hydra.sweep.subdir=a_${a}/b_${b}` a cell dir's parent is its `a`
    group, not the sweep root: every cell must still share the one start in
    the root, and the cleanup must find it there. No `working_dir`, so the
    cleanup has only the job dirs to search from."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    root = tmp_path / "s"

    try:
        _GridWF().run(
            a=multirun([0, 1]),
            b=multirun([0, 1, 2, 3]),
            max_total_seconds=_BUDGET,
            overrides=[
                f"hydra.sweep.dir='{root.as_posix()}'",
                "hydra.sweep.subdir=a_${a}/b_${b}",
            ],
            **_JOBLIB,
        )
    except FileNotFoundError as e:
        # Post-processing does not support a nested subdir (as on main: it
        # looks for multirun.yaml in the first job's parent), so run() can
        # raise once the launch has returned; the budget and its cleanup are
        # done by then.
        assert "multirun.yaml" in str(e)
    status = {
        d.relative_to(root).as_posix(): read_cell_status(d)["status"]
        for d in root.glob("a_*/b_*")
    }
    completed = sorted(k for k, v in status.items() if v == "completed")
    assert len(status) == 8
    assert sorted(k for k, v in status.items() if v != "completed") == sorted(
        k for k, v in status.items() if v == "skipped"
    )
    # One budget for the sweep. Before the deadline each worker starts one
    # cell, the first two dispatched (a_0's); with a start per `a` group, the
    # first a_1 cell would start a second clock and run.
    assert completed and set(completed) <= {"a_0/b_0", "a_0/b_1"}, status
    assert sorted(root.rglob(".mushin_budget_start.*")) == []


def test_a_cell_finds_the_sweep_root_under_a_nested_subdir(tmp_path):
    """The cell reads the depth of `hydra.sweep.subdir` from its live job
    config, which resolves the job's own keys (`a_${a}`), so it climbs to the
    sweep root, not to its `a` group. (In the joblib test above, each worker
    keeps the deadline of its first cell, which can hide a wrong root.)"""
    root = tmp_path / "s"
    roots = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            roots.append(wf_mod._cell_sweep_root())
            return dict(v=float(10 * a + b))

    try:
        W().run(
            a=multirun([0, 1]),
            b=multirun([0, 1]),
            working_dir=str(root),
            overrides=["hydra.sweep.subdir=a_${a}/b_${b}"],
        )
    except FileNotFoundError as e:  # post-processing, as above
        assert "multirun.yaml" in str(e)
    assert roots == [root] * 4


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
        working_dir=str(root),
        **_JOBLIB,
    )

    # A budget no cell reaches, so the cell meant to fail always runs: seed 1
    # fails under on_error="raise", after the other seven completed.
    with pytest.raises(RuntimeError, match="boom"):
        _FailsOnceWF().run(max_total_seconds=60, **kwargs)
    stale = _start_files(root)
    assert len(stale) == 1
    assert len(_completed_cells(root)) == 7

    # The stale start is older than this run's budget (seven half-second cells
    # on two workers ran since): had the resume read it, it would have skipped
    # seed 1, the one cell it has to run.
    wf = _FailsOnceWF()
    wf.run(max_total_seconds=_BUDGET, resume=True, **kwargs)
    assert wf.is_complete
    assert len(_completed_cells(root)) == 8
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


def _record(value):
    """A start record as its creator writes it: the value and a newline."""
    return f"{value!r}\n"


def test_the_start_record_written_by_a_cell_is_complete(tmp_path, monkeypatch):
    root = tmp_path / "s"
    _cell(root, 0, monkeypatch)
    copy = _copy(_runner(max_total_seconds=60))
    before = time.time()
    assert copy._start_budget() is True
    text = copy.budget_start_path(root).read_text(encoding="ascii")
    assert text.endswith("\n") and before <= float(text) <= time.time()


def test_a_complete_plausible_record_is_trusted_over_the_mtime(tmp_path):
    """The file is named by the run's uuid, so a complete record in it is this
    run's start. On NFS the mtime comes from the server's clock: a minute of
    server skew must not move the budget. Only a record no clock could have
    written falls back to the mtime."""
    path = tmp_path / ".mushin_budget_start.x"
    start = time.time() - 60  # a minute off the mtime the write sets
    path.write_text(_record(start))
    assert wf_mod._read_budget_start(path) == start

    implausible = (1.0, time.time() + 3 * 86400, float("inf"), float("nan"))
    for text in [_record(v) for v in implausible] + ["x\n"]:
        path.write_text(text)
        assert wf_mod._read_budget_start(path) == path.stat().st_mtime, text


def test_an_incomplete_record_is_not_trusted(tmp_path):
    """Without its newline a record may be cut short (``1761400000`` of
    ``1761400000.25``): after the retries, the mtime stands in."""
    path = tmp_path / ".mushin_budget_start.x"
    path.write_text(repr(time.time() - 2.0))  # plausible, but no newline
    assert wf_mod._read_budget_start(path) == path.stat().st_mtime


def test_an_incomplete_read_is_retried_before_the_mtime(tmp_path, monkeypatch):
    """Caught between the creator's create and write, or in the middle of the
    write, a reader retries for a few milliseconds: the mtime may be truncated
    to the second, which would shorten a sub-second budget."""
    path = tmp_path / ".mushin_budget_start.x"
    path.touch()
    start = path.stat().st_mtime - 2.0  # the start, unlike the mtime
    reads = iter(["", repr(start)[:5], _record(start)])  # empty, partial, done
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == path:
            return next(reads)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert wf_mod._read_budget_start(path) == start
    assert next(reads, None) is None  # all three reads were needed


def test_reading_a_missing_or_empty_start_wastes_no_sleep(tmp_path, monkeypatch):
    """A retry cannot bring a missing file back, and no read follows the last
    sleep: Windows' timer rounds each 2 ms sleep up to about 16 ms."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert wf_mod._read_budget_start(tmp_path / "gone") is None
    assert sleeps == []

    empty = tmp_path / ".mushin_budget_start.x"
    empty.touch()
    assert wf_mod._read_budget_start(empty) == empty.stat().st_mtime
    assert len(sleeps) == wf_mod._BUDGET_START_READS - 1


def test_an_undecodable_start_file_reads_as_its_mtime(tmp_path):
    path = tmp_path / ".mushin_budget_start.x"
    path.write_bytes(b"\x81\x8d\xff")  # not ASCII; every record is
    assert wf_mod._read_budget_start(path) == path.stat().st_mtime


def test_both_clocks_are_read_after_the_file_io(tmp_path, monkeypatch):
    """The read's retries, or a slow NFS stat, must not be taken off the
    budget: the deadline is the shared start plus the budget, however long
    reading the start took."""
    root = tmp_path / "s"
    _cell(root, 0, monkeypatch)
    copy = _copy(_runner(max_total_seconds=2.0))
    copy.budget_start_path(root).write_text(_record(time.time()))  # another cell
    real_read = wf_mod._read_budget_start

    def slow_read(path):
        time.sleep(1.2)
        return real_read(path)

    monkeypatch.setattr(wf_mod, "_read_budget_start", slow_read)
    assert copy._start_budget() is False
    remaining = copy._known_deadline() - time.monotonic()
    assert 0.3 < remaining <= 0.8, remaining  # 2.0 s from the start, 1.2 s gone


def test_a_vanished_start_file_falls_back_to_this_process(tmp_path, monkeypatch):
    """A start file that exists at the create but is gone at the read is no
    start to share: like an uncreatable one, the process times the budget
    from this cell and warns; this cell did not start the sweep's clock."""
    real_open = os.open

    def open_finding_a_start(path, *args, **kwargs):
        if ".mushin_budget_start." in os.fspath(path):
            raise FileExistsError(17, "File exists", os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_finding_a_start)
    _cell(tmp_path / "s", 0, monkeypatch)
    copy = _copy(_runner(max_total_seconds=60))
    with pytest.warns(UserWarning, match="vanished"):
        assert copy._start_budget() is False
    assert copy._known_deadline() is not None
    assert copy._budget_expired() is False


def test_an_unexpected_budget_error_never_fails_the_cell(tmp_path, monkeypatch):
    """The budget step runs before the runner's fail-soft guard: an error in it
    falls back to this process's own deadline, with a warning, and the cell
    runs."""

    def broken_root():
        raise IndexError("no sweep root above /")

    monkeypatch.setattr(wf_mod, "_cell_sweep_root", broken_root)
    root = tmp_path / "s"
    runner = _runner(max_total_seconds=0.2)
    with pytest.warns(UserWarning, match="shared budget failed"):
        assert _copy(runner)(_cell(root, 0, monkeypatch)) == {"v": 0.0}
    time.sleep(0.3)
    assert isinstance(_copy(runner)(_cell(root, 1, monkeypatch)), _SkippedRun)


def test_a_runner_pickled_by_an_earlier_mushin_keeps_its_own_budget(
    tmp_path, monkeypatch
):
    """A runner pickled before the budget was shared (a submitit array queued
    before an upgrade) has no `budget_id`: it must not fail before the
    fail-soft guard, and it times a budget of its own, as it did."""
    root = tmp_path / "s"
    runner = _runner(max_total_seconds=0.2)
    del runner.budget_id, runner._shared_budget
    runner._deadline = None  # what that version kept on the runner
    old = _copy(runner)
    assert old(_cell(root, 0, monkeypatch)) == {"v": 0.0}
    time.sleep(0.3)
    assert isinstance(old(_cell(root, 1, monkeypatch)), _SkippedRun)
    assert sorted(root.rglob(".mushin_budget_start.*")) == []


def test_the_deadline_table_keeps_only_recent_runs(monkeypatch):
    """loky keeps its worker processes across run() calls, so the per-process
    table of deadlines keeps only the most recent runs."""
    monkeypatch.setattr(wf_mod, "_BUDGET_DEADLINES", {})
    kept = wf_mod._BUDGET_DEADLINES_KEPT
    for i in range(kept + 10):
        wf_mod._remember_deadline(f"run{i}", float(i))
    assert len(wf_mod._BUDGET_DEADLINES) == kept
    assert "run0" not in wf_mod._BUDGET_DEADLINES
    assert f"run{kept + 9}" in wf_mod._BUDGET_DEADLINES
