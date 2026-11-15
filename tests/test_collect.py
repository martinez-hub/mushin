# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`run(..., wait=False)` submits a sweep and returns; `mushin.collect`
builds its dataset once the jobs have ended (#345).

These tests run no job. They check what `run()` refuses before it writes
anything, and how collect judges a launch's jobs before it reads any: a
SLURM job by submitit's own `sacct` reader (its command faked here, as
SLURM would print), a `submitit_local` job by its process. A launch whose
jobs have all ended goes on to its jobs' submission files, which these
tests do not write, so collect stops there, having written nothing.
Running jobs are in tests/test_submitit_local.py."""

from __future__ import annotations

import ast
import contextlib
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from hydra_zen import make_config

import mushin
from _scheduler import fail_closed
from mushin import _jobs, multirun
from mushin.workflows import (
    MultiRunMetricsWorkflow,
    _mushin_launchers_available,
    _TaskRunner,
)

R = "d" * 32  # the run id of the hand-written sweeps below

posix = pytest.mark.skipif(
    sys.platform == "win32", reason="a pid is probed on POSIX only"
)
needs_launchers = pytest.mark.skipif(
    not _mushin_launchers_available(), reason="needs the `slurm` extra"
)


@pytest.fixture(autouse=True)
def _no_scheduler(monkeypatch):
    # Collect asks SLURM through submitit's sacct reader, never `_jobs._run`.
    fail_closed(monkeypatch)


@mushin.sweep
def _task(lr):
    return dict(v=float(lr))


@mushin.sweep
def _waits(lr, wait):
    """A task with a parameter of its own named `wait`."""
    return dict(v=float(lr) * wait)


@mushin.sweep
def _keeps_wait(lr, wait):
    """A task with a `wait` of its own, of any type."""
    return dict(v=float(lr))


class _WaitField(MultiRunMetricsWorkflow):
    """Its config has a field named `wait` (`make_config(wait=1)`)."""

    @staticmethod
    def task(lr):
        return dict(v=float(lr))


class _PreTaskWaits(MultiRunMetricsWorkflow):
    """Its pre_task has a parameter named `wait`."""

    @staticmethod
    def pre_task(wait):
        pass

    @staticmethod
    def task(lr):
        return dict(v=float(lr))


# What run(wait=False) refuses, before anything is written.


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({}, "launcher=", id="default launcher"),
        pytest.param({"launcher": "joblib"}, "launcher=", id="joblib"),
        pytest.param(
            {"launcher": "submitit_local", "working_dir": None},
            "working_dir",
            id="no working_dir",
            marks=needs_launchers,
        ),
        pytest.param(
            {"launcher": "submitit_local", "working_dir": "${hydra:runtime.cwd}/s"},
            "working_dir",
            id="working_dir resolved only in the launch",
            marks=needs_launchers,
        ),
        pytest.param(
            {"launcher": "submitit_local", "sweeper": "nevergrad"},
            "sweeper",
            id="sweeper",
            marks=needs_launchers,
        ),
        pytest.param(
            {"launcher": "submitit_local", "overrides": ["hydra/sweeper=optuna"]},
            "sweeper",
            id="raw sweeper",
            marks=needs_launchers,
        ),
    ],
)
def test_wait_false_is_refused_where_nothing_could_collect(
    tmp_path, monkeypatch, kwargs, match
):
    monkeypatch.chdir(tmp_path)  # Hydra's default dir, for the case with none
    wd = tmp_path / "s"
    sweep = {"lr": multirun([0.1, 0.2]), "working_dir": str(wd), "wait": False}
    with pytest.raises(ValueError, match=match):
        _task.run(**{**sweep, **kwargs})
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_wait_must_be_a_bool(tmp_path):
    wd = tmp_path / "s"
    with pytest.raises(TypeError, match="`wait` must be True or False"):
        _task.run(lr=multirun([0.1]), working_dir=str(wd), wait="no")
    assert not wd.exists()


@pytest.mark.parametrize(
    "sweep",
    [
        pytest.param(_waits, id="task parameter"),
        pytest.param(_PreTaskWaits(), id="pre_task parameter"),
        pytest.param(_WaitField(make_config(wait=1)), id="config field"),
    ],
)
def test_wait_is_refused_where_it_names_the_tasks_own_value(tmp_path, sweep):
    """`wait` became run()'s own parameter: a value meant for the task's
    `wait` is refused, not taken as run()'s, and the message says how to
    pass it now."""
    wd = tmp_path / "s"
    # True first: a bool passes run()'s type check, so only the refusal
    # stops it.
    for wait in (True, 2):
        with pytest.raises(ValueError, match=re.escape('overrides=["++wait=')):
            sweep.run(lr=multirun([1.0]), wait=wait, working_dir=str(wd))
    assert not wd.exists()
    # The migration: the job's `wait` as an override.
    sweep.run(lr=multirun([1.0]), overrides=["++wait=2"], working_dir=str(wd))
    workflow = getattr(sweep, "workflow", sweep)
    assert [cfg["wait"] for cfg in workflow.cfgs] == [2]


@pytest.mark.parametrize(
    "wait, received",
    [
        pytest.param(2, [2], id="int"),
        pytest.param(None, [None], id="None"),
        pytest.param("a,b", ["a,b"], id="string"),
        pytest.param('say "hi" in C:\\data', ['say "hi" in C:\\data'], id="quotes"),
        pytest.param(multirun([0.5, 2.0]), [0.5, 2.0], id="multirun"),
    ],
)
def test_the_wait_hint_passes_the_value_as_run_would(tmp_path, wait, received):
    """The override the refusal suggests, pasted as written, gives the task
    the value run(wait=...) was given: written as run() writes a keyword's
    value (a string quoted, a `multirun` swept), not as its text, which
    gave a string "None", split "a,b", and swept nothing (review of #414)."""
    wd = tmp_path / "s"
    with pytest.raises(ValueError, match=re.escape('overrides=["++wait=')) as info:
        _keeps_wait.run(lr=multirun([1.0]), wait=wait, working_dir=str(wd))
    written = re.search(r"`overrides=(\[.*\])`", str(info.value)).group(1)
    _keeps_wait.run(
        lr=multirun([1.0]), overrides=ast.literal_eval(written), working_dir=str(wd)
    )
    assert [cfg["wait"] for cfg in _keeps_wait.workflow.cfgs] == received


def test_a_runner_pickled_by_an_earlier_mushin_waits():
    """A runner unpickled from a job queued before the fields existed
    neither leaves its jobs to collect nor carries a run's lineage."""
    runner = _TaskRunner.__new__(_TaskRunner)
    runner.__setstate__({"max_total_seconds": None})
    assert runner.detached is False and runner.lineage is None


# Collect with nothing pending: the reload's view, and nothing written.


def _files(wd: Path) -> dict:
    return {p: p.stat().st_mtime_ns for p in wd.rglob("*") if p.is_file()}


def test_collect_with_nothing_pending_reloads_and_writes_nothing(tmp_path):
    wd = tmp_path / "s"
    ds = _task.run(lr=multirun([0.1, 0.2]), working_dir=str(wd))
    before = _files(wd)
    again = mushin.collect(wd)
    assert again["v"].values.tolist() == ds["v"].values.tolist()
    assert _files(wd) == before
    assert "collect" in mushin.__all__


def test_collect_needs_a_sweep_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        mushin.collect(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="multirun.yaml"):
        mushin.collect(tmp_path)


# A detached launch, written by hand: run R's sweep over `lr`, its cells
# each sent to a job, none of whose submission files exist.


def _detached(wd: Path, cells, *, executor="slurm", host="login1", detached=True):
    """``wd`` as a `run(wait=False)` leaves it before collect: the sweep's
    multirun.yaml naming run R, and a launch file naming ``cells``
    (``(num, job_id)``)."""
    wd.mkdir(parents=True, exist_ok=True)
    # The jobs' folder, as submitit makes it on submission.
    (wd / ".submitit").mkdir(exist_ok=True)
    nums = [n for n, _ in cells]
    values = ",".join(str(n / 10) for n in range(max(nums) + 1))
    (wd / "multirun.yaml").write_text(
        "hydra:\n"
        "  job:\n"
        f"    mushin_run_id: '{R}'\n"
        "  overrides:\n"
        "    task:\n"
        f"    - lr=choice({values})\n",
        encoding="utf-8",
    )
    return _jobs.record_launch(
        wd,
        run_id=R,
        executor=executor,
        folder=str(wd / ".submitit"),
        clusters=None,
        host=host,
        depth=1,
        cells=[
            _jobs.LaunchCell(n, j, {"lr": n / 10}, overrides=(f"lr={n / 10}",))
            for n, j in cells
        ],
        detached=detached,
        submitit_folder=str(wd / ".submitit" / "%j"),
    )


def _untouched(wd: Path) -> list[str]:
    """What is in ``wd``: only what `_detached` wrote, for a collect that
    must write nothing."""
    return sorted(p.name for p in wd.iterdir() if p.name != ".submitit")


@pytest.fixture
def sacct(monkeypatch):
    """submitit's own sacct reader, its command answering from ``table``
    (job id -> state) as SLURM prints it; ``table["fail"] = True`` makes
    the command fail, as where sacct is missing. A fresh watcher, so no
    state carries over from another test."""
    slurm = pytest.importorskip("submitit.slurm.slurm")
    from submitit.core import core

    monkeypatch.setattr(slurm.SlurmJob, "watcher", slurm.SlurmInfoWatcher(delay_s=600))
    # No wait for a result file that never comes: these jobs wrote none.
    monkeypatch.setattr(core.Job, "_results_timeout_s", 0)
    table: dict = {}

    def command(self):
        if table.get("fail"):
            return [sys.executable, "-c", "raise SystemExit(1)"]
        out = "JobID|State|NodeList\n" + "".join(
            f"{job}|{state}|n1\n" for job, state in table.items()
        )
        return [sys.executable, "-c", f"import sys; sys.stdout.write({out!r})"]

    monkeypatch.setattr(slurm.SlurmInfoWatcher, "_make_command", command)
    return table


@needs_launchers
@pytest.mark.parametrize(
    "state", ["PENDING", "RUNNING", "PREEMPTED", "REQUEUED", "SUSPENDED"]
)
def test_collect_refuses_while_a_slurm_job_has_not_ended(tmp_path, sacct, state):
    """SUSPENDED is not an end, though submitit's own `done()` takes it
    for one: collect counts only SLURM's end states, so it never records
    a live job's cell."""
    wd = tmp_path / "s"
    path = _detached(wd, [(0, "100_0"), (1, "100_1")])
    sacct.update({"100_0": "COMPLETED", "100_1": state})
    with pytest.raises(RuntimeError, match="not ended") as caught:
        mushin.collect(wd)
    assert f"1 of 2 jobs ({state}: 1)" in str(caught.value)
    assert path.exists() and _untouched(wd) == ["multirun.yaml", "mushin_jobs"]


@needs_launchers
def test_collect_refuses_when_sacct_gives_no_answer(tmp_path, sacct):
    wd = tmp_path / "s"
    _detached(wd, [(0, "100_0")])
    sacct["fail"] = True
    with pytest.raises(RuntimeError, match="UNKNOWN") as caught:
        mushin.collect(wd)
    assert "sacct" in str(caught.value)
    assert _untouched(wd) == ["multirun.yaml", "mushin_jobs"]


@needs_launchers
@pytest.mark.parametrize(
    "state",
    ["COMPLETED", "FAILED", "OUT_OF_MEMORY", "CANCELLED by 1000", "TIMEOUT"],
)
def test_an_ended_slurm_job_is_collected(tmp_path, sacct, state):
    """A job SLURM says has ended, with or without a result, passes the
    check: collect goes on to its submission file (not written here), and
    stops there with nothing written."""
    wd = tmp_path / "s"
    path = _detached(wd, [(0, "100_0"), (1, "100_1")])
    sacct.update({"100_0": "COMPLETED", "100_1": state})
    with pytest.raises(RuntimeError, match="_submitted.pkl") as caught:
        mushin.collect(wd)
    assert "not ended" not in str(caught.value)
    assert path.exists() and _untouched(wd) == ["multirun.yaml", "mushin_jobs"]


# A stand-in for submitit's local controller, as in tests/test_guard.py: its
# command line holds what the real one's does, and it runs until the file
# its first argument names exists.
_HOLD = """\
import pathlib, sys, time
release = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 60
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.02)
"""


@contextlib.contextmanager
def _controller(tmp_path: Path, folder: str):
    release = tmp_path / f"release-{time.monotonic_ns()}"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(release)]
        + ["-m", "submitit.local._local", f"{folder}/%j"]
    )
    try:
        yield proc
    finally:
        release.touch()
        proc.kill()
        proc.wait()


@posix
@needs_launchers
def test_collect_judges_a_local_job_by_its_process(tmp_path, monkeypatch):
    from submitit.core import core

    monkeypatch.setattr(core.Job, "_results_timeout_s", 0)
    wd = tmp_path / "s"
    with _controller(tmp_path, str(wd / ".submitit")) as proc:
        path = _detached(
            wd, [(0, str(proc.pid))], executor="local", host=socket.gethostname()
        )
        with pytest.raises(RuntimeError, match="not ended"):
            mushin.collect(wd)
    # Its process has ended: collect goes on to its submission file.
    with pytest.raises(RuntimeError, match="_submitted.pkl"):
        mushin.collect(wd)
    assert path.exists() and _untouched(wd) == ["multirun.yaml", "mushin_jobs"]


@needs_launchers
def test_collect_refuses_a_local_launch_of_another_host(tmp_path):
    wd = tmp_path / "s"
    _detached(wd, [(0, "12345")], executor="local", host="elsewhere")
    with pytest.raises(RuntimeError, match="'elsewhere'"):
        mushin.collect(wd)
    assert _untouched(wd) == ["multirun.yaml", "mushin_jobs"]


@needs_launchers
def test_collect_leaves_a_waiting_drivers_launch_alone(tmp_path, sacct):
    """A launch a driver waits on (or did, until it was killed) is that
    driver's to finish, or a resume's: collect would race it over the same
    records."""
    wd = tmp_path / "s"
    path = _detached(wd, [(0, "100_0")], detached=False)
    with pytest.raises(RuntimeError, match="resume=True"):
        mushin.collect(wd)
    assert path.exists() and _untouched(wd) == ["multirun.yaml", "mushin_jobs"]


@needs_launchers
def test_collect_refuses_a_run_whose_launch_file_is_gone(tmp_path, sacct):
    """Cells 0 and 1 were sent to jobs, but only cell 1's launch file is
    left (another was removed by hand, or by `cancel`): a dataset of the
    rest would read as the whole sweep."""
    wd = tmp_path / "s"
    _detached(wd, [(1, "100_1")])
    with pytest.raises(RuntimeError, match=r"cells \[0\]"):
        mushin.collect(wd)


def test_a_detached_launch_file_round_trips(tmp_path):
    wd = tmp_path / "s"
    _detached(wd, [(0, "100_0"), (1, "100_1")])
    (read,), bad = _jobs.read_launches(wd)
    assert bad == [] and read.detached
    assert read.submitit_folder == str(wd / ".submitit" / "%j")
    assert [c.overrides for c in read.cells] == [("lr=0.0",), ("lr=0.1",)]
    # A launch whose cells all settled in the driver, under max_batch_size:
    # it names no job, only the cells collect composes.
    path = _jobs.record_launch(
        wd,
        run_id=R,
        executor="slurm",
        folder=str(wd / ".submitit"),
        clusters=None,
        host="login1",
        depth=1,
        cells=[],
        detached=True,
        submitit_folder=str(wd / ".submitit" / "%j"),
        settled=[_jobs.SettledCell(2, ("lr=0.2",), "skipped")],
    )
    assert path.name == f"{R}.2.json"
    launches, bad = _jobs.read_launches(wd)
    assert bad == [] and [len(x.cells) for x in launches] == [2, 0]
    assert launches[1].settled == (_jobs.SettledCell(2, ("lr=0.2",), "skipped"),)
    # Nothing to cancel or to show as queued; a run's check reads it as over.
    data = json.loads(path.read_text())
    data["settled"][0]["value"] = "other"
    path.write_text(json.dumps(data))
    assert len(_jobs.read_launches(wd)[1]) == 1
