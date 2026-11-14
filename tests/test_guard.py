# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`run()` refuses a `working_dir` whose earlier jobs may still be queued or
running (#345), so two drivers never run one sweep at once.

mushin's submitit launchers name each submission's jobs in a launch file
under ``<working_dir>/mushin_jobs/``. A run asks SLURM about the jobs of
those files (faked here, as in tests/test_cancel.py), or, for a
``submitit_local`` launch, looks for their processes. Every test runs the
default launcher: the check is the same whatever launcher the new run uses.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

import mushin
from _scheduler import (
    PURGED,
    R0,
    R1,
    R2,
    SCONTROL,
    fail_closed,
    fake_slurm,
    job,
    launch,
    shown,
)
from mushin import _jobs, multirun

posix = pytest.mark.skipif(
    sys.platform == "win32", reason="a pid is probed on POSIX only"
)


@pytest.fixture(autouse=True)
def _no_scheduler(monkeypatch):
    fail_closed(monkeypatch)


@pytest.fixture
def slurm(monkeypatch):
    return fake_slurm(monkeypatch)


@mushin.sweep
def _task(lr):
    return dict(v=float(lr))


@mushin.sweep
def _boom(lr):
    raise RuntimeError("boom")


def _run(wd: Path):
    return _task.run(lr=multirun([0.1, 0.2]), working_dir=str(wd))


def _written(wd: Path) -> list[str]:
    """What a run wrote in ``wd``: nothing but the launch files a test put
    there, for one refused before it launched."""
    return sorted(p.name for p in wd.iterdir() if p.name != "mushin_jobs")


def _record(wd: Path, name: str, *, run, status, num=None, job_id=None):
    """A cell's record, as the runner writes it."""
    fields = {"combo": {"lr": 0.0}, "status": status, "run_id": run}
    if num is not None:
        fields["job_num"] = num
    if job_id is not None:
        fields["job_id"] = job_id
    cell = wd / name
    cell.mkdir(parents=True)
    (cell / "mushin_cell_status.json").write_text(json.dumps(fields))


def _ours(wd: Path) -> str:
    return str(wd / ".submitit")


# SLURM launches.


def test_a_run_refuses_a_dir_whose_earlier_jobs_are_live(tmp_path, slurm):
    wd = tmp_path / "s"
    path = launch(wd, R1, [(0, "100_0"), (1, "100_1")])
    slurm.jobs["100"] = shown(
        job("RUNNING", _ours(wd), task=0), job("PENDING", _ours(wd), task="1")
    )
    with pytest.raises(ValueError, match="queued or running") as caught:
        _run(wd)
    message = str(caught.value)
    # It says to wait or cancel, and names the file to delete.
    assert "mushin.cancel" in message and str(path) in message and "100" in message
    assert _written(wd) == [] and path.exists()
    assert slurm.asked == ["100"] and slurm.scancels == []


def test_a_preempted_job_counts_as_live(tmp_path, slurm):
    """SLURM may requeue a preempted job, which then runs its cell again."""
    wd = tmp_path / "s"
    path = launch(wd, R1, [(0, "100")])
    slurm.jobs["100"] = shown(job("PREEMPTED", _ours(wd)))
    with pytest.raises(ValueError, match="queued or running"):
        _run(wd)
    assert _written(wd) == [] and path.exists()


def test_only_the_live_launch_is_named(tmp_path, slurm):
    """A launch each of whose cells has a record of its run's end, by job
    id or by job number, is over: SLURM is not asked about it."""
    wd = tmp_path / "s"
    over = launch(wd, R1, [(0, "100_0"), (1, "100_1")])
    _record(wd, "0", run=R1, num=0, status="completed")
    _record(wd, "1", run=R1, job_id="100_1", status="failed")
    live = launch(wd, R2, [(2, "200")])
    slurm.jobs["200"] = shown(job("RUNNING", _ours(wd), job="200"))
    with pytest.raises(ValueError) as caught:
        _run(wd)
    assert live.name in str(caught.value) and over.name not in str(caught.value)
    assert slurm.asked == ["200"]


@pytest.mark.parametrize(
    "record",
    [dict(run=R1, status="running"), dict(run=R0, status="completed")],
    ids=["running", "another run's"],
)
def test_a_record_ends_a_job_only_if_its_run_says_it_ended(tmp_path, slurm, record):
    """A `running` record of the launch's run, or an end another run
    recorded in the cell's dir, leaves the job to SLURM."""
    wd = tmp_path / "s"
    launch(wd, R1, [(0, "100")])
    _record(wd, "0", num=0, job_id="100", **record)
    slurm.jobs["100"] = shown(job("RUNNING", _ours(wd)))
    with pytest.raises(ValueError, match="queued or running"):
        _run(wd)
    assert slurm.asked == ["100"]


def test_an_unknown_scheduler_warns_and_runs(tmp_path, slurm, monkeypatch):
    """Where SLURM's commands are not found, the jobs cannot be checked:
    the run warns and goes ahead, and the launch file stays."""
    wd = tmp_path / "s"
    path = launch(wd, R1, [(0, "100")])
    slurm.jobs["100"] = shown(job("RUNNING", _ours(wd)))
    with pytest.raises(ValueError, match="queued or running"):
        _run(wd)
    monkeypatch.setattr(_jobs, "_which", lambda name: None)
    with pytest.warns(UserWarning, match="no scontrol") as caught:
        ds = _run(wd)
    assert any(str(path) in str(w.message) for w in caught)
    assert ds["v"].values.tolist() == [0.1, 0.2]
    assert path.exists()


def test_a_purged_job_does_not_block(tmp_path, slurm):
    """SLURM forgets a job some minutes after it ends. Once this run's own
    launch has returned, it removes the launch files of the jobs it found
    over."""
    wd = tmp_path / "s"
    gone = launch(wd, R1, [(0, "100")])
    live = launch(wd, R2, [(1, "200")])
    slurm.jobs["100"] = PURGED
    slurm.jobs["200"] = shown(job("RUNNING", _ours(wd), job="200"))
    with pytest.raises(ValueError) as caught:
        _run(wd)
    assert live.name in str(caught.value) and gone.name not in str(caught.value)
    assert gone.exists() and live.exists()
    slurm.jobs["200"] = shown(job("COMPLETED", _ours(wd), job="200"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = _run(wd)
    assert not [w for w in caught if "mushin_jobs" in str(w.message)]
    assert ds["v"].values.tolist() == [0.1, 0.2]
    assert not gone.exists() and not live.exists()


def test_launch_files_stay_when_the_run_raises(tmp_path, slurm):
    """A run that raises has superseded no earlier launch: the files of the
    jobs it found over stay."""
    wd = tmp_path / "s"
    gone = launch(wd, R1, [(0, "100")])
    slurm.jobs["100"] = PURGED
    with pytest.raises(RuntimeError, match="boom"):
        _boom.run(lr=multirun([0.1]), working_dir=str(wd))
    assert slurm.asked == ["100"] and gone.exists()


@pytest.mark.parametrize(
    "answer",
    [
        (1, "", "slurm_load_jobs error: Unable to contact slurm controller\n"),
        subprocess.TimeoutExpired([SCONTROL], 10),
        (0, "nothing a job line holds\n", ""),
    ],
    ids=["failed", "timeout", "unreadable"],
)
def test_a_scheduler_that_cannot_answer_refuses(tmp_path, slurm, answer):
    """scontrol is there but gives no answer to read: as `mushin.cancel`
    cancels nothing then, the run does not go ahead."""
    wd = tmp_path / "s"
    path = launch(wd, R1, [(0, "100")])
    slurm.jobs["100"] = answer
    with pytest.raises(RuntimeError, match="could not tell") as caught:
        _run(wd)
    assert str(path) in str(caught.value)
    assert _written(wd) == [] and path.exists()


def test_a_launch_to_other_clusters_is_refused_unchecked(tmp_path, slurm):
    """Plain scontrol does not see another cluster's jobs, so it would call
    them purged: they are not asked about, and the run does not go ahead.
    `mushin.cancel` names the command that cancels them and removes the
    file; the run then goes ahead."""
    wd = tmp_path / "s"
    path = launch(wd, R1, [(0, "100_0")], clusters="gpu2")
    slurm.jobs["100"] = PURGED
    with pytest.raises(RuntimeError, match="gpu2") as caught:
        _run(wd)
    assert str(path) in str(caught.value) and "mushin.cancel" in str(caught.value)
    assert slurm.asked == [] and _written(wd) == []
    with pytest.warns(UserWarning, match="scancel --clusters=gpu2"):
        mushin.cancel(wd)
    assert _run(wd)["v"].values.tolist() == [0.1, 0.2]


def test_a_job_that_is_not_this_sweeps_warns_and_runs(tmp_path, slurm):
    """SLURM reuses ids: a job whose script and logs are not under the
    sweep's submitit folder is another's. The run warns and goes ahead, and
    the launch file stays, as `mushin.cancel` keeps it."""
    wd = tmp_path / "s"
    path = launch(wd, R1, [(0, "100")])
    slurm.jobs["100"] = shown(job("RUNNING", _ours(wd)))
    with pytest.raises(ValueError, match="queued or running"):
        _run(wd)
    slurm.jobs["100"] = shown(job("RUNNING", "/home/other/.submitit"))
    with pytest.warns(UserWarning, match="not this sweep's"):
        ds = _run(wd)
    assert ds["v"].values.tolist() == [0.1, 0.2]
    assert path.exists()


def test_a_dir_with_no_launch_asks_nothing(tmp_path):
    """The scheduler fails closed here: a run into a dir with no launch, or
    an empty `mushin_jobs/`, asks it nothing."""
    wd = tmp_path / "s"
    (wd / "mushin_jobs").mkdir(parents=True)
    assert _run(wd)["v"].values.tolist() == [0.1, 0.2]
    assert _run(wd)["v"].values.tolist() == [0.1, 0.2]


# submitit_local launches: a job's id is its controller's process id on the
# host that submitted it.

# A stand-in for submitit's local controller: its command line holds what
# the real one's does (`python -m submitit.local._local <folder>/%j`), and it
# runs until the file its first argument names exists.
_HOLD = """\
import pathlib, sys, time
release = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 60
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.02)
"""


@contextlib.contextmanager
def _controller(tmp_path: Path, folder: str):
    """A running stand-in controller of jobs under ``folder``, and the file
    that ends it. It is killed and reaped on the way out."""
    release = tmp_path / f"release-{time.monotonic_ns()}"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(release)]
        + ["-m", "submitit.local._local", f"{folder}/%j"]
    )
    try:
        yield proc, release
    finally:
        release.touch()
        proc.kill()
        proc.wait()


def _zombie(pid: int) -> bool:
    """Whether ``pid`` has exited and its parent has not reaped it, as the
    system reports it (the test's own reading, not mushin's)."""
    if sys.platform.startswith("linux"):
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rpartition(")")[2].split()[0] == "Z"
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True)
    return out.stdout.decode().strip().startswith("Z")


def _until(condition, seconds=30) -> bool:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _table(monkeypatch, read) -> None:
    """Stand ``read`` in for the process table mushin reads: a dict of each
    pid's state and command line, or None when it cannot be read. Not
    `raising`: each test that sets it also asserts what the real table
    gives, which a renamed seam would leave in place, failing the test."""
    monkeypatch.setattr(_jobs, "_processes", read, raising=False)


def _local(wd: Path, pid: int, host: str | None = None) -> Path:
    return launch(
        wd,
        R1,
        [(0, str(pid))],
        executor="local",
        host=socket.gethostname() if host is None else host,
    )


@posix
def test_a_running_local_job_blocks(tmp_path):
    wd = tmp_path / "s"
    with _controller(tmp_path, _ours(wd)) as (proc, _):
        path = _local(wd, proc.pid)
        with pytest.raises(ValueError, match="queued or running") as caught:
            _run(wd)
        message = str(caught.value)
        assert str(proc.pid) in message and str(path) in message
        assert _written(wd) == [] and path.exists()
    # Ended and reaped: the run goes ahead, and removes the file.
    assert _run(wd)["v"].values.tolist() == [0.1, 0.2]
    assert not path.exists()


@posix
@pytest.mark.parametrize("holder", ["another program", "another sweep"])
def test_a_pid_another_process_now_holds_does_not_block(tmp_path, holder):
    """The system gives an ended job's pid to new processes: only a process
    running submitit's controller on the launch's folder is its job. A
    sibling folder (`.submitit2`) is not the launch's."""
    wd = tmp_path / "s"
    with (
        _controller(tmp_path, _ours(wd)) as (mine, _),
        _controller(tmp_path, _ours(wd) + "2") as (sibling, _),
    ):
        path = _local(wd, mine.pid)
        with pytest.raises(ValueError, match="queued or running"):
            _run(wd)
        path.unlink()
        path = _local(wd, os.getpid() if holder == "another program" else sibling.pid)
        assert _run(wd)["v"].values.tolist() == [0.1, 0.2]
        assert not path.exists()


@posix
def test_a_zombie_local_job_has_ended(tmp_path):
    """A job's controller that has exited stays in the process table until
    its parent reaps it, which the driver that submitted it may not do while
    it lives: it has ended."""
    wd = tmp_path / "s"
    with _controller(tmp_path, _ours(wd)) as (proc, release):
        path = _local(wd, proc.pid)
        with pytest.raises(ValueError, match="queued or running"):
            _run(wd)
        (read,), _ = _jobs.read_launches(wd)
        assert not _jobs.ended_locally(read)
        release.touch()
        assert _until(lambda: _zombie(proc.pid))
        # As `show` and `cancel` judge it, by the process's state alone.
        assert _jobs.ended_locally(read)
        assert _run(wd)["v"].values.tolist() == [0.1, 0.2]
        assert _zombie(proc.pid)  # the run did not reap it
        assert not path.exists()


@posix
def test_a_zombie_has_ended_whatever_its_command_line_reads(tmp_path, monkeypatch):
    """The process table's state, not only its command line, says a zombie
    has ended (some systems keep a zombie's command line)."""
    wd = tmp_path / "s"
    path = _local(wd, os.getpid())
    command = f"python -m submitit.local._local {_ours(wd)}/%j"
    _table(monkeypatch, lambda pids: {os.getpid(): ("S", command)})
    with pytest.raises(ValueError, match="queued or running"):
        _run(wd)
    _table(monkeypatch, lambda pids: {os.getpid(): ("Z", command)})
    assert _run(wd)["v"].values.tolist() == [0.1, 0.2]
    assert not path.exists()


@posix
def test_a_process_table_that_cannot_be_read_counts_as_running(tmp_path, monkeypatch):
    """A pid that exists, but whose process cannot be read, is not known to
    have ended."""
    wd = tmp_path / "s"
    path = _local(wd, os.getpid())
    _table(monkeypatch, lambda pids: None)
    with pytest.raises(ValueError, match="queued or running"):
        _run(wd)
    monkeypatch.undo()
    fail_closed(monkeypatch)
    assert _run(wd)["v"].values.tolist() == [0.1, 0.2]  # this process is not a job
    assert not path.exists()


@posix
def test_a_local_launch_from_another_host_warns_and_runs(tmp_path, monkeypatch):
    """A local job's pid means something only on the host that submitted
    it: elsewhere the run warns and goes ahead, and the file stays."""
    wd = tmp_path / "s"
    command = f"python -m submitit.local._local {_ours(wd)}/%j"
    _table(monkeypatch, lambda pids: {os.getpid(): ("S", command)})
    here = _local(wd, os.getpid())
    with pytest.raises(ValueError, match="queued or running"):
        _run(wd)
    here.unlink()
    path = _local(wd, os.getpid(), host=socket.gethostname() + "-other")
    with pytest.warns(UserWarning, match="-other") as caught:
        assert _run(wd)["v"].values.tolist() == [0.1, 0.2]
    assert any(str(path) in str(w.message) for w in caught)
    assert path.exists()


@posix
def test_show_and_cancel_count_a_zombie_as_ended(tmp_path, monkeypatch):
    """The process-table state is shared: `show` no longer lists the cells
    of a local launch whose jobs are zombies, and `cancel` removes it."""
    wd = tmp_path / "s"
    command = "python -m submitit.local._local /elsewhere/%j"
    path = _local(wd, os.getpid())
    _table(monkeypatch, lambda pids: {os.getpid(): ("S", command)})
    assert [r["status"] for r in mushin.show(wd).rows] == ["queued"]
    _table(monkeypatch, lambda pids: {os.getpid(): ("Z", command)})
    assert mushin.show(wd).rows == []
    mushin.cancel(wd)
    assert not path.exists()
