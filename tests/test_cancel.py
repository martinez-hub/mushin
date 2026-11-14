# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.cancel(working_dir)` cancels a sweep's SLURM jobs that have not
ended (#345).

It reads the launch files mushin's submitit launchers write under
``<working_dir>/mushin_jobs/``. The scheduler is faked here: `_jobs._which`
finds the commands and `_jobs._run` answers from a table and records each
call, so no SLURM is needed, and any call a test did not expect fails it."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import pytest

import mushin
from mushin import _jobs

R0, R1, R2 = "a" * 32, "b" * 32, "c" * 32  # run ids
SCONTROL, SCANCEL = "/opt/slurm/bin/scontrol", "/opt/slurm/bin/scancel"
TERMINAL = "COMPLETED"
PURGED = (1, "", "slurm_load_jobs error: Invalid job id specified\n")


@pytest.fixture(autouse=True)
def _no_scheduler(monkeypatch):
    """Fail closed: a test that did not set up the scheduler asks it
    nothing, not even where its commands are."""

    def asked(*args, **kwargs):
        raise AssertionError(f"the scheduler was asked: {args!r}")

    monkeypatch.setattr(_jobs, "_run", asked)
    monkeypatch.setattr(_jobs, "_which", asked)


class Scheduler:
    """`_jobs._run` answering from tables. ``jobs[id]`` is scontrol's answer
    for ``id`` and ``scancel_answer`` scancel's: ``(exit code, stdout,
    stderr)``, or an exception to raise. An id not in ``jobs`` fails the
    test. Every command is recorded in ``calls``."""

    def __init__(self):
        self.jobs: dict = {}
        self.scancel_answer = (0, "", "")
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        assert kwargs.get("timeout") and kwargs.get("check") is False
        if argv[0] == SCONTROL:
            *command, job_id = argv
            assert command == [SCONTROL, "--all", "--oneliner", "show", "job"]
            answer = self.jobs[job_id]
        elif argv[0] == SCANCEL:
            answer = self.scancel_answer
        else:
            raise AssertionError(f"unexpected command {argv!r}")
        if isinstance(answer, BaseException):
            raise answer
        code, out, err = answer
        return subprocess.CompletedProcess(argv, code, out, err)

    @property
    def asked(self) -> list[str]:
        """The ids scontrol was asked about, in order."""
        return [argv[-1] for argv in self.calls if argv[0] == SCONTROL]

    @property
    def scancels(self) -> list[list[str]]:
        """The ids of each scancel call."""
        return [argv[1:] for argv in self.calls if argv[0] == SCANCEL]


@pytest.fixture
def slurm(monkeypatch):
    sched = Scheduler()
    tools = {"scontrol": SCONTROL, "scancel": SCANCEL}
    monkeypatch.setattr(_jobs, "_which", tools.get)
    monkeypatch.setattr(_jobs, "_run", sched)
    return sched


def _launch(wd, run, cells, *, executor="slurm", folder=None, **fields):
    """A launch file as the launcher writes it, naming ``cells`` (``(num,
    job_id)``)."""
    defaults = dict(clusters=None, host="login1", depth=1)
    return _jobs.record_launch(
        wd,
        run_id=run,
        executor=executor,
        folder=str(wd / ".submitit") if folder is None else folder,
        cells=[_jobs.LaunchCell(n, j, {"lr": n / 10}) for n, j in cells],
        **{**defaults, **fields},
    )


def _job(state, folder, *, job="100", task=None, command=None, out=None, err=None):
    """One line of `scontrol --oneliner show job <job>`, as SLURM prints it,
    for a job submitit wrote under ``folder``: task ``task`` of array
    ``job`` (an index, or a range of pending tasks shown as one line), or
    the plain job ``job`` when ``task`` is None."""
    if task is None:
        ids = [f"JobId={job}"]
        logs = f"{folder}/%j/%j_0_log"
    else:
        # A started task has an id of its own; pending ones show the array's.
        own = str(task).isdigit()
        ids = [
            f"JobId={int(job) + 1 + int(task) if own else job}",
            f"ArrayJobId={job}",
            f"ArrayTaskId={task}",
        ]
        logs = f"{folder}/%A_%a/%A_%a_0_log"
    fields = [
        *ids,
        "JobName=submitit",
        "UserId=me(1000)",
        "Priority=1",
        f"JobState={state}",
        "Reason=None",
        "TimeLimit=01:00:00",
        f"Command={command or folder + '/.submission_file_0f3c.sh'}",
        "WorkDir=/home/me",
        f"StdErr={err or logs + '.err'}",
        "StdIn=/dev/null",
        f"StdOut={out or logs + '.out'}",
        "Power=",
    ]
    return " ".join(fields)


def _shown(*lines):
    return (0, "".join(line + "\n" for line in lines), "")


def _dead_pid() -> int:
    """A process id that names no process: one this test started and
    reaped."""
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    return done.pid


def test_no_launch_files(tmp_path, capsys):
    assert mushin.cancel(tmp_path) == []
    assert "nothing to cancel" in capsys.readouterr().out


def test_a_missing_working_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mushin.cancel(tmp_path / "nope")


def test_names_off_the_pattern_are_ignored(tmp_path, capsys):
    """A writer's temporary sibling and any other file are not launches."""
    jobs = tmp_path / "mushin_jobs"
    jobs.mkdir()
    others = [jobs / f"{R1}.0.json.123.abc.tmp", jobs / "notes.json"]
    for path in others:
        path.write_text("{", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mushin.cancel(tmp_path) == []
    assert all(path.exists() for path in others)
    assert "nothing to cancel" in capsys.readouterr().out


def test_cancels_only_live_arrays_of_the_sweep(tmp_path, slurm, capsys):
    folder = str(tmp_path / ".submitit")
    live = _launch(tmp_path, R1, [(0, "100_0"), (1, "100_1"), (2, "100_2")])
    ended = _launch(tmp_path, R1, [(3, "200")])
    slurm.jobs["100"] = _shown(
        _job("RUNNING", folder, task=0),
        _job("PENDING", folder, task="1-2"),
    )
    slurm.jobs["200"] = _shown(_job(TERMINAL, folder, job="200"))
    assert mushin.cancel(tmp_path) == ["100"]
    assert slurm.asked == ["100", "200"]
    assert slurm.scancels == [["100"]]
    assert not live.exists() and not ended.exists()
    assert "100" in capsys.readouterr().out


def test_output_moved_by_additional_parameters(tmp_path, slurm):
    """`additional_parameters={"output": ...}` moves StdOut and StdErr out of
    submitit's folder; the job script, `Command=`, stays in it."""
    folder = str(tmp_path / ".submitit")
    _launch(tmp_path, R1, [(0, "100_0")])
    slurm.jobs["100"] = _shown(
        _job(
            "RUNNING",
            folder,
            task=0,
            out="/elsewhere/100.out",
            err="/elsewhere/100.err",
        )
    )
    assert mushin.cancel(tmp_path) == ["100"]
    assert slurm.scancels == [["100"]]


@pytest.mark.parametrize("where", ["another sweep", "a sibling folder"])
def test_a_reused_id_is_left_alone(tmp_path, slurm, where):
    """A job id SLURM gave another job since: none of its paths lie under the
    launch's folder (`/s/.submitit2` is not under `/s/.submitit`), so it is
    not cancelled, and its launch is kept. A launch whose job is gone beside
    it is removed."""
    folder = str(tmp_path / ".submitit")
    other = "/home/you/.submitit" if where == "another sweep" else folder + "2"
    reused = _launch(tmp_path, R1, [(0, "100_0"), (1, "100_1")])
    purged = _launch(tmp_path, R2, [(2, "300")])
    slurm.jobs["100"] = _shown(_job("RUNNING", other, task=0))
    slurm.jobs["300"] = PURGED
    with pytest.warns(UserWarning, match="not this sweep's") as caught:
        assert mushin.cancel(tmp_path) == []
    assert "100" in str(caught[0].message)
    assert slurm.scancels == []
    assert reused.exists() and not purged.exists()


def test_a_folder_with_a_space(tmp_path, slurm):
    wd = tmp_path / "my sweep"
    wd.mkdir()
    folder = str(wd / ".submitit")
    _launch(wd, R1, [(0, "100_0")])
    slurm.jobs["100"] = _shown(_job("RUNNING", folder, task=0))
    assert mushin.cancel(wd) == ["100"]


def test_a_collapsed_pending_array(tmp_path, slurm):
    """scontrol shows the pending tasks of an array as one line."""
    folder = str(tmp_path / ".submitit")
    _launch(tmp_path, R1, [(n, f"100_{n}") for n in range(100)])
    slurm.jobs["100"] = _shown(_job("PENDING", folder, task="1-99"))
    assert mushin.cancel(tmp_path) == ["100"]
    assert slurm.scancels == [["100"]]


def test_a_purged_id_is_gone(tmp_path, slurm, capsys):
    """SLURM forgets a job some minutes after it ends: scontrol then calls
    its id invalid. The job is over, and so is its launch."""
    path = _launch(tmp_path, R1, [(0, "100_0")])
    slurm.jobs["100"] = PURGED
    assert mushin.cancel(tmp_path) == []
    assert slurm.scancels == []
    assert not path.exists()
    assert "no job" in capsys.readouterr().out


@pytest.mark.parametrize(
    "answer",
    [
        subprocess.TimeoutExpired([SCONTROL], 10),
        OSError("exec format error"),
        (1, "", "slurm_load_jobs error: Unable to contact slurm controller\n"),
        (0, "", ""),
        (0, "No jobs in the system\n", ""),
        (0, "JobId=200 ArrayJobId=200 JobName=submitit\n", ""),  # no JobState
    ],
    ids=["timeout", "oserror", "error", "empty", "no-JobId", "no-JobState"],
)
def test_an_unverifiable_id_cancels_nothing(tmp_path, slurm, answer):
    """One id the scheduler does not answer for: nothing is cancelled and no
    launch removed, though the other id is live and another launch would
    be removed. The error names the id and the command to run by hand,
    which leaves out a job that is gone and one that is not the sweep's."""
    folder = str(tmp_path / ".submitit")
    paths = [
        _launch(tmp_path, R1, [(0, "100_0")]),
        _launch(tmp_path, R1, [(1, "200_0")]),
        _launch(tmp_path, R2, [(2, "300")]),
        _launch(tmp_path, R2, [(3, "400")], clusters="gpu2"),
        _launch(tmp_path, R2, [(4, "500")]),
    ]
    slurm.jobs["100"] = _shown(_job("RUNNING", folder, task=0))
    slurm.jobs["200"] = answer
    slurm.jobs["300"] = PURGED
    slurm.jobs["500"] = _shown(_job("RUNNING", "/home/you/.submitit", job="500"))
    with pytest.raises(RuntimeError, match="200") as raised:
        with pytest.warns(UserWarning) as caught:
            mushin.cancel(tmp_path)
    assert "`scancel 100 200`" in str(raised.value)
    messages = [str(w.message) for w in caught]
    assert any("--clusters" in m for m in messages)
    assert any("not this sweep's" in m for m in messages)
    assert not any("removed" in m for m in messages)
    assert slurm.scancels == []
    assert all(path.exists() for path in paths)


@pytest.mark.parametrize("missing", ["scontrol", "scancel"])
def test_a_missing_command_names_the_scancel_to_run(tmp_path, monkeypatch, missing):
    paths = [
        _launch(tmp_path, R1, [(0, "100_0"), (1, "100_1")]),
        _launch(tmp_path, R1, [(2, "200")]),
    ]
    tools = {"scontrol": SCONTROL, "scancel": SCANCEL}
    del tools[missing]
    monkeypatch.setattr(_jobs, "_which", tools.get)
    with pytest.raises(RuntimeError, match=f"finds no {missing}") as raised:
        mushin.cancel(tmp_path)
    assert "`scancel 100 200`" in str(raised.value)
    assert all(path.exists() for path in paths)


def test_dry_run(tmp_path, slurm, capsys):
    folder = str(tmp_path / ".submitit")
    paths = [
        _launch(tmp_path, R1, [(0, "100_0")]),
        _launch(tmp_path, R1, [(1, "200")]),
        _launch(tmp_path, R2, [(2, "300")], clusters="gpu2"),
    ]
    slurm.jobs["100"] = _shown(_job("PENDING", folder, task=0))
    slurm.jobs["200"] = PURGED
    with pytest.warns(UserWarning, match="--clusters"):
        assert mushin.cancel(tmp_path, dry_run=True) == ["100"]
    assert slurm.asked == ["100", "200"] and slurm.scancels == []
    assert all(path.exists() for path in paths)
    assert "100" in capsys.readouterr().out


def test_scancel_naming_an_invalid_id_is_not_an_error(tmp_path, slurm):
    """A job that ended between the check and scancel."""
    folder = str(tmp_path / ".submitit")
    path = _launch(tmp_path, R1, [(0, "100_0")])
    slurm.jobs["100"] = _shown(_job("RUNNING", folder, task=0))
    slurm.scancel_answer = (
        1,
        "",
        "scancel: error: Kill job error on job id 100: Invalid job id specified\n",
    )
    assert mushin.cancel(tmp_path) == ["100"]
    assert not path.exists()


@pytest.mark.parametrize(
    "answer",
    [
        (1, "", "scancel: error: Unable to contact slurm controller\n"),
        (1, "", ""),
        subprocess.TimeoutExpired([SCANCEL], 10),
    ],
    ids=["error", "silent", "timeout"],
)
def test_a_scancel_that_fails_raises_and_keeps_the_launches(tmp_path, slurm, answer):
    folder = str(tmp_path / ".submitit")
    paths = [
        _launch(tmp_path, R1, [(0, "100_0")]),
        _launch(tmp_path, R1, [(1, "200")], clusters="gpu2"),
    ]
    slurm.jobs["100"] = _shown(_job("RUNNING", folder, task=0))
    slurm.scancel_answer = answer
    with pytest.raises(RuntimeError, match="scancel"):
        with pytest.warns(UserWarning, match="--clusters") as caught:
            mushin.cancel(tmp_path)
    assert not any("removed" in str(w.message) for w in caught)
    assert slurm.scancels == [["100"]]
    assert all(path.exists() for path in paths)


def test_a_local_launch_is_left_to_its_host(tmp_path, slurm):
    """A local launch's job ids are process ids on the driver's host, never
    SLURM's: one whose process lives, or whose host is another, is skipped
    with a warning and kept. A SLURM launch beside it is still cancelled."""
    import socket

    folder = str(tmp_path / ".submitit")
    here = socket.gethostname()
    local = [
        _launch(tmp_path, R0, [(0, str(os.getpid()))], executor="local", host=here),
        _launch(tmp_path, R0, [(1, "4242")], executor="local", host=here + "-other"),
    ]
    _launch(tmp_path, R1, [(2, "100_0")])
    slurm.jobs["100"] = _shown(_job("RUNNING", folder, task=0))
    with pytest.warns(UserWarning, match="process ids") as caught:
        assert mushin.cancel(tmp_path) == ["100"]
    assert len([w for w in caught if "process ids" in str(w.message)]) == 2
    assert slurm.asked == ["100"] and slurm.scancels == [["100"]]
    assert all(path.exists() for path in local)


@pytest.mark.skipif(sys.platform == "win32", reason="a pid is probed on POSIX only")
def test_an_ended_local_launch_is_dropped(tmp_path, capsys):
    """On its host, a local launch none of whose processes lives is over (a
    killed driver's): it is removed, with no warning and no scheduler
    asked."""
    import socket

    path = _launch(
        tmp_path,
        R0,
        [(0, str(_dead_pid()))],
        executor="local",
        host=socket.gethostname(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mushin.cancel(tmp_path) == []
    assert not path.exists()
    assert "nothing to cancel" in capsys.readouterr().out


def test_a_clusters_launch_is_left_to_the_user(tmp_path):
    """Plain scontrol does not see another cluster's jobs, so a launch sent
    with `--clusters` is not checked: the warning names the command, and
    the launch is removed."""
    path = _launch(tmp_path, R1, [(0, "100_0"), (1, "100_1")], clusters="gpu2")
    with pytest.warns(UserWarning, match=r"scancel --clusters=gpu2 100\b"):
        assert mushin.cancel(tmp_path) == []
    assert not path.exists()


def test_an_unreadable_launch_warns_and_the_rest_proceed(tmp_path, slurm):
    folder = str(tmp_path / ".submitit")
    bad = tmp_path / "mushin_jobs" / f"{R0}.0.json"
    _launch(tmp_path, R1, [(1, "100_0")])
    bad.write_text("{not json", encoding="utf-8")
    slurm.jobs["100"] = _shown(_job("RUNNING", folder, task=0))
    with pytest.warns(UserWarning, match=bad.name):
        assert mushin.cancel(tmp_path) == ["100"]
    assert bad.exists()


def test_a_one_cell_launch_is_a_plain_job(tmp_path, slurm):
    folder = str(tmp_path / ".submitit")
    _launch(tmp_path, R1, [(4, "300")])
    slurm.jobs["300"] = _shown(_job("PENDING", folder, job="300"))
    assert mushin.cancel(tmp_path) == ["300"]
    assert slurm.asked == ["300"] and slurm.scancels == [["300"]]


def test_show_after_cancel(tmp_path, slurm):
    """Once `cancel` removes a launch, `show` lists a cell that never started
    as its dir's earlier record, or not at all, instead of `queued`."""
    from mushin._resume import STATUS_FILE, write_cell_status

    folder = str(tmp_path / ".submitit")
    earlier = tmp_path / "0"
    earlier.mkdir()
    write_cell_status(
        earlier,
        status="failed",
        combo={"lr": 0.0},
        attempt=1,
        job_num=0,
        choices={},
        run_id=R0,
        job_id="90_0",
    )
    old = 1_700_000_000 * 10**9
    os.utime(earlier / STATUS_FILE, ns=(old, old))
    _launch(tmp_path, R1, [(0, "100_0"), (1, "100_1")])
    statuses = [(r["lr"], r["status"]) for r in mushin.show(tmp_path).rows]
    assert statuses == [(0.0, "queued"), (0.1, "queued")]
    slurm.jobs["100"] = _shown(_job("PENDING", folder, task="0-1"))
    assert mushin.cancel(tmp_path) == ["100"]
    statuses = [(r["lr"], r["status"]) for r in mushin.show(tmp_path).rows]
    assert statuses == [(0.0, "failed")]
