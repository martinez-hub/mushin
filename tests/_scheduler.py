# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A fake SLURM for the tests that read a sweep's launch files (#345):
`mushin.cancel` (tests/test_cancel.py) and `run()`'s refusal of a
`working_dir` whose earlier jobs may still run (tests/test_guard.py).

`_jobs._which` finds the commands and `_jobs._run` answers from a table and
records each call, so no SLURM is needed. Each module's fixtures call
`fail_closed`, autouse, so a test that did not set up the scheduler (with
`fake_slurm`) asks it nothing, not even where its commands are."""

from __future__ import annotations

import subprocess
import sys

from mushin import _jobs

R0, R1, R2 = "a" * 32, "b" * 32, "c" * 32  # run ids
SCONTROL, SCANCEL = "/opt/slurm/bin/scontrol", "/opt/slurm/bin/scancel"
TERMINAL = "COMPLETED"
PURGED = (1, "", "slurm_load_jobs error: Invalid job id specified\n")


def fail_closed(monkeypatch) -> None:
    """Make any question to the scheduler fail the test."""

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
        self.envs: list = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        self.envs.append(kwargs.get("env"))
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


def fake_slurm(monkeypatch) -> Scheduler:
    """Answer the scheduler's questions from a `Scheduler`'s tables."""
    sched = Scheduler()
    tools = {"scontrol": SCONTROL, "scancel": SCANCEL}
    monkeypatch.setattr(_jobs, "_which", tools.get)
    monkeypatch.setattr(_jobs, "_run", sched)
    return sched


def launch(wd, run, cells, *, executor="slurm", folder=None, **fields):
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


def job(state, folder, *, job="100", task=None, command=None, out=None, err=None):
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


def shown(*lines):
    return (0, "".join(line + "\n" for line in lines), "")


def dead_pid() -> int:
    """A process id that names no process: one this test started and
    reaped."""
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    return done.pid
