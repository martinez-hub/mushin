# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Tasks for tests/test_submitit_local.py and test_submitit_launcher.py.

They live in a module of their own because submitit pickles a task by
reference: each job is a fresh `python -m submitit.core._submit` process that
imports this module by name, so it must be importable there (the tests put
tests/ on that process's PYTHONPATH).
"""

from __future__ import annotations

import os
import sys

from omegaconf import OmegaConf

import mushin


@mushin.sweep
def record_pid(seed):
    return dict(v=float(seed), pid=float(os.getpid()))


@mushin.sweep
def fail_on_one(seed):
    if seed == 1:
        raise RuntimeError("boom")
    return dict(v=float(seed))


@mushin.sweep
def die_on_two(seed):
    if seed == 2:
        # Ends the process with no Python error handling, as an out-of-memory
        # kill or `scancel` ends a job on a cluster.
        os._exit(9)
    return dict(v=float(seed))


@mushin.sweep
def die_loudly(seed):
    """Writes a long error stream, its cause at the end, then dies."""
    if seed == 1:
        sys.stderr.write("x" * 1_000_000 + "\nthe cause, last\n")
        sys.stderr.flush()
        os._exit(9)
    return dict(v=float(seed))


@mushin.sweep
def die_on_zero_and_two(seed):
    # The first job and the last die, so a launcher that stopped waiting at
    # the first dead job would lose the cell between them.
    if seed in (0, 2):
        os._exit(9)
    return dict(v=float(seed))


def die_plainly(seed):
    """Not a mushin task: a plain Hydra task function that dies."""
    if seed == 1:
        os._exit(9)
    return seed


@mushin.sweep
def die_when_told(seed):
    """Dies on the seed `MUSHIN_TEST_DIE` names: set for one run, unset for
    the resume that follows."""
    if os.environ.get("MUSHIN_TEST_DIE") == str(seed):
        os._exit(9)
    return dict(v=float(seed), pid=float(os.getpid()))


def _die_before_start(num):
    """A `hydra.sweep.subdir` resolver. Inside a job whose number is in
    `MUSHIN_TEST_DIE_BEFORE_START`, it ends the process while Hydra computes
    the cell's dir, before making it (and before mushin's runner starts).
    The driver evaluates the subdir too, outside any job, where it returns
    the default subdir."""
    doomed = os.environ.get("MUSHIN_TEST_DIE_BEFORE_START", "").split(",")
    if os.environ.get("SUBMITIT_EXECUTOR") and str(num) in doomed:
        os._exit(9)
    return str(num)


# Registered on import, which a job does when it unpickles a task from here.
OmegaConf.register_new_resolver(
    "mushin_test_die_before_start", _die_before_start, replace=True
)
# Quoted: Hydra's override grammar takes a nested interpolation only so.
DIE_BEFORE_START = (
    "hydra.sweep.subdir='${mushin_test_die_before_start:${hydra.job.num}}'"
)


@mushin.sweep
def second_task_dies(seed):
    """In a job of two tasks, the second dies on seed 1 after the first has
    started, and the first returns only once it has, so the cell's last
    record is the first task's completed one. Both tasks run mushin's runner
    in the cell's dir: submitit's local tasks set none of the variables
    `_multi_rank_world` reads a rank from."""
    import time

    import submitit

    started = f"rank1_started_{seed}"
    if submitit.JobEnvironment().global_rank == 1:
        open(started, "w").close()
        if seed == 1:
            os._exit(9)
    else:
        deadline = time.monotonic() + 50
        while not os.path.exists(started) and time.monotonic() < deadline:
            time.sleep(0.05)
    return dict(v=float(seed))


@mushin.sweep
def unpicklable_result(seed):
    """On seed 0, returns a value a JSON sidecar holds only as text and pickle
    cannot hold at all, so submitit cannot save the job's result."""
    import threading

    if seed == 0:
        return dict(v=float(seed), lock=threading.Lock())
    return dict(v=float(seed))


@mushin.sweep
def first_task_reports(seed):
    """In a job of two tasks, the first returns metrics at once and the
    second returns None once the first has written them."""
    import time

    import submitit

    if submitit.JobEnvironment().global_rank == 0:
        return dict(v=float(seed))
    deadline = time.monotonic() + 20  # well under the job's time limit
    while not os.path.exists("mushin_metrics.json") and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)  # past the first task's completed record too
    return None
