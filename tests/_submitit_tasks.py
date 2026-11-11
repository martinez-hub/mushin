# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Tasks for tests/test_submitit_local.py.

They live in a module of their own because submitit pickles a task by
reference: each job is a fresh `python -m submitit.core._submit` process that
imports this module by name, so it must be importable there (the tests put
tests/ on that process's PYTHONPATH).
"""

from __future__ import annotations

import os

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
        # kill or `scancel` ends a job on a cluster. The last seed of the tests'
        # grid, so the driver has every other job's result when it meets this
        # one, and no job is left running if it aborts there.
        os._exit(9)
    return dict(v=float(seed))
