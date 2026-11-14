# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The launch files of mushin's submitit launchers (#345).

A cell's record exists only once its job has started, so nothing in a cell's
dir tells a job still in the scheduler's queue. Each submission therefore
writes one file, ``<working_dir>/mushin_jobs/<run id>.<first job number>.json``,
naming the jobs it sent, and its driver removes the file once it has waited on
them. `mushin.show` reads these files to list queued cells.

A driver writes only its own files and never rewrites one: there is no
read-modify-write anywhere, so two drivers on one ``working_dir`` (a resume
while an earlier run's jobs still wait) need no lock. A file is written once,
by `_atomic_write_json` (a temporary file, then a rename), so its mtime is the
submission time on the filesystem clock that also stamps the cells' records.

Standard library only: the launcher imports this module in the driver, and
`show` on a login node.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

log = logging.getLogger(__name__)

#: The dir under a sweep's ``working_dir`` that holds its launch files. It has
#: no status record, so whatever walks the sweep root for cells skips it.
JOBS_DIR = "mushin_jobs"

#: The launch file format; a reader ignores any other.
SCHEMA = 1

# A launch file's name; a writer's temporary sibling (``<name>.<pid>.<hex>.tmp``)
# does not match, and is not read.
_NAME = re.compile(r"(?P<run_id>[0-9a-f]{32})\.\d+\.json")
# A SLURM job id (``<array>_<index>`` for an array task) or a local process
# id: digits only, so an id never reaches a command line as anything else.
_JOB_ID = re.compile(r"\d+(?:_\d+)?")
_EXECUTORS = frozenset({"slurm", "local"})


@dataclass(frozen=True)
class LaunchCell:
    """One job a launch submitted: the cell's Hydra job number ``num``, the
    scheduler's ``job_id``, and ``combo``, the cell's swept params as its
    overrides give them (for display; a record's combo may differ, as a
    typed field converts a value)."""

    num: int
    job_id: str
    combo: dict


@dataclass(frozen=True)
class Launch:
    """A launch file as read: ``path``, the ``run_id`` of the ``run()`` that
    submitted it, its submitit ``executor`` (``"slurm"`` or ``"local"``),
    submitit's id-independent log ``folder``, the SLURM ``clusters`` the
    jobs were sent to (None for the default one), the driver's ``host``, the
    ``depth`` of the cells' dirs below the sweep root (`hydra.sweep.subdir`'s
    parts), the file's ``mtime_ns`` and its ``cells``."""

    path: Path
    run_id: str
    executor: str
    folder: str
    clusters: str | None
    host: str
    depth: int
    mtime_ns: int
    cells: tuple[LaunchCell, ...]


def record_launch(
    root,
    *,
    run_id: str,
    executor: str,
    folder: str,
    clusters: str | None,
    host: str,
    depth: int,
    cells: Iterable[LaunchCell],
) -> Path:
    """Write the launch file for ``cells``, the jobs one submission sent, in
    submission order, and return its path. Named by the first job's number:
    Hydra numbers each job once per run, so no two launches of a run share
    a name."""
    from ._sweep_io import _atomic_write_json, _json_normalize

    cells = list(cells)
    payload = {
        "schema": SCHEMA,
        "run_id": run_id,
        "executor": executor,
        "folder": folder,
        "clusters": clusters,
        "host": host,
        "depth": depth,
        "cells": [
            {
                "num": cell.num,
                "job_id": cell.job_id,
                # As a record stores its combo: a NaN tagged, as a null is
                # a swept None.
                "combo": _json_normalize(cell.combo, tag_nan=True),
            }
            for cell in cells
        ],
    }
    path = Path(root) / JOBS_DIR / f"{run_id}.{cells[0].num}.json"
    _atomic_write_json(path, payload)
    return path


def forget_launch(path) -> None:
    """Remove a launch file, if it is still there. Its driver does once it
    has waited on the launch's jobs; a failure is logged, since a sweep
    must not fail over a view of its queue."""
    from ._sweep_io import _while_held

    try:
        # On Windows, a `show` with the file open makes the unlink raise
        # PermissionError for a moment; retried as for a sidecar (#357).
        _while_held(lambda: Path(path).unlink(missing_ok=True))
    except OSError as exc:
        log.warning(
            f"mushin could not remove the launch file {path} ({exc!r}); "
            "`mushin.show` may list its cells as queued."
        )


def read_launches(root) -> tuple[list[Launch], list[Path]]:
    """The launch files under sweep root ``root``: those that are valid, and
    the paths of those that are not (unreadable, or of another shape). A
    file removed between the listing and its read is neither: its launch is
    over."""
    from ._sweep_io import restore_combo_value

    folder = Path(root) / JOBS_DIR
    try:
        paths = sorted(p for p in folder.iterdir() if _NAME.fullmatch(p.name))
    except OSError:  # no launch was ever recorded here
        return [], []
    launches, bad = [], []
    for path in paths:
        try:
            mtime_ns = path.stat().st_mtime_ns
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, RecursionError):
            bad.append(path)
            continue
        launch = _launch(path, data, mtime_ns, restore_combo_value)
        if launch is None:
            bad.append(path)
        else:
            launches.append(launch)
    return launches, bad


def _launch(path: Path, data: Any, mtime_ns: int, restore) -> Launch | None:
    """``data`` as a Launch, or None when it is not one as written."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    run_id, cells = data.get("run_id"), data.get("cells")
    folder, clusters = data.get("folder"), data.get("clusters")
    host, depth = data.get("host"), data.get("depth")
    if not (
        run_id == _NAME.fullmatch(path.name)["run_id"]
        and data.get("executor") in _EXECUTORS
        and isinstance(folder, str)
        and _absolute(folder)
        and (clusters is None or (isinstance(clusters, str) and clusters))
        and isinstance(host, str)
        and host
        and type(depth) is int
        and depth >= 1
        and isinstance(cells, list)
        and cells
    ):
        return None
    read, nums = [], set()
    for cell in cells:
        if not isinstance(cell, dict):
            return None
        num, job_id, combo = cell.get("num"), cell.get("job_id"), cell.get("combo")
        if not (
            type(num) is int  # not a bool
            and num >= 0
            and num not in nums
            and isinstance(job_id, str)
            and _JOB_ID.fullmatch(job_id)
            and isinstance(combo, dict)
        ):
            return None
        nums.add(num)
        try:
            combo = {k: restore(v) for k, v in combo.items()}
        except RecursionError:  # nested too deep to restore: as stored
            pass
        read.append(LaunchCell(num, job_id, combo))
    return Launch(
        path=path,
        run_id=run_id,
        executor=data["executor"],
        folder=folder,
        clusters=clusters,
        host=host,
        depth=depth,
        mtime_ns=mtime_ns,
        cells=tuple(read),
    )


def _absolute(folder: str) -> bool:
    # A POSIX path or a Windows one, whichever system reads it: a cluster's
    # sweep dir may be read from a desktop over a network share.
    return PurePosixPath(folder).is_absolute() or PureWindowsPath(folder).is_absolute()


def ended_locally(launch: Launch) -> bool:
    """Whether ``launch`` is a local one, submitted from this host, none of
    whose jobs still runs. A local job's id is its controller's process id
    on the driver's host, so there, and only there, a launch whose
    processes have all exited is over, though no driver removed its file
    (the driver was killed). A process that has exited but that its parent
    has not reaped still counts as running, as does a pid the system has
    since given another process: the launch is then read as before."""
    import socket  # here, not at the top: the module stays cheap to import

    if launch.executor != "local" or launch.host != socket.gethostname():
        return False
    # On Windows `os.kill` terminates the process it is given: never probe
    # there (submitit's local executor does not run there either).
    if sys.platform == "win32":
        return False
    # A local job's id is a bare pid; anything else cannot be probed.
    if not all(cell.job_id.isdigit() for cell in launch.cells):
        return False
    return not any(_alive(int(cell.job_id)) for cell in launch.cells)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Not known to have exited: a process of another user (PermissionError),
    # or an id no process can have, which only a hand-edited file holds.
    except (OSError, OverflowError):
        return True
    return True
