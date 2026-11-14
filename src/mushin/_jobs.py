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

`mushin.cancel` reads the same files to cancel the jobs a sweep left queued
or running (a driver killed mid-sweep leaves its files behind).

Standard library only: the launcher imports this module in the driver, `show`
on a login node, and `import mushin` loads it for `cancel`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

from ._slurm import _SCONTROL_TIMEOUT_S

log = logging.getLogger(__name__)

# How `cancel` finds and runs SLURM's commands: seams the tests replace, so
# no test needs SLURM or ever reaches a real one.
_which = shutil.which
_run = subprocess.run

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


def cancel(working_dir, *, dry_run: bool = False) -> list[str]:
    """Cancel a sweep's SLURM jobs that are still queued or running.

    Each submission of ``run(launcher="submitit_slurm")`` names the jobs it
    sent in a file under ``<working_dir>/mushin_jobs/``, which its driver
    removes once it has waited on them, so the files that remain name jobs
    that may not have ended: a running sweep's, or those of a driver killed
    mid-sweep (Ctrl-C, a lost login session). ``cancel`` asks SLURM about
    each of their job arrays and runs ``scancel`` once, for those still
    queued or running. Call it where ``scontrol`` and ``scancel`` reach the
    cluster, such as a login node, with the driver running or not.

    Parameters
    ----------
    working_dir : str or os.PathLike
        The sweep's ``working_dir``.
    dry_run : bool
        Return the ids it would cancel, and cancel and remove nothing.

    Returns
    -------
    list of str
        The ids cancelled, one per job array (or per job, for a submission
        of one cell); with ``dry_run``, the ids it would cancel.

    Raises
    ------
    FileNotFoundError
        ``working_dir`` is not a directory.
    RuntimeError
        ``scontrol`` or ``scancel`` is not found, SLURM's answer about a job
        cannot be read (it took too long, failed, or is not in the expected
        form), or ``scancel`` fails. In the first two cases nothing is
        cancelled, and the message names the ``scancel`` command to run by
        hand.

    Notes
    -----
    A job is cancelled only while SLURM lists it as not ended and its
    script or log paths lie under the sweep's submitit folder. A job id
    SLURM has since given another job is left alone, with a warning. A job
    SLURM no longer lists (it forgets one some minutes after it ends) is
    over.

    Once a submission's jobs are cancelled or over, its file is removed,
    so `mushin.show` no longer lists its cells as ``queued``: a cell whose
    job never started shows its dir's earlier record, or no row. A driver
    still running records the cancelled cells as failed, and a later
    ``resume=True`` runs them again.

    Two kinds of submission are not checked. A ``launcher="submitit_local"``
    job's id is a process id on the machine that submitted it, not SLURM's:
    it is skipped with a warning, and its file is removed only on that
    machine, once none of its processes runs. A submission sent to other
    clusters (``--clusters``) is out of plain ``scontrol``'s sight: the
    warning names the ``scancel --clusters=...`` command to run, and its
    file is removed.
    """
    from ._console import echo

    root = Path(working_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"mushin.cancel: {str(root)!r} is not a directory")
    launches, bad = read_launches(root)
    if bad:
        warnings.warn(
            f"mushin.cancel could not read the launch files {[p.name for p in bad]} "
            f"in {str(root / JOBS_DIR)!r}; the jobs they name, if any, are not "
            "cancelled.",
            UserWarning,
            stacklevel=2,
        )
    # Launch files to remove once every job is known to be over or
    # cancelled, and the SLURM launches to ask about, with their arrays.
    over: list[Path] = []
    asked: list[tuple[Launch, list[str]]] = []
    for launch in launches:
        # A task of an array is `<array>_<index>`; scancel takes the array.
        arrays = list(dict.fromkeys(c.job_id.split("_")[0] for c in launch.cells))
        if launch.executor == "local":
            # A process id, not SLURM's: never passed to scancel. On its
            # host, a launch none of whose processes lives is over.
            if ended_locally(launch):
                over.append(launch.path)
            else:
                warnings.warn(
                    f"mushin.cancel leaves the jobs of {launch.path.name} alone: "
                    "a submitit_local job's ids are process ids on the host that "
                    f"submitted it ({launch.host!r}), not SLURM jobs.",
                    UserWarning,
                    stacklevel=2,
                )
            continue
        if launch.clusters is not None:
            # Plain `scontrol` sees only the default cluster's jobs, so these
            # cannot be checked; the file goes with the others (nothing here
            # would ever retire it), and the user is told how to cancel them.
            # Warned now, so a call that raises later still names the command.
            warnings.warn(
                f"mushin.cancel cannot check the jobs of {launch.path.name}, sent to "
                f"SLURM clusters {launch.clusters!r}. If they are still queued or "
                f"running, cancel them with `scancel --clusters={launch.clusters} "
                f"{' '.join(arrays)}`.",
                UserWarning,
                stacklevel=2,
            )
            over.append(launch.path)
            continue
        asked.append((launch, arrays))

    ids = list(dict.fromkeys(a for _, arrays in asked for a in arrays))
    if not ids:
        if not dry_run:
            _forget_all(over)
        echo(f"mushin.cancel: nothing to cancel in {root}")
        return []
    scontrol, scancel = _which("scontrol"), _which("scancel")
    if scontrol is None or scancel is None:
        missing = "scontrol" if scontrol is None else "scancel"
        raise RuntimeError(
            f"mushin.cancel needs SLURM's scontrol and scancel, and finds no "
            f"{missing} here; nothing was cancelled. Where SLURM is, check that "
            "each job is this sweep's (`scontrol show job <id>`) and cancel them "
            f"with `scancel {' '.join(ids)}`."
        )

    answers = {array: _ask(scontrol, array) for array in ids}
    to_cancel: set[str] = set()
    for launch, arrays in asked:
        verdicts = {}
        for array in arrays:
            kind, detail = answers[array]
            verdicts[array] = _judge(detail, launch.folder) if kind == "shown" else kind
        for array, verdict in verdicts.items():
            if verdict == "live":
                to_cancel.add(array)
            elif verdict == "not ours":
                warnings.warn(
                    f"mushin.cancel leaves job {array} alone: SLURM lists it, but "
                    f"its script and logs are not under {launch.folder!r}, so it is "
                    "not this sweep's (SLURM may have given its id to another job). "
                    f"Its launch file {launch.path.name} is kept.",
                    UserWarning,
                    stacklevel=2,
                )
        # A launch with a job that is not this sweep's keeps its file: the
        # check may have misread a job that is, and the file still names it.
        if all(v in ("gone", "live") for v in verdicts.values()):
            over.append(launch.path)
    unknown = {a: detail for a, (kind, detail) in answers.items() if kind == "unknown"}
    if unknown:
        # Never names a job that is gone or not this sweep's.
        by_hand = [a for a in ids if a in to_cancel or a in unknown]
        why = "; ".join(f"job {a}: {reason}" for a, reason in unknown.items())
        raise RuntimeError(
            f"mushin.cancel could not check every job, so it cancelled nothing "
            f"({why}). Try again, or check them (`scontrol show job <id>`) and "
            f"cancel them by hand: `scancel {' '.join(by_hand)}`."
        )

    cancelled = [a for a in ids if a in to_cancel]
    if dry_run:
        if cancelled:
            echo(f"mushin.cancel would cancel: {' '.join(cancelled)} (dry run)")
        else:
            echo("mushin.cancel: no job of this sweep is queued or running (dry run)")
        return cancelled
    if cancelled:
        _scancel(scancel, cancelled)
    _forget_all(over)
    if cancelled:
        echo(f"mushin.cancel cancelled: {' '.join(cancelled)}")
    else:
        echo("mushin.cancel: no job of this sweep is queued or running")
    return cancelled


# The states of a job that has ended (SLURM's JOB STATE CODES). Any other,
# a state this list does not know included, counts as not ended. PREEMPTED is
# not here: SLURM may requeue such a job, and cancelling one that did end is
# harmless (scancel's "already completed" is tolerated).
_ENDED = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "REVOKED",
        "TIMEOUT",
    }
)
# A job's line of `scontrol --oneliner show job`, and its state. Fields are
# `Key=value` tokens: a field name starts the line or follows whitespace, so
# `ArrayJobId=` is not `JobId=`.
_JOB_LINE = re.compile(r"(?:^|\s)JobId=\S")
_STATE = re.compile(r"(?:^|\s)JobState=(\S+)")


def _ask(scontrol: str, array: str) -> tuple[str, Any]:
    """scontrol's account of job ``array``: ``("shown", lines)``, a line per
    job (or per run of pending array tasks); ``("gone", None)`` once SLURM
    has forgotten it; ``("unknown", why)`` for an answer that tells
    neither."""
    try:
        # `--all`, as `_slurm` asks: a job in a hidden partition is listed too.
        out = _run(
            [scontrol, "--all", "--oneliner", "show", "job", array],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SCONTROL_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "unknown", f"scontrol gave no answer in {_SCONTROL_TIMEOUT_S} s"
    except (OSError, subprocess.SubprocessError) as exc:
        return "unknown", f"scontrol could not run ({exc!r})"
    stdout, stderr = out.stdout or "", out.stderr or ""
    if out.returncode != 0:
        # SLURM's answer for a job it has purged, some minutes after it ended.
        if "invalid job id" in f"{stdout}\n{stderr}".lower():
            return "gone", None
        message = _first_line(stderr) or _first_line(stdout) or "no message"
        return "unknown", f"scontrol failed with exit code {out.returncode}: {message}"
    lines = [line for line in stdout.splitlines() if _JOB_LINE.search(line)]
    if not lines:
        return "unknown", f"scontrol listed no job: {_first_line(stdout) or 'nothing'}"
    if not all(_STATE.search(line) for line in lines):
        return "unknown", "scontrol listed a job with no JobState"
    return "shown", lines


def _judge(lines: list[str], folder: str) -> str:
    """``"live"``, ``"gone"`` or ``"not ours"``: what scontrol's ``lines``
    for one array say of a launch whose jobs submitit wrote under
    ``folder``."""
    if all(_STATE.search(line)[1] in _ENDED for line in lines):
        return "gone"
    # A folder at or just under the filesystem's root (`/`, `/scratch`) would
    # take nearly any job's paths as under it, so its jobs cannot be told from
    # another's: not cancelled. The check is only as strict as the folder:
    # one that several sweeps share cannot tell their jobs apart either.
    if len(PurePath(folder).parts) < 3:
        return "not ours"
    # Matched as text, so a folder holding a space matches, and on a path
    # boundary, so `/s/.submitit2` is not under `/s/.submitit`. `Command=` is
    # submitit's job script in that folder, there even when
    # `additional_parameters` moves the logs.
    under = re.escape(folder.rstrip("/") + "/")
    ours = re.compile(rf"(?:^|\s)(?:Command|StdOut|StdErr)={under}")
    return "live" if all(ours.search(line) for line in lines) else "not ours"


def _scancel(scancel: str, ids: list[str]) -> None:
    """Cancel ``ids``; raise when scancel fails for any but a job that ended
    since it was checked."""
    command = f"scancel {' '.join(ids)}"
    try:
        out = _run(
            [scancel, *ids],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SCONTROL_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"mushin.cancel: `{command}` did not finish in {_SCONTROL_TIMEOUT_S} s, "
            "so some jobs may not be cancelled. Check them with `squeue --me`, "
            "and call mushin.cancel again."
        ) from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"mushin.cancel: `{command}` could not run ({exc!r})."
        ) from exc
    if out.returncode == 0:
        return
    lines = [line for line in (out.stderr or "").splitlines() if line.strip()]
    # A job that ended between the check and scancel is not an error: SLURM
    # says its id is invalid (purged) or that it is already completing.
    ended = ("invalid job id", "already completing or completed")
    other = [line for line in lines if not any(e in line.lower() for e in ended)]
    if other or not lines:
        message = " ".join(other) or "no message"
        raise RuntimeError(
            f"mushin.cancel: `{command}` failed with exit code {out.returncode}: "
            f"{message}. Check the jobs with `squeue --me`."
        )


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _forget_all(paths: Iterable[Path]) -> None:
    for path in paths:
        forget_launch(path)
