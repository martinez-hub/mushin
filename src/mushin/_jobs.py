# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The launch files of mushin's submitit launchers (#345).

A cell's record exists only once its job has started, so nothing in a cell's
dir tells a job still in the scheduler's queue. Each submission therefore
writes one file, ``<working_dir>/mushin_jobs/<run id>.<first job number>.json``,
naming the jobs it sent, and its driver removes the file once it has waited on
them. `mushin.show` reads these files to list queued cells.

A driver writes only its own files and never rewrites one: there is no
read-modify-write anywhere, so the files need no lock even when two drivers
share one ``working_dir``. A file is written once, by `_atomic_write_json` (a
temporary file, then a rename), so its mtime is the submission time on the
filesystem clock that also stamps the cells' records.

`mushin.cancel` reads the same files to cancel the jobs a sweep left queued
or running (a driver killed mid-sweep leaves its files behind), and `run()`
reads them to refuse a ``working_dir`` whose earlier jobs may still run
(`refuse_live_launches`): two drivers would each submit the sweep's cells,
and two jobs would then write one cell.

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
    parts), the file's ``mtime_ns``, its ``cells``, and the ``tasks`` of each
    job (1 for a file from before the field)."""

    path: Path
    run_id: str
    executor: str
    folder: str
    clusters: str | None
    host: str
    depth: int
    mtime_ns: int
    cells: tuple[LaunchCell, ...]
    tasks: int = 1


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
    tasks: int = 1,
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
        "tasks": tasks,
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
    tasks = data.get("tasks", 1)
    if not (
        type(tasks) is int
        and tasks >= 1
        and run_id == _NAME.fullmatch(path.name)["run_id"]
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
        tasks=tasks,
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
    has not reaped (a zombie) has ended. A pid the system has since given
    another process still counts as running: the launch is then read as
    before."""
    return launch.executor == "local" and _local_state(launch, launch.cells) == "ended"


def _local_state(launch: Launch, cells, *, own_command: bool = False) -> str:
    """``"ended"`` when none of ``cells``' jobs of local ``launch`` still
    runs, ``"live"`` when one may, and ``"unknown"`` where that cannot be
    told: on another host than the one that submitted it, on Windows, or
    for an id that is not a pid. With ``own_command``, a process counts as
    a job only if its command line is the launch's controller's."""
    import socket  # here, not at the top: the module stays cheap to import

    if launch.host != socket.gethostname():
        return "unknown"
    # On Windows `os.kill` terminates the process it is given: never probe
    # there (submitit's local executor does not run there either).
    if sys.platform == "win32":
        return "unknown"
    # A local job's id is a bare pid; anything else cannot be probed.
    if not all(cell.job_id.isdigit() for cell in cells):
        return "unknown"
    pids = [int(cell.job_id) for cell in cells]
    folder = launch.folder if own_command else None
    return "live" if _any_alive(pids, folder) else "ended"


#: The module a submitit_local job's controller runs, as submitit's
#: `start_controller` starts it: `python -m submitit.local._local <folder>`,
#: the executor's folder (``<working_dir>/.submitit/%j`` by default), which
#: lies under the launch's ``folder``.
_LOCAL_CONTROLLER = "submitit.local._local"
#: How long reading the process table with ``ps`` may take.
_PS_TIMEOUT_S = 10


def _any_alive(pids: list[int], folder: str | None) -> bool:
    """Whether any of ``pids`` names a process that still runs: one that
    exists and is not a zombie. A job's controller is a child of the driver
    that submitted it, so once it exits it stays a zombie until that driver
    reaps it, which a driver interrupted mid-wait in a live process (Ctrl-C
    in a notebook) may never do. It is not reaped here either: that would
    take the exit status from the driver's own handle on it. With
    ``folder``, a process counts only if its command line runs submitit's
    local controller on a path under ``folder``, since the system may have
    given an ended job's pid to any process. A pid whose process cannot be
    read counts as running: it is not known to have ended."""
    existing = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass  # another user's process: it exists
        # An id no process can have, which only a hand-edited file holds.
        except (OSError, OverflowError):
            return True
        existing.append(pid)
    if not existing:
        return False
    found = _processes(existing)
    if found is None:
        return True
    for pid in existing:
        if pid not in found:
            continue  # it exited after `os.kill` found it
        state, command = found[pid]
        if state.startswith("Z"):
            continue
        if folder is None or _runs_controller(command, folder):
            return True
    return False


def _processes(pids: list[int]) -> dict[int, tuple[str, str]] | None:
    """Each of ``pids`` that names a process, mapped to its state code
    (``Z`` for a zombie) and its command line, its arguments joined by
    spaces; None when the process table cannot be read. From ``/proc`` on
    Linux; elsewhere (macOS) from ``ps``, run here rather than through
    `_run`, which is the scheduler's (the tests fail it closed)."""
    if sys.platform.startswith("linux") and os.path.isdir("/proc/self"):
        found = {}
        for pid in pids:
            try:
                stat = Path(f"/proc/{pid}/stat").read_text(
                    encoding="utf-8", errors="replace"
                )
                argv = Path(f"/proc/{pid}/cmdline").read_bytes()
            except (FileNotFoundError, ProcessLookupError):
                continue  # it has exited
            except OSError:
                return None
            # `<pid> (<name>) <state> ...`: the name may hold spaces and
            # parentheses, so the state is the first field after the last `)`.
            state = stat.rpartition(")")[2].split()[:1]
            if not state:
                return None
            # Arguments end in NUL; a zombie's command line is empty.
            command = argv.replace(b"\0", b" ").decode("utf-8", "replace").strip()
            found[pid] = (state[0], command)
        return found
    try:
        out = subprocess.run(
            # A flag per column: POSIX leaves a comma after `=` open to
            # reading as part of the header.
            ["ps", "-ww", "-o", "pid=", "-o", "stat=", "-o", "command="]
            + ["-p", ",".join(map(str, pids))],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # ps exits 1, printing nothing, when none of the pids names a process.
    if out.returncode not in (0, 1) or out.stderr.strip():
        return None
    found = {}
    for line in out.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 2 or not fields[0].isdigit():
            return None
        # A zombie's command reads `<defunct>` on macOS.
        found[int(fields[0])] = (fields[1], fields[2] if len(fields) > 2 else "")
    return found


def _runs_controller(command: str, folder: str) -> bool:
    """Whether ``command`` runs submitit's local controller on a path under
    ``folder``. Read as text: ``ps`` joins the arguments with spaces, and
    with submitit's ``setup`` the process is a shell given the controller's
    command as one argument, its folder quoted. On a path boundary, so
    ``/s/.submitit2`` is not under ``/s/.submitit``."""
    if _LOCAL_CONTROLLER not in command:
        return False
    under = re.escape(folder.rstrip("/"))
    return re.search(rf"{under}(?:/|['\"\s]|$)", command) is not None


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
        "LAUNCH_FAILED",
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
            env=_scheduler_env(),
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


def _scheduler_env() -> dict[str, str]:
    """The environment for ``scontrol`` and ``scancel``: this process's,
    without what changes the jobs they act on. A shell's ``SCANCEL_STATE``
    or ``SCANCEL_PARTITION`` filters the ids scancel is given, so a job the
    check found live would stay running with no error, and ``SLURM_CLUSTERS``
    sends both commands to another cluster, where the sweep's ids read as
    gone. ``SCONTROL_*`` goes too, for what scontrol reports (#405)."""
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SCANCEL_", "SCONTROL_")) and k != "SLURM_CLUSTERS"
    }


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
            env=_scheduler_env(),
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


#: A record's statuses once its job has run its cell to an end, or its
#: driver has recorded the job's end.
_RECORDED_ENDS = frozenset({"completed", "failed", "skipped"})


def refuse_live_launches(working_dir) -> list[Path]:
    """Refuse a run into ``working_dir`` while jobs an earlier run submitted
    there may still be queued or running (#345). Return the launch files of
    those found over, for `run()` to remove once its own launch has returned:
    the new run has then superseded them, and nothing else would remove
    them, so every later run would ask about them again.

    Two drivers on one sweep (a resume started while the earlier run's jobs
    still wait, or the same script run twice) each submit the cells they
    find unfinished, and two jobs then write one cell. A launch's job has
    ended once its cell holds a record of the launch's run that says so;
    otherwise:

    - a SLURM job is asked about with ``scontrol``, as `cancel` asks
      (`_ask`, `_judge`). One not ended (PREEMPTED included: SLURM may
      requeue it) refuses the run with a ValueError, and one SLURM no longer
      lists is over. One SLURM lists that is not this sweep's (its id
      reused) does not refuse the run: it warns, and the launch file is
      kept, as `cancel` keeps it. When scontrol is found but gives no answer
      to read (it failed, or took too long), the run is refused with a
      RuntimeError, as `cancel` then cancels nothing; so is a launch sent to
      other clusters (``--clusters``), whose jobs plain scontrol would call
      purged.
    - a local job's id is its controller's process id: on the host that
      submitted it, a process that runs submitit's local controller on the
      launch's folder, and is not a zombie, refuses the run.

    Where the jobs cannot be checked at all (no scontrol here, or a local
    launch of another host), the run warns and goes ahead: the check is a
    net for the likely mistake, not a lock, and refusing would block every
    run on a machine without SLURM whose dir once ran there. Each message
    names the launch file to delete once its jobs are known to be over."""
    root = Path(working_dir)
    launches, bad = read_launches(root)
    if bad:
        warnings.warn(
            f"mushin could not read the launch files {[str(p) for p in bad]}: if "
            "they name jobs of an earlier run that are still queued or running, "
            "those jobs and this run write the same cells.",
            UserWarning,
            stacklevel=3,
        )
    if not launches:
        return []
    recorded = _recorded_ends(root)
    ended: list[Path] = []
    # A line per launch for each message: those that refuse the run (live,
    # or not readable), then those it only warns of.
    live: list[str] = []
    unreadable: list[str] = []
    unchecked: list[str] = []
    others: list[str] = []
    asked: list[tuple[Launch, list[str]]] = []
    for launch in launches:
        # A cell's recorded end is its job's only for a job of one task:
        # each task of a job of several writes the cell's record, so one
        # task's end leaves the others to the scheduler (#410).
        cells = [
            cell
            for cell in launch.cells
            if launch.tasks > 1
            or (
                (launch.run_id, cell.job_id) not in recorded
                and (launch.run_id, cell.num) not in recorded
            )
        ]
        if not cells:
            ended.append(launch.path)
        elif launch.executor == "local":
            state = _local_state(launch, cells, own_command=True)
            ids = " ".join(cell.job_id for cell in cells)
            if state == "ended":
                ended.append(launch.path)
            elif state == "live":
                live.append(f"{launch.path}: local process ids {ids} on this host")
            else:
                unchecked.append(
                    f"{launch.path}: local process ids {ids} on host {launch.host!r}"
                )
        elif launch.clusters is not None:
            # Plain scontrol, asked, would call these jobs purged, and so let
            # the run go ahead while they run; `cancel` retires the file.
            unreadable.append(
                f"{launch.path}: sent to SLURM clusters {launch.clusters!r}, which "
                "plain scontrol does not see"
            )
        else:
            # A task of an array is `<array>_<index>`: scontrol is asked
            # about the array, as `cancel` asks.
            arrays = list(dict.fromkeys(c.job_id.split("_")[0] for c in cells))
            asked.append((launch, arrays))
    if asked:
        scontrol = _which("scontrol")
        if scontrol is None:
            unchecked.extend(
                f"{launch.path}: SLURM jobs {' '.join(arrays)}, and no scontrol here"
                for launch, arrays in asked
            )
        else:
            answers: dict[str, tuple[str, Any]] = {}
            stalled = None
            for _, arrays in asked:
                for array in arrays:
                    if array in answers:
                        continue
                    if stalled is not None:
                        # One answer that could not be read already refuses
                        # the run: the rest are not asked, each with its own
                        # wait for a scheduler that is not answering (#410).
                        answers[array] = (
                            "unknown",
                            f"not asked, as scontrol did not answer for job {stalled}",
                        )
                        continue
                    answers[array] = _ask(scontrol, array)
                    if answers[array][0] == "unknown":
                        stalled = array
            for launch, arrays in asked:
                # The arrays of each verdict: "live", "gone", "not ours" or,
                # when scontrol's answer could not be read, "unknown".
                having: dict[str, list[str]] = {}
                for array in arrays:
                    kind, detail = answers[array]
                    verdict = _judge(detail, launch.folder) if kind == "shown" else kind
                    having.setdefault(verdict, []).append(array)
                if "live" in having:
                    live.append(f"{launch.path}: SLURM jobs {' '.join(having['live'])}")
                elif "unknown" in having:
                    why = "; ".join(
                        f"job {a}: {answers[a][1]}" for a in having["unknown"]
                    )
                    unreadable.append(f"{launch.path}: {why}")
                elif "not ours" in having:
                    others.append(
                        f"{launch.path}: SLURM jobs {' '.join(having['not ours'])}, "
                        f"whose script and logs are not under {launch.folder!r}"
                    )
                else:
                    ended.append(launch.path)
    where = str(root)
    if live:
        also = (
            "\nNor can it tell whether these have ended:\n" + _lines(unreadable)
            if unreadable
            else ""
        )
        raise ValueError(
            f"mushin: jobs an earlier run submitted in working_dir {where!r} may "
            "still be queued or running, and would write the same cells as this "
            f"run:\n{_lines(live)}{also}\nWait for them to end (a driver still "
            "waiting on them finishes its run, and a `resume=True` then reuses "
            f"the cells they completed), or cancel them: `mushin.cancel({where!r})` "
            "for SLURM jobs, or stop the processes of local ones. If they are not "
            "this sweep's jobs, delete the launch file named and run again."
        )
    if unreadable:
        raise RuntimeError(
            "mushin could not tell whether jobs an earlier run submitted in "
            f"working_dir {where!r} have ended, and two runs on one sweep would "
            f"write the same cells:\n{_lines(unreadable)}\nTry again, or check "
            "them with `scontrol show job <id>` (with `--clusters` for jobs sent "
            "to other clusters). Once they have ended or are cancelled "
            f"(`mushin.cancel({where!r})`), or if they are not this sweep's "
            "jobs, delete the launch file named and run again."
        )
    if unchecked:
        warnings.warn(
            "mushin cannot check whether these jobs of an earlier run in "
            f"working_dir {where!r} have ended, and runs anyway:\n"
            f"{_lines(unchecked)}\nIf they are still queued or running, they "
            "write the same cells as this run. Once they have ended, delete the "
            "launch file named.",
            UserWarning,
            stacklevel=3,
        )
    if others:
        warnings.warn(
            "mushin found jobs under the ids an earlier run in working_dir "
            f"{where!r} submitted that are not this sweep's (SLURM may have "
            f"given the ids to other jobs), and runs anyway:\n{_lines(others)}\n"
            "Once you are sure, delete the launch file named.",
            UserWarning,
            stacklevel=3,
        )
    return ended


def _lines(lines: list[str]) -> str:
    return "\n".join(f"  {line}" for line in lines)


def _recorded_ends(root: Path) -> set[tuple[str, Any]]:
    """``(run id, job id)`` and ``(run id, job number)`` of each record in a
    direct child of ``root`` (where `show` reads records) that says its
    job's end, indexed as `show`'s rule 1 indexes a record of a job that
    ran. A cell's dir further down is not read: its job is asked about."""
    from ._resume import read_cell_status

    ends: set[tuple[str, Any]] = set()
    try:
        dirs = [p for p in root.iterdir() if p.is_dir() and p.name != JOBS_DIR]
    except OSError:
        return ends
    for d in dirs:
        record = read_cell_status(d)
        if record is None or record.get("status") not in _RECORDED_ENDS:
            continue
        # Only as written (a str, or an int): a hand-edited value may not
        # even hash.
        run = record.get("run_id")
        if not isinstance(run, str):
            continue
        if isinstance(record.get("job_id"), str):
            ends.add((run, record["job_id"]))
        if type(record.get("job_num")) is int:  # not a bool
            ends.add((run, record["job_num"]))
    return ends
