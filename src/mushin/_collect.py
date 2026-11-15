# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.collect`: finish a run submitted with ``run(..., wait=False)``
(#345).

Such a run submits its jobs, names them in its launch files (`_jobs`) and
returns. Until it is collected it is in the state of a waited run whose
driver stopped mid-wait: its launch files are kept, no manifest is written,
and the budget start file stays, which `show`, `cancel` and resume already
read. Collect does the rest of the driver's work once every job has ended.
It rebuilds the jobs from their files and waits on them with the launcher's
own `_wait`, so a job that ended without a result becomes a failed cell by
the waited run's rules, of which there is one copy. It gives the cells the
driver settled without a job the returns the driver gave them, and hands
every return to the workflow's `jobs_post_process`, as `run()` does, so the
manifest, the failures and the dataset are a waited run's.

Standard library at module level: `import mushin` loads this module, and
submitit, Hydra and the launcher plugin are imported only by a collect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def collect(working_dir, metrics_filename="mushin_metrics.json"):
    """Build the dataset of a sweep submitted with ``run(..., wait=False)``,
    once its jobs have ended.

    ``run(launcher="submitit_slurm", wait=False, working_dir=...)`` submits
    the sweep's jobs and returns at once, so the driver need not stay
    connected while they queue and run. Collect then finishes the run as a
    waited ``run()`` would have: each cell whose job ended without a result
    (out of memory, ``scancel``, a time limit) is recorded failed, the sweep
    manifest is written, the run's launch files and budget start file are
    removed, and the dataset is returned. Call it from a new Python session
    if you like: it needs no import of the script that ran the sweep. For a
    relative ``working_dir``, call it from the directory ``run()`` was
    called in.

    Parameters
    ----------
    working_dir : str or os.PathLike
        The run's ``working_dir``.
    metrics_filename : str or Sequence[str], optional
        For a run with nothing left to collect (collected already, run with
        ``wait=True``, or its jobs cancelled with `mushin.cancel`): the
        metrics files ``load_from_dir`` reads. Collecting such a run reloads
        it and writes nothing.

    Returns
    -------
    xarray.Dataset
        The sweep's dataset, as ``run()`` returns it.

    Raises
    ------
    FileNotFoundError
        ``working_dir`` is not a sweep's directory.
    RuntimeError
        A job of the run has not ended: queued, running, or requeued after
        preemption (nothing is written; collect again later). Or a job's
        end cannot be read here: ``sacct`` gives no answer (the state reads
        ``UNKNOWN``), or a ``submitit_local`` run was submitted from another
        host. Or the jobs' files cannot be loaded here: their submissions,
        or a result whose type does not import here (nothing is written;
        collect where it does). Or ``working_dir`` is not the dir the run
        was submitted in (a copy or a move of it), the run's
        ``submitit_folder`` lies outside it, or the run's ``working_dir``
        was relative and this process's working directory is not the one
        ``run()`` was called in (nothing is written). Or,
        under ``on_error="raise"``, a cell failed: its own error is
        raised, once every cell is recorded, as a waited run raises it.
    ImportError
        The ``slurm`` extra is not installed.

    Notes
    -----
    A SLURM job's end is read from ``sacct``, as submitit reads it; a
    ``submitit_local`` job's from its process, on the host that submitted
    it.

    Collect unpickles the jobs' submission and result files under the
    sweep's ``.submitit/`` folder, which hold the task and what it
    returned, so collect a ``working_dir`` you trust, with the task's code
    and the types it returns importable as in the jobs, and keep
    ``.submitit/`` until the run is collected. While it
    waits on the jobs, collect sets Hydra's process-wide state to the
    driver's, and restores it after.

    A job that died gives no result: collect waits for those files once,
    as long as submitit waits for one (15 seconds), then records the cells.

    Examples
    --------
    >>> summary = sweep.run(seed=mushin.multirun([0, 1, 2]),
    ...                     launcher="submitit_slurm", working_dir="runs/a",
    ...                     wait=False)  # doctest: +SKIP
    >>> # later, perhaps from a new session
    >>> ds = mushin.collect("runs/a")  # doctest: +SKIP
    """
    from .workflows import MultiRunMetricsWorkflow

    wf = MultiRunMetricsWorkflow().collect(working_dir, metrics_filename)
    return wf.to_xarray()


def collect_into(wf, working_dir, metrics_filename):
    """`MultiRunMetricsWorkflow.collect`: finish the run in ``working_dir``
    into ``wf``, and return it."""
    from ._jobs import read_launches

    root = Path(working_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"mushin.collect: {str(root)!r} is not a directory")
    root = root.resolve()
    run_id = _run_id(root)
    # A workflow collected on keeps nothing of a run it held before, whatever
    # collect does next: raises, reloads, or post-processes.
    _forget_run(wf)
    wf.working_dir = root
    launches, bad = read_launches(root)
    if run_id is not None:
        unread = [p.name for p in bad if p.name.startswith(f"{run_id}.")]
        if unread:
            raise RuntimeError(
                f"mushin.collect could not read this run's launch files {unread} in "
                f"{str(root / 'mushin_jobs')!r}, which name its jobs; nothing was "
                "written."
            )
    ours = [launch for launch in launches if launch.run_id == run_id]
    if run_id is None or not any(launch.cells for launch in ours):
        # Nothing left to collect: the run was collected, waited on, or its
        # jobs cancelled (`cancel` removes a launch once its jobs are over).
        wf.load_from_dir(root, metrics_filename)
        return wf
    # The run is not finished until a collect succeeds: until then its
    # `to_xarray` raises, saying to collect.
    wf._awaiting_collect = True
    waited = [launch.path.name for launch in ours if not launch.detached]
    if waited:
        raise RuntimeError(
            f"mushin.collect: the launches {waited} in {str(root)!r} were not "
            "submitted with wait=False: a driver is waiting on their jobs, or was "
            "until it was stopped. Let a running driver finish; otherwise, once "
            "the jobs have ended, run the sweep again with `resume=True`, which "
            "reuses the cells they completed (or `mushin.cancel` them)."
        )
    _refuse_missing(root, ours)
    plugin = _plugin()
    for launch in ours:
        if launch.cells and launch.submitit_folder is None:
            raise RuntimeError(
                f"mushin.collect cannot rebuild the jobs of {launch.path.name}, "
                "written without their submitit folder."
            )
        if launch.cells:
            # A copy or a move of the run's dir names the original's jobs.
            _refuse_elsewhere(root, launch.submitit_folder, folder=True)
    rebuilt = {launch.path: plugin.rebuild_jobs(launch) for launch in ours}
    _refuse_unended(root, ours, rebuilt, plugin)
    plugin.await_results([job for jobs in rebuilt.values() for job in jobs])
    # Every submission is loaded before any cell is recorded, so a file that
    # will not load stops collect with nothing written.
    loaded = {
        launch.path: plugin.load_submission(launch, rebuilt[launch.path])
        for launch in ours
        if launch.cells
    }
    # Where a dead or settled cell is recorded: with the jobs' files moved
    # into this dir by hand, still the original.
    for launcher, state in loaded.values():
        _refuse_elsewhere(root, plugin.sweep_dir(launcher, state), folder=False)
    # And every result: one that does not load here would be read as a job
    # that gave none, and a completed cell recorded failed.
    plugin.load_results([job for jobs in rebuilt.values() for job in jobs])
    # A batch whose cells all settled composes them with another batch's
    # launcher, which is the same one, sent with the same config.
    anyone = next(iter(loaded.values()))
    returns: dict[int, Any] = {}
    for launch in ours:
        launcher, state = loaded.get(launch.path, anyone)
        returns.update(
            plugin.collect_launch(launcher, state, launch, rebuilt[launch.path])
        )
    # Every launch of a run carries the same pickled runner.
    runner = anyone[0].task_function
    jobs = [returns[num] for num in sorted(returns)]
    _finish(wf, root, ours, runner, jobs)
    return wf


def _finish(wf, root: Path, launches, runner, jobs) -> None:
    """What `run()` does once its launch returns, for the returns ``jobs``
    (in job order) of ``launches``, the run's, whose ``runner`` was
    pickled with its jobs."""
    from hydra.core.utils import JobStatus

    from ._jobs import forget_launch
    from .workflows import _remove_budget_start

    if runner.on_error == "raise":
        failed = [job for job in jobs if job.status != JobStatus.COMPLETED]
        if failed:
            # As the sweeper raises it once the batch is waited on: every
            # cell is recorded, no manifest is written, and the budget start
            # is left. The jobs have ended, so the launch is done with, and a
            # second collect reloads, as `load_from_dir` after a raising run.
            for launch in launches:
                forget_launch(launch.path)
            raise failed[0]._return_value
    # As run() leaves the workflow for `jobs_post_process`.
    wf.working_dir = root
    wf._on_error = runner.on_error
    wf._task_resolves = getattr(runner.task, "_resolve", False) is True
    lineage = runner.lineage or {}
    wf._notes, wf._tags = lineage.get("notes"), lineage.get("tags")
    wf._lineage_from_prior = False
    # Before the post-processing, which can raise on its own, as in run().
    _remove_budget_start(runner, jobs, str(root))
    wf.jobs = jobs
    wf.jobs_post_process()
    wf._awaiting_collect = False
    for launch in launches:
        forget_launch(launch.path)


def _forget_run(wf) -> None:
    """Reset what a workflow holds of a run, as `run()` does before it
    launches and `load_from_dir` before it loads."""
    wf._multirun_task_overrides = {}
    wf._interpolated_cells = None
    wf._interpolated_records = {}
    wf._interpolated_decision = None
    wf._target_dir_multirun_overrides = None
    wf._loaded_job_choices = None
    wf._job_metrics_by_index = None
    wf._run_stamps = None
    wf._manifest = None
    wf._metrics_by_combo = {}
    wf.jobs = []
    wf.cfgs = []
    wf.metrics = {}
    wf.failures = []
    wf.skipped = []
    wf.multirun_working_dirs = None


def _run_id(root: Path) -> str | None:
    """The run ``root``'s multirun.yaml names (`run()` sets it); None for a
    sweep from before runs were named."""
    from hydra_zen import load_from_yaml

    from ._hydra_yaml import hydra_run_stamp

    path = root / "multirun.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"mushin.collect: no multirun.yaml in {str(root)!r}: not a sweep's "
            "working_dir, or none of its cells was launched."
        )
    return hydra_run_stamp(load_from_yaml(path))


def _refuse_elsewhere(root: Path, path: str, folder: bool) -> None:
    """Raise, writing nothing, unless ``path`` lies under ``root``: the
    jobs' submitit folder (``folder``), which the launch file names as an
    absolute path, or the run's sweep dir, which their submission names as
    ``run()`` was given it and collect, recording a cell there, makes
    absolute against this process's working directory, so ``path`` is
    resolved that way. In a copy or a move of the dir a run was submitted
    in, both name the original, and for a relative ``working_dir``
    collected from another dir, the sweep dir is one under that dir:
    collect would read one dir's jobs, or record their cells in another.
    Compared by path components, not by text: ``sweep-orig`` is not under
    ``sweep``."""
    where = Path(path).resolve()
    if where.is_relative_to(root):
        return
    shown = repr(path)
    if not Path(path).is_absolute():
        shown += f", which from this process's working directory is {str(where)!r}"
    if folder:
        what = "keep their files in"
        advice = (
            "Collect a copied or moved run in the dir it was submitted in. A run "
            "whose `submitit_folder` lies outside its `working_dir` cannot be "
            "collected: once its jobs have ended, run the sweep again with "
            "`resume=True`, which reuses the cells they completed."
        )
    else:
        what = "record their cells in"
        advice = (
            "Collect a copied or moved run in the dir it was submitted in, and one "
            "given a relative `working_dir` from the dir its run() was called in."
        )
    raise RuntimeError(
        f"mushin.collect: this run was submitted in another dir than "
        f"{str(root)!r}, as its jobs {what} {shown}; nothing was written. {advice}"
    )


def _refuse_missing(root: Path, launches) -> None:
    """Refuse a run some of whose cells no launch file names: its sweep has
    more cells (as its multirun.yaml gives the grid) than its launches hold,
    so a launch file was removed (by hand, or by `cancel` once its jobs were
    cancelled), and a dataset of the rest would read as the whole sweep.
    Hydra numbers a sweep's cells 0, 1, ... across its batches."""
    import math

    from hydra_zen import load_from_yaml

    from .workflows import BaseWorkflow, _override_texts, multirun

    named = {cell.num for launch in launches for cell in launch.cells}
    named |= {cell.num for launch in launches for cell in launch.settled}
    try:
        task = load_from_yaml(root / "multirun.yaml").hydra.overrides.task
        parsed = BaseWorkflow._parse_overrides(_override_texts(task))
        cells = math.prod(len(v) for v in parsed.values() if isinstance(v, multirun))
    except Exception:  # noqa: BLE001 - only gaps among the named are seen
        cells = max(named) + 1
    missing = sorted(set(range(max(cells, max(named) + 1))) - named)
    if missing:
        raise RuntimeError(
            f"mushin.collect: no launch file in {str(root / 'mushin_jobs')!r} names "
            f"cells {missing[:10]}{' ...' if len(missing) > 10 else ''} of this run, "
            "so it cannot be collected whole; nothing was written. "
            "`mushin.show` lists what its cells recorded, and a run with "
            "`resume=True` reuses the cells that completed and runs the rest."
        )


def _refuse_unended(root: Path, launches, rebuilt, plugin) -> None:
    """Raise unless every job of ``launches`` has ended, writing nothing:
    `_wait` would record a cell whose job still runs as dead. A SLURM job
    has ended once each of its tasks wrote its result or SLURM's accounting
    gives an end state (`_jobs._ENDED`; submitit's own test also takes, for
    one, SUSPENDED and other states a job resumes from); a local job, once
    its process is no longer running, which only its host can tell."""
    from collections import Counter

    from ._jobs import _ENDED, _local_state

    states: Counter = Counter()
    hosts = set()
    total = 0
    for launch in launches:
        total += len(launch.cells)
        if not launch.cells:
            continue
        if launch.executor == "local":
            state = _local_state(launch, launch.cells, own_command=True)
            if state == "unknown":
                hosts.add(launch.host)
            elif state == "live":
                states["RUNNING"] += len(launch.cells)
            continue
        for state in plugin.slurm_states(rebuilt[launch.path]):
            word = state.split()[0] if state.split() else "UNKNOWN"
            if state != "ENDED" and word not in _ENDED:
                states[word] += 1
    if hosts:
        raise RuntimeError(
            f"mushin.collect: this run's submitit_local jobs ran on {sorted(hosts)}, "
            "and only that host can tell whether they have ended. Collect it "
            "there; nothing was written."
        )
    if states:
        count = sum(states.values())
        listed = ", ".join(f"{s}: {n}" for s, n in sorted(states.items()))
        hint = (
            " UNKNOWN is a job sacct did not report: it may not be listed yet, "
            "or sacct may not be available or reach the job's cluster here."
            if "UNKNOWN" in states
            else ""
        )
        raise RuntimeError(
            f"mushin.collect: {count} of {total} jobs ({listed}) of the run in "
            f"{str(root)!r} have not ended; nothing was written. Collect again "
            f"once they have (`mushin.show` lists the cells).{hint}"
        )


def _plugin():
    """mushin's submitit launcher module, as Hydra's plugin scan left it
    (the scan executes the module again, rebinding what it defines)."""
    import importlib

    from hydra.core.plugins import Plugins

    Plugins.instance()
    try:
        module = importlib.import_module("hydra_plugins.mushin_submitit.launcher")
    except ImportError:  # pragma: no cover - the package ships it
        module = None
    if module is None or getattr(module, "rebuild_jobs", None) is None:
        raise ImportError(
            "mushin.collect needs submitit and hydra-submitit-launcher: pip "
            "install 'mushin-py[slurm]'"
        )
    return module
