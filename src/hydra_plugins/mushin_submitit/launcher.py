# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The launchers behind `run(launcher="submitit_slurm")` and `"submitit_local"`.

They are hydra-submitit-launcher's own launchers, changed in three places.
They submit only the cells that need a job: a cell `sample=` leaves out, a
resume reuses or a `cache_dir` holds is settled in the driver first, by the
runner's own rules. And they wait on every job: the plugin waits in a list
comprehension, so the first job that ends without a result (killed out of
memory, on a failed node, by `scancel` or at a time limit before Python could
act, or raised outside the task, as when it cannot import it) raises from the
driver while the later jobs still run, unobserved, and the sweep records
nothing. Here a job that ends without a result comes back to mushin as a
failed cell, which `on_error` then handles as it does a task that raised,
unless the cell's own record holds the result the job could not return
(#345). And with ``cells_per_job`` above 1, each job runs several cells, one
after another, so a sweep of many short cells costs fewer jobs (#348).

`run()` sends the plugin's two names here (`mushin_submitit_slurm` and
`mushin_submitit_local`); their config is the plugin's, field for field, plus
``cells_per_job``, so `launcher_config` and `submitit_slurm_config` work
unchanged.

For mushin's runner, each launch also names its jobs in a launch file
under `<working_dir>/mushin_jobs/` until it has waited on them, so
`mushin.show` can list the cells whose jobs have not started (`queued`).

Under `run(..., wait=False)` a launch records its jobs and returns without
waiting on them; `mushin.collect` later rebuilds the jobs from their files
and waits on them with this launcher's own `_wait`, in another process
(`rebuild_jobs` and the functions after it), so a detached run's cells are
judged by the rules a waited run's are.

Hydra imports every module under `hydra_plugins` whenever any Hydra
application starts, and reports an ImportError there as a warning (an error
under `-W error`). So this module imports nothing heavy, and without the
plugin it defines nothing. mushin itself is imported only on a launch of
mushin's runner.
"""

import contextlib
import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    from hydra_plugins.hydra_submitit_launcher import config as _plugin_config
    from hydra_plugins.hydra_submitit_launcher import (
        submitit_launcher as _plugin_launcher,
    )
except ImportError:  # no `slurm` extra: nothing to define
    _plugin_launcher = None

if _plugin_launcher is not None:
    from hydra.core.config_store import ConfigStore
    from hydra.core.singleton import Singleton
    from hydra.core.utils import JobReturn, filter_overrides
    from omegaconf import OmegaConf, open_dict, read_write

    class _MushinSubmititMixin:
        """Submit as the plugin does, but only the cells that need a job,
        several to a job when ``cells_per_job`` says so (#348); wait on every
        job (#345).

        Holds nothing but its config: the launcher is pickled into every job
        (`map_array(self, ...)`, or a `_CellGroup` holding it) and into
        `checkpoint()`'s resubmission."""

        def __init__(self, *, cells_per_job=1, **params):
            # Kept out of the plugin's params: `_submit` hands those to
            # submitit's `update_parameters`, which refuses a key it does not
            # know (#348).
            super().__init__(**params)
            self._cells_per_job = cells_per_job

        def setup(self, *, hydra_context, task_function, config):
            """The plugin's `setup`, then the refusals of ``cells_per_job``:
            Hydra sets the launcher up before it writes anything for the
            sweep (``multirun.yaml``), and `launcher_config` and a raw
            ``hydra.launcher.cells_per_job`` override both arrive here."""
            super().setup(
                hydra_context=hydra_context,
                task_function=task_function,
                config=config,
            )
            cells = self._cells_per_job
            # Not a bool: `True` is an int, but not a count anyone means.
            if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
                raise ValueError(
                    f"cells_per_job must be a whole number >= 1, got {cells!r}"
                )
            if cells == 1:
                return
            tasks = _tasks_per_job(self.params)
            if tasks > 1:
                # Every task of a job runs every cell (#365), so several
                # tasks cannot share out a job's cells.
                raise ValueError(
                    f"cells_per_job={cells} needs a job of one task; this "
                    f"launcher config gives each job {tasks} tasks "
                    "(tasks_per_node x nodes). Use cells_per_job=1 for a "
                    "cell of several tasks, such as DDP."
                )
            if not getattr(task_function, "_mushin_collects_dead_jobs", False):
                # A dead job's cells are sorted out from their records, which
                # only mushin's runner writes.
                raise ValueError(
                    f"cells_per_job={cells} runs only a mushin sweep's cells "
                    "(`run()`), not a plain Hydra task function."
                )

        def launch(self, job_overrides, initial_job_idx):
            runner = self.task_function
            # `run(wait=False)` (#345): the jobs are recorded for
            # `mushin.collect`, and left running.
            detached = getattr(runner, "_mushin_collects_dead_jobs", False) and (
                getattr(runner, "detached", False) is True
            )
            if detached:
                _refuse_another_sweeper(self.config)
            settled = self._settled_in_driver(job_overrides, initial_job_idx)
            numbers = [initial_job_idx + i for i in range(len(job_overrides))]
            if len(settled) == len(numbers):
                # Nothing to submit: no executor (on SLURM it wants `srun`),
                # but the sweep dir as `_submit` leaves it.
                sweep_dir = Path(str(self.config.hydra.sweep.dir))
                sweep_dir.mkdir(parents=True, exist_ok=True)
                if "mode" in self.config.hydra.sweep:
                    mode = int(str(self.config.hydra.sweep.mode), 8)
                    os.chmod(sweep_dir, mode=mode)
                if detached:
                    # Recorded for collect all the same: another batch may
                    # leave its jobs running. run() removes the file when
                    # none does.
                    self._record_launch([], [], settled, job_overrides, initial_job_idx)
                return [settled[n] for n in numbers]
            jobs, job_params = self._submit(
                job_overrides, initial_job_idx, skip=settled
            )
            launch = self._record_launch(
                jobs,
                job_params,
                settled if detached else None,
                job_overrides,
                initial_job_idx,
            )
            if detached:
                if launch is not None:
                    _hold(jobs)
                    returns = {
                        p[2]: _pending_return(job)
                        for job, p in zip(jobs, job_params, strict=True)
                    }
                    returns.update(settled)
                    return [returns[n] for n in numbers]
                # Nothing would find these jobs later.
                log.warning(
                    "mushin could not record this launch's jobs for "
                    "`mushin.collect`, so it waits on them, as with wait=True."
                )
            try:
                waited = self._wait(jobs, job_params)
            except Exception:
                # `_wait` raises only once it has visited every job, but a
                # job it could not tell had ended may still be queued: then
                # the launch stays. Without one (a plain Hydra task, or a
                # write that failed) the scheduler is not asked again.
                if launch is not None and all(_ended(job) for job in jobs):
                    _forget(launch)
                raise
            # A BaseException (Ctrl-C, SystemExit) leaves the launch: its
            # jobs may still be queued, and `show` keeps listing them so.
            _forget(launch)
            returns = dict(zip([p[2] for p in job_params], waited, strict=True))
            returns.update(settled)
            return [returns[n] for n in numbers]

        def _settled_in_driver(self, job_overrides, initial_job_idx):
            """The JobReturns of the cells mushin's runner says need no job
            (left out by ``sample=``, reused by a resume, or a ``cache_dir``
            hit), by job number, so only the rest are submitted, under their
            own job numbers (#345).

            A cell is composed in the driver only when it may need no job, and
            not at all when the sweep has Hydra callbacks, which `run_job`
            calls for each job. Every cell is decided before any is recorded,
            and a settled cell whose dir a cell left to its job reads its
            prior result from is not recorded but submitted, as before (a
            resume over a renumbered grid finds cells' prior results in dirs
            other cells now take). A cell that cannot be settled here, for
            whatever reason, is left to its job, which decides as before; the
            driver warns once. Warnings from judging a cell are the job's to
            give, and are not shown here."""
            import warnings

            runner = self.task_function
            settles = getattr(runner, "settles_in_driver", None)
            if not (
                getattr(runner, "_mushin_collects_dead_jobs", False)
                and settles is not None
                and settles()
                and not OmegaConf.select(self.config, "hydra.callbacks")
            ):
                return {}
            from mushin._resume import judging_quietly

            settled, writes, own, reads, errors = {}, {}, {}, {}, []
            environment = 0
            reuses = runner._reuses()
            for i, overrides in enumerate(job_overrides):
                idx = initial_job_idx + i
                if not runner.may_settle(idx, reuses):
                    continue
                try:
                    with self._composed(overrides, idx, f"driver_{idx}") as ret:
                        with judging_quietly():
                            own[idx], reads[idx] = runner.dirs_of(ret)
                            result = runner.settle_in_driver(ret)
                            if result is None and runner.reads_environment(ret):
                                environment += 1
                        if result is not None:
                            ret.status, ret.return_value, writes[idx] = result
                            settled[idx] = ret
                except Exception as exc:  # noqa: BLE001 - its job decides
                    errors.append((idx, exc))
            # The prior dirs the jobs will read; all but the settled cells'
            # when a cell's is unknown.
            if any(idx not in reads for idx, _ in errors):
                taken = {reads[n] for n in settled}
                protected = runner.prior_dirs() - taken
            else:
                protected = {d for n, d in reads.items() if n not in settled}
            # A cell submitted for that reads its own prior dir in its job, which
            # no settled cell may then take either: drop until none does.
            kept = 0
            while dropped := [n for n in settled if own[n] in protected]:
                for idx in dropped:
                    del settled[idx]
                    kept += 1
                    if reads[idx] is not None:
                        protected.add(reads[idx])
            for idx in list(settled):
                try:
                    for write in writes[idx]:
                        write()
                except Exception as exc:  # noqa: BLE001 - its job rewrites the dir
                    errors.append((idx, exc))
                    del settled[idx]
            if errors:
                idx, exc = errors[0]
                try:
                    warnings.warn(
                        f"{len(errors)} of {len(job_overrides)} cells could not "
                        f"be settled before submitting (#{idx}: {exc!r}); they "
                        "are submitted, and their jobs decide as before.",
                        UserWarning,
                        stacklevel=3,
                    )
                except Warning:  # warnings as errors: settling is best effort
                    pass
            if environment:
                log.info(
                    f"{environment} of {len(job_overrides)} cells read the "
                    "environment (${oc.env:...}), which their jobs' may hold "
                    "differently: they are submitted, and their jobs decide."
                )
            if kept:
                log.info(
                    f"{kept} of {len(job_overrides)} cells that need no job are "
                    "submitted anyway: their dirs hold the prior results other "
                    "cells' jobs read."
                )
            if settled:
                listed = ", ".join(f"#{n}" for n in sorted(settled))
                log.info(
                    f"{len(settled)} of {len(job_overrides)} cells need no job "
                    f"(left out by sample=, reused or cached): {listed}"
                )
            return settled

        def _submit(self, job_overrides, initial_job_idx, skip=()):
            """hydra-submitit-launcher 1.2.0's `launch()` up to its submission,
            but for the cells in ``skip``, which need no job
            (`tests/test_submitit_launcher.py` pins that body, so a
            plugin release that changes it fails until this is compared)."""
            import submitit

            assert self.config is not None

            num_jobs = len(job_overrides)
            assert num_jobs > 0
            params = self.params
            # build executor
            init_params = {"folder": self.params["submitit_folder"]}
            specific_init_keys = {"max_num_timeout"}

            init_params.update(
                **{
                    f"{self._EXECUTOR}_{x}": y
                    for x, y in params.items()
                    if x in specific_init_keys
                }
            )
            init_keys = specific_init_keys | {"submitit_folder"}
            executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

            # specify resources/parameters
            baseparams = set(OmegaConf.structured(_plugin_config.BaseQueueConf).keys())
            params = {
                x if x in baseparams else f"{self._EXECUTOR}_{x}": y
                for x, y in params.items()
                if x not in init_keys
            }
            executor.update_parameters(**params)

            log.info(
                f"Submitit '{self._EXECUTOR}' sweep output dir : "
                f"{self.config.hydra.sweep.dir}"
            )
            sweep_dir = Path(str(self.config.hydra.sweep.dir))
            sweep_dir.mkdir(parents=True, exist_ok=True)
            if "mode" in self.config.hydra.sweep:
                mode = int(str(self.config.hydra.sweep.mode), 8)
                os.chmod(sweep_dir, mode=mode)

            job_params: list[Any] = []
            for idx, overrides in enumerate(job_overrides):
                idx = initial_job_idx + idx
                # mushin's: a cell settled in the driver needs no job.
                if idx in skip:
                    continue
                lst = " ".join(filter_overrides(overrides))
                log.info(f"\t#{idx} : {lst}")
                job_params.append(
                    (
                        list(overrides),
                        "hydra.sweep.dir",
                        idx,
                        f"job_id_for_{idx}",
                        Singleton.get_state(),
                    )
                )

            # mushin's: several cells per job (#348).
            if self._cells_per_job > 1:
                return self._submit_groups(executor, job_params), job_params
            jobs = executor.map_array(self, *zip(*job_params))  # noqa: B905
            return jobs, job_params

        def _submit_groups(self, executor, job_params):
            """Submit the cells of ``job_params`` (one per cell, in order) as
            jobs of up to ``cells_per_job`` consecutive cells, in one array,
            each running a `_CellGroup`; return one job per cell, a job's
            `Job` repeated for each of its cells, so the launch file, `_wait`
            and `launch()` stay keyed by cell (#348)."""
            size = self._cells_per_job
            groups = [job_params[i : i + size] for i in range(0, len(job_params), size)]
            jobs = executor.map_array(_CellGroup(self), groups)
            return [job for job, group in zip(jobs, groups, strict=True) for _ in group]

        def _record_launch(
            self, jobs, job_params, settled=None, job_overrides=(), initial_job_idx=0
        ) -> Path | None:
            """Write the launch file that names ``jobs`` (#345), and return
            its path; None for a task function other than mushin's runner,
            which `show` does not read, and when the file cannot be written:
            it is a view of the queue, and never fails a sweep. Stores
            nothing on the launcher, which every job unpickles.

            With ``settled`` (the returns of the cells the driver settled,
            by job number), the launch is detached (`run(wait=False)`): the
            file also holds what `mushin.collect` needs, each cell's
            overrides (from ``job_overrides``, numbered from
            ``initial_job_idx``), what each settled cell returned,
            submitit's folder, and the sweep dir as resolved here."""
            runner = self.task_function
            run_id = getattr(runner, "run_id", None)
            if not getattr(runner, "_mushin_collects_dead_jobs", False) or not run_id:
                return None
            try:
                import socket

                from submitit.core.utils import JobPaths

                from mushin._jobs import LaunchCell, SettledCell, record_launch

                detached = settled is not None
                combos = runner.submitted_combos([p[0] for p in job_params])
                cells = [
                    LaunchCell(
                        params[2],
                        str(job.job_id),
                        combo,
                        tuple(params[0]) if detached else None,
                    )
                    for job, params, combo in zip(jobs, job_params, combos, strict=True)
                ]
                kept = [
                    SettledCell(
                        num,
                        tuple(job_overrides[num - initial_job_idx]),
                        _settled_value(ret),
                    )
                    for num, ret in sorted((settled or {}).items())
                ]
                # The folder every job's paths start with (its id comes
                # after), as submitit wrote them into the scripts it
                # submitted: the scheduler's record of a job names it, which
                # tells this sweep's jobs from others under a reused id.
                folder = JobPaths.get_first_id_independent_folder(
                    self.params["submitit_folder"]
                )
                root = Path(str(self.config.hydra.sweep.dir))
                return record_launch(
                    root,
                    run_id=run_id,
                    executor=self._EXECUTOR,
                    folder=str(folder),
                    clusters=_clusters(self.params),
                    host=socket.gethostname(),
                    depth=_cell_depth(self.config),
                    cells=cells,
                    # Each task of a job of several writes its cell's record,
                    # so the guard asks about such a job whatever its cells
                    # record (#410).
                    tasks=max((getattr(j, "num_tasks", 1) for j in jobs), default=1),
                    detached=detached,
                    # The template each job's paths come from (`%j` and
                    # all), as submitit made it absolute.
                    submitit_folder=str(
                        Path(str(self.params["submitit_folder"]))
                        .expanduser()
                        .absolute()
                    )
                    if detached
                    else None,
                    settled=kept,
                    # The dir the cells are in, as this process resolved
                    # `hydra.sweep.dir`: collect, in another process, may
                    # resolve an interpolation in it (`${oc.env:...}`) or a
                    # relative one to another dir, or none (review of #414).
                    sweep_dir=str(root.resolve()) if detached else None,
                )
            except Exception as exc:  # noqa: BLE001 - the sweep goes on
                log.warning(
                    "mushin could not record this launch's jobs, so "
                    f"`mushin.show` will not list them as queued: {exc!r}"
                )
                return None

        def _wait(self, jobs, job_params):
            """Every cell's return, in launch order.

            ``jobs`` and ``job_params`` are per cell, in launch order. The
            cells of a job of several (``cells_per_job``, #348) are
            consecutive and share one `Job` object, whose `results()` is
            read once (submitit caches nothing, and waits on a job with no
            result before it raises), and whose outcome stands for each of
            them. Such a job returns a `_GroupReturns`, from which each cell
            takes its own return, as a one-cell job's is its first result. A
            cell in it with no return of its own comes back as a dead job's
            cell: one that raised (`_Unreturned`) as a job that pickled an
            error in place of its result, one not run (or missing) as a job
            that gave no result.

            For mushin's runner, a job that ended without a result comes
            back as its cell's record says (`_dead_return`), each of its
            cells so: a failed cell, or the result the cell recorded, and the
            later jobs are still waited on. An error that cannot become one
            (a job not known to have ended, or a cell that cannot be
            composed here) is raised once every job has been waited on. Any
            other task function sees the first error at once, as the plugin
            raises it (it never runs several cells per job)."""
            from submitit.core.utils import FailedJobError, UncompletedJobError

            collects = getattr(self.task_function, "_mushin_collects_dead_jobs", False)
            outcomes: dict = {}
            returns, unrecorded = [], None
            for job, params in zip(jobs, job_params, strict=True):
                if id(job) not in outcomes:
                    outcomes[id(job)] = _outcome(job, collects)
                results, error, ended = outcomes[id(job)]
                if results is not None:
                    # The job gave its results, so nothing here may send its
                    # cell down the dead-job path but the cell's own entry.
                    first = results[0]
                    if not isinstance(first, _GroupReturns):
                        several = collects and len(results) > 1
                        returns.append(
                            self._job_return(job, results) if several else first
                        )
                        continue
                    entry = first.get(params[2])
                    if not isinstance(entry, _Unreturned) and entry is not None:
                        returns.append(entry)
                        continue
                    ended = True
                    if entry is not None and entry.why == "raised":
                        error = FailedJobError(entry.text)
                    elif entry is not None:
                        error = UncompletedJobError(entry.text)
                    else:
                        error = UncompletedJobError(
                            f"submitit job {job.job_id} returned nothing for "
                            f"cell #{params[2]}"
                        )
                try:
                    if not ended:
                        # Recording a cell whose job may still run would
                        # overwrite what that job writes.
                        raise error
                    returns.append(self._dead_return(job, params[0], params[2], error))
                except Exception as exc:  # noqa: BLE001 - raised after the loop
                    unrecorded = unrecorded or exc
                    returns.append(None)
            if unrecorded is not None:
                raise unrecorded
            return returns

        def _job_return(self, job, results):
            """The return a job of several tasks of mushin's runner stands
            for: its first task's, unless a task failed (raised: Hydra returns
            that as a FAILED return, or, under ``on_error="nan"``, the runner
            returns its sentinel), when the cell failed: the first failing
            task's, the first task's when it failed. Under ``"nan"`` a FAILED
            one comes back as the runner's failure, as a dead job's does, so
            the sweep goes on. The cell's record is made to agree (#365)."""
            from hydra.core.utils import JobStatus

            from mushin.workflows import _FailedRun

            failed = [r for r in results if _task_failed(r)]
            if not failed:
                return results[0]
            runner = self.task_function
            runner.record_a_failed_task(failed[0], str(job.job_id))
            ret = failed[0]
            if ret.status != JobStatus.COMPLETED and runner.on_error != "raise":
                ret.status = JobStatus.COMPLETED
                ret._return_value = _FailedRun(ret._return_value)
            return ret

        @contextlib.contextmanager
        def _composed(self, overrides, idx, job_id):
            """The JobReturn of the job for ``overrides``, composed as the
            plugin's `__call__` and Hydra's `run_job` compose it
            (`tests/test_submitit_launcher.py` pins the part of `run_job`
            mirrored here), in the driver, with the global HydraConfig set to
            the job's config inside and restored after. ``job_id`` is the
            scheduler's, or ``driver_<n>`` for a cell the driver settles."""
            from hydra.core.hydra_config import HydraConfig

            assert self.hydra_context is not None and self.config is not None
            outer = HydraConfig.instance().cfg
            try:
                config = self.hydra_context.config_loader.load_sweep_config(
                    self.config, list(overrides)
                )
                with open_dict(config.hydra.job) as hydra_job:
                    hydra_job.id = job_id
                    config.hydra.job.num = idx
                HydraConfig.instance().set_config(config)
                output_dir = os.path.join(
                    str(OmegaConf.select(config, "hydra.sweep.dir")),
                    str(OmegaConf.select(config, "hydra.sweep.subdir")),
                )
                with read_write(config.hydra.runtime):
                    with open_dict(config.hydra.runtime):
                        config.hydra.runtime.output_dir = os.path.abspath(output_dir)
                HydraConfig.instance().set_config(config)

                ret = JobReturn()
                task_cfg = copy.deepcopy(config)
                with read_write(task_cfg):
                    with open_dict(task_cfg):
                        del task_cfg["hydra"]
                ret.cfg = task_cfg
                ret.hydra_cfg = copy.deepcopy(HydraConfig.instance().cfg)
                ret.overrides = OmegaConf.to_container(config.hydra.overrides.task)
                # run_job's rule; mushin sets chdir, so this is the cell dir.
                chdir = config.hydra.job.chdir
                ret.working_dir = output_dir if chdir else os.getcwd()
                yield ret
            finally:
                HydraConfig.instance().cfg = outer

        def _dead_return(self, job, overrides, idx, exc):
            """The JobReturn for a job that ended without a result, with the
            status and value mushin's runner gives it once it has recorded
            the cell (`_TaskRunner.record_dead_cell`)."""
            from mushin.workflows import _DeadJob

            runner = self.task_function
            # The summary reads the whole message; the cell keeps a bounded one.
            dead = _DeadJob(
                _bounded(exc, job.paths.folder),
                _summary(job, exc, self.params),
                keepable=_keepable(job, exc),
                job=_entry_of(job),
            )
            with self._composed(overrides, idx, str(job.job_id)) as ret:
                # Inside: the cell's config-group choices come from the
                # HydraConfig the composition set.
                ret.status, ret.return_value = runner.record_dead_cell(ret, dead)
            if isinstance(ret.return_value, dict):
                log.info(
                    f"submitit job {job.job_id} gave no result; its cell's "
                    "record holds one, which stands for it (#345)"
                )
            return ret

    def _outcome(job, collects):
        """A job's ``(results, error, ended)``: its results, or the error
        reading them raised and whether submitit knows the job has ended.
        Any task function but mushin's runner gets the error raised, as the
        plugin raises it."""
        from submitit.core.utils import FailedJobError, UncompletedJobError

        try:
            return job.results(), None, True
        except (UncompletedJobError, FailedJobError) as exc:
            # submitit raises these once the job has ended.
            if not collects:
                raise
            return None, exc, True
        except Exception as exc:  # noqa: BLE001 - e.g. an unreadable result
            if not collects:
                raise
            return None, exc, _ended(job)

    def _tasks_per_job(params) -> int:
        """How many tasks a launcher config gives each job: ``tasks_per_node``
        times ``nodes``, each 1 when absent or unreadable. An ``ntasks`` in
        ``additional_parameters`` is not seen."""

        def count(name: str) -> int:
            try:
                return max(int(params.get(name) or 1), 1)
            except (TypeError, ValueError):
                return 1

        return count("tasks_per_node") * count("nodes")

    def _entry_of(job) -> dict | None:
        """The job's id and log folder for its cell's record (#345); None
        when submitit cannot give them."""
        try:
            return {"job_id": str(job.job_id), "logs": str(job.paths.folder)}
        except Exception:  # noqa: BLE001 - the record names no job
            return None

    def _forget(launch) -> None:
        """Remove a launch's file, once its jobs are known to have ended;
        nothing for a launch that wrote none."""
        if launch is not None:
            from mushin._jobs import forget_launch

            forget_launch(launch)

    # The processes of the local jobs a detached launch left running: dropped
    # while its child runs, a `Popen` warns (a ResourceWarning, an error
    # under `-W error`) at whatever moment it is collected. Held instead, and
    # reaped once exited at the next detached launch, so a long-lived driver
    # (a notebook) does not gather zombies; until then an exited one is a
    # zombie, which `mushin._jobs` counts as ended. Hydra's plugin scan
    # executes this module again in its own namespace (on every Hydra state
    # swap, as collect's), so the list there is kept, not replaced.
    _HELD: list = globals().setdefault("_HELD", [])

    def _hold(jobs) -> None:
        _HELD[:] = [p for p in _HELD if p.poll() is None]
        _HELD.extend(
            p for p in (getattr(job, "_process", None) for job in jobs) if p is not None
        )

    def _pending_return(job):
        """The return of a cell whose job a detached launch left running."""
        from hydra.core.utils import JobStatus

        from mushin.workflows import _Pending

        ret = JobReturn()
        ret.status = JobStatus.COMPLETED
        ret.return_value = _Pending(str(job.job_id))
        return ret

    def _settled_value(ret) -> str:
        """What the driver settled a cell with, as a launch file records it
        (`mushin._jobs.SETTLED_VALUES`)."""
        from mushin.workflows import _SkippedRun

        value = ret.return_value
        if isinstance(value, _SkippedRun):
            return "skipped"
        return "none" if value is None else "metrics"

    def _refuse_another_sweeper(config) -> None:
        """`run(wait=False)` checks the sweeper it is given; this is for one
        a config's defaults list chooses. Only the basic sweeper submits
        every cell at once: another chooses its next batch from the last
        one's results, which a detached launch does not have."""
        from mushin.workflows import _BASIC_SWEEPER

        target = OmegaConf.select(config, "hydra.sweeper._target_")
        if target != _BASIC_SWEEPER:
            raise ValueError(
                f"`wait=False` needs Hydra's basic sweeper, and this sweep's is "
                f"{target!r}: another sweeper chooses its next cells from the "
                "results of the last, which are not there until the jobs end."
            )

    def rebuild_jobs(launch) -> list:
        """The submitit jobs of a detached ``launch`` (`mushin._jobs.Launch`),
        one per job, in the order its cells first name them, rebuilt from
        their ids as submitit builds them, with the launch's tasks per job:
        their paths, results and logs are the jobs'. The cells of a job of
        several (``cells_per_job``, #348) all name its one id, and it is
        rebuilt once: collect counts, asks the scheduler about and reads
        each job once, and `collect_launch` hands its one object to each of
        its cells, as `_wait` takes them. A SLURM job is registered with
        submitit's shared sacct reader. A rebuilt local job has no process
        to poll, so whether it has ended is told by its pid
        (`mushin._jobs`)."""
        if launch.executor == "slurm":
            from submitit.slurm.slurm import SlurmJob as job_class
        else:
            from submitit.local.local import LocalJob as job_class
        tasks = range(launch.tasks)
        return [
            job_class(folder=launch.submitit_folder, job_id=job_id, tasks=tasks)
            for job_id in dict.fromkeys(cell.job_id for cell in launch.cells)
        ]

    def slurm_states(jobs) -> list:
        """Each SLURM job's state from one sacct call for them all (through
        submitit's reader, whose answer `_wait` then waits by), or
        ``"ENDED"`` for a job each of whose tasks wrote its result. A state
        sacct does not give (it failed, or does not list the job) is
        ``"UNKNOWN"``."""
        from submitit.slurm.slurm import SlurmJob

        if jobs:
            SlurmJob.watcher.update()
        states = []
        for job in jobs:
            tasks = list(job._sub_jobs) or [job]
            if all(task.paths.result_pickle.exists() for task in tasks):
                states.append("ENDED")
            else:
                states.append(job.watcher.get_state(job.job_id, mode="cache"))
        return states

    def await_results(jobs) -> None:
        """Wait once for the result files of ``jobs`` (all ended) that are
        not there yet, as long as submitit waits for one, then have each job
        read its result at once. submitit waits for each job's file in
        turn, for a filesystem slow to show it, so a sweep whose every job
        died (one that cannot import its task, say) would keep collect for
        that long per job; one shared wait covers a slow filesystem as
        well. `_results_timeout_s` is submitit's own, read per job
        (`tests/test_submitit_local.py` pins it)."""
        import time

        tasks = [task for job in jobs for task in (list(job._sub_jobs) or [job])]
        missing = [t for t in tasks if not t.paths.result_pickle.exists()]
        if missing:
            deadline = time.monotonic() + max(t._results_timeout_s for t in missing)
            while missing and time.monotonic() < deadline:
                time.sleep(0.5)
                for folder in {t.paths.folder for t in missing}:
                    try:
                        # As submitit does: refreshes a network filesystem's
                        # view of the folder.
                        folder.chmod(folder.stat().st_mode)
                    except OSError:
                        pass
                missing = [t for t in missing if not t.paths.result_pickle.exists()]
        for job in jobs:
            for task in (job, *job._sub_jobs):
                task._results_timeout_s = 0

    def load_submission(launch, jobs):
        """The launcher and the Hydra state the driver sent ``launch``'s
        jobs with, from the first of their submission files that loads;
        raises, naming the files, when none does."""
        from submitit.core.utils import DelayedSubmission

        errors = []
        for job in jobs:
            path = job.paths.submitted_pickle
            try:
                submitted = DelayedSubmission.load(path)
                function, args = submitted.function, submitted.args
                if isinstance(function, _CellGroup):
                    # A job of several cells (#348), as submitted or as its
                    # `checkpoint()` requeued it: a `_CellGroup` holding the
                    # launcher, called with the list of its cells' args,
                    # each what a one-cell job is called with.
                    function, args = function.launcher, args[0][0]
                return function, args[4]
            except Exception as exc:  # noqa: BLE001 - the next may load
                errors.append(f"{path}: {exc!r}")
        raise RuntimeError(
            f"mushin.collect could not load the submission of any job of "
            f"{launch.path.name} ({'; '.join(errors[:3])}). It needs the "
            "sweep's `.submitit/` files as the jobs left them, and the task's "
            "code importable here, as in the jobs; nothing was written."
        )

    def sweep_dir(launcher, state) -> str:
        """The ``hydra.sweep.dir`` ``launcher`` was sent with, resolved as
        `_composed` resolves it for a cell, under the Hydra state it was sent
        with: a dead or settled cell's dir starts with it. Relative when
        ``run()`` was given a relative ``working_dir``. For a launch file
        from before its ``sweep_dir``, which collect pins instead
        (`pin_sweep_dir`)."""
        with _as_submitted(state):
            return str(OmegaConf.select(launcher.config, "hydra.sweep.dir"))

    def pin_sweep_dir(launcher, path: str) -> None:
        """Make ``path`` the sweep dir of ``launcher`` and of every cell it
        composes: the dir the driver resolved ``hydra.sweep.dir`` to, as
        the launch file records it (review of #414). Otherwise collect
        resolves the submitted value again, in this process: an
        interpolation (``${oc.env:SWEEP_ROOT}``) may give another dir here
        or none, and a relative one names a dir under this process's
        working directory. `_composed` builds each cell's config anew from
        the launcher's Hydra overrides (`load_sweep_config`), not from its
        config, so the ``hydra.sweep.dir`` override is replaced too, not
        only the config's node."""
        from mushin.workflows import _to_override_element

        config = launcher.config
        key = "hydra.sweep.dir"
        # The `hydra` node is read-only: Hydra's `HydraConfig.set_config`
        # made it so in the driver, and the pickle kept the flag.
        with read_write(config.hydra):
            OmegaConf.update(config, key, path, merge=False)
            # As written, as `load_sweep_config` reads them: an override
            # holding `${...}` is an interpolation to OmegaConf, which
            # iterating the list would resolve, here.
            overrides = [
                text
                for text in OmegaConf.to_container(
                    config.hydra.overrides.hydra, resolve=False
                )
                if text.partition("=")[0].strip().lstrip("+~") != key
            ]
            overrides.append(f"{key}={_to_override_element(path)}")
            config.hydra.overrides.hydra = overrides

    def load_results(jobs) -> None:
        """Load every result file the tasks of ``jobs`` wrote, and raise,
        naming the files, when one does not load. `_wait` reads a result it
        cannot load as a job that gave none, and a cell of a job of several
        tasks is then recorded failed, its metrics file and cache entry
        deleted, though the job completed. A result holds what the task
        returned, so one whose type comes from a module only the jobs import
        loads there and not here: refused before anything is written, the
        run can still be collected where it loads. `_wait` loads each again,
        through submitit's `job.results()`, which takes none loaded
        before."""
        from submitit.core.utils import pickle_load

        errors = []
        for job in jobs:
            for task in list(job._sub_jobs) or [job]:
                path = task.paths.result_pickle
                if not path.exists():
                    continue  # the task died: `_wait` records its cell
                try:
                    pickle_load(path)
                except Exception as exc:  # noqa: BLE001 - raised below
                    errors.append(f"{path}: {exc!r}")
        if errors:
            more = f" and {len(errors) - 3} more" if len(errors) > 3 else ""
            raise RuntimeError(
                f"mushin.collect could not load these results of the run's jobs: "
                f"{'; '.join(errors[:3])}{more}. A result holds what the task "
                "returned, so collect where the task's code and the types it "
                "returns import as in the jobs; nothing was written."
            )

    @contextlib.contextmanager
    def _as_submitted(state):
        """Hydra's singletons and OmegaConf's resolvers as the driver sent
        them with the jobs, as each job's own `__call__` sets them; this
        process's are restored after."""
        from omegaconf.basecontainer import BaseContainer

        saved = (Singleton._instances, BaseContainer._resolvers)
        try:
            Singleton.set_state(state)
            yield
        finally:
            Singleton._instances, BaseContainer._resolvers = saved

    def collect_launch(launcher, state, launch, jobs) -> dict:
        """The returns of ``launch``'s cells, by job number: its jobs'
        (``jobs``, from `rebuild_jobs`) from ``launcher``'s own `_wait` (a
        dead job's cell recorded as the driver would have), and the cells
        the driver settled, composed as it composed them, with what they
        returned."""
        from hydra.core.utils import JobStatus

        from mushin._sweep_io import read_metrics_sidecar
        from mushin.workflows import _job_dir, _SkippedRun

        returns = {}
        with _as_submitted(state):
            if launch.cells:
                # Each cell's job, as the launch gave them to `_wait`: the
                # cells of a job of several (#348) share its one object,
                # whose result `_wait` reads once and splits per cell.
                by_id = {str(job.job_id): job for job in jobs}
                params = [
                    (
                        list(cell.overrides),
                        "hydra.sweep.dir",
                        cell.num,
                        f"job_id_for_{cell.num}",
                        state,
                    )
                    for cell in launch.cells
                ]
                cells_jobs = [by_id[cell.job_id] for cell in launch.cells]
                waited = launcher._wait(cells_jobs, params)
                returns.update(zip([c.num for c in launch.cells], waited, strict=True))
            for cell in launch.settled:
                with launcher._composed(
                    cell.overrides, cell.num, f"driver_{cell.num}"
                ) as ret:
                    pass
                if cell.value == "skipped":
                    value = _SkippedRun("not sampled")
                elif cell.value == "none":
                    value = None
                else:
                    # The driver wrote it into the cell's dir from the result
                    # it reused, and no job writes that dir.
                    value = read_metrics_sidecar(_job_dir(ret))
                    if value is None:
                        raise RuntimeError(
                            f"mushin.collect: cell #{cell.num}'s metrics file, "
                            f"which its run reused, is gone from {_job_dir(ret)}."
                        )
                ret.status, ret.return_value = JobStatus.COMPLETED, value
                returns[cell.num] = ret
        return returns

    def _clusters(params) -> str | None:
        """The SLURM clusters a launch's jobs were sent to, as sbatch reads
        them (`--clusters`, or `SBATCH_CLUSTERS`), or None for the default
        one: `squeue` and `scontrol` see another cluster's jobs only when
        told it."""
        extra = params.get("additional_parameters") or {}
        for key in ("clusters", "cluster"):
            value = extra.get(key)
            if isinstance(value, list | tuple):
                value = ",".join(map(str, value))
            if value:
                return str(value)
        return os.environ.get("SBATCH_CLUSTERS") or None

    def _cell_depth(config) -> int:
        """How many levels below the sweep root `hydra.sweep.subdir` puts a
        cell's dir (1 for Hydra's ``${hydra.job.num}``, 2 for
        ``a_${a}/b_${b}``), counted on the subdir's text: the driver cannot
        resolve it for a job (it may read the job's own keys), and resolving
        it would run its resolvers outside a job. `show` reads only the
        root's direct children, so it lists no queued cell of a deeper
        launch."""
        raw = OmegaConf.to_container(config.hydra.sweep, resolve=False)["subdir"]
        return max(len(Path(str(raw)).parts), 1)

    def _task_failed(ret) -> bool:
        """Whether one task's return is a failure, as `jobs_post_process`
        judges a job's: not COMPLETED (its error is not read, which would
        raise it), or the runner's sentinel."""
        from hydra.core.utils import JobStatus

        from mushin.workflows import _FailedRun

        return getattr(ret, "status", None) != JobStatus.COMPLETED or isinstance(
            getattr(ret, "_return_value", None), _FailedRun
        )

    def _keepable(job, exc) -> bool:
        """Whether a completed record the job's cell holds can stand for the
        job's result: a job of one task (every task of a job runs mushin's
        runner in the cell's dir, and the result is the first task's) that
        gave no result, not one that pickled an error in its place."""
        from submitit.core.utils import FailedJobError

        return getattr(job, "num_tasks", 1) == 1 and not isinstance(exc, FailedJobError)

    def _ended(job) -> bool:
        """Whether submitit knows the job has ended; False when it cannot
        tell."""
        try:
            return bool(job.done())
        except Exception:  # noqa: BLE001 - the scheduler may not answer
            return False

    # submitit puts a dead job's whole error stream in its message, and a long
    # run can write megabytes there (a progress bar redrawn with `\r` is one
    # line), so mushin keeps its start and its end, where the cause usually is.
    _HEAD_CHARS, _TAIL_CHARS = 2_000, 20_000

    def _log_of(job, params) -> str:
        """The job's log; for a job of several tasks (several nodes), the
        folder of their logs, as submitit's message names the failed task."""
        paths = job.paths
        if getattr(job, "num_tasks", 1) > 1:
            return f"{paths.folder} (one per task)"
        return str(paths.stdout if params.get("stderr_to_stdout") else paths.stderr)

    def _bounded(exc, folder):
        """``exc``, or, for a submitit error whose message is longer than
        mushin keeps, a new one of the same type with the message's start and
        end. The new one has no traceback: the original's frames hold the
        whole message. The cut text may be the job's stderr or its stdout,
        so the marker names their folder."""
        from submitit.core.utils import FailedJobError, UncompletedJobError

        message = str(exc)
        if not isinstance(exc, (UncompletedJobError, FailedJobError)) or (
            len(message) <= _HEAD_CHARS + _TAIL_CHARS
        ):
            return exc
        cut = len(message) - _HEAD_CHARS - _TAIL_CHARS
        return type(exc)(
            f"{message[:_HEAD_CHARS]}\n[... {cut} characters cut by mushin; "
            f"the job's logs are in {folder} ...]\n{message[-_TAIL_CHARS:]}"
        )

    def _summary(job, exc, params) -> str:
        """One line, shaped like the `repr` of a task's exception, for the
        manifest, the failures list and the warning; submitit's own message,
        with the job's error stream (its start and end when long), goes to
        the cell's mushin_error.txt. submitit raises a FailedJobError only
        for a job that pickled an error instead of a result (it could not
        import the task, say, or save its result); the line says so and
        leaves the error to that file, since reading it from the traceback's
        text named the wrong exception for some tracebacks."""
        from submitit.core.utils import FailedJobError

        try:
            state = job.state
        except Exception:  # noqa: BLE001 - the scheduler may not answer
            state = "unknown"
        if isinstance(exc, FailedJobError):
            what = "raised an error instead of returning"
        else:
            what = "gave no result"
        message = (
            f"submitit job {job.job_id} {what} (state: {state}); "
            f"its log: {_log_of(job, params)}"
        )
        return f"{type(exc).__name__}({message!r})"

    class _Unreturned:
        """A cell of a job of several with no return of its own (#348):
        ``why`` is ``"raised"`` (its Hydra job raised outside its task, or
        its return would not load once pickled, so it would take the whole
        job's result down with it) or ``"not run"`` (an error between two
        cells, a time limit's signal say, stopped the job before it), and
        ``text`` the traceback."""

        def __init__(self, why: str, text: str):
            self.why, self.text = why, text

    class _GroupReturns(dict):
        """What a job of several cells returns: each cell's return, a
        `JobReturn` or an `_Unreturned`, by its Hydra job number (#348).
        `_wait` tells it from a one-cell job's `JobReturn` by its type."""

    class _CellGroup:
        """The callable each job of several cells runs (#348): the cells'
        Hydra jobs, one after another, each through the launcher's own
        `__call__` (so through Hydra's `run_job`, as a one-cell job runs),
        skipping those in ``done``, which a requeued job carries from its
        `checkpoint()`.

        A cell's error never stops the job: Hydra returns a task that raised
        as a FAILED return, and an error outside the task (in `run_job`, or
        a return that would not load) is the cell's `_Unreturned`, so the
        job's later cells still run and its result can still be saved, as
        the other cells' one-cell jobs would. Only an `Exception` is caught:
        submitit's requeue exits with `SystemExit`, and the local
        controller stops a job with `SIGINT`.

        ``done`` maps a finished cell's job number to ``(how, value)``: its
        return (``"returned"``), or the traceback of a cell that
        ``"raised"``."""

        def __init__(self, launcher, done=()):
            self.launcher = launcher
            self.done = dict(done)

        def __call__(self, cells):
            import pickle
            import traceback

            import cloudpickle

            done = dict(self.done)
            try:
                for params in cells:
                    num = params[2]
                    if num in done:
                        continue  # finished before a requeue
                    try:
                        ret = self.launcher(*params)
                        # Loaded back as the driver will load it: a return
                        # that pickles but does not unpickle (an exception
                        # whose __init__ takes two arguments) would make the
                        # whole job's result unreadable. The copy is kept: a
                        # task that refills and returns one dict for every
                        # cell would otherwise leave each cell the last's
                        # values by the time the job returns (#412).
                        ret = pickle.loads(cloudpickle.dumps(ret))
                        done[num] = ("returned", ret)
                    except Exception:  # noqa: BLE001 - the cell's, not the job's
                        done[num] = ("raised", traceback.format_exc())
                    self.done = dict(done)
            except Exception:  # noqa: BLE001 - e.g. a signal's error between cells
                # Raised outside any cell's Hydra job: the job stops here,
                # and returns what it finished.
                text = traceback.format_exc()
                for params in cells:
                    done.setdefault(params[2], ("not run", text))
            # Built only now: the first cell's `run_job` runs Hydra's plugin
            # scan, which executes this module again and rebinds its classes,
            # and cloudpickle pickles an instance of a class since rebound by
            # value, as a class the driver's `_wait` would not know.
            return _GroupReturns(
                {
                    num: value if how == "returned" else _Unreturned(how, value)
                    for num, (how, value) in done.items()
                }
            )

        def checkpoint(self, cells):
            """The plugin's `checkpoint`, carrying the cells this job
            finished, so a requeued job skips them and runs again only the
            one it was in, as that cell's one-cell job would."""
            import submitit

            return submitit.helpers.DelayedSubmission(
                _CellGroup(self.launcher, self.done), cells
            )

    class MushinSlurmLauncher(_MushinSubmititMixin, _plugin_launcher.SlurmLauncher):
        pass

    class MushinLocalLauncher(_MushinSubmititMixin, _plugin_launcher.LocalLauncher):
        pass

    # The plugin's configs, plus mushin's one field: how many cells each job
    # runs, one after another (#348). The launcher's `__init__` takes it out
    # of the params submitit sees.
    @dataclass
    class MushinSlurmQueueConf(_plugin_config.SlurmQueueConf):
        _target_: str = "hydra_plugins.mushin_submitit.launcher.MushinSlurmLauncher"
        cells_per_job: int = 1

    @dataclass
    class MushinLocalQueueConf(_plugin_config.LocalQueueConf):
        _target_: str = "hydra_plugins.mushin_submitit.launcher.MushinLocalLauncher"
        cells_per_job: int = 1

    ConfigStore.instance().store(
        group="hydra/launcher",
        name="mushin_submitit_slurm",
        node=MushinSlurmQueueConf(),
        provider="mushin",
    )
    ConfigStore.instance().store(
        group="hydra/launcher",
        name="mushin_submitit_local",
        node=MushinLocalQueueConf(),
        provider="mushin",
    )
