# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The launchers behind `run(launcher="submitit_slurm")` and `"submitit_local"`.

They are hydra-submitit-launcher's own launchers, changed in two places.
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
(#345).

`run()` sends the plugin's two names here (`mushin_submitit_slurm` and
`mushin_submitit_local`); their config is the plugin's, field for field, so
`launcher_config` and `submitit_slurm_config` work unchanged.

Hydra imports every module under `hydra_plugins` whenever any Hydra
application starts, and reports an ImportError there as a warning (an error
under `-W error`). So this module imports nothing heavy, and without the
plugin it defines nothing. mushin itself is imported only when a job died
or a job of several tasks returned.
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
        """Submit as the plugin does, but only the cells that need a job;
        wait on every job (#345).

        Holds no state of its own: the launcher is pickled into every job
        (`map_array(self, ...)`) and into `checkpoint()`'s resubmission."""

        def launch(self, job_overrides, initial_job_idx):
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
                return [settled[n] for n in numbers]
            jobs, job_params = self._submit(
                job_overrides, initial_job_idx, skip=settled
            )
            waited = self._wait(jobs, job_params)
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

            jobs = executor.map_array(self, *zip(*job_params))  # noqa: B905
            return jobs, job_params

        def _wait(self, jobs, job_params):
            """Every job's result, in submission order.

            For mushin's runner, a job that ended without a result comes
            back as its cell's record says (`_dead_return`): a failed cell,
            or the result the cell recorded, and the later jobs are still
            waited on. An error that cannot become one (a job not known to have
            ended, or a cell that cannot be composed here) is raised once
            every job has been waited on. Any other task function sees the
            first error at once, as the plugin raises it."""
            from submitit.core.utils import FailedJobError, UncompletedJobError

            collects = getattr(self.task_function, "_mushin_collects_dead_jobs", False)
            returns, unrecorded = [], None
            for job, params in zip(jobs, job_params, strict=True):
                try:
                    results = job.results()
                except (UncompletedJobError, FailedJobError) as exc:
                    # submitit raises these once the job has ended.
                    if not collects:
                        raise
                    error, ended = exc, True
                except Exception as exc:  # noqa: BLE001 - e.g. an unreadable result
                    if not collects:
                        raise
                    error, ended = exc, _ended(job)
                else:
                    # Outside the try: the job gave its results, so nothing
                    # here may send it down the dead-job path.
                    several = collects and len(results) > 1
                    returns.append(
                        self._job_return(job, results) if several else results[0]
                    )
                    continue
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

    class MushinSlurmLauncher(_MushinSubmititMixin, _plugin_launcher.SlurmLauncher):
        pass

    class MushinLocalLauncher(_MushinSubmititMixin, _plugin_launcher.LocalLauncher):
        pass

    @dataclass
    class MushinSlurmQueueConf(_plugin_config.SlurmQueueConf):
        _target_: str = "hydra_plugins.mushin_submitit.launcher.MushinSlurmLauncher"

    @dataclass
    class MushinLocalQueueConf(_plugin_config.LocalQueueConf):
        _target_: str = "hydra_plugins.mushin_submitit.launcher.MushinLocalLauncher"

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
