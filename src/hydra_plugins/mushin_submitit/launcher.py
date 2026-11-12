# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The launchers behind `run(launcher="submitit_slurm")` and `"submitit_local"`.

They are hydra-submitit-launcher's own launchers, changed in one place: the
plugin waits on its jobs in a list comprehension, so the first job that ends
without a result (killed out of memory, on a failed node, by `scancel` or at
a time limit before Python could act, or raised outside the task, as when it
cannot import it) raises from the driver while the later jobs still run,
unobserved, and the sweep records nothing. Here every job is waited on,
and a job that ends without a result comes back to mushin as a failed cell,
which `on_error` then handles as it does a task that raised (#345).

`run()` sends the plugin's two names here (`mushin_submitit_slurm` and
`mushin_submitit_local`); their config is the plugin's, field for field, so
`launcher_config` and `submitit_slurm_config` work unchanged.

Hydra imports every module under `hydra_plugins` whenever any Hydra
application starts, and reports an ImportError there as a warning (an error
under `-W error`). So this module imports nothing heavy, and without the
plugin it defines nothing. mushin itself is imported only when a job died.
"""

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
        """Submit as the plugin does; wait on every job (#345).

        Holds no state of its own: the launcher is pickled into every job
        (`map_array(self, ...)`) and into `checkpoint()`'s resubmission."""

        def launch(self, job_overrides, initial_job_idx):
            jobs, job_params = self._submit(job_overrides, initial_job_idx)
            return self._wait(jobs, job_params)

        def _submit(self, job_overrides, initial_job_idx):
            """hydra-submitit-launcher 1.2.0's `launch()` up to its submission,
            unchanged (`tests/test_submitit_launcher.py` pins that body, so a
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

            For mushin's runner, a job that ended without a result is a
            failed cell (`_dead_return`), and the later jobs are still waited
            on. An error that cannot become one (a job not known to have
            ended, or a cell that cannot be composed here) is raised once
            every job has been waited on. Any other task function sees the
            first error at once, as the plugin raises it."""
            from submitit.core.utils import FailedJobError, UncompletedJobError

            collects = getattr(self.task_function, "_mushin_collects_dead_jobs", False)
            returns, unrecorded = [], None
            for job, params in zip(jobs, job_params, strict=True):
                try:
                    returns.append(job.results()[0])
                    continue
                except (UncompletedJobError, FailedJobError) as exc:
                    # submitit raises these once the job has ended.
                    if not collects:
                        raise
                    error, ended = exc, True
                except Exception as exc:  # noqa: BLE001 - e.g. an unreadable result
                    if not collects:
                        raise
                    error, ended = exc, _ended(job)
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

        def _dead_return(self, job, overrides, idx, exc):
            """The JobReturn for a job that ended without a result, with the
            status and value mushin's runner gives it once it has recorded
            the cell (`_TaskRunner.record_dead_cell`). The job is composed
            as the plugin's `__call__` and Hydra's `run_job` compose it
            (`tests/test_submitit_launcher.py` pins the part of `run_job`
            mirrored here), in the driver, with the global HydraConfig
            restored after."""
            from hydra.core.hydra_config import HydraConfig

            from mushin.workflows import _DeadJob

            assert self.hydra_context is not None and self.config is not None
            runner = self.task_function
            # The summary reads the whole message; the cell keeps a bounded one.
            dead = _DeadJob(
                _bounded(exc, _log_of(job, self.params)),
                _summary(job, exc, self.params),
            )
            outer = HydraConfig.instance().cfg
            try:
                config = self.hydra_context.config_loader.load_sweep_config(
                    self.config, list(overrides)
                )
                with open_dict(config.hydra.job) as hydra_job:
                    hydra_job.id = str(job.job_id)
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
                # Inside the try: the cell's config-group choices come from
                # the HydraConfig set above.
                ret.status, ret.return_value = runner.record_dead_cell(ret, dead)
            finally:
                HydraConfig.instance().cfg = outer
            return ret

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
    _TRACE_LINE_CHARS = 200

    def _log_of(job, params) -> str:
        """The job's log; for a job of several tasks (several nodes), the
        folder of their logs, as submitit's message names the failed task."""
        paths = job.paths
        if getattr(job, "num_tasks", 1) > 1:
            return f"{paths.folder} (one per task)"
        return str(paths.stdout if params.get("stderr_to_stdout") else paths.stderr)

    def _bounded(exc, log_file):
        """``exc``, or, for a submitit error whose message is longer than
        mushin keeps, one of the same type with the message's start and end."""
        from submitit.core.utils import FailedJobError, UncompletedJobError

        message = str(exc)
        if not isinstance(exc, (UncompletedJobError, FailedJobError)) or (
            len(message) <= _HEAD_CHARS + _TAIL_CHARS
        ):
            return exc
        cut = len(message) - _HEAD_CHARS - _TAIL_CHARS
        return type(exc)(
            f"{message[:_HEAD_CHARS]}\n[... {cut} characters cut by mushin; "
            f"the whole log: {log_file} ...]\n{message[-_TAIL_CHARS:]}"
        )

    def _last_trace_line(message: str) -> str:
        """The error line of the traceback a FailedJobError carries (between
        submitit's dashed rules), on one line; empty if there is none."""
        parts = message.split("\n----------------------\n")
        if len(parts) < 3:
            return ""
        lines = [line.strip() for line in parts[1].splitlines() if line.strip()]
        line = lines[-1] if lines else ""
        if len(line) > _TRACE_LINE_CHARS:
            line = line[: _TRACE_LINE_CHARS - 3] + "..."
        return line

    def _summary(job, exc, params) -> str:
        """One line, shaped like the `repr` of a task's exception, for the
        manifest, the failures list and the warning; submitit's own message,
        with the job's error stream (its start and end when long), goes to
        the cell's mushin_error.txt. A FailedJobError means the job raised
        outside the task's own error handling (it could not import the task,
        or could not save its result), so its line names that error."""
        from submitit.core.utils import FailedJobError

        try:
            state = job.state
        except Exception:  # noqa: BLE001 - the scheduler may not answer
            state = "unknown"
        if isinstance(exc, FailedJobError):
            raised = _last_trace_line(str(exc))
            what = f"raised {raised}" if raised else "raised an error"
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
