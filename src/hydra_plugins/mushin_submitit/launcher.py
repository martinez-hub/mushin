# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The launchers behind `run(launcher="submitit_slurm")` and `"submitit_local"`.

They are hydra-submitit-launcher's own launchers, changed in one place: the
plugin waits on its jobs in a list comprehension, so the first job that ends
without a result (killed at its time limit, out of memory, on a failed node,
or exited outside Python) raises from the driver while the later jobs still
run, unobserved, and the sweep records nothing. Here every job is waited on,
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
    from hydra.core.utils import JobReturn, JobStatus, filter_overrides
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
            """Every job's result, in submission order. A job that ends
            without one is a failed cell for mushin's runner; for any other
            task function it raises, as the plugin does."""
            from submitit.core.utils import FailedJobError, UncompletedJobError

            collects = getattr(self.task_function, "_mushin_collects_dead_jobs", False)
            returns = []
            for job, params in zip(jobs, job_params, strict=True):
                try:
                    returns.append(job.results()[0])
                except (UncompletedJobError, FailedJobError) as exc:
                    if not collects:
                        raise
                    overrides, _, idx = params[0], params[1], params[2]
                    returns.append(self._dead_return(job, overrides, idx, exc))
            return returns

        def _dead_return(self, job, overrides, idx, exc):
            """The JobReturn the job would have sent had it run, carrying a
            `_DeadJob`. COMPLETED, not FAILED: Hydra's sweeper re-raises a
            FAILED return's exception whatever `on_error` says. Composed as
            the plugin's `__call__` and Hydra's `run_job` compose a job,
            here in the driver, with the global HydraConfig restored after."""
            from hydra.core.hydra_config import HydraConfig

            from mushin.workflows import _DeadJob

            assert self.hydra_context is not None and self.config is not None
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
            finally:
                HydraConfig.instance().cfg = outer
            ret.status = JobStatus.COMPLETED
            ret.return_value = _DeadJob(exc, _summary(job, exc, self.params))
            return ret

    def _summary(job, exc, params) -> str:
        """One line for the manifest, the failures list and the warning;
        submitit's own message, with its log tail, goes to the cell's
        mushin_error.txt."""
        try:
            state = job.state
        except Exception:  # noqa: BLE001 - the scheduler may not answer
            state = "unknown"
        paths = job.paths
        log_file = paths.stdout if params.get("stderr_to_stdout") else paths.stderr
        return (
            f"{type(exc).__name__}: submitit job {job.job_id} ended without a "
            f"result (state: {state}); its log: {log_file}"
        )

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
