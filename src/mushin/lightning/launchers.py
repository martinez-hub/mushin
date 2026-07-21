# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from time import sleep
from typing import Any, TypeVar

import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra_zen import load_from_yaml
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import TrainerFn
from torch import distributed

from .._compatibility import PL_VERSION, Version

R = TypeVar("R")

# Env vars mushin itself set (single-node subprocess launcher). Under an external
# launcher (SLURM/torchrun) these are scheduler-owned and mushin sets none, so
# teardown leaves them alone.
_MUSHIN_SET_ENV: set[str] = set()


def _set_env(name: str, value: str) -> None:
    os.environ[name] = value
    _MUSHIN_SET_ENV.add(name)


def _setup_environment() -> None:
    if distributed.is_initialized():
        distributed.destroy_process_group()


def _validate_external_world_size(
    num_nodes, num_processes, cluster_environment
) -> None:
    """Under an external launcher (SLURM/torchrun), fail fast if the number of
    launched processes doesn't match num_nodes x devices-per-node — the #1
    multi-node footgun (a mismatch otherwise hangs at rendezvous, OOMs, or
    silently runs single-GPU). No-op for the single-node subprocess path."""
    if (
        cluster_environment is None
        or not cluster_environment.creates_processes_externally
    ):
        return
    expected = int(num_nodes) * int(num_processes)
    actual = int(cluster_environment.world_size())
    if actual != expected:
        raise RuntimeError(
            f"distributed world size mismatch: the launcher started {actual} process(es), "
            f"but the Trainer expects num_nodes={num_nodes} x devices={num_processes} "
            f"= {expected}. For DDP, set the launcher's tasks-per-node equal to "
            f"GPUs-per-node (== Trainer `devices`). See the multi-node guide."
        )


def _teardown() -> None:
    # Remove only the env vars mushin set itself, so consecutive multirun jobs
    # start fresh without stomping scheduler-owned vars under SLURM/torchrun.
    for name in list(_MUSHIN_SET_ENV):
        os.environ.pop(name, None)
    _MUSHIN_SET_ENV.clear()
    # PL_GLOBAL_SEED is Lightning's, not scheduler-owned; safe to reset each job.
    os.environ.pop("PL_GLOBAL_SEED", None)


def _global_rank(node_rank: int, num_processes: int, local_rank: int) -> int:
    """Global rank = node_rank * (GPUs per node) + local_rank."""
    return int(node_rank) * int(num_processes) + int(local_rank)


def _subprocess_call(
    local_rank: int, global_rank: int, testing: bool, predicting: bool
) -> None:
    env_copy = os.environ.copy()
    env_copy["LOCAL_RANK"] = f"{local_rank}"
    # CWD is the Hydra working directory
    cwd = os.getcwd()
    os_cwd = (
        f'"{cwd}"'  # this is needed to handle characters like `=` in the directory name
    )

    command = [
        sys.executable,
        "-m",
        "mushin.lightning._pl_main",
    ]
    hydra_cfg = HydraConfig.get()

    hydra_output = (
        os.path.join(cwd, hydra_cfg.output_subdir)
        if hydra_cfg.output_subdir is not None
        else cwd
    )

    # Validate that minimal configuration requirements
    config = Path(hydra_output) / "config.yaml"
    if not config.exists():
        raise FileNotFoundError(
            f"{config} not found; HydraDDP re-launches each rank from the "
            "job's saved Hydra config and cannot proceed without it."
        )
    cfg = load_from_yaml(config)
    if "trainer" not in cfg or "module" not in cfg:
        raise ConfigAttributeError(
            "Missing configurations `trainer` and `module` are required for use with HydraDDP.  See documentation for further details."
        )

    # create the command for CLI
    command += ["-cp", hydra_output, "-cn", "config.yaml"]

    # Set flag to run Trainer.fit or Trainer.test in `_pl_main.py`
    command += ["++pl_testing=" + ("false" if not testing else "true")]

    # Set flag to run Trainer.fit or Trainer.test in `_pl_main.py`
    command += ["++pl_predicting=" + ("false" if not predicting else "true")]

    # Set flag for local rank
    command += [f"++pl_local_rank={local_rank}"]

    command += [
        f"hydra.run.dir={os_cwd}",
        f"hydra.output_subdir=.pl_hydra_rank_{global_rank}",
        f"hydra.job.name={hydra_cfg.job.name}",
    ]
    return subprocess.Popen(command, env=env_copy, cwd=cwd)


if PL_VERSION >= Version(1, 6, 0):
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.strategies.fsdp import FSDPStrategy
    from pytorch_lightning.strategies.launchers.subprocess_script import (
        _SubprocessScriptLauncher,
    )

    # Private Lightning helpers mirrored from the base launcher's lifecycle:
    # the observer reaps child ranks if rank 0 dies (no orphaned GPU
    # processes), and the thread cap stops N ranks from each spawning
    # cpu_count() OMP/torch threads. Guarded so a future PL relocation
    # degrades to the old behavior instead of breaking imports.
    try:
        from pytorch_lightning.strategies.launchers.subprocess_script import (
            _launch_process_observer,
        )
    except ImportError:  # pragma: no cover
        _launch_process_observer = None
    try:
        from lightning_fabric.utilities.distributed import (
            _set_num_threads_if_needed,
        )
    except ImportError:  # pragma: no cover
        _set_num_threads_if_needed = None

    class _HydraReattachMixin:
        """Hydra-aware launcher behavior shared by ``HydraDDP`` and ``HydraFSDP``:
        reattach each rank via the job's saved ``config.yaml`` (not ``sys.argv``,
        which in a Hydra sweep would re-run the wrong job), fail fast on an external
        world-size mismatch, and scope env-var teardown to what mushin set so
        consecutive multirun jobs start clean without stomping scheduler-owned vars.
        """

        def setup_environment(self) -> None:
            _setup_environment()
            # Validate BEFORE super().setup_environment(): that is where Lightning
            # initializes the process group / rendezvous, which is exactly what
            # hangs when the launcher started the wrong number of ranks. Failing
            # fast here turns a hang into a legible error.
            _validate_external_world_size(
                self.num_nodes, self.num_processes, self.cluster_environment
            )
            super().setup_environment()

        def _configure_launcher(self) -> None:
            if self.cluster_environment is None:  # pragma: no cover
                raise TypeError(f"{type(self).__name__}.cluster_environment is None")
            if not self.cluster_environment.creates_processes_externally:
                self._launcher = _HydraReattachLauncher(
                    self.cluster_environment, self.num_processes, self.num_nodes
                )
                self._rank_0_will_call_children_scripts = True

        def teardown(self) -> None:
            """Additional teardown so consecutive Hydra multirun jobs start fresh."""
            super().teardown()
            _teardown()

    class HydraDDP(_HydraReattachMixin, DDPStrategy):  # type: ignore
        """DDP Strategy that supports Hydra run and multirun jobs.

        This strategy assumes a PyTorch Lightning `Trainer.fit` or `Trainer.test` has been configured
        to execute via Hydra.  It requires that Hydra saves a `config.yaml` in the current working directory with the following keys/properties set::

           ├── Config
           │    ├── trainer: A `pytorch_lightning.Trainer` configuration
           │    ├── module: A `pytorch_lightning.LightningModule` configuration
           │    ├── datamodule: [OPTIONAL] A `pytorch_lightning.LightningDataModule` configuration

        This strategy will launch a child subprocesses for additional GPU beyond the first using the following base command::

           python -m mushin.lightning._pl_main -cp <path to config.yaml> -cn config.yaml

        Examples
        --------

        First define a Hydra configuration using hydra-zen:

        >>> import pytorch_lightning as pl
        ... from hydra_zen import builds, make_config,
        ... from mushin import HydraDDP
        ... from mushin.testing.lightning import SimpleLightningModule
        ...
        ... TrainerConfig = builds(
        ...     pl.Trainer,
        ...     accelerator="auto",
        ...     gpus=2,
        ...     max_epochs=1,
        ...     fast_dev_run=True,
        ...     strategy=builds(HydraDDP),
        ...     populate_full_signature=True
        ... )
        ...
        ... ModuleConfig = builds(SimpleLightningModule)
        ...
        ... Config = make_config(
        ...     trainer=TrainerConfig,
        ...     module=ModuleConfig
        ... )

        Next, define a task function to execute the Hydra job:

        >>> from hydra_zen import instantiate
        >>> def task_function(cfg):
        ...     obj = instantiate(cfg)
        ...     obj.trainer.fit(obj.module)

        Launch the Hydra+Lightning DDP job

        >>> from hydra_zen import launch
        >>> job = launch(Config, task_function)

        ``HydraDDP`` also supports ``LightningDataModule`` configuration.

        >>> DataModuleConfig = ... # A LightningDataModule config
        >>> Config = make_config(
        ...     trainer=TrainerConfig,
        ...     module=ModuleConfig
        ...     datamodule=DataModuleconfig
        ... )

        Next define a task function to execute the Hydra job:

        >>> from hydra_zen import instantiate
        >>> def task_function(cfg):
        ...     obj = instantiate(cfg)
        ...     obj.trainer.fit(obj.module, datamodule=obj.datamodule)

        Launch the Hydra+Lightning DDP job:

        >>> from hydra_zen import launch
        >>> job = launch(Config, task_function)
        """

    class HydraFSDP(_HydraReattachMixin, FSDPStrategy):  # type: ignore
        """Fully-Sharded Data Parallel strategy that works under Hydra ``--multirun``.

        Like :class:`HydraDDP`, but for sharded training: it replaces Lightning's
        stock ``FSDPStrategy`` subprocess launcher (which re-execs the script with
        ``sys.argv`` and, in a sweep, spawns the wrong job) with mushin's launcher,
        which reattaches each rank via the job's saved ``config.yaml``. FSDP shards
        parameters/gradients/optimizer state across ranks; mushin's results and
        significance analysis are unchanged from a single-GPU run.

        Requires Hydra to save a ``config.yaml`` (with ``trainer`` and ``module``
        keys) in the job's output dir — the same contract as :class:`HydraDDP`.
        Configure it with hydra-zen, e.g. ``strategy=builds(HydraFSDP)`` on a
        ``builds(pl.Trainer, ...)`` config.
        """

    class _HydraReattachLauncher(_SubprocessScriptLauncher):
        @property
        def is_interactive_compatible(self) -> bool:  # pragma: no cover
            return True

        def launch(
            self,
            function: Callable[..., R],
            *args: Any,
            trainer: Trainer,
            **kwargs: Any,
        ) -> R:
            """Creates new processes, then calls the given function.

            Parameters
            ----------
            function : Callable[[...], ReturnType]
                A callback function to execute after all processes have been created.
                It is up to the implementation of this function to synchronize the processes, e.g., with barriers.

            *args : Any
                Optional positional arguments to be passed to the given function.

            trainer : pytorch_lightning.Trainer
                Optional reference to the pytorch_lightning.Trainer`.

            **kwargs : Any
                Optional keyword arguments to be passed to the given function.

            Returns
            -------
            ReturnType
            """
            del trainer  # unused
            if not self.cluster_environment.creates_processes_externally:
                testing = function.__name__ == "_test_impl"
                predicting = function.__name__ == "_predict_impl"
                self._call_children_scripts(testing=testing, predicting=predicting)
                # Reap child ranks if this (rank 0) process dies, instead of
                # leaving them orphaned holding GPU memory.
                if _launch_process_observer is not None:
                    _launch_process_observer(self.procs)
            # Cap intra-op threads so N ranks don't each spawn cpu_count()
            # thread pools (mirrors the base launcher).
            if _set_num_threads_if_needed is not None:
                _set_num_threads_if_needed(num_processes=self.num_processes)

            return function(*args, **kwargs)

        def _call_children_scripts(self, testing: bool, predicting: bool):
            # bookkeeping of spawned processes
            self._check_can_spawn_children()
            # Track the children like the base class: kill()/signal forwarding
            # iterate self.procs, and the process observer watches it.
            self.procs = []

            # DDP Environment variables (tracked so _teardown clears only these)
            _set_env("MASTER_ADDR", self.cluster_environment.main_address)
            _set_env("MASTER_PORT", str(self.cluster_environment.main_port))
            _set_env("NODE_RANK", str(self.cluster_environment.node_rank()))
            _set_env("LOCAL_RANK", str(self.cluster_environment.local_rank()))
            _set_env("WORLD_SIZE", f"{self.num_processes * self.num_nodes}")

            node_rank = self.cluster_environment.node_rank()
            for local_rank in range(1, self.num_processes):
                proc = _subprocess_call(
                    local_rank,
                    _global_rank(node_rank, self.num_processes, local_rank),
                    testing,
                    predicting,
                )
                self.procs.append(proc)

                # starting all processes at once can cause issues
                # with dataloaders delay between 1-10 seconds
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)

else:  # pragma: no cover
    from pytorch_lightning.plugins.training_type.ddp import DDPPlugin  # type: ignore

    class HydraDDP(DDPPlugin):
        """DDP Strategy that supports Hydra run and multirun jobs.

        This strategy assumes a PyTorch Lightning `Trainer.fit` or `Trainer.test` has been configured
        to execute via Hydra.  It requires that Hydra saves a `config.yaml` in the current working directory with the following keys/properties set::

           ├── Config
           │    ├── trainer: A `pytorch_lightning.Trainer` configuration
           │    ├── module: A `pytorch_lightning.LightningModule` configuration
           │    ├── datamodule: [OPTIONAL] A `pytorch_lightning.LightningDataModule` configuration

        This strategy will launch a child subprocesses for additional GPU beyond the first using the following base command::

           python -m mushin.lightning._pl_main -cp <path to config.yaml> -cn config.yaml

        Examples
        --------

        First define a Hydra configuration using hydra-zen:

        >>> import pytorch_lightning as pl
        ... from hydra_zen import builds, make_config,
        ... from mushin import HydraDDP
        ... from mushin.testing.lightning import SimpleLightningModule
        ...
        ... TrainerConfig = builds(
        ...     pl.Trainer,
        ...     accelerator="auto",
        ...     gpus=2,
        ...     max_epochs=1,
        ...     fast_dev_run=True,
        ...     strategy=builds(HydraDDP),
        ...     populate_full_signature=True
        ... )
        ...
        ... ModuleConfig = builds(SimpleLightningModule)
        ...
        ... Config = make_config(
        ...     trainer=TrainerConfig,
        ...     module=ModuleConfig
        ... )

        Next define a task function to execute the Hydra job:

        >>> from hydra_zen import instantiate
        >>> def task_function(cfg):
        ...     obj = instantiate(cfg)
        ...     obj.trainer.fit(obj.module)

        Launch the Hydra+Lightning DDP job:

        >>> from hydra_zen import launch
        >>> job = launch(Config, task_function)

        ``HydraDDP`` also supports ``LightningDataModule`` configuration.

        >>> DataModuleConfig = ... # A LightningDataModule config
        >>> Config = make_config(
        ...     trainer=TrainerConfig,
        ...     module=ModuleConfig
        ...     datamodule=DataModuleconfig
        ... )

        Next, define a task function to execute the Hydra job:

        >>> from hydra_zen import instantiate
        >>> def task_function(cfg):
        ...     obj = instantiate(cfg)
        ...     obj.trainer.fit(obj.module, datamodule=obj.datamodule)

        Launch the Hydra+Lightning DDP job:

        >>> from hydra_zen import launch
        >>> job = launch(Config, task_function)
        """

        def setup_environment(self) -> None:
            _setup_environment()
            super().setup_environment()

        def _call_children_scripts(self):
            if self.lightning_module is None:  # pragma: no cover
                raise TypeError("HydraDDP.lightning_module is None")

            if self.lightning_module.trainer is None:  # pragma: no cover
                raise TypeError("HydraDDP.lightning_module.trainer is None")

            if self.cluster_environment is None:  # pragma: no cover
                raise TypeError("HydraDDP.cluster_environment is None")

            # bookkeeping of spawned processes
            self._check_can_spawn_children()

            # DDP Environment variables
            os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
            os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())

            # allow the user to pass the node rank
            os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
            os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())
            os.environ["WORLD_SIZE"] = f"{self.num_processes * self.num_nodes}"

            self.interactive_ddp_procs = []
            node_rank = self.cluster_environment.node_rank()
            for local_rank in range(1, self.num_processes):
                testing = self.lightning_module.trainer.state.fn == TrainerFn.TESTING
                predicting = (
                    self.lightning_module.trainer.state.fn == TrainerFn.PREDICTING
                )
                _subprocess_call(
                    local_rank,
                    _global_rank(node_rank, self.num_processes, local_rank),
                    testing=testing,
                    predicting=predicting,
                )

                # starting all processes at once can cause issues
                # with dataloaders delay between 1-10 seconds
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)

            self._rank_0_has_called_call_children_scripts = True

        def teardown(self) -> None:
            """Performs additional teardown steps for PL to allow for Hydra multirun jobs."""
            super().teardown()
            _teardown()
