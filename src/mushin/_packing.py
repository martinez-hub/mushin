# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Opt-in GPU packing: co-locate small sweep jobs on shared devices."""

from __future__ import annotations

import os
import warnings


def pin_gpu_round_robin(num_gpus: int, job_index: int | None = None) -> int:
    """Pin this process to a single GPU (round-robin) via ``CUDA_VISIBLE_DEVICES``.

    Call this at the TOP of your task function, before any CUDA use (importing
    torch is fine; the first CUDA op is not). It maps a job to one physical GPU
    with ``job_index % num_gpus`` and returns the chosen index.

    Parameters
    ----------
    num_gpus : int
        Number of physical GPUs to spread jobs across (must be >= 1).
    job_index : int or None
        The job's index. Defaults to the current Hydra sweep index
        (``HydraConfig.get().hydra.job.num``); pass it explicitly outside Hydra.

    Notes
    -----
    This only maps a job to a device. Running ``num_gpus * jobs_per_gpu`` jobs
    concurrently (so ``jobs_per_gpu`` land on each GPU) is set by your launcher,
    e.g. ``hydra.launcher.n_jobs``. Placement does not change results, only
    scheduling.
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1; got {num_gpus}")

    if job_index is None:
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf

        # hydra.job.num is populated only by the --multirun sweep launcher; in
        # single-run (plain @hydra.main) it is MISSING even though HydraConfig is
        # initialized. Catch both so the user gets this clear message instead of an
        # opaque OmegaConf MissingMandatoryValue.
        no_index = RuntimeError(
            "pin_gpu_round_robin: no Hydra multirun job index (hydra.job.num) is "
            "available — it is set only inside a Hydra --multirun sweep, not in "
            "single-run mode or outside Hydra. Pass job_index=... explicitly."
        )
        if not HydraConfig.initialized():
            raise no_index
        job = HydraConfig.get().hydra.job
        if OmegaConf.is_missing(job, "num"):
            raise no_index
        job_index = int(job.num)

    gpu = job_index % num_gpus

    try:
        import torch

        if torch.cuda.is_initialized():
            warnings.warn(
                "pin_gpu_round_robin: CUDA is already initialized, so setting "
                "CUDA_VISIBLE_DEVICES now has no effect on this process. Call it at "
                "the top of your task function, before any CUDA use.",
                UserWarning,
                stacklevel=2,
            )
    except ImportError:  # pragma: no cover
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return gpu
