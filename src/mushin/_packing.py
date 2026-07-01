# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Opt-in GPU packing: co-locate small sweep jobs on shared devices."""

from __future__ import annotations

import os


def pin_gpu_round_robin(num_gpus: int, job_index: int | None = None) -> int:
    """Pin this process to a single GPU (round-robin) via ``CUDA_VISIBLE_DEVICES``.

    Call this at the TOP of your task function, before any CUDA use (importing
    torch is fine; the first CUDA op is not). It maps a job to one physical GPU
    with ``job_index % num_gpus`` and returns that round-robin slot.

    Parameters
    ----------
    num_gpus : int
        Number of physical GPUs to spread jobs across (must be >= 1). If
        ``CUDA_VISIBLE_DEVICES`` is already set (e.g. by SLURM/containers),
        ``num_gpus`` must not exceed the number of devices it exposes.
    job_index : int or None
        The job's index. Defaults to the current Hydra sweep index
        (``HydraConfig.get().job.num``); pass it explicitly outside Hydra.

    Returns
    -------
    int
        The round-robin slot ``job_index % num_gpus``. ``CUDA_VISIBLE_DEVICES``
        is set to the physical device for that slot: the slot number itself, or —
        when an allocation is already present — the slot-th entry of that list.

    Raises
    ------
    ValueError
        If ``num_gpus < 1``, or if it exceeds an existing ``CUDA_VISIBLE_DEVICES``
        allocation.
    RuntimeError
        If ``job_index`` is None and no Hydra multirun index is available, or if
        CUDA is already initialized in this process (pinning cannot take effect).

    Notes
    -----
    This only maps a job to a device. Running ``num_gpus * jobs_per_gpu`` jobs
    concurrently (so ``jobs_per_gpu`` land on each GPU) is set by your launcher,
    e.g. ``hydra.launcher.n_jobs``. Placement does not change results, only
    scheduling.

    Each job must run in a *fresh* process: once CUDA is initialized, its visible
    devices are fixed, so a reused worker keeps the first job's GPU. If your
    launcher reuses workers across jobs this raises rather than silently mispin;
    see the GPU-packing guide (Ray handles fractional-GPU packing without this
    constraint).
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1; got {num_gpus}")

    if job_index is None:
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf

        # HydraConfig.get() returns the HydraConf node itself, so the job lives at
        # `.job` (as in Hydra's own `HydraConfig.get().job.name`), not under a
        # `.hydra` wrapper. `job.num` is populated only by the --multirun sweep
        # launcher; in single-run (plain @hydra.main) it stays MISSING even though
        # HydraConfig is initialized. Catch both so the user gets this clear
        # message instead of an opaque OmegaConf error.
        no_index = RuntimeError(
            "pin_gpu_round_robin: no Hydra multirun job index (hydra.job.num) is "
            "available — it is set only inside a Hydra --multirun sweep, not in "
            "single-run mode or outside Hydra. Pass job_index=... explicitly."
        )
        if not HydraConfig.initialized():
            raise no_index
        job = HydraConfig.get().job
        if OmegaConf.is_missing(job, "num"):
            raise no_index
        job_index = int(job.num)

    try:
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError(
                "pin_gpu_round_robin: CUDA is already initialized in this process, "
                "so setting CUDA_VISIBLE_DEVICES now has no effect. Call it at the "
                "top of your task function, before any CUDA use. In a sweep, make "
                "sure each job runs in a fresh worker process — reused workers keep "
                "the first job's GPU. See the GPU-packing guide."
            )
    except ImportError:  # pragma: no cover
        pass

    slot = job_index % num_gpus

    # Respect an existing allocation: schedulers/containers often restrict this
    # process to a device subset (e.g. "4,5" or GPU UUIDs). Index into that pool
    # instead of overwriting it with a bare ordinal that would escape onto the
    # wrong physical devices.
    existing = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [d for d in existing.split(",") if d != ""]
    if visible:
        if num_gpus > len(visible):
            raise ValueError(
                f"num_gpus={num_gpus} exceeds the {len(visible)} device(s) already "
                f"visible via CUDA_VISIBLE_DEVICES={existing!r}"
            )
        device = visible[slot]
    else:
        device = str(slot)

    os.environ["CUDA_VISIBLE_DEVICES"] = device
    return slot
