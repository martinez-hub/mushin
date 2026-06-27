# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Helpers for multi-node training on SLURM/Elastic clusters."""

from __future__ import annotations

import os
from typing import Any

from pytorch_lightning import seed_everything


def submitit_slurm_config(
    *,
    nodes: int,
    gpus_per_node: int,
    cpus_per_task: int = 1,
    partition: str | None = None,
    timeout_min: int = 60,
    mem_gb: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a ``hydra-submitit-launcher`` SLURM config for multi-node DDP.

    ``tasks_per_node`` is derived as ``gpus_per_node`` so the two can never desync
    (DDP needs one SLURM task per GPU). Returns a plain dict you wire into your
    Hydra ``hydra/launcher`` config (see the multi-node guide); it submits nothing.
    Extra keyword args (e.g. ``account``, ``qos``) pass through verbatim.
    """
    if int(nodes) < 1 or int(gpus_per_node) < 1:
        raise ValueError(
            f"`nodes` and `gpus_per_node` must be >= 1; got nodes={nodes}, "
            f"gpus_per_node={gpus_per_node}"
        )
    if "tasks_per_node" in extra:
        raise ValueError(
            "`tasks_per_node` is derived from `gpus_per_node` (one DDP task per "
            "GPU) and must not be overridden — that desync is the footgun this "
            "helper exists to prevent."
        )
    cfg: dict[str, Any] = {
        "nodes": int(nodes),
        "gpus_per_node": int(gpus_per_node),
        "tasks_per_node": int(gpus_per_node),  # one DDP rank per GPU
        "cpus_per_task": int(cpus_per_task),
        "timeout_min": int(timeout_min),
    }
    if partition is not None:
        cfg["partition"] = partition
    if mem_gb is not None:
        cfg["mem_gb"] = int(mem_gb)
    cfg.update(extra)
    return cfg


def seed_everything_per_rank(base: int, workers: bool = True) -> int:
    """Seed each process with ``base + global_rank`` so a multi-GPU/-node run is as
    reproducible as a single-GPU run (each rank gets a distinct but deterministic
    seed). Reads the global rank from ``RANK`` (preferred) or ``SLURM_PROCID``,
    defaulting to 0. Returns the seed used."""
    rank_str = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"
    seed = int(base) + int(rank_str)
    seed_everything(seed, workers=workers)
    return seed
