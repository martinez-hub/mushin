# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Helpers for multi-node training on SLURM/Elastic clusters."""

from __future__ import annotations

import os
from pathlib import Path
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
    (DDP needs one SLURM task per GPU). Returns a plain dict — pass it to
    ``run(launcher="submitit_slurm", launcher_config=...)`` (see the multi-node
    guide); it submits nothing. Extra keyword args (e.g. ``account``, ``qos``,
    ``constraint``) pass through verbatim — including preemption knobs such as
    ``signal_delay_s`` (grace signal before the kill, time to checkpoint) and
    ``additional_parameters={"requeue": True}`` for SLURM auto-requeue.
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


def _rank_from_env() -> int:
    """Best-effort global rank: ``RANK`` (torchrun/external launchers), then
    ``SLURM_PROCID``, then ``LOCAL_RANK`` (the only variable a plain
    single-node HydraDDP child exports — equal to the global rank there).
    A malformed value degrades to the next source rather than crashing."""
    for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        v = os.environ.get(var)
        if v:
            try:
                return int(v)
            except ValueError:
                continue
    return 0


def seed_everything_per_rank(base: int, workers: bool = True) -> int:
    """Seed each process with ``base + global_rank`` so a multi-GPU/-node run is as
    reproducible as a single-GPU run (each rank gets a distinct but deterministic
    seed). Reads the global rank from ``RANK`` (preferred), ``SLURM_PROCID``,
    or ``LOCAL_RANK`` (single-node), defaulting to 0. Returns the seed used."""
    rank = _rank_from_env()
    seed = int(base) + rank
    seed_everything(seed, workers=workers)
    # Persist the effective seed: if it lives only in-process, the exact run
    # can never be re-seeded identically from its artifacts. Best-effort (a
    # read-only cwd must not break training); rank in the filename so DDP
    # ranks sharing a dir don't clobber rank 0's record.
    try:
        import json

        name = "mushin_seed.json" if rank == 0 else f"mushin_seed_rank{rank}.json"
        Path(name).write_text(json.dumps({"seed": seed, "rank": rank}))
    except OSError:
        pass
    return seed
