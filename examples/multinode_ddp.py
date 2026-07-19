# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Multi-node DDP training via ``HydraDDP`` + submitit on a SLURM cluster.

The launcher (submitit) starts one process per GPU across nodes; ``HydraDDP``
coordinates them into a single NCCL DDP world. The contract is
``tasks_per_node == gpus_per_node == Trainer(devices=...)`` and
``world_size == nodes * gpus_per_node`` — ``submitit_slurm_config`` derives
``tasks_per_node`` from ``gpus_per_node`` so they can't desync, and a mismatch
fails fast with a clear error instead of hanging at NCCL rendezvous.

Requires a real SLURM cluster with >=2 GPU nodes and ``hydra-submitit-launcher``.
Not run in CI. Fill in your ``partition``/``account`` below.

Run it (submits the SLURM job for you):
  pip install hydra-submitit-launcher
  python examples/multinode_ddp.py

For single-node multi-GPU, set ``nodes=1`` and ``gpus_per_node`` to that node's
GPU count. See the Multi-node training guide for the full runbook.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytorch_lightning as pl
from hydra_zen import builds, instantiate, launch, make_config

import mushin
from mushin import HydraDDP, MetricsCallback, submitit_slurm_config
from mushin.testing.lightning import SimpleLightningModule

NODES = 2
GPUS_PER_NODE = 1  # == Trainer devices == launcher tasks_per_node
PARTITION = "CHANGE_ME"  # your SLURM partition
ACCOUNT = "CHANGE_ME"  # your SLURM account

TrainerConfig = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=GPUS_PER_NODE,
    num_nodes=NODES,
    strategy=builds(HydraDDP),
    callbacks=[builds(MetricsCallback)],
    max_epochs=1,
    limit_train_batches=2,
    num_sanity_val_steps=0,
    enable_progress_bar=False,
    enable_checkpointing=False,
    logger=False,
    populate_full_signature=True,
)
# HydraDDP re-execs each rank against this saved config, so `trainer` and
# `module` must be declarative Hydra config keys (not built imperatively).
Config = make_config(trainer=TrainerConfig, module=builds(SimpleLightningModule))


def task_fn(cfg):
    print(
        "RANK",
        {k: os.environ.get(k) for k in ("SLURM_PROCID", "SLURM_NTASKS")},
        "host",
        socket.gethostname(),
        flush=True,
    )
    obj = instantiate(cfg)
    obj.trainer.fit(obj.module)


if __name__ == "__main__":
    slurm = submitit_slurm_config(
        nodes=NODES,
        gpus_per_node=GPUS_PER_NODE,  # tasks_per_node derived == GPUS_PER_NODE
        cpus_per_task=4,
        partition=PARTITION,
        account=ACCOUNT,
        timeout_min=15,
        mem_gb=16,
    )
    overrides = [
        "hydra/launcher=submitit_slurm",
        "hydra.sweep.dir=multinode_ddp_runs",
        "hydra.sweep.subdir=0",
        "+trial=0",
    ] + [f"hydra.launcher.{k}={v}" for k, v in slurm.items()]

    launch(Config, task_fn, overrides=overrides, multirun=True, version_base="1.1")

    metrics = sorted(str(p) for p in Path("multinode_ddp_runs").glob("**/*metrics*"))
    loaded = mushin.load_experiment("multinode_ddp_runs")
    count = len(loaded) if isinstance(loaded, list) else 1
    print(f"DONE — {count} experiment(s), metrics: {metrics}")
