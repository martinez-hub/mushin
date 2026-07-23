# Multi-node training (SLURM / Elastic)

`HydraDDP` runs single-node multi-GPU out of the box. For **multi-node**, delegate
process launching to a cluster scheduler — Hydra's
[`hydra-submitit-launcher`](https://hydra.cc/docs/plugins/submitit_launcher/) +
Lightning's auto-detected `SLURMEnvironment`. Under an external launcher,
`HydraDDP` steps aside and Lightning uses the scheduler-launched ranks directly.

## The one contract: one SLURM task per GPU

For DDP there must be exactly one process (rank) per GPU:

```
tasks_per_node == gpus_per_node == Trainer `devices` per node
world_size     == nodes * gpus_per_node
```

Get this wrong and you hang at rendezvous (too few tasks), OOM (too many), or
silently run single-GPU. `mushin.submitit_slurm_config` derives `tasks_per_node`
from `gpus_per_node` so they cannot desync, and `HydraDDP` **fails fast** with a
clear error if the launched world size doesn't match `num_nodes x devices`.

## Building the launcher config

```python
from mushin import submitit_slurm_config

slurm = submitit_slurm_config(
    nodes=2,
    gpus_per_node=4,        # -> tasks_per_node=4, one rank per GPU
    cpus_per_task=8,
    partition="gpu",
    timeout_min=120,
    mem_gb=64,
)
```

Hand the dict straight to `run()` (install the plugin first:
`pip install hydra-submitit-launcher`) — no hand-rolled `hydra.launcher.*`
overrides:

```python
wf.run(
    working_dir="runs",
    launcher="submitit_slurm",
    launcher_config=slurm,
)
```

(Outside the workflow API, the equivalent raw overrides are
`hydra/launcher=submitit_slurm` plus `hydra.launcher.<key>=<value>` per field.)
The Trainer must use a matching `devices`/`num_nodes`:

```python
import pytorch_lightning as pl
from hydra_zen import builds
from mushin import HydraDDP

TrainerConfig = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=4,        # == gpus_per_node
    num_nodes=2,      # == nodes
    strategy=builds(HydraDDP),
    populate_full_signature=True,
)
```

## Reproducibility

Seed each rank deterministically so a 64-GPU run reproduces a 1-GPU run:

```python
from mushin import seed_everything_per_rank

seed_everything_per_rank(1234)   # each rank: 1234 + global_rank
```

## Metrics

`MetricsCallback` writes its output (`fit_metrics.pt` / `test_metrics.pt`, one per
stage) only on global rank 0, so the N ranks don't clobber the file on a shared
filesystem; `load_experiment` reads it back as usual.

## Preemption & resume

When SLURM preempts (or times out) a cell's job, the sweep driver observes a
failed job: under the default `on_error="raise"` the sweep aborts with that
error; under `on_error="nan"` it records the failure and finishes the rest.
Either way, recovery is one command — re-run the same sweep with
`resume=True` (same `working_dir`): completed cells are reused from their
sidecars, and only the preempted/missing cells re-run. Inside a long cell, use
the [`mushin_resume` checkpoint contract](resilience.md) so the re-run
continues from `last.ckpt` instead of epoch 0.

To have SLURM requeue preempted jobs automatically, pass the scheduler knobs
through `submitit_slurm_config(**extra)` — e.g.
`signal_delay_s=120` (submitit's grace signal before the kill, time to write a
checkpoint) and `additional_parameters={"requeue": True}`.

Two multi-rank cautions:

- `max_total_seconds` is **disabled** for cells that run under a multi-rank
  launch (each rank would keep its own deadline; a rank that stops while its
  siblings train would hang DDP at rendezvous — a warning is emitted). Bound
  multi-rank jobs with the scheduler's own `timeout_min` instead.
- Every rank of a cell runs the task function, so per-cell files
  (status/metrics/provenance sidecars) are written once per rank — writes are
  atomic and carry the same values, so this is benign; `MetricsCallback`'s
  `.pt` files are rank-0-only by design.

## Runbook (the merge gate)

To validate a real multi-node run on your cluster:

1. `pip install hydra-submitit-launcher` on the cluster.
2. Build the launcher config with `submitit_slurm_config(nodes=2, gpus_per_node=<G>, partition=<P>, account=<A>)`.
3. Configure the Trainer with `devices=<G>`, `num_nodes=2`, `strategy=builds(HydraDDP)`.
4. Launch your hydra-zen workflow with `hydra/launcher=submitit_slurm` and `--multirun`.
5. Confirm: the job completes, `fit_metrics.pt` exists exactly once per job dir,
   `load_experiment` aggregates the results, and a mismatched `tasks_per_node`
   raises the fail-fast world-size error rather than hanging.

This is the condition for merging the multi-node branch.
