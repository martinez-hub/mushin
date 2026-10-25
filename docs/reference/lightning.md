# lightning

::: mushin.lightning.HydraDDP

::: mushin.lightning.HydraFSDP

::: mushin.lightning.MetricsCallback

## Clusters and GPUs

Helpers for multi-process runs: seeding per rank, a SLURM launcher config for
multi-node jobs, and pinning small jobs to one GPU each. The
[multi-node](../guides/multinode.md) and [packing](../guides/packing.md) guides
show when to use them.

::: mushin.seed_everything_per_rank

::: mushin.submitit_slurm_config

::: mushin.pin_gpu_round_robin
