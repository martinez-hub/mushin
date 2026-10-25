# lightning

::: mushin.lightning.HydraDDP
    options:
      show_bases: false

::: mushin.lightning.HydraFSDP
    options:
      show_bases: false

::: mushin.lightning.MetricsCallback

## Clusters and GPUs

Helpers for multi-process runs: seeding per rank, a SLURM launcher config for
multi-node jobs, and pinning small jobs to one GPU each. The
[multi-node](../guides/multinode.md) and [packing](../guides/packing.md) guides
show when to use them.

::: mushin.lightning.seed_everything_per_rank
    options:
      heading_level: 3

::: mushin.lightning.submitit_slurm_config
    options:
      heading_level: 3

::: mushin.pin_gpu_round_robin
    options:
      heading_level: 3
