Multi-node DDP support: `submitit_slurm_config` (derives `tasks_per_node` from
`gpus_per_node`) and `seed_everything_per_rank` helpers, a fail-fast check that the
launched world size matches `num_nodes x devices`, `MetricsCallback` now writes only
on global rank 0, and `_teardown` clears only mushin-set env vars (leaving
scheduler-owned vars alone under SLURM/torchrun). See the new multi-node guide.
