# Sharded training under Hydra (FSDP)

`HydraDDP` is **data-parallel**: it replicates the whole model on each GPU. When a
model does **not** fit on one GPU you need **sharding** — splitting parameters,
gradients, and optimizer state across GPUs — which is Lightning's `FSDPStrategy`.
`HydraFSDP` is the sharded counterpart of `HydraDDP`: it makes FSDP work correctly
under Hydra `--multirun`.

## Why not stock `FSDPStrategy` under `--multirun`?

Lightning's `FSDPStrategy` (like `DDPStrategy`) launches ranks by **re-executing
the script with `sys.argv`**. Inside a Hydra sweep that argv is the *sweep*
command, so child ranks re-run the wrong job. `HydraFSDP` instead re-execs each
rank against the job's saved `config.yaml` (via `mushin.lightning._pl_main`),
exactly as `HydraDDP` does for DDP — so a `--multirun` sweep runs each job
correctly.

## When to shard vs DDP

| Question | Use |
|---|---|
| Model fits on one GPU | `HydraDDP` |
| Model / optimizer state does **not** fit on one GPU | `HydraFSDP` |

## Configure with hydra-zen

```python
import pytorch_lightning as pl
from hydra_zen import builds, make_config

from mushin import HydraFSDP

TrainerConf = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=4,
    strategy=builds(HydraFSDP),
    populate_full_signature=True,
)
Config = make_config(trainer=TrainerConf, module=...)  # your LightningModule config
```

`HydraFSDP` requires Hydra to save a `config.yaml` with `trainer` and `module`
keys in the job dir (the same contract as `HydraDDP`). See
`examples/sharding_fsdp_multirun.py` for a runnable 2-GPU sweep (needs real GPUs,
not run in CI).

## Rank-launch stagger

`HydraDDP`/`HydraFSDP` wait 1 second between spawning each child rank (it
avoids dataloader startup contention). For short jobs in large sweeps that
per-cell tax adds up — set `MUSHIN_DDP_LAUNCH_DELAY=0` (any float works) to
tune or disable it.

## DeepSpeed

DeepSpeed ZeRO is the same idea; you would wrap `DeepSpeedStrategy` the same way.
mushin does not ship a `HydraDeepSpeed` class or the `deepspeed` dependency —
configure `DeepSpeedStrategy` directly if you need it.

## mushin is unchanged

Sharding only changes the Trainer `strategy`. `MetricsCallback`, checkpoints,
`load_experiment`, and the `compare`/significance pipeline are identical — mushin
compares sharded and unsharded runs the same way.
