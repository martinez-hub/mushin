# Sharded training (FSDP / DeepSpeed)

`HydraDDP` is **data-parallel**: it replicates the whole model on each GPU and
splits the batch, so it assumes the model fits on one GPU. When a model does
**not** fit, you need **sharding** — splitting parameters, gradients, and
optimizer state across GPUs. That is a **Lightning Strategy**
(`FSDPStrategy`, `DeepSpeedStrategy`), not something Hydra or mushin implements.
Hydra/hydra-zen configures it; mushin analyzes the results exactly as before.

## When to shard vs DDP

| Question | Use |
|---|---|
| Model fits on one GPU, want more throughput | `HydraDDP` (data-parallel) |
| Model (or its optimizer state) does **not** fit on one GPU | FSDP / DeepSpeed (sharded) |

## FSDP via hydra-zen

`FSDPStrategy` is built into PyTorch — no extra dependency. Configure it like any
other Trainer field with hydra-zen:

```python
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from hydra_zen import builds

from mushin import DistributedTeardown

TrainerConfig = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=4,
    strategy=builds(FSDPStrategy, sharding_strategy="FULL_SHARD"),
    callbacks=[builds(DistributedTeardown)],   # see "Multirun hygiene" below
    populate_full_signature=True,
)
```

## Multirun hygiene

Lightning's `FSDPStrategy`/`DeepSpeedStrategy` do **not** destroy the
`torch.distributed` process group when a Trainer run ends. Under Hydra
`--multirun` (several jobs in one process), the leftover group makes the next
job's `init_process_group` fail. Add mushin's `DistributedTeardown` callback —
it destroys the group (and clears leaked rank/seed env vars) at the end of each
job, so the next one starts clean:

```python
from mushin import DistributedTeardown

trainer = pl.Trainer(..., strategy=FSDPStrategy(...), callbacks=[DistributedTeardown()])
```

It is idempotent and a no-op on CPU / single-process runs, so it is always safe
to include. Do not pair it with `HydraDDP`, which performs its own teardown.

## DeepSpeed

DeepSpeed ZeRO follows the same pattern — `builds(DeepSpeedStrategy, stage=3, ...)`
— but needs the `deepspeed` package (Linux + GPU), which mushin does not depend
on. The `DistributedTeardown` callback covers it too.

## mushin is unchanged

Sharding only changes the Trainer `strategy`. Your `MetricsCallback`,
checkpoints, `load_experiment`, and the `compare`/significance pipeline are
identical — mushin compares sharded and unsharded runs the same way.

See `examples/sharding_fsdp_demo.py` for a runnable 2-GPU example (it needs real
GPUs, so it is not run in CI).
