# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Runnable FSDP sharding demo (requires >=2 real GPUs; not run in CI).

Run:  python examples/sharding_fsdp_demo.py

Shows the mushin pieces for sharded training: an FSDP-configured Trainer plus the
``DistributedTeardown`` callback so consecutive runs (e.g. a Hydra ``--multirun``)
leave a clean process group. The model/data are intentionally tiny; the point is
the wiring, not the workload. mushin's analysis layer (compare/significance) is
unchanged from a single-GPU run.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import FSDPStrategy
from torch.utils.data import DataLoader, TensorDataset

from mushin import DistributedTeardown


class _TinyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        )

    def training_step(self, batch, _idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x).squeeze(-1), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("This demo needs >=2 GPUs (FSDP shards across ranks).")

    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 8, generator=g)
    y = x.sum(dim=1)
    loader = DataLoader(TensorDataset(x, y), batch_size=32)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy=FSDPStrategy(sharding_strategy="FULL_SHARD"),
        callbacks=[DistributedTeardown()],
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(_TinyModule(), loader)
    print("FSDP run complete; process group torn down for the next job.")


if __name__ == "__main__":
    main()
