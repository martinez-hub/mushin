# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""FSDP under Hydra multirun via HydraFSDP (requires >=2 GPUs; not run in CI).

Run a 2-job sweep:
  python examples/sharding_fsdp_multirun.py --multirun +run=a,b

Each Hydra job saves its config.yaml; HydraFSDP re-execs the ranks against THAT
config (not the sweep argv), so both jobs run correctly in one process. The model
is tiny — the point is the launcher wiring, not the workload. mushin's analysis
(compare/significance) is unchanged from a single-GPU run.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from hydra_zen import builds, make_config, store, zen
from torch.utils.data import DataLoader, TensorDataset

from mushin import HydraFSDP


class _TinyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        )

    def train_dataloader(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(256, 8, generator=g)
        y = x.sum(dim=1)
        return DataLoader(TensorDataset(x, y), batch_size=32)

    def training_step(self, batch, _idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x).squeeze(-1), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


TrainerConf = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=2,
    strategy=builds(HydraFSDP),
    max_epochs=1,
    enable_checkpointing=False,
    logger=False,
    populate_full_signature=False,
)
Config = make_config(trainer=TrainerConf, module=builds(_TinyModule))


def _task(trainer: pl.Trainer, module: pl.LightningModule) -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("This demo needs >=2 GPUs (FSDP shards across ranks).")
    trainer.fit(module)


if __name__ == "__main__":
    store(Config, name="config")
    store.add_to_hydra_store()
    zen(_task).hydra_main(config_name="config", config_path=None, version_base="1.1")
