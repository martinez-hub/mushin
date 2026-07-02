# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import torch
from torch.utils.data import DataLoader, TensorDataset


class _CountingModule(__import__("pytorch_lightning").LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)
        self.opt_steps = 0

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def optimizer_step(self, *args, **kwargs):
        self.opt_steps += 1
        return super().optimizer_step(*args, **kwargs)


class _DataModule(__import__("pytorch_lightning").LightningDataModule):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        ds = TensorDataset(torch.randn(16, 4), torch.randn(16, 1))
        return DataLoader(ds, batch_size=self.batch_size)


def test_applied_accumulation_and_device_batch_take_effect(monkeypatch, tmp_path):
    import pytorch_lightning as pl
    from pytorch_lightning.tuner.tuning import Tuner

    from mushin._tuning import tune_batch_size

    # Search is patched to "find" device batch 2; real fit runs afterward.
    monkeypatch.setattr(Tuner, "scale_batch_size", lambda self, *a, **k: 2)

    module = _CountingModule()
    dm = _DataModule()
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # effective 8 on 1 device => per_device_total 8; device_batch min(2, 8) = 2;
    # accumulate = round(8 / 2) = 4.
    pin = tune_batch_size(
        trainer,
        module,
        dm,
        effective_batch_size=8,
        num_devices=1,
        pin_path=tmp_path / "pin.yaml",
        retune=True,
    )
    assert pin.device_batch == 2 and pin.accumulate_grad_batches == 4
    assert dm.batch_size == 2  # applied

    trainer.fit(module, datamodule=dm)
    # 16 samples / batch 2 = 8 batches; accumulate 4 => 2 optimizer steps.
    assert module.opt_steps == 2
