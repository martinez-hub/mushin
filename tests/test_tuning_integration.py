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


class _BatchRecorder(__import__("pytorch_lightning").LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)
        self.lr = lr
        self.seen_batches = []
        self.step_lrs = []

    def training_step(self, batch, _):
        x, y = batch
        self.seen_batches.append(int(x.shape[0]))
        self.step_lrs.append(self.optimizers().param_groups[0]["lr"])
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


class _DataModule64(__import__("pytorch_lightning").LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        ds = TensorDataset(torch.randn(64, 4), torch.randn(64, 1))
        return DataLoader(ds, batch_size=self.batch_size)


def test_real_tuner_fit_uses_capped_batch(tmp_path):
    """After a REAL scale_batch_size search, an immediate fit uses the applied
    (capped) device batch, not the search's final trial batch: fit rebuilds the
    dataloader from the datamodule's updated attribute."""
    import pytorch_lightning as pl

    from mushin._tuning import tune_batch_size

    m, dm = _BatchRecorder(), _DataModule64()
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    pin = tune_batch_size(
        trainer,
        m,
        dm,
        effective_batch_size=8,
        num_devices=1,
        pin_path=tmp_path / "bp.yaml",
        retune=True,
        max_trials=4,
        steps_per_trial=1,
        init_val=2,
    )
    m.seen_batches.clear()  # isolate fit from the tuner's own trial steps
    trainer.fit(m, datamodule=dm)
    assert m.seen_batches  # fit actually ran
    assert set(m.seen_batches) == {pin.device_batch}


def test_real_fit_uses_applied_learning_rate(tmp_path):
    """An immediate fit steps with the applied learning rate: fit rebuilds the
    optimizer from the module's updated attribute (uses the pinned path so the test
    does not depend on lr_find producing a suggestion)."""
    import pytorch_lightning as pl

    from mushin._tuning import _write_pin, tune_learning_rate

    _write_pin(tmp_path / "lr.yaml", {"learning_rate": 0.05})
    m, dm = _BatchRecorder(lr=0.001), _DataModule64(batch_size=8)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    tune_learning_rate(trainer, m, dm, pin_path=tmp_path / "lr.yaml")
    assert m.lr == 0.05
    m.step_lrs.clear()
    trainer.fit(m, datamodule=dm)
    assert m.step_lrs and all(abs(x - 0.05) < 1e-9 for x in m.step_lrs)
