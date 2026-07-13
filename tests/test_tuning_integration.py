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
    assert pin.device_batch * pin.accumulate_grad_batches * pin.num_devices == 8
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


def test_real_tuner_fit_uses_divisor_batch(tmp_path):
    """After a REAL scale_batch_size search, tune_batch_size applies the largest
    divisor of the per-device target that fits (not the search's own final trial
    batch), so device_batch * accumulate_grad_batches * num_devices lands exactly
    on effective_batch_size; an immediate fit uses that applied device batch:
    fit rebuilds the dataloader from the datamodule's updated attribute.

    found_max is deterministic here: with 64 samples and this trivial CPU model,
    power-mode doubling from init_val=2 across max_trials=4 trials (2, 4, 8, 16)
    never OOMs, so the search always settles on found_max=16. per_device_total =
    8 // 1 = 8, whose largest divisor <= 16 is 8 itself (accumulate=1).
    """
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
    assert pin.device_batch * pin.accumulate_grad_batches * pin.num_devices == 8
    assert pin.device_batch == 8 and pin.accumulate_grad_batches == 1

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


def test_real_lr_find_pins_a_finite_learning_rate(tmp_path):
    """A REAL Tuner.lr_find run (no monkeypatch, no pre-written pin) pins a finite,
    positive learning rate and applies it. Uses signal-bearing data so lr_find has a
    well-behaved loss curve; if it still yields no suggestion, the real path must
    surface the documented RuntimeError (never a silent/garbage value)."""
    import math

    import pytorch_lightning as pl

    from mushin._tuning import _read_pin, tune_learning_rate

    class _SignalDM(pl.LightningDataModule):
        def train_dataloader(self):
            x = torch.randn(256, 4)
            w = torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
            y = x @ w + 0.01 * torch.randn(256, 1)
            return DataLoader(TensorDataset(x, y), batch_size=16)

    m = _BatchRecorder(lr=0.001)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    try:
        pin = tune_learning_rate(
            trainer,
            m,
            _SignalDM(),
            pin_path=tmp_path / "lr.yaml",
            num_training=8,
            min_lr=1e-5,
            max_lr=1.0,
        )
    except RuntimeError as e:
        assert "no learning-rate suggestion" in str(e)
        return
    assert pin.learning_rate > 0 and math.isfinite(pin.learning_rate)
    assert m.lr == pin.learning_rate
    assert _read_pin(tmp_path / "lr.yaml")["learning_rate"] == pin.learning_rate
