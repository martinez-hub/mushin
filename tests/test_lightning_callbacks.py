# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import Trainer

from mushin.lightning import MetricsCallback
from mushin.testing.lightning import SimpleLightningModule


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("testing", [True, False])
def test_metrics_callback(testing):
    trainer = Trainer(max_epochs=1, callbacks=[MetricsCallback()])
    module = SimpleLightningModule()

    if testing:
        trainer.test(module)
        metric_files = list(Path(".").glob("**/test_metrics.pt"))
    else:
        trainer.fit(module)
        metric_files = list(Path(".").glob("**/fit_metrics.pt"))

    assert len(metric_files) == 1
    metrics = torch.load(metric_files[0], weights_only=False)
    assert isinstance(metrics, dict)

    if testing:
        assert "test_tensor_metric" in metrics
    else:
        assert "fit_tensor_metric" in metrics
        assert "val_tensor_metric" in metrics

    for k, v in metrics.items():
        assert not isinstance(v, torch.Tensor)


def _make_fake_trainer(callback_metrics, sanity_checking=False):
    """Return a minimal fake Trainer-like object for unit testing MetricsCallback."""
    trainer = MagicMock()
    trainer.sanity_checking = sanity_checking
    trainer.callback_metrics = callback_metrics
    return trainer


def _make_fake_module(current_epoch):
    """Return a minimal fake LightningModule-like object."""
    module = MagicMock()
    module.current_epoch = current_epoch
    return module


@pytest.mark.usefixtures("cleandir")
def test_bug1_metric_series_alignment():
    """BUG 1: a metric absent in epoch 0 must be NaN-padded so all series stay
    the same length and list-index == epoch holds."""
    cb = MetricsCallback()

    # Epoch 0: only val_acc is logged
    trainer0 = _make_fake_trainer({"val_acc": torch.tensor(0.5)})
    cb.on_validation_end(trainer0, _make_fake_module(current_epoch=0))

    # Epoch 1: both val_acc and val_loss are logged
    trainer1 = _make_fake_trainer(
        {"val_acc": torch.tensor(0.6), "val_loss": torch.tensor(0.2)}
    )
    cb.on_validation_end(trainer1, _make_fake_module(current_epoch=1))

    val = cb.val_metrics

    # All series must have the same length == 2 (number of epochs recorded)
    lengths = {k: len(v) for k, v in val.items()}
    assert all(n == 2 for n in lengths.values()), (
        f"Series lengths are not equal — desync detected: {lengths}"
    )

    # epoch axis: must be [0, 1]
    assert val["epoch"] == [0, 1], f"epoch series wrong: {val['epoch']}"

    # val_acc: both values present
    assert val["val_acc"][0] == pytest.approx(0.5)
    assert val["val_acc"][1] == pytest.approx(0.6)

    # val_loss: first entry must be NaN (absent in epoch 0), second 0.2
    assert math.isnan(val["val_loss"][0]), (
        f"val_loss[0] should be NaN (metric absent in epoch 0), got {val['val_loss'][0]}"
    )
    assert val["val_loss"][1] == pytest.approx(0.2)


@pytest.mark.usefixtures("cleandir")
def test_bug2_user_epoch_key_does_not_double_append():
    """BUG 2: if a user logs a metric literally named 'epoch', the callback-owned
    epoch axis must not be double-appended; all series stay equal length == 1."""
    cb = MetricsCallback()

    trainer = _make_fake_trainer(
        {"epoch": torch.tensor(0.0), "val_acc": torch.tensor(0.9)}
    )
    cb.on_validation_end(trainer, _make_fake_module(current_epoch=0))

    val = cb.val_metrics

    # epoch axis must have exactly ONE entry (not two from double-append)
    assert len(val["epoch"]) == 1, (
        f"epoch series has {len(val['epoch'])} entries — double-append detected"
    )

    # All series must be the same length
    lengths = {k: len(v) for k, v in val.items()}
    assert all(n == 1 for n in lengths.values()), (
        f"Series lengths are not equal after user 'epoch' key: {lengths}"
    )
