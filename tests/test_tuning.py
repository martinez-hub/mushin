# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest


def test_pin_roundtrip(tmp_path):
    from mushin._tuning import _read_pin, _write_pin

    path = tmp_path / "pin.yaml"
    assert _read_pin(path) is None  # absent -> None

    _write_pin(
        path, {"device_batch": 64, "effective_batch_size": 256, "num_devices": 1}
    )
    assert path.exists()
    got = _read_pin(path)
    assert got == {"device_batch": 64, "effective_batch_size": 256, "num_devices": 1}


def test_write_pin_creates_parent_dirs(tmp_path):
    from mushin._tuning import _read_pin, _write_pin

    path = tmp_path / "nested" / "dir" / "pin.yaml"
    _write_pin(path, {"learning_rate": 0.001})
    assert _read_pin(path) == {"learning_rate": 0.001}


def test_default_pin_path_uses_trainer_log_dir():
    from types import SimpleNamespace

    from mushin._tuning import _default_pin_path

    trainer = SimpleNamespace(log_dir="/tmp/run7", default_root_dir="/tmp/root")
    assert (
        str(_default_pin_path(trainer, "mushin_batch_pin.yaml"))
        == "/tmp/run7/mushin_batch_pin.yaml"
    )


def test_default_pin_path_falls_back_to_default_root_dir():
    from types import SimpleNamespace

    from mushin._tuning import _default_pin_path

    trainer = SimpleNamespace(log_dir=None, default_root_dir="/tmp/root")
    assert str(_default_pin_path(trainer, "x.yaml")) == "/tmp/root/x.yaml"


def test_batchpin_and_lrpin_are_frozen_dataclasses():
    import dataclasses

    from mushin._tuning import BatchPin, LRPin

    bp = BatchPin(
        device_batch=64,
        accumulate_grad_batches=4,
        effective_batch_size=256,
        num_devices=1,
        drift=0,
    )
    lp = LRPin(learning_rate=0.001)
    assert dataclasses.is_dataclass(bp) and dataclasses.is_dataclass(lp)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bp.device_batch = 1
