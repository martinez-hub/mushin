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


def _make_trainer():
    import pytorch_lightning as pl

    return pl.Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=False
    )


class _DM:
    def __init__(self, batch_size=2):
        self.batch_size = batch_size


def _patch_scale(monkeypatch, return_value, counter=None):
    """Patch Tuner.scale_batch_size to return a fixed value (no real search)."""
    from pytorch_lightning.tuner.tuning import Tuner

    def fake(self, *a, **k):
        if counter is not None:
            counter["n"] += 1
        return return_value

    monkeypatch.setattr(Tuner, "scale_batch_size", fake)


def test_batch_exact_when_max_meets_target(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 512)  # >= per_device_total(256)
    dm = _DM()
    pin = tune_batch_size(
        _make_trainer(),
        object(),
        dm,
        effective_batch_size=256,
        num_devices=1,
        pin_path=tmp_path / "p.yaml",
        retune=True,
    )
    assert (pin.device_batch, pin.accumulate_grad_batches) == (256, 1)
    assert pin.effective_batch_size == 256 and pin.drift == 0
    assert dm.batch_size == 256  # applied to the datamodule


def test_batch_accumulation_clean_divisor(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 64)  # 256/64 == 4 exactly
    trainer = _make_trainer()
    pin = tune_batch_size(
        trainer,
        object(),
        _DM(),
        effective_batch_size=256,
        num_devices=1,
        pin_path=tmp_path / "p.yaml",
        retune=True,
    )
    assert (pin.device_batch, pin.accumulate_grad_batches) == (64, 4)
    assert pin.effective_batch_size == 256 and pin.drift == 0
    assert trainer.accumulate_grad_batches == 4  # applied to the trainer


def test_batch_drift_warns_when_not_divisible(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 100)  # 256/100 -> round(2.56)=3 -> 300
    with pytest.warns(UserWarning, match="differs from requested"):
        pin = tune_batch_size(
            _make_trainer(),
            object(),
            _DM(),
            effective_batch_size=256,
            num_devices=1,
            pin_path=tmp_path / "p.yaml",
            retune=True,
        )
    assert pin.device_batch == 100 and pin.accumulate_grad_batches == 3
    assert pin.effective_batch_size == 300 and pin.drift == 44


def test_batch_safety_margin_backs_off_found_max(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    # found 100, margin 0.2 -> floor(80)=80 -> min(80, 240)=80; round(240/80)=3 -> 240 (no drift)
    _patch_scale(monkeypatch, 100)
    pin = tune_batch_size(
        _make_trainer(),
        object(),
        _DM(),
        effective_batch_size=240,
        num_devices=1,
        safety_margin=0.2,
        pin_path=tmp_path / "p.yaml",
        retune=True,
    )
    assert pin.device_batch == 80 and pin.accumulate_grad_batches == 3
    assert pin.effective_batch_size == 240 and pin.drift == 0


def test_batch_num_devices_divides_effective(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 512)  # per_device_total = 256/4 = 64
    pin = tune_batch_size(
        _make_trainer(),
        object(),
        _DM(),
        effective_batch_size=256,
        num_devices=4,
        pin_path=tmp_path / "p.yaml",
        retune=True,
    )
    assert pin.device_batch == 64 and pin.accumulate_grad_batches == 1
    assert pin.num_devices == 4 and pin.effective_batch_size == 256


def test_batch_invalid_inputs(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    # validation raises before any pin I/O, so the path is never written
    _patch_scale(monkeypatch, 128)
    t, dm, pin = _make_trainer(), _DM(), tmp_path / "p.yaml"
    with pytest.raises(ValueError, match="divisible"):
        tune_batch_size(
            t,
            object(),
            dm,
            effective_batch_size=250,
            num_devices=4,
            pin_path=pin,
            retune=True,
        )
    with pytest.raises(ValueError, match="safety_margin"):
        tune_batch_size(
            t,
            object(),
            dm,
            effective_batch_size=256,
            num_devices=1,
            safety_margin=1.0,
            pin_path=pin,
            retune=True,
        )
    with pytest.raises(ValueError, match="effective_batch_size"):
        tune_batch_size(
            t,
            object(),
            dm,
            effective_batch_size=0,
            num_devices=1,
            pin_path=pin,
            retune=True,
        )


def test_batch_none_from_tuner_raises(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, None)
    with pytest.raises(RuntimeError, match="no batch size"):
        tune_batch_size(
            _make_trainer(),
            object(),
            _DM(),
            effective_batch_size=256,
            num_devices=1,
            pin_path=tmp_path / "p.yaml",
            retune=True,
        )


def test_batch_pin_invalid_device_batch_raises(tmp_path):
    from mushin._tuning import _write_pin, tune_batch_size

    pin_path = tmp_path / "bad.yaml"
    _write_pin(
        pin_path, {"device_batch": 0, "effective_batch_size": 256, "num_devices": 1}
    )
    with pytest.raises(ValueError, match="device_batch"):
        tune_batch_size(
            _make_trainer(),
            object(),
            _DM(),
            effective_batch_size=256,
            num_devices=1,
            pin_path=pin_path,
        )


def test_batch_pin_context_mismatch_warns(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    counter = {"n": 0}
    _patch_scale(monkeypatch, 128, counter)
    pin_path = tmp_path / "batch.yaml"
    # first tune records effective_batch_size=256, num_devices=1
    tune_batch_size(
        _make_trainer(),
        object(),
        _DM(),
        effective_batch_size=256,
        num_devices=1,
        pin_path=pin_path,
    )
    assert counter["n"] == 1

    # re-run with a DIFFERENT target reuses the pinned device_batch but warns
    with pytest.warns(UserWarning, match="was recorded for"):
        pin = tune_batch_size(
            _make_trainer(),
            object(),
            _DM(),
            effective_batch_size=512,
            num_devices=1,
            pin_path=pin_path,
        )
    assert counter["n"] == 1  # still no new search
    assert pin.device_batch == 128 and pin.accumulate_grad_batches == 4
    assert pin.effective_batch_size == 512 and pin.drift == 0  # 128*4*1 == 512


def test_batch_pin_roundtrip_skips_search(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    counter = {"n": 0}
    _patch_scale(monkeypatch, 128, counter)
    pin_path = tmp_path / "batch.yaml"
    a1 = _make_trainer()
    r1 = tune_batch_size(
        a1, object(), _DM(), effective_batch_size=256, num_devices=1, pin_path=pin_path
    )
    assert counter["n"] == 1 and pin_path.exists()

    a2 = _make_trainer()
    r2 = tune_batch_size(
        a2, object(), _DM(), effective_batch_size=256, num_devices=1, pin_path=pin_path
    )
    assert counter["n"] == 1  # search NOT called again
    assert r2.device_batch == r1.device_batch
    assert a2.accumulate_grad_batches == r1.accumulate_grad_batches  # still applied

    r3 = tune_batch_size(
        _make_trainer(),
        object(),
        _DM(),
        effective_batch_size=256,
        num_devices=1,
        pin_path=pin_path,
        retune=True,
    )
    assert counter["n"] == 2  # retune forces a fresh search
    assert r3.device_batch == 128


class _Mod:
    def __init__(self):
        self.lr = 0.0


class _FakeFinder:
    def __init__(self, value):
        self._value = value

    def suggestion(self):
        return self._value


def _patch_lr_find(monkeypatch, suggestion, counter=None):
    from pytorch_lightning.tuner.tuning import Tuner

    def fake(self, *a, **k):
        if counter is not None:
            counter["n"] += 1
        return _FakeFinder(suggestion)

    monkeypatch.setattr(Tuner, "lr_find", fake)


def test_lr_sets_attr_and_writes_pin(monkeypatch, tmp_path):
    from mushin._tuning import tune_learning_rate

    _patch_lr_find(monkeypatch, 0.0123)
    pin_path = tmp_path / "lr.yaml"
    mod = _Mod()
    pin = tune_learning_rate(_make_trainer(), mod, None, pin_path=pin_path)
    assert pin.learning_rate == 0.0123
    assert mod.lr == 0.0123
    assert pin_path.exists()


def test_lr_pin_roundtrip_skips_search(monkeypatch, tmp_path):
    from mushin._tuning import tune_learning_rate

    counter = {"n": 0}
    _patch_lr_find(monkeypatch, 0.05, counter)
    pin_path = tmp_path / "lr.yaml"
    tune_learning_rate(_make_trainer(), _Mod(), None, pin_path=pin_path)
    assert counter["n"] == 1

    mod2 = _Mod()
    pin2 = tune_learning_rate(_make_trainer(), mod2, None, pin_path=pin_path)
    assert counter["n"] == 1  # search skipped
    assert mod2.lr == 0.05 and pin2.learning_rate == 0.05

    tune_learning_rate(_make_trainer(), _Mod(), None, pin_path=pin_path, retune=True)
    assert counter["n"] == 2  # retune forces a fresh search


def test_lr_custom_attr_name(monkeypatch, tmp_path):
    from mushin._tuning import tune_learning_rate

    _patch_lr_find(monkeypatch, 0.007)

    class M:
        learning_rate = 0.0

    m = M()
    tune_learning_rate(
        _make_trainer(), m, None, pin_path=tmp_path / "lr.yaml", lr_attr="learning_rate"
    )
    assert m.learning_rate == 0.007


def test_lr_none_suggestion_raises(monkeypatch, tmp_path):
    from mushin._tuning import tune_learning_rate

    _patch_lr_find(monkeypatch, None)
    with pytest.raises(RuntimeError, match="no learning-rate suggestion"):
        tune_learning_rate(_make_trainer(), _Mod(), None, pin_path=tmp_path / "lr.yaml")


def test_exports():
    import mushin
    from mushin import tune_batch_size, tune_learning_rate

    assert "tune_batch_size" in mushin.__all__
    assert "tune_learning_rate" in mushin.__all__
    assert callable(tune_batch_size) and callable(tune_learning_rate)
