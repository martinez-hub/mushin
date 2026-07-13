# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
from pathlib import Path

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


def test_default_pin_path_prefers_default_root_dir():
    from types import SimpleNamespace

    from mushin._tuning import _default_pin_path

    trainer = SimpleNamespace(default_root_dir="/tmp/root", log_dir="/tmp/run7")
    assert (
        _default_pin_path(trainer, "mushin_batch_pin.yaml")
        == Path("/tmp/root") / "mushin_batch_pin.yaml"
    )


def test_default_pin_path_falls_back_to_log_dir():
    from types import SimpleNamespace

    from mushin._tuning import _default_pin_path

    trainer = SimpleNamespace(default_root_dir=None, log_dir="/tmp/run7")
    assert _default_pin_path(trainer, "x.yaml") == Path("/tmp/run7") / "x.yaml"


def test_default_pin_path_requires_explicit_in_multirun(monkeypatch):
    from types import SimpleNamespace

    import hydra.core.hydra_config as hc
    from hydra.types import RunMode

    from mushin._tuning import _default_pin_path

    class _FakeHC:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            return SimpleNamespace(mode=RunMode.MULTIRUN)

    monkeypatch.setattr(hc, "HydraConfig", _FakeHC)
    trainer = SimpleNamespace(default_root_dir="/tmp/root", log_dir=None)
    with pytest.raises(RuntimeError, match="multirun"):
        _default_pin_path(trainer, "mushin_batch_pin.yaml")


def test_default_pin_path_ok_in_single_run(monkeypatch):
    from types import SimpleNamespace

    import hydra.core.hydra_config as hc
    from hydra.types import RunMode

    from mushin._tuning import _default_pin_path

    class _FakeHC:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            return SimpleNamespace(mode=RunMode.RUN)

    monkeypatch.setattr(hc, "HydraConfig", _FakeHC)
    trainer = SimpleNamespace(default_root_dir="/tmp/root", log_dir=None)
    assert _default_pin_path(trainer, "x.yaml") == Path("/tmp/root") / "x.yaml"


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


def test_batch_exact_hparams_updated(monkeypatch, tmp_path):
    """tune_batch_size also updates .hparams when batch_size is stored there."""
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 512)

    class DMWithHparams:
        def __init__(self):
            self.batch_size = 2
            self.hparams = {"batch_size": 2}

    dm = DMWithHparams()
    tune_batch_size(
        _make_trainer(),
        object(),
        dm,
        effective_batch_size=256,
        num_devices=1,
        pin_path=tmp_path / "p.yaml",
        retune=True,
    )
    assert dm.batch_size == 256
    assert dm.hparams["batch_size"] == 256


def test_batch_hparams_only_datamodule_is_target(monkeypatch, tmp_path):
    """A datamodule exposing batch_size ONLY via .hparams (no direct attribute, as
    with save_hyperparameters()) is still the apply target, not skipped for the module."""
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 512)

    class DMHparamsOnly:
        def __init__(self):
            self.hparams = {"batch_size": 2}  # no direct self.batch_size

    class Mod:
        pass

    dm = DMHparamsOnly()
    module = Mod()
    tune_batch_size(
        _make_trainer(),
        module,
        dm,
        effective_batch_size=256,
        num_devices=1,
        pin_path=tmp_path / "p.yaml",
        retune=True,
    )
    assert dm.hparams["batch_size"] == 256  # applied to the datamodule's hparams
    assert not hasattr(module, "batch_size")  # not misapplied to the module


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


def test_batch_num_devices_multiplies_nodes(monkeypatch, tmp_path):
    """num_devices inference multiplies per-node devices by num_nodes."""
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 512)

    trainer = _make_trainer()
    # Patch num_devices and num_nodes as read-only properties on this trainer instance
    monkeypatch.setattr(type(trainer), "num_devices", property(lambda self: 2))
    monkeypatch.setattr(type(trainer), "num_nodes", property(lambda self: 2))

    pin = tune_batch_size(
        trainer,
        object(),
        _DM(),
        effective_batch_size=8,
        pin_path=tmp_path / "p.yaml",
        retune=True,
    )
    # 2 devices * 2 nodes = 4 total; per_device_total = 8 / 4 = 2
    assert pin.num_devices == 4
    assert pin.device_batch == 2


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


def test_batch_pin_clamped_on_scale_out(monkeypatch, tmp_path):
    # A pin found on 1 device must be clamped to the per-device target when reused
    # on more devices, or accumulation bottoms out at 1 and the effective overshoots.
    from mushin._tuning import _write_pin, tune_batch_size

    pin_path = tmp_path / "batch.yaml"
    _write_pin(
        pin_path, {"device_batch": 256, "effective_batch_size": 256, "num_devices": 1}
    )

    # reused on 4 devices: per_device_total = 256/4 = 64, so device_batch clamps to 64
    with pytest.warns(UserWarning, match="was recorded for"):
        pin = tune_batch_size(
            _make_trainer(),
            object(),
            _DM(),
            effective_batch_size=256,
            num_devices=4,
            pin_path=pin_path,
        )
    assert pin.device_batch == 64 and pin.accumulate_grad_batches == 1
    assert pin.effective_batch_size == 256 and pin.drift == 0  # not 1024


def test_batch_rejects_ambiguous_owners(monkeypatch, tmp_path):
    # Both module and datamodule expose batch_arg -> ambiguous which one the tuner
    # scaled; reject rather than apply to the wrong object.
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 128)

    class ModWithBatch:
        def __init__(self):
            self.batch_size = 2

    with pytest.raises(ValueError, match="both the module and datamodule"):
        tune_batch_size(
            _make_trainer(),
            ModWithBatch(),
            _DM(),
            effective_batch_size=256,
            num_devices=1,
            pin_path=tmp_path / "p.yaml",
            retune=True,
        )


def test_batch_pin_no_owner_raises(tmp_path):
    # Reused pin but neither module nor datamodule exposes batch_arg: applying would
    # create a dead attribute and silently keep the stale batch. Reject instead.
    from mushin._tuning import _write_pin, tune_batch_size

    pin_path = tmp_path / "p.yaml"
    _write_pin(
        pin_path, {"device_batch": 8, "effective_batch_size": 8, "num_devices": 1}
    )

    class Bare:
        pass

    with pytest.raises(ValueError, match="neither the module nor the datamodule"):
        tune_batch_size(
            _make_trainer(),
            Bare(),
            None,
            effective_batch_size=8,
            num_devices=1,
            pin_path=pin_path,
        )


def test_batch_rejects_existing_accumulation_scheduler(monkeypatch, tmp_path):
    # A GradientAccumulationScheduler drives accumulation; combining it with a
    # non-1 accumulate_grad_batches makes Lightning crash at fit. Fail early.
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import GradientAccumulationScheduler

    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 128)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[GradientAccumulationScheduler(scheduling={0: 2})],
    )
    with pytest.raises(ValueError, match="GradientAccumulationScheduler"):
        tune_batch_size(
            trainer,
            object(),
            _DM(),
            effective_batch_size=256,
            num_devices=1,
            pin_path=tmp_path / "p.yaml",
            retune=True,
        )


def test_batch_rejects_existing_batch_size_finder(monkeypatch, tmp_path):
    # A Lightning BatchSizeFinder callback would run its own search at fit and
    # overwrite the pinned batch; reject the combination.
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import BatchSizeFinder

    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 128)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[BatchSizeFinder()],
    )
    with pytest.raises(ValueError, match="BatchSizeFinder"):
        tune_batch_size(
            trainer,
            object(),
            _DM(),
            effective_batch_size=256,
            num_devices=1,
            pin_path=tmp_path / "p.yaml",
            retune=True,
        )


def test_lr_rejects_existing_lr_finder(tmp_path):
    # A Lightning LearningRateFinder callback would run its own range test at fit
    # and move off the pinned LR; reject the combination.
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateFinder

    from mushin._tuning import tune_learning_rate

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[LearningRateFinder()],
    )
    with pytest.raises(ValueError, match="LearningRateFinder"):
        tune_learning_rate(trainer, _Mod(), None, pin_path=tmp_path / "lr.yaml")


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


def test_lr_hparams_updated(monkeypatch, tmp_path):
    """tune_learning_rate also updates .hparams when lr is stored there."""
    from mushin._tuning import tune_learning_rate

    _patch_lr_find(monkeypatch, 0.0456)

    class ModWithHparams:
        def __init__(self):
            self.lr = 0.0
            self.hparams = {"lr": 0.0}

    mod = ModWithHparams()
    pin = tune_learning_rate(_make_trainer(), mod, None, pin_path=tmp_path / "lr.yaml")
    assert mod.lr == 0.0456
    assert mod.hparams["lr"] == 0.0456
    assert pin.learning_rate == 0.0456


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


def test_lr_pin_invalid_learning_rate_raises(tmp_path):
    from mushin._tuning import _write_pin, tune_learning_rate

    pin_path = tmp_path / "lr.yaml"
    _write_pin(pin_path, {"learning_rate": 0.0})
    with pytest.raises(ValueError, match="learning_rate"):
        tune_learning_rate(
            _make_trainer(),
            _Mod(),
            None,
            pin_path=pin_path,
        )


def test_lr_pin_non_finite_learning_rate_raises(tmp_path):
    from mushin._tuning import _write_pin, tune_learning_rate

    pin_path = tmp_path / "lr.yaml"
    _write_pin(pin_path, {"learning_rate": float("inf")})
    with pytest.raises(ValueError, match="learning_rate"):
        tune_learning_rate(_make_trainer(), _Mod(), None, pin_path=pin_path)


def test_lr_missing_owner_raises(tmp_path):
    # lr_attr not present on the module (misspelled/renamed): reject rather than
    # create a dead attribute configure_optimizers never reads.
    from mushin._tuning import tune_learning_rate

    class Bare:
        pass

    with pytest.raises(ValueError, match="does not expose"):
        tune_learning_rate(_make_trainer(), Bare(), None, pin_path=tmp_path / "lr.yaml")


def test_batch_no_pin_written_when_owner_invalid(monkeypatch, tmp_path):
    # A call that fails owner validation must not leave a sidecar behind (deferred
    # write), or a later non-retune run would read an unverified pin.
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 128)

    class ModWithBatch:
        def __init__(self):
            self.batch_size = 2

    pin_path = tmp_path / "p.yaml"
    with pytest.raises(ValueError, match="both the module and datamodule"):
        tune_batch_size(
            _make_trainer(),
            ModWithBatch(),
            _DM(),
            effective_batch_size=256,
            num_devices=1,
            pin_path=pin_path,
            retune=True,
        )
    assert not pin_path.exists()


def test_exports():
    import mushin
    from mushin import tune_batch_size, tune_learning_rate

    assert "tune_batch_size" in mushin.__all__
    assert "tune_learning_rate" in mushin.__all__
    assert callable(tune_batch_size) and callable(tune_learning_rate)


@pytest.mark.parametrize(
    "target, cap, expected",
    [
        (128, 100, 64),   # divisors of 128 <=100 -> 64
        (128, 128, 128),  # cap == target -> target itself
        (512, 600, 512),  # cap > target -> target (accumulate would be 1)
        (512, 300, 256),  # largest power-of-two divisor <=300
        (100, 7, 5),      # divisors of 100 <=7 -> 5
        (17, 4, 1),       # prime target, small cap -> 1
        (1, 1, 1),        # degenerate
        (128, 1, 1),      # cap 1 -> 1 divides everything
    ],
)
def test_largest_divisor_leq(target, cap, expected):
    from mushin._tuning import _largest_divisor_leq

    d = _largest_divisor_leq(target, cap)
    assert d == expected
    assert target % d == 0  # it is always an exact divisor
    assert 1 <= d <= cap
