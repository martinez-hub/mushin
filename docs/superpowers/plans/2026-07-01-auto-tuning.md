# Reproducibility-preserving auto-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in helpers — `tune_batch_size` and `tune_learning_rate` — that run Lightning's `Tuner` once, record the result to a sidecar pin file so re-runs reuse it deterministically, and apply it while protecting reproducibility (pin the effective batch, set `accumulate_grad_batches`).

**Architecture:** A new self-contained module `src/mushin/_tuning.py` (sibling to `src/mushin/_packing.py`) holds two frozen result dataclasses (`BatchPin`, `LRPin`), shared sidecar read/write helpers, and the two public functions. Both functions follow the same flow: if a pin file exists and `retune` is False, read the pinned value and skip the search; otherwise run `pytorch_lightning.tuner.tuning.Tuner`, write the pin, and apply. `tune_batch_size` maximizes the device batch (`device_batch = min(found_max, effective/num_devices)`) and sets `trainer.accumulate_grad_batches` to reach ~the effective batch, recording any drift. The helpers are exported from `mushin`.

**Tech Stack:** Python 3.10+, PyTorch Lightning 2.6 (`Tuner.scale_batch_size`, `Tuner.lr_find`), OmegaConf (sidecar YAML), pytest + monkeypatch (hermetic — the Tuner is patched, no GPU/real search in CI).

---

## Verified facts (already checked against the installed environment; rely on these)

- `pytorch_lightning` is **2.6.5**. `from pytorch_lightning.tuner.tuning import Tuner`.
- `Tuner(trainer).scale_batch_size(model, datamodule=..., batch_arg_name="batch_size", mode="power", steps_per_trial=3, init_val=2, max_trials=25, margin=0.05, max_val=8192) -> Optional[int]` (returns the found batch, or None).
- `Tuner(trainer).lr_find(model, datamodule=..., update_attr=True, ...) -> Optional[_LRFinder]`; `_LRFinder.suggestion() -> Optional[float]`.
- **`trainer.accumulate_grad_batches = N` set AFTER construction IS honored live by `trainer.fit`** (verified: 8 batches ÷ accumulate=4 → 2 optimizer steps). No `GradientAccumulationScheduler` needed.
- `trainer.num_devices` exists (int, e.g. 1). `trainer.log_dir` is a usable path pre-fit (falls back through `default_root_dir`).
- `OmegaConf.save(OmegaConf.create({...}), path)` / `OmegaConf.load(path)` round-trips plain dicts.

## Conventions (match the existing codebase)

- Every source file starts with the two-line MIT header (copy from `src/mushin/_packing.py:1-2`).
- Test files start with the header from `tests/test_packing.py:1-2` and `import` inside each test function (that file's style).
- Commits: no `Co-Authored-By` / no Claude attribution. Stage named files only — never `git add -A` (the repo has an untracked `.worktrees/`).
- Run tools via `uv run` (e.g. `uv run pytest`, `uv run ruff`).

---

## Task 1: Module skeleton — dataclasses + sidecar I/O + default path

**Files:**
- Create: `src/mushin/_tuning.py`
- Test: `tests/test_tuning.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tuning.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest


def test_pin_roundtrip(tmp_path):
    from mushin._tuning import _read_pin, _write_pin

    path = tmp_path / "pin.yaml"
    assert _read_pin(path) is None  # absent -> None

    _write_pin(path, {"device_batch": 64, "effective_batch_size": 256, "num_devices": 1})
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
    assert str(_default_pin_path(trainer, "mushin_batch_pin.yaml")) == "/tmp/run7/mushin_batch_pin.yaml"


def test_default_pin_path_falls_back_to_default_root_dir():
    from types import SimpleNamespace

    from mushin._tuning import _default_pin_path

    trainer = SimpleNamespace(log_dir=None, default_root_dir="/tmp/root")
    assert str(_default_pin_path(trainer, "x.yaml")) == "/tmp/root/x.yaml"


def test_batchpin_and_lrpin_are_frozen_dataclasses():
    import dataclasses

    from mushin._tuning import BatchPin, LRPin

    bp = BatchPin(device_batch=64, accumulate_grad_batches=4, effective_batch_size=256, num_devices=1, drift=0)
    lp = LRPin(learning_rate=0.001)
    assert dataclasses.is_dataclass(bp) and dataclasses.is_dataclass(lp)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bp.device_batch = 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tuning.py -q`
Expected: FAIL / collection error — `No module named 'mushin._tuning'`.

- [ ] **Step 3: Write the module skeleton**

Create `src/mushin/_tuning.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Opt-in, reproducibility-preserving auto-tuning: find once, then pin.

``tune_batch_size`` and ``tune_learning_rate`` run PyTorch Lightning's ``Tuner``
a single time, record the result to a sidecar YAML file, and apply it. On a
later run the sidecar is read and the (stochastic, real-step) search is skipped,
so the same config yields the same result regardless of hardware.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchPin:
    """Result of :func:`tune_batch_size`."""

    device_batch: int
    accumulate_grad_batches: int
    effective_batch_size: int  # the ACTUAL realized value (device * accum * devices)
    num_devices: int
    drift: int  # actual_effective - requested_effective (0 when they match)


@dataclass(frozen=True)
class LRPin:
    """Result of :func:`tune_learning_rate`."""

    learning_rate: float


def _default_pin_path(trainer, filename: str) -> Path:
    base = getattr(trainer, "log_dir", None) or getattr(trainer, "default_root_dir", None) or "."
    return Path(base) / filename


def _read_pin(path) -> dict | None:
    """Return the sidecar's contents as a plain dict, or None if it is absent."""
    from omegaconf import OmegaConf

    p = Path(path)
    if not p.exists():
        return None
    return dict(OmegaConf.to_container(OmegaConf.load(p), resolve=True))


def _write_pin(path, mapping: dict) -> None:
    from omegaconf import OmegaConf

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(dict(mapping)), p)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tuning.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/_tuning.py tests/test_tuning.py
git commit -m "feat: auto-tuning module skeleton (pin I/O + result dataclasses)"
```

---

## Task 2: `tune_batch_size` — search-or-pin, batch math, apply

**Files:**
- Modify: `src/mushin/_tuning.py` (append the function)
- Test: `tests/test_tuning.py` (append tests)

Batch math (locked in the spec — *maximize device batch, approximate effective*):
- `per_device_total = effective_batch_size // num_devices` (require even division).
- `B = max(1, floor(found_max * (1 - safety_margin)))`; `device_batch = min(B, per_device_total)`.
- `accumulate = max(1, round(per_device_total / device_batch))`.
- `actual_effective = device_batch * accumulate * num_devices`; `drift = actual_effective - effective_batch_size`; `warn` if `drift != 0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tuning.py`:

```python
def _make_trainer():
    import pytorch_lightning as pl

    return pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False)


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
        _make_trainer(), object(), dm,
        effective_batch_size=256, num_devices=1, pin_path=tmp_path / "p.yaml",
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
        trainer, object(), _DM(),
        effective_batch_size=256, num_devices=1, pin_path=tmp_path / "p.yaml", retune=True,
    )
    assert (pin.device_batch, pin.accumulate_grad_batches) == (64, 4)
    assert pin.effective_batch_size == 256 and pin.drift == 0
    assert trainer.accumulate_grad_batches == 4  # applied to the trainer


def test_batch_drift_warns_when_not_divisible(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 100)  # 256/100 -> round(2.56)=3 -> 300
    with pytest.warns(UserWarning, match="differs from requested"):
        pin = tune_batch_size(
            _make_trainer(), object(), _DM(),
            effective_batch_size=256, num_devices=1, pin_path=tmp_path / "p.yaml", retune=True,
        )
    assert pin.device_batch == 100 and pin.accumulate_grad_batches == 3
    assert pin.effective_batch_size == 300 and pin.drift == 44


def test_batch_safety_margin_backs_off_found_max(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    # found 100, margin 0.2 -> floor(80)=80 -> min(80, 240)=80; round(240/80)=3 -> 240 (no drift)
    _patch_scale(monkeypatch, 100)
    pin = tune_batch_size(
        _make_trainer(), object(), _DM(),
        effective_batch_size=240, num_devices=1, safety_margin=0.2,
        pin_path=tmp_path / "p.yaml", retune=True,
    )
    assert pin.device_batch == 80 and pin.accumulate_grad_batches == 3
    assert pin.effective_batch_size == 240 and pin.drift == 0


def test_batch_num_devices_divides_effective(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, 512)  # per_device_total = 256/4 = 64
    pin = tune_batch_size(
        _make_trainer(), object(), _DM(),
        effective_batch_size=256, num_devices=4, pin_path=tmp_path / "p.yaml", retune=True,
    )
    assert pin.device_batch == 64 and pin.accumulate_grad_batches == 1
    assert pin.num_devices == 4 and pin.effective_batch_size == 256


def test_batch_invalid_inputs(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    # validation raises before any pin I/O, so the path is never written
    _patch_scale(monkeypatch, 128)
    t, dm, pin = _make_trainer(), _DM(), tmp_path / "p.yaml"
    with pytest.raises(ValueError, match="divisible"):
        tune_batch_size(t, object(), dm, effective_batch_size=250, num_devices=4,
                        pin_path=pin, retune=True)
    with pytest.raises(ValueError, match="safety_margin"):
        tune_batch_size(t, object(), dm, effective_batch_size=256, num_devices=1,
                        safety_margin=1.0, pin_path=pin, retune=True)
    with pytest.raises(ValueError, match="effective_batch_size"):
        tune_batch_size(t, object(), dm, effective_batch_size=0, num_devices=1,
                        pin_path=pin, retune=True)


def test_batch_none_from_tuner_raises(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    _patch_scale(monkeypatch, None)
    with pytest.raises(RuntimeError, match="no batch size"):
        tune_batch_size(_make_trainer(), object(), _DM(), effective_batch_size=256,
                        num_devices=1, pin_path=tmp_path / "p.yaml", retune=True)


def test_batch_pin_roundtrip_skips_search(monkeypatch, tmp_path):
    from mushin._tuning import tune_batch_size

    counter = {"n": 0}
    _patch_scale(monkeypatch, 128, counter)
    pin_path = tmp_path / "batch.yaml"
    a1 = _make_trainer()
    r1 = tune_batch_size(a1, object(), _DM(), effective_batch_size=256, num_devices=1, pin_path=pin_path)
    assert counter["n"] == 1 and pin_path.exists()

    a2 = _make_trainer()
    r2 = tune_batch_size(a2, object(), _DM(), effective_batch_size=256, num_devices=1, pin_path=pin_path)
    assert counter["n"] == 1  # search NOT called again
    assert r2.device_batch == r1.device_batch
    assert a2.accumulate_grad_batches == r1.accumulate_grad_batches  # still applied

    r3 = tune_batch_size(_make_trainer(), object(), _DM(), effective_batch_size=256,
                         num_devices=1, pin_path=pin_path, retune=True)
    assert counter["n"] == 2  # retune forces a fresh search
    assert r3.device_batch == 128
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tuning.py -k batch -q`
Expected: FAIL — `tune_batch_size` does not exist (ImportError).

- [ ] **Step 3: Implement `tune_batch_size`**

Append to `src/mushin/_tuning.py`:

```python
def tune_batch_size(
    trainer,
    module,
    datamodule=None,
    *,
    effective_batch_size: int,
    pin_path=None,
    num_devices: int | None = None,
    safety_margin: float = 0.0,
    batch_arg: str = "batch_size",
    retune: bool = False,
    **scale_kwargs,
) -> BatchPin:
    """Pin the effective batch; find the largest device batch and set accumulation.

    Runs ``Tuner.scale_batch_size`` once to find the largest device batch that
    fits, caps it at ``effective_batch_size / num_devices``, and sets
    ``trainer.accumulate_grad_batches`` so the realized effective batch is as
    close as possible to the requested one. The found device batch is written to
    ``pin_path``; a later call reads it and skips the search (``retune=True``
    forces a fresh search).

    Parameters
    ----------
    effective_batch_size : int
        The pinned, hardware-independent quantity
        ``device_batch * accumulate_grad_batches * num_devices``. Must be
        divisible by ``num_devices``.
    pin_path : str, Path, or None
        Sidecar YAML. Defaults to ``<trainer.log_dir>/mushin_batch_pin.yaml``.
    num_devices : int or None
        Defaults to ``trainer.num_devices``.
    safety_margin : float
        Fraction in ``[0, 1)`` to back the found max off by (OOM noise).
    batch_arg : str
        Attribute on the datamodule (or module) the tuner scales and the helper
        sets; forwarded to the tuner as ``batch_arg_name``.
    retune : bool
        Ignore any existing pin and search again.
    **scale_kwargs
        Forwarded to ``Tuner.scale_batch_size`` (e.g. ``mode``, ``steps_per_trial``).

    Returns
    -------
    BatchPin
        With the ACTUAL realized ``effective_batch_size`` and the ``drift`` from
        the requested value.
    """
    from pytorch_lightning.tuner.tuning import Tuner

    if effective_batch_size < 1:
        raise ValueError(f"effective_batch_size must be >= 1; got {effective_batch_size}")
    if not (0.0 <= safety_margin < 1.0):
        raise ValueError(f"safety_margin must be in [0, 1); got {safety_margin}")
    if num_devices is None:
        num_devices = max(1, int(getattr(trainer, "num_devices", 1) or 1))
    if num_devices < 1:
        raise ValueError(f"num_devices must be >= 1; got {num_devices}")
    if effective_batch_size % num_devices != 0:
        raise ValueError(
            f"effective_batch_size={effective_batch_size} must be divisible by "
            f"num_devices={num_devices}; choose an effective batch that divides evenly."
        )
    per_device_total = effective_batch_size // num_devices

    if pin_path is None:
        pin_path = _default_pin_path(trainer, "mushin_batch_pin.yaml")

    pin = None if retune else _read_pin(pin_path)
    if pin is not None:
        device_batch = int(pin["device_batch"])
    else:
        found_max = Tuner(trainer).scale_batch_size(
            module, datamodule=datamodule, batch_arg_name=batch_arg, **scale_kwargs
        )
        if found_max is None:
            raise RuntimeError(
                "tune_batch_size: Tuner.scale_batch_size returned no batch size. "
                "Check that the model or datamodule exposes the "
                f"'{batch_arg}' attribute, or pass an explicit pin file."
            )
        backed_off = max(1, math.floor(found_max * (1.0 - safety_margin)))
        device_batch = min(backed_off, per_device_total)
        _write_pin(
            pin_path,
            {
                "device_batch": device_batch,
                "effective_batch_size": effective_batch_size,
                "num_devices": num_devices,
            },
        )

    accumulate = max(1, round(per_device_total / device_batch))
    actual_effective = device_batch * accumulate * num_devices
    drift = actual_effective - effective_batch_size
    if drift != 0:
        warnings.warn(
            f"tune_batch_size: realized effective batch {actual_effective} differs "
            f"from requested {effective_batch_size} (drift {drift:+d}): the device "
            f"batch {device_batch} does not divide the per-device target "
            f"{per_device_total}. The actual value is recorded in the returned BatchPin.",
            UserWarning,
            stacklevel=2,
        )

    # apply: device batch on the datamodule (else the module); accumulation on trainer
    if datamodule is not None and hasattr(datamodule, batch_arg):
        setattr(datamodule, batch_arg, device_batch)
    else:
        setattr(module, batch_arg, device_batch)
    trainer.accumulate_grad_batches = accumulate

    return BatchPin(
        device_batch=device_batch,
        accumulate_grad_batches=accumulate,
        effective_batch_size=actual_effective,
        num_devices=num_devices,
        drift=drift,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tuning.py -k batch -q`
Expected: PASS (8 batch tests).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/_tuning.py tests/test_tuning.py
git commit -m "feat: tune_batch_size (find device batch, pin effective, set accumulation)"
```

---

## Task 3: `tune_learning_rate` — record-and-pin the LR finder

**Files:**
- Modify: `src/mushin/_tuning.py` (append the function)
- Test: `tests/test_tuning.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tuning.py`:

```python
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
    tune_learning_rate(_make_trainer(), m, None, pin_path=tmp_path / "lr.yaml", lr_attr="learning_rate")
    assert m.learning_rate == 0.007


def test_lr_none_suggestion_raises(monkeypatch, tmp_path):
    from mushin._tuning import tune_learning_rate

    _patch_lr_find(monkeypatch, None)
    with pytest.raises(RuntimeError, match="no learning-rate suggestion"):
        tune_learning_rate(_make_trainer(), _Mod(), None, pin_path=tmp_path / "lr.yaml")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tuning.py -k lr -q`
Expected: FAIL — `tune_learning_rate` does not exist.

- [ ] **Step 3: Implement `tune_learning_rate`**

Append to `src/mushin/_tuning.py`:

```python
def tune_learning_rate(
    trainer,
    module,
    datamodule=None,
    *,
    pin_path=None,
    lr_attr: str = "lr",
    retune: bool = False,
    **lr_find_kwargs,
) -> LRPin:
    """Record-and-pin Lightning's LR finder.

    Runs ``Tuner.lr_find`` once, writes the suggested learning rate to
    ``pin_path``, and sets ``module.<lr_attr>``. A later call reads the pin and
    skips the (stochastic) range test; ``retune=True`` forces a fresh search.
    Learning rate is hardware-independent, so there is no device math.

    Parameters
    ----------
    pin_path : str, Path, or None
        Sidecar YAML. Defaults to ``<trainer.log_dir>/mushin_lr_pin.yaml``.
    lr_attr : str
        Attribute on the module set to the found learning rate.
    retune : bool
        Ignore any existing pin and search again.
    **lr_find_kwargs
        Forwarded to ``Tuner.lr_find`` (e.g. ``min_lr``, ``max_lr``, ``num_training``).
    """
    from pytorch_lightning.tuner.tuning import Tuner

    if pin_path is None:
        pin_path = _default_pin_path(trainer, "mushin_lr_pin.yaml")

    pin = None if retune else _read_pin(pin_path)
    if pin is not None:
        lr = float(pin["learning_rate"])
    else:
        # update_attr=False: we set the attribute ourselves, uniformly, whether
        # the value came from the finder or a pin.
        finder = Tuner(trainer).lr_find(
            module, datamodule=datamodule, update_attr=False, **lr_find_kwargs
        )
        lr = finder.suggestion() if finder is not None else None
        if lr is None:
            raise RuntimeError(
                "tune_learning_rate: Tuner.lr_find found no learning-rate suggestion. "
                "Widen the range (min_lr/max_lr) or increase num_training."
            )
        lr = float(lr)
        _write_pin(pin_path, {"learning_rate": lr})

    setattr(module, lr_attr, lr)
    return LRPin(learning_rate=lr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tuning.py -k lr -q`
Expected: PASS (4 lr tests).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/_tuning.py tests/test_tuning.py
git commit -m "feat: tune_learning_rate (record-and-pin the LR finder)"
```

---

## Task 4: Export from `mushin`

**Files:**
- Modify: `src/mushin/__init__.py`
- Test: `tests/test_tuning.py` (append one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tuning.py`:

```python
def test_exports():
    import mushin
    from mushin import tune_batch_size, tune_learning_rate

    assert "tune_batch_size" in mushin.__all__
    assert "tune_learning_rate" in mushin.__all__
    assert callable(tune_batch_size) and callable(tune_learning_rate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tuning.py::test_exports -q`
Expected: FAIL — `ImportError: cannot import name 'tune_batch_size'`.

- [ ] **Step 3: Add the imports and `__all__` entries**

In `src/mushin/__init__.py`, next to the packing import (`from ._packing import pin_gpu_round_robin`), add:

```python
from ._tuning import tune_batch_size, tune_learning_rate
```

In the `__all__` list, add the two names next to `"pin_gpu_round_robin"`:

```python
    "pin_gpu_round_robin",
    "tune_batch_size",
    "tune_learning_rate",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tuning.py::test_exports -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/__init__.py tests/test_tuning.py
git commit -m "feat: export tune_batch_size and tune_learning_rate from mushin"
```

---

## Task 5: End-to-end apply-path integration test (real Trainer + real fit)

Locks the flagged risk (that the applied `accumulate_grad_batches` and device batch actually take effect) against regressions, without running the real Tuner search — the search is monkeypatched, but a real 1-epoch CPU `fit` runs and the optimizer-step count is asserted.

**Files:**
- Test: `tests/test_tuning_integration.py`

- [ ] **Step 1: Write the test**

Create `tests/test_tuning_integration.py`:

```python
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
        max_epochs=1, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )

    # effective 8 on 1 device => per_device_total 8; device_batch min(2, 8) = 2;
    # accumulate = round(8 / 2) = 4.
    pin = tune_batch_size(
        trainer, module, dm, effective_batch_size=8, num_devices=1,
        pin_path=tmp_path / "pin.yaml", retune=True,
    )
    assert pin.device_batch == 2 and pin.accumulate_grad_batches == 4
    assert dm.batch_size == 2  # applied

    trainer.fit(module, datamodule=dm)
    # 16 samples / batch 2 = 8 batches; accumulate 4 => 2 optimizer steps.
    assert module.opt_steps == 2
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_tuning_integration.py -q`
Expected: PASS. (If it fails on the optimizer-step count, the Lightning version does not honor a post-construction `accumulate_grad_batches` — STOP and report; the apply mechanism must change, not the assertion.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_tuning_integration.py
git commit -m "test: end-to-end apply-path check for tune_batch_size (real fit)"
```

---

## Task 6: Docs guide + nav + changelog

**Files:**
- Create: `docs/guides/auto-tuning.md`
- Modify: `mkdocs.yml` (nav)
- Create: `changes/+auto-tuning.added.md`

- [ ] **Step 1: Write the guide**

Create `docs/guides/auto-tuning.md`:

```markdown
# Auto-tuning batch size and learning rate

PyTorch Lightning's `Tuner` can find the largest **device** batch that fits
(`scale_batch_size`) and a learning rate (`lr_find`). Used naively they hurt
reproducibility: the largest device batch depends on the GPU, so the same config
gives different results on a 24 GB vs an 80 GB card, and a sweep silently
compares methods at different batch sizes.

mushin adds two opt-in helpers that keep the convenience while protecting
reproducibility. Both **find once, then pin**: they write the found value to a
small sidecar YAML file and, on a later run, read it and skip the search.

## `tune_batch_size`: pin the effective batch

Pin the **effective** batch (`device_batch x accumulate_grad_batches x
num_devices`) — the hardware-independent, scientifically meaningful quantity.
The helper finds the largest device batch that fits, then sets
`accumulate_grad_batches` to reach your target. Call it before `fit`:

```python
from mushin import tune_batch_size

pin = tune_batch_size(trainer, module, datamodule, effective_batch_size=256)
print(pin.device_batch, pin.accumulate_grad_batches, pin.effective_batch_size)
trainer.fit(module, datamodule=datamodule)
```

The device batch is maximized for throughput, so the realized effective batch
can differ slightly from the target when the device batch doesn't divide it
evenly — the helper **records the actual value** in `pin.effective_batch_size`
and **warns** when it drifts. The found device batch is written to
`<trainer.log_dir>/mushin_batch_pin.yaml` (override with `pin_path=`); commit it
to make re-runs deterministic. Pass `retune=True` to search again — for example
when you deliberately move to hardware where the pinned batch no longer fits.

Use `safety_margin=` (e.g. `0.1`) to back the found maximum off from the OOM
edge, and `num_devices=` if it should not come from the trainer.

## `tune_learning_rate`: record-and-pin the LR finder

```python
from mushin import tune_learning_rate

pin = tune_learning_rate(trainer, module, datamodule)  # sets module.lr
trainer.fit(module, datamodule=datamodule)
```

Learning rate is hardware-independent, so there is no device math — pinning just
makes the stochastic range test skip on re-runs and reuse the exact found value.
The suggestion is written to `<trainer.log_dir>/mushin_lr_pin.yaml` and set on
`module.lr` (use `lr_attr=` for a different attribute).

## Caveats

- **Opt-in and explicit.** Both run real training steps and mutate then reset
  trainer/model state — call them deliberately, not on by default.
- **Tune on a single device.** The pinned device batch and recomputed
  accumulation then apply at scale; running the finder itself under DDP is not
  orchestrated for you.
- **Reproducibility is preserved by the pin.** Re-runs reuse the recorded device
  batch regardless of hardware. A pinned batch that no longer fits on smaller
  hardware is a deliberate `retune=True` decision, not a silent change.
```

- [ ] **Step 2: Add the nav entry**

In `mkdocs.yml`, under `Guides:` add a line after the Packing entry (`- Packing small jobs: guides/packing.md`):

```yaml
      - Auto-tuning batch & LR: guides/auto-tuning.md
```

- [ ] **Step 3: Add the changelog fragment**

Create `changes/+auto-tuning.added.md`:

```markdown
`tune_batch_size` and `tune_learning_rate`: opt-in, reproducibility-preserving
auto-tuning helpers. `tune_batch_size` pins the effective batch, finds the
largest device batch that fits, and sets `accumulate_grad_batches` to reach it
(recording any drift); `tune_learning_rate` records-and-pins Lightning's LR
finder. Both write the found value to a sidecar YAML so re-runs skip the search
and stay deterministic across hardware. New "Auto-tuning batch size and learning
rate" guide.
```

- [ ] **Step 4: Verify the docs build**

Run: `uv run mkdocs build --strict`
Expected: builds with no warnings/errors.

- [ ] **Step 5: Commit**

```bash
git add docs/guides/auto-tuning.md mkdocs.yml changes/+auto-tuning.added.md
git commit -m "docs: auto-tuning guide + nav + changelog"
```

---

## Task 7: Full verification

- [ ] **Step 1: Full suite + lint + spellcheck + docs**

```bash
uv run pytest tests/test_tuning.py tests/test_tuning_integration.py -v
uv run pytest -q
uv run ruff check src/mushin/_tuning.py tests/test_tuning.py tests/test_tuning_integration.py
uv run ruff format src/mushin/_tuning.py tests/test_tuning.py tests/test_tuning_integration.py
uv run codespell src/mushin/_tuning.py tests/test_tuning.py tests/test_tuning_integration.py docs/guides/auto-tuning.md
uv run mkdocs build --strict
```

Expected: all green; full suite passes with the new tests included.

- [ ] **Step 2: Commit any formatting changes**

```bash
git add -u src/mushin tests
git commit -m "style: ruff format for auto-tuning" || echo "nothing to format"
```

---

## Notes for the reviewer / final step

- **No cluster gate.** Every path is hermetic (the Tuner is monkeypatched; the one real `fit` is a tiny CPU run). This merges to `main` via the normal CI + Codex review gate — unlike PRs #50/#58/#59, it is not held for hardware.
- After all tasks: dispatch a final whole-branch review, then push `auto-tuning` and open a PR against `main`; watch CI + Codex and fix findings before merge (no `Co-Authored-By`).
