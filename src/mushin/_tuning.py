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
    base = (
        getattr(trainer, "log_dir", None)
        or getattr(trainer, "default_root_dir", None)
        or "."
    )
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
        raise ValueError(
            f"effective_batch_size must be >= 1; got {effective_batch_size}"
        )
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
