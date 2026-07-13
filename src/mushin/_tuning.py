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
    """Result of :func:`tune_batch_size`. The effective batch is always exact."""

    device_batch: int
    accumulate_grad_batches: int
    effective_batch_size: int  # always == the requested value (exact divisor selection)
    num_devices: int


@dataclass(frozen=True)
class LRPin:
    """Result of :func:`tune_learning_rate`."""

    learning_rate: float


def _default_pin_path(trainer, filename: str) -> Path:
    # Inside a Hydra --multirun sweep the default directory is shared across all
    # jobs, so distinct configs would clobber (and silently reuse) one another's
    # pins. Require an explicit, per-config pin_path there instead of guessing.
    from hydra.core.hydra_config import HydraConfig

    if HydraConfig.initialized():
        from hydra.types import RunMode

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            raise RuntimeError(
                "auto-tuning: no pin_path given inside a Hydra --multirun sweep. The "
                "default pin directory is shared across sweep jobs, so different "
                "configs would overwrite and silently reuse each other's pins. Pass "
                "an explicit per-config pin_path=... (and commit it for reproducibility)."
            )
    base = (
        getattr(trainer, "default_root_dir", None)
        or getattr(trainer, "log_dir", None)
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


def _set_attr(target, name: str, value) -> None:
    """Set ``name`` on ``target``, also updating ``.hparams`` if it contains the key.

    ``pytorch_lightning.utilities.parsing.lightning_setattr`` requires a
    ``_trainer`` attribute (it is designed for attached modules); this helper
    provides the same hparams-awareness for standalone objects — used by
    :func:`tune_batch_size` and :func:`tune_learning_rate` when applying the
    found value before training begins.
    """
    setattr(target, name, value)
    hparams = getattr(target, "hparams", None)
    if isinstance(hparams, dict) and name in hparams:
        hparams[name] = value


def _has_attr(target, name: str) -> bool:
    """True if ``target`` exposes ``name`` directly or via a ``.hparams`` mapping.

    A datamodule that stores the batch size only through ``save_hyperparameters()``
    has no direct attribute, so a plain ``hasattr`` would miss it and the value
    would be applied to the wrong object.
    """
    if hasattr(target, name):
        return True
    hparams = getattr(target, "hparams", None)
    return isinstance(hparams, dict) and name in hparams


def _largest_divisor_leq(target: int, cap: int) -> int:
    """Largest ``d`` with ``d`` dividing ``target`` and ``1 <= d <= cap``.

    Used to pick a device batch that divides the per-device target exactly, so the
    realized effective batch equals the requested one with no drift. ``d == 1``
    always divides ``target``, so a value in ``[1, cap]`` always exists. For the
    round batch sizes researchers use (256/512/1024, many divisors) this returns a
    value at or just below ``cap``; only pathological (near-prime) targets fall far.
    """
    d = min(int(target), int(cap))
    while target % d != 0:
        d -= 1
    return d


def tune_batch_size(
    trainer,
    module,
    datamodule=None,
    *,
    effective_batch_size: int,
    pin_path=None,
    num_devices: int | None = None,
    batch_arg: str = "batch_size",
    retune: bool = False,
    **scale_kwargs,
) -> BatchPin:
    """Pin the effective batch; find the largest fitting device batch, set accumulation.

    Runs ``Tuner.scale_batch_size`` once to find the largest device batch that
    fits (``found_max``), then chooses ``device_batch`` = the largest divisor of
    ``effective_batch_size / num_devices`` that is ``<= found_max`` and sets
    ``trainer.accumulate_grad_batches`` accordingly. Because ``device_batch``
    divides the per-device target exactly, the realized effective batch always
    equals ``effective_batch_size`` on any hardware — no drift.

    ``found_max`` (the raw hardware probe) is written to ``pin_path``; a later run
    reads it and skips the search, re-deriving ``device_batch``/accumulation for
    that run's ``effective_batch_size``/``num_devices``. ``retune=True`` forces a
    fresh search.

    Parameters
    ----------
    effective_batch_size : int
        The pinned, hardware-independent quantity
        ``device_batch * accumulate_grad_batches * num_devices``. Must be
        divisible by ``num_devices``.
    pin_path : str, Path, or None
        Sidecar YAML storing ``found_max_device_batch``. Defaults to
        ``<trainer.default_root_dir>/mushin_batch_pin.yaml``. Inside a Hydra
        ``--multirun`` an explicit ``pin_path`` is required (the default dir is
        shared across jobs).
    num_devices : int or None
        Defaults to ``trainer.num_devices * trainer.num_nodes``.
    batch_arg : str
        Attribute on the module (or datamodule) the tuner scales and this helper
        sets; forwarded to the tuner as ``batch_arg_name``.
    retune : bool
        Ignore any existing pin and search again.
    **scale_kwargs
        Forwarded to ``Tuner.scale_batch_size`` (e.g. ``mode``, ``steps_per_trial``).

    Returns
    -------
    BatchPin
    """
    from pytorch_lightning.tuner.tuning import Tuner

    if effective_batch_size < 1:
        raise ValueError(
            f"effective_batch_size must be >= 1; got {effective_batch_size}"
        )
    if num_devices is None:
        per_node = int(getattr(trainer, "num_devices", 1) or 1)
        nodes = int(getattr(trainer, "num_nodes", 1) or 1)
        num_devices = max(1, per_node * nodes)
    if effective_batch_size % num_devices != 0:
        raise ValueError(
            f"effective_batch_size={effective_batch_size} must be divisible by "
            f"num_devices={num_devices}; choose an effective batch that divides evenly."
        )
    per_device_total = effective_batch_size // num_devices

    # Validate the batch owner up front so we never write a pin for a call that
    # would then apply the value to a dead attribute. Apply to the SAME owner
    # Lightning's finder scales (module first, then datamodule).
    module_has = _has_attr(module, batch_arg)
    dm_has = datamodule is not None and _has_attr(datamodule, batch_arg)
    if not module_has and not dm_has:
        raise ValueError(
            f"neither the module nor the datamodule exposes '{batch_arg}', so the "
            "tuned batch size cannot be applied to anything the dataloader reads. "
            f"Expose '{batch_arg}' on your module (or datamodule), or set batch_arg=."
        )

    if pin_path is None:
        pin_path = _default_pin_path(trainer, "mushin_batch_pin.yaml")

    pin = None if retune else _read_pin(pin_path)
    if pin is not None:
        found_max = int(pin["found_max_device_batch"])
        if found_max < 1:
            raise ValueError(
                f"pin file {pin_path} has an invalid found_max_device_batch="
                f"{found_max} (must be >= 1); delete it or pass retune=True."
            )
    else:
        found_max = Tuner(trainer).scale_batch_size(
            module, datamodule=datamodule, batch_arg_name=batch_arg, **scale_kwargs
        )
        if found_max is None:
            raise RuntimeError(
                "tune_batch_size: Tuner.scale_batch_size returned no batch size. "
                f"Check that the model or datamodule exposes the '{batch_arg}' "
                "attribute, or pass an explicit pin file."
            )
        found_max = int(found_max)
        _write_pin(pin_path, {"found_max_device_batch": found_max})

    device_batch = _largest_divisor_leq(per_device_total, found_max)
    accumulate = per_device_total // device_batch  # exact: device_batch divides target
    if device_batch < found_max and device_batch * 2 <= found_max:
        warnings.warn(
            f"tune_batch_size: chosen device_batch={device_batch} is well below the "
            f"{found_max} that fits, because effective_batch_size={effective_batch_size} "
            f"(per-device target {per_device_total}) has no larger divisor that fits. "
            "Pick a rounder effective_batch_size (more divisors) for better GPU use.",
            UserWarning,
            stacklevel=2,
        )

    target = module if module_has else datamodule
    _set_attr(target, batch_arg, device_batch)
    trainer.accumulate_grad_batches = accumulate

    return BatchPin(
        device_batch=device_batch,
        accumulate_grad_batches=accumulate,
        effective_batch_size=effective_batch_size,
        num_devices=num_devices,
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
        Sidecar YAML. Defaults to ``<trainer.default_root_dir>/mushin_lr_pin.yaml``.
    lr_attr : str
        Attribute on the module set to the found learning rate.
    retune : bool
        Ignore any existing pin and search again.
    **lr_find_kwargs
        Forwarded to ``Tuner.lr_find`` (e.g. ``min_lr``, ``max_lr``, ``num_training``).
    """
    from pytorch_lightning.callbacks import LearningRateFinder
    from pytorch_lightning.tuner.tuning import Tuner

    # A LearningRateFinder callback re-runs its stochastic range test at fit and can
    # move the model off the pinned LR, breaking determinism. Reject the combination.
    if any(
        isinstance(cb, LearningRateFinder)
        for cb in getattr(trainer, "callbacks", []) or []
    ):
        raise ValueError(
            "the Trainer already has a Lightning LearningRateFinder callback, which "
            "would run its own range test at fit and move the model off the tuned/"
            "pinned learning rate. Use tune_learning_rate or that callback, not both."
        )

    # Validate the LR owner up front: a misspelled/renamed lr_attr would otherwise
    # have _set_attr create a dead attribute while configure_optimizers keeps reading
    # the old field, so LRPin would report a value that never took effect.
    if not _has_attr(module, lr_attr):
        raise ValueError(
            f"the module does not expose '{lr_attr}', so the tuned learning rate "
            "cannot be applied to anything configure_optimizers reads. Define "
            f"self.{lr_attr} on the module, or pass lr_attr=."
        )

    if pin_path is None:
        pin_path = _default_pin_path(trainer, "mushin_lr_pin.yaml")

    pin = None if retune else _read_pin(pin_path)
    if pin is not None:
        lr = float(pin["learning_rate"])
        if not (lr > 0 and math.isfinite(lr)):
            raise ValueError(
                f"pin file {pin_path} has an invalid learning_rate={lr} "
                "(must be > 0); delete it or pass retune=True to re-tune."
            )
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

    _set_attr(module, lr_attr, lr)
    return LRPin(learning_rate=lr)
