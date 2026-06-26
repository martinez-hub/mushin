"""Normalize a system value (a callable or a hydra-zen config) into a callable."""

from __future__ import annotations

import dataclasses
from typing import Any

from hydra_zen import instantiate
from omegaconf import DictConfig, ListConfig


def _is_config(value: Any) -> bool:
    """True if `value` is a hydra-zen / OmegaConf config (has a _target_)."""
    if isinstance(value, (DictConfig, ListConfig)):
        return "_target_" in value
    if dataclasses.is_dataclass(value):  # builds(...) returns a dataclass type/instance
        return hasattr(value, "_target_")
    return isinstance(value, dict) and "_target_" in value


def as_system(value: Any):
    """Return a callable `system(inputs, seed)`. hydra-zen configs are instantiated
    once here (so a heavy model loads a single time and is reused across seeds)."""
    if _is_config(value):
        value = instantiate(value)
    if not callable(value):
        raise TypeError(
            "each system must be a callable system(inputs, seed) -> outputs, "
            f"or a hydra-zen config that instantiates to one; got {type(value)!r}"
        )
    return value
