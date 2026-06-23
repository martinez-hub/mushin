# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Read-only logic for the mushin MCP server (transport-agnostic)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def create_server() -> None:  # pragma: no cover
    """Placeholder — full implementation in a later task."""
    raise NotImplementedError("create_server is not yet implemented")


def _to_jsonable(obj: Any) -> Any:
    """Convert torch/numpy/omegaconf values into JSON-serializable Python."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if OmegaConf.is_config(obj):
        return _to_jsonable(OmegaConf.to_container(obj, resolve=True))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    return str(obj)
