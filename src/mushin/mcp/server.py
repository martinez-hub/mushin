# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Read-only logic for the mushin MCP server (transport-agnostic)."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from mushin._utils import Experiment, load_experiment


def _flatten(value: Any, prefix: str = "") -> dict:
    """Flatten a nested (already JSON-able) dict into dotted keys."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = value
    return out


def _as_list(exps: Any) -> list:
    return [exps] if isinstance(exps, Experiment) else list(exps)


def _describe_experiment(path: str | Path, root: str | Path | None = None) -> dict:
    """Summarize swept params, metric keys, and run/checkpoint counts."""
    p = _resolve(path, root)
    exps = _as_list(load_experiment(p))
    metric_keys = sorted({k for e in exps for k in (e.metrics or {})})
    flats = [_flatten(_to_jsonable(e.cfg)) for e in exps if e.cfg is not None]
    swept: dict[str, list] = {}
    if flats:
        for k in sorted(set().union(*(set(f) for f in flats))):
            uniq: list = []
            for f in flats:
                v = f.get(k)
                if v not in uniq:
                    uniq.append(v)
            if len(uniq) > 1:
                swept[k] = uniq
    return {
        "path": str(p),
        "num_runs": len(exps),
        "metric_keys": metric_keys,
        "swept_params": swept,
        "num_checkpoints": [len(e.ckpts) for e in exps],
    }


def _reduce_metrics(per_run: list[dict], how: str) -> dict:
    """Reduce numeric metric leaves across runs by 'mean' or 'std'."""
    if how not in {"mean", "std"}:
        raise ValueError(f"unknown reduce '{how}'; use 'mean' or 'std'")
    flats = [_flatten(r) for r in per_run]
    out: dict[str, float] = {}
    for k in sorted(set().union(*(set(f) for f in flats))) if flats else []:
        vals = [
            f[k]
            for f in flats
            if isinstance(f.get(k), (int, float)) and not isinstance(f.get(k), bool)
        ]
        if not vals:
            continue
        if how == "mean":
            out[k] = sum(vals) / len(vals)
        else:
            out[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return out


def _get_metrics(
    path: str | Path,
    metrics: list[str] | None = None,
    reduce: str | None = None,
    root: str | Path | None = None,
) -> dict:
    """Return per-run metrics, optionally filtered and reduced across runs."""
    p = _resolve(path, root)
    exps = _as_list(load_experiment(p))
    per_run = []
    for e in exps:
        m = _to_jsonable(e.metrics or {})
        if metrics is not None:
            m = {k: v for k, v in m.items() if k in metrics}
        per_run.append(m)
    result = {"path": str(p), "num_runs": len(exps), "per_run": per_run}
    if reduce is not None:
        result["reduced"] = _reduce_metrics(per_run, reduce)
    return result


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


class RootError(ValueError):
    """Raised when a requested path escapes the configured --root."""


def _resolve(path: str | Path, root: str | Path | None) -> Path:
    """Resolve ``path`` to an absolute Path, enforcing ``root`` containment."""
    p = Path(path).expanduser().resolve()
    if root is not None:
        root = Path(root).expanduser().resolve()
        if p != root and root not in p.parents:
            raise RootError(f"{p} is outside the configured root {root}")
    return p


def _list_experiments(root: str | Path | None = None) -> dict:
    """List run directories (those containing a ``.hydra/`` child) under ``root``."""
    base = _resolve(root if root is not None else Path.cwd(), root)
    if not base.exists():
        raise FileNotFoundError(f"{base} not found")
    runs = sorted(str(p.parent) for p in base.glob("**/.hydra"))
    return {"root": str(base), "runs": runs, "count": len(runs)}
