# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Read-only logic for the mushin MCP server (transport-agnostic)."""

from __future__ import annotations

import io
import math
import pickle
import statistics
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra_zen import load_from_yaml
from omegaconf import OmegaConf

from mushin._utils import Experiment

# Globals an MCP-loaded metrics file may legitimately reference. Everything here
# is a pure-data container or numpy array/scalar constructor — no callable that
# could run code. Anything outside this set is refused, so a malicious *.pt under
# --root cannot execute arbitrary pickle.
_SAFE_GLOBALS = {("collections", "OrderedDict"), ("collections", "defaultdict")}
_BUILTIN_MODULES = {"builtins", "__builtin__"}
_SAFE_BUILTINS = {"list", "dict", "tuple", "set", "frozenset", "bytearray", "complex"}
_SAFE_NUMPY_NAMES = {
    "ndarray", "dtype",
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool_", "complex64", "complex128", "intc", "intp", "longlong",
}


def _is_safe_global(module: str, name: str) -> bool:
    if (module, name) in _SAFE_GLOBALS:
        return True
    if module in _BUILTIN_MODULES and name in _SAFE_BUILTINS:
        return True
    if module == "numpy" and name in _SAFE_NUMPY_NAMES:
        return True
    # numpy array/scalar reconstruction (data constructors, numpy 1.x and 2.x paths)
    if name in {"_reconstruct", "scalar"} and module.endswith("multiarray"):
        return True
    # numpy stores raw array bytes via codecs.encode(bytes, "latin1")
    if (module, name) == ("_codecs", "encode"):
        return True
    return False


class _DataOnlyUnpickler(pickle.Unpickler):
    """Unpickler that reconstructs only pure data (safe containers + numpy)."""

    def find_class(self, module: str, name: str):
        if _is_safe_global(module, name):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked unsafe global {module}.{name}")

    def persistent_load(self, pid):
        # Persistent ids appear only for torch storages (tensors); refuse them so
        # tensor payloads are handled by the weights_only path, never here.
        raise pickle.UnpicklingError("persistent storage not allowed in data-only load")


def _data_only_load(path: Path):
    """Load a ``torch.save`` file allowing only safe, pure-data objects."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith("data.pkl")]
            if not names:
                raise pickle.UnpicklingError("no data.pkl in archive")
            raw = archive.read(names[0])
    else:  # legacy (non-zip) torch.save format
        raw = Path(path).read_bytes()
    return _DataOnlyUnpickler(io.BytesIO(raw)).load()


def _safe_load_pt(path: Path):
    """Best-effort safe load of a torch ``*.pt`` file.

    Tries torch's ``weights_only`` loader first (safely handles tensors on any
    torch version), then a data-only unpickler for pure-data payloads such as
    ``MetricsCallback``'s ``defaultdict`` metrics. Never executes pickled code;
    raises if neither safe path can read the file.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return _data_only_load(path)


def _flatten(value: Any, prefix: str = "") -> dict:
    """Flatten a nested (already JSON-able) dict into dotted keys."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = value
    return out


def _job_sort_key(hydra_dir: Path):
    """Sort key so numeric Hydra job dirs (0,1,2,...,10) order numerically."""
    name = hydra_dir.parent.name
    return (0, int(name)) if name.isdigit() else (1, name)


def _within_root(target: Path, root: Path | None) -> bool:
    """True if ``target`` (after resolving symlinks) is inside ``root``."""
    if root is None:
        return True
    resolved = target.resolve()
    return resolved == root or root in resolved.parents


def _load_runs(p: Path, root: str | Path | None = None) -> list[Experiment]:
    """Load experiment runs at ``p`` without executing pickled code.

    Job directories are ordered numerically so the ``job`` index matches Hydra
    job numbers. When ``root`` is set, each discovered config/metric/checkpoint
    is re-checked for containment (resolving symlinks) so artifacts escaping
    ``root`` are refused.
    """
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    rootp = Path(root).expanduser().resolve() if root is not None else None
    runs: list[Experiment] = []
    for hydra_dir in sorted(p.glob("**/.hydra"), key=_job_sort_key):
        run_dir = hydra_dir.parent
        cfg_file = hydra_dir / "config.yaml"
        cfg = (
            load_from_yaml(cfg_file)
            if cfg_file.exists() and _within_root(cfg_file, rootp)
            else None
        )
        metrics: dict = {}
        for f in sorted(run_dir.glob("*.pt")):
            if not _within_root(f, rootp):
                continue
            try:
                metrics[f.stem] = _safe_load_pt(f)
            except Exception:
                # Fail closed: never fall back to unsafe (pickle-executing)
                # loading; just skip metrics we cannot safely read.
                continue
        ckpts = [
            str(c.resolve())
            for c in run_dir.glob("**/*.ckpt")
            if _within_root(c, rootp)
        ]
        runs.append(Experiment(str(run_dir.parent), cfg, ckpts, metrics))
    if not runs:
        raise FileNotFoundError(f"no experiment found at {p} (no .hydra directory)")
    return runs


def _describe_experiment(path: str | Path, root: str | Path | None = None) -> dict:
    """Summarize swept params, metric keys, and run/checkpoint counts."""
    p = _resolve(path, root)
    exps = _load_runs(p, root)
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
    """Return per-run metrics, optionally filtered and reduced across runs.

    ``metrics``, when provided, matches flattened metric leaves by full dotted
    path (e.g. ``"metrics.accuracy"``) or trailing leaf name (e.g.
    ``"accuracy"``), not by ``.pt`` filename stems.
    """
    p = _resolve(path, root)
    exps = _load_runs(p, root)
    per_run = []
    wanted = set(metrics) if metrics is not None else None
    for e in exps:
        m = _to_jsonable(e.metrics or {})
        if wanted is not None:
            flat = _flatten(m)
            m = {
                k: v
                for k, v in flat.items()
                if k in wanted or k.rsplit(".", 1)[-1] in wanted
            }
        per_run.append(m)
    result = {"path": str(p), "num_runs": len(exps), "per_run": per_run}
    if reduce is not None:
        result["reduced"] = _reduce_metrics(per_run, reduce)
    return result


def _get_config(
    path: str | Path,
    job: int | None = None,
    root: str | Path | None = None,
) -> dict:
    """Return the resolved Hydra config for one run (``job``) or all runs."""
    p = _resolve(path, root)
    cfgs = [_to_jsonable(e.cfg) for e in _load_runs(p, root)]
    if job is not None:
        if not 0 <= job < len(cfgs):
            raise ValueError(
                f"job {job} out of range; experiment has {len(cfgs)} run(s) at {p}"
            )
        return {"path": str(p), "job": job, "config": cfgs[job]}
    if len(cfgs) == 1:
        return {"path": str(p), "config": cfgs[0]}
    return {"path": str(p), "configs": cfgs}


def _read_dataset(path: str | Path, root: str | Path | None = None) -> dict:
    """Summarize a saved netCDF dataset: dims, coords, data_vars, basic stats."""
    import xarray as xr

    p = _resolve(path, root)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    with xr.open_dataset(p) as ds:
        data_vars = {}
        for name, da in ds.data_vars.items():
            entry = {
                "dims": list(da.dims),
                "shape": list(da.shape),
                "dtype": str(da.dtype),
            }
            try:
                entry["mean"] = float(da.mean().item())
                entry["min"] = float(da.min().item())
                entry["max"] = float(da.max().item())
            except (TypeError, ValueError):
                pass
            data_vars[str(name)] = entry
        return {
            "path": str(p),
            "dims": {str(k): int(v) for k, v in ds.sizes.items()},
            "coords": {str(k): _to_jsonable(v.values) for k, v in ds.coords.items()},
            "data_vars": data_vars,
        }


def create_server(root: str | Path | None = None):
    """Build the FastMCP stdio server. Importing ``mcp`` requires Python >= 3.10."""
    from mcp.server.fastmcp import FastMCP

    rootp = Path(root).expanduser().resolve() if root is not None else None
    mcp = FastMCP("mushin")

    @mcp.tool()
    def list_experiments(root_dir: str | None = None) -> dict:
        """List experiment run directories (those containing a .hydra/ child)."""
        return _list_experiments(root_dir, rootp)

    @mcp.tool()
    def describe_experiment(path: str) -> dict:
        """Summarize an experiment: swept params, metric keys, run/ckpt counts."""
        return _describe_experiment(path, rootp)

    @mcp.tool()
    def get_metrics(
        path: str,
        metrics: list[str] | None = None,
        reduce: str | None = None,
    ) -> dict:
        """Per-run metrics, optionally reduced ('mean'/'std') across runs."""
        return _get_metrics(path, metrics, reduce, rootp)

    @mcp.tool()
    def get_config(path: str, job: int | None = None) -> dict:
        """Resolved Hydra config for one run (job index) or all runs."""
        return _get_config(path, job, rootp)

    @mcp.tool()
    def read_dataset(path: str) -> dict:
        """Summarize a saved netCDF dataset: dims, coords, data_vars, stats."""
        return _read_dataset(path, rootp)

    return mcp


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


def _list_experiments(
    base: str | Path | None = None, root: str | Path | None = None
) -> dict:
    """List run directories (those containing a ``.hydra/`` child) under ``base``.

    ``root``, when set, confines ``base``: a ``base`` outside ``root`` is rejected.
    """
    target = base if base is not None else (root if root is not None else Path.cwd())
    p = _resolve(target, root)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    runs = sorted(str(d.parent) for d in p.glob("**/.hydra"))
    return {"root": str(p), "runs": runs, "count": len(runs)}
