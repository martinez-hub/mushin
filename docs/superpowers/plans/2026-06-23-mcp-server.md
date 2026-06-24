# mushin-mcp Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `mushin-mcp`, a read-only MCP server that lets Claude Code (or any MCP client) load and analyze completed mushin experiments.

**Architecture:** A new `src/mushin/mcp/` subpackage. Pure, transport-agnostic logic functions (`_list_experiments`, `_describe_experiment`, `_get_metrics`, `_get_config`, `_read_dataset`) built on `mushin._utils.load_experiment` and `xarray`. A thin `create_server()` wraps each as a FastMCP stdio tool. The logic functions have no `mcp` dependency, so they are unit-tested on every supported Python; only `create_server` imports `mcp`.

**Tech Stack:** Python, the official `mcp` SDK (FastMCP, stdio), `load_experiment`, `omegaconf`, `xarray`, `torch`/`numpy` (already mushin deps).

---

## Important constraints (read before starting)

- The `mcp` SDK requires Python **>= 3.10**; mushin supports **>= 3.9**. The `mcp`
  dependency therefore carries a `python_version >= '3.10'` marker, and every test
  that touches `create_server`/`__main__` must `pytest.importorskip("mcp")`.
- `Experiment.metrics` is a dict keyed by the `.pt` filename stem; its values are
  whatever was saved (typically nested dicts of torch tensors). All tool output
  must pass through `_to_jsonable`.
- `Experiment.cfg` is an OmegaConf container; convert with `_to_jsonable` before
  flattening/diffing.
- `load_experiment(path)` returns a single `Experiment` for a one-run dir and a
  `list[Experiment]` for a multirun dir. Logic functions must normalize to a list.

## File structure

- Create `src/mushin/mcp/__init__.py` — package marker, re-exports `create_server`.
- Create `src/mushin/mcp/server.py` — helpers + the five logic functions + `create_server`.
- Create `src/mushin/mcp/__main__.py` — argparse entry point (`--root`), runs stdio server.
- Create `tests/test_mcp/test_server.py` — unit tests for every logic function + a guarded smoke test.
- Modify `pyproject.toml` — add `mcp` optional extra, `[project.scripts]` entry, dev-group `mcp`.
- Modify `README.md` — add an "Analyze experiments from Claude Code (MCP)" section.
- Create `docs/mcp.md` — install + `claude mcp add` + example prompts.
- Modify `CHANGELOG.md` — `[Unreleased]` bullet.

---

### Task 1: Package scaffold + packaging metadata

**Files:**
- Create: `src/mushin/mcp/__init__.py`
- Modify: `pyproject.toml` (`[project.optional-dependencies]`, new `[project.scripts]`, `[dependency-groups].dev`)

- [ ] **Step 1: Create the package marker**

Create `src/mushin/mcp/__init__.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Read-only MCP server exposing mushin experiment analysis."""

from .server import create_server

__all__ = ["create_server"]
```

- [ ] **Step 2: Add the optional extra and console script**

In `pyproject.toml`, change the `[project.optional-dependencies]` block from:

```toml
[project.optional-dependencies]
viz = ["matplotlib >= 3.3"]
netcdf = ["netCDF4 >= 1.5.8"]
```

to:

```toml
[project.optional-dependencies]
viz = ["matplotlib >= 3.3"]
netcdf = ["netCDF4 >= 1.5.8"]
# The MCP SDK requires Python >= 3.10; mushin still supports 3.9, so gate it.
mcp = ["mcp >= 1.2 ; python_version >= '3.10'"]

[project.scripts]
mushin-mcp = "mushin.mcp.__main__:main"
```

- [ ] **Step 3: Add mcp to the dev dependency group**

In `pyproject.toml`, inside `[dependency-groups]` `dev = [ ... ]`, add this line alongside the other dev deps:

```toml
    "mcp >= 1.2 ; python_version >= '3.10'",
```

- [ ] **Step 4: Verify the package imports as a namespace (logic not yet present)**

Run: `python -c "import mushin; print('ok')"`
Expected: prints `ok` (we have not imported `mushin.mcp` yet — its `__init__` imports `server`, written in Task 7; do not import `mushin.mcp` until then).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/__init__.py pyproject.toml
git commit -m "build: add mushin[mcp] extra and mushin-mcp console script"
```

---

### Task 2: `_to_jsonable` helper

**Files:**
- Create: `src/mushin/mcp/server.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcp/test_server.py`:

```python
# SPDX-License-Identifier: MIT
import numpy as np
import torch
from omegaconf import OmegaConf

from mushin.mcp.server import _to_jsonable


def test_to_jsonable_scalar_tensor():
    assert _to_jsonable(torch.tensor(0.5)) == 0.5


def test_to_jsonable_tensor_array():
    assert _to_jsonable(torch.tensor([1.0, 2.0])) == [1.0, 2.0]


def test_to_jsonable_numpy_and_nested():
    out = _to_jsonable({"a": np.float32(1.5), "b": [np.int64(2)]})
    assert out == {"a": 1.5, "b": [2]}


def test_to_jsonable_omegaconf():
    cfg = OmegaConf.create({"lr": 0.1, "nested": {"seed": 0}})
    assert _to_jsonable(cfg) == {"lr": 0.1, "nested": {"seed": 0}}


def test_to_jsonable_non_finite_float_becomes_string():
    assert _to_jsonable(float("inf")) == "inf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mushin.mcp.server'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/mushin/mcp/server.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add _to_jsonable conversion helper"
```

---

### Task 3: `_resolve` root-containment helper

**Files:**
- Modify: `src/mushin/mcp/server.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp/test_server.py`:

```python
import pytest

from mushin.mcp.server import RootError, _resolve


def test_resolve_no_root_returns_absolute(tmp_path):
    target = tmp_path / "exp"
    target.mkdir()
    assert _resolve(target, None) == target.resolve()


def test_resolve_inside_root_ok(tmp_path):
    root = tmp_path
    target = tmp_path / "exp"
    target.mkdir()
    assert _resolve(target, root) == target.resolve()


def test_resolve_outside_root_raises(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(RootError):
        _resolve(outside, root)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -k resolve -q`
Expected: FAIL — `ImportError: cannot import name 'RootError'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/mcp/server.py`, add to the imports `from pathlib import Path` and `from typing import Optional, Union` (extend the existing `from typing import Any` line to `from typing import Any, Optional, Union`). Then add below `_to_jsonable`:

```python
class RootError(ValueError):
    """Raised when a requested path escapes the configured --root."""


def _resolve(path: Union[str, Path], root: Optional[Union[str, Path]]) -> Path:
    """Resolve ``path`` to an absolute Path, enforcing ``root`` containment."""
    p = Path(path).expanduser().resolve()
    if root is not None:
        root = Path(root).expanduser().resolve()
        if p != root and root not in p.parents:
            raise RootError(f"{p} is outside the configured root {root}")
    return p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -k resolve -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add _resolve with --root containment"
```

---

### Task 4: Experiment fixture + `_list_experiments`

**Files:**
- Modify: `src/mushin/mcp/server.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing test (with a reusable fixture builder)**

Append to `tests/test_mcp/test_server.py`:

```python
from pathlib import Path

from omegaconf import OmegaConf


def _make_experiment(base: Path, lrs=(0.1, 0.2)) -> Path:
    """Build a minimal 2-run Hydra multirun layout under ``base``."""
    for i, lr in enumerate(lrs):
        run = base / str(i)
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create({"lr": lr, "seed": 0}),
            run / ".hydra" / "config.yaml",
        )
        torch.save({"accuracy": torch.tensor(0.8 + 0.1 * i)}, run / "metrics.pt")
    return base


def test_list_experiments_finds_runs(tmp_path):
    from mushin.mcp.server import _list_experiments

    base = _make_experiment(tmp_path / "exp")
    out = _list_experiments(base)
    assert out["count"] == 2
    assert sorted(Path(r).name for r in out["runs"]) == ["0", "1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -k list_experiments -q`
Expected: FAIL — `ImportError: cannot import name '_list_experiments'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/mcp/server.py`, add `from mushin._utils import Experiment, load_experiment` to the imports, then add:

```python
def _list_experiments(root: Optional[Union[str, Path]] = None) -> dict:
    """List run directories (those containing a ``.hydra/`` child) under ``root``."""
    base = _resolve(root if root is not None else Path.cwd(), root)
    if not base.exists():
        raise FileNotFoundError(f"{base} not found")
    runs = sorted(str(p.parent) for p in base.glob("**/.hydra"))
    return {"root": str(base), "runs": runs, "count": len(runs)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -k list_experiments -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add _list_experiments tool logic"
```

---

### Task 5: `_describe_experiment` (+ `_flatten`)

**Files:**
- Modify: `src/mushin/mcp/server.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp/test_server.py`:

```python
def test_describe_experiment_reports_sweep(tmp_path):
    from mushin.mcp.server import _describe_experiment

    base = _make_experiment(tmp_path / "exp")
    out = _describe_experiment(base)
    assert out["num_runs"] == 2
    assert "metrics" in out["metric_keys"]
    assert out["swept_params"]["lr"] == [0.1, 0.2]
    assert "seed" not in out["swept_params"]  # constant across runs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -k describe -q`
Expected: FAIL — `ImportError: cannot import name '_describe_experiment'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/mcp/server.py`, add:

```python
def _flatten(value: Any, prefix: str = "") -> dict:
    """Flatten a nested (already JSON-able) dict into dotted keys."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = value
    return out


def _as_list(exps) -> list:
    return [exps] if isinstance(exps, Experiment) else list(exps)


def _describe_experiment(
    path: Union[str, Path], root: Optional[Union[str, Path]] = None
) -> dict:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -k describe -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add _describe_experiment tool logic"
```

---

### Task 6: `_get_metrics` (+ reduce)

**Files:**
- Modify: `src/mushin/mcp/server.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp/test_server.py`:

```python
def test_get_metrics_per_run_and_reduce(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    out = _get_metrics(base, reduce="mean")
    assert out["num_runs"] == 2
    # metrics saved as metrics.pt -> {"metrics": {"accuracy": ...}}
    assert out["per_run"][0]["metrics"]["accuracy"] == pytest.approx(0.8, abs=1e-5)
    assert out["reduced"]["metrics.accuracy"] == pytest.approx(0.85, abs=1e-5)


def test_get_metrics_filter(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    out = _get_metrics(base, metrics=["does-not-exist"])
    assert out["per_run"][0] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -k get_metrics -q`
Expected: FAIL — `ImportError: cannot import name '_get_metrics'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/mcp/server.py`, add `import statistics` to the imports, then add:

```python
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
    path: Union[str, Path],
    metrics: Optional[list[str]] = None,
    reduce: Optional[str] = None,
    root: Optional[Union[str, Path]] = None,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -k get_metrics -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add _get_metrics with cross-run reduce"
```

---

### Task 7: `_get_config` and `_read_dataset`

**Files:**
- Modify: `src/mushin/mcp/server.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp/test_server.py`:

```python
def test_get_config_returns_configs(tmp_path):
    from mushin.mcp.server import _get_config

    base = _make_experiment(tmp_path / "exp")
    out = _get_config(base)
    assert [c["lr"] for c in out["configs"]] == [0.1, 0.2]


def test_get_config_single_job(tmp_path):
    from mushin.mcp.server import _get_config

    base = _make_experiment(tmp_path / "exp")
    out = _get_config(base, job=1)
    assert out["config"]["lr"] == 0.2


def test_read_dataset_summarizes(tmp_path):
    import xarray as xr

    from mushin.mcp.server import _read_dataset

    ds = xr.Dataset(
        {"accuracy": ("lr", [0.8, 0.9])},
        coords={"lr": [0.1, 0.2]},
    )
    nc = tmp_path / "result.nc"
    ds.to_netcdf(nc, engine="scipy")

    out = _read_dataset(nc)
    assert out["dims"] == {"lr": 2}
    assert out["coords"]["lr"] == [0.1, 0.2]
    assert out["data_vars"]["accuracy"]["max"] == pytest.approx(0.9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -k "get_config or read_dataset" -q`
Expected: FAIL — `ImportError: cannot import name '_get_config'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/mcp/server.py`, add:

```python
def _get_config(
    path: Union[str, Path],
    job: Optional[int] = None,
    root: Optional[Union[str, Path]] = None,
) -> dict:
    """Return the resolved Hydra config for one run (``job``) or all runs."""
    p = _resolve(path, root)
    cfgs = [_to_jsonable(e.cfg) for e in _as_list(load_experiment(p))]
    if job is not None:
        return {"path": str(p), "job": job, "config": cfgs[job]}
    if len(cfgs) == 1:
        return {"path": str(p), "config": cfgs[0]}
    return {"path": str(p), "configs": cfgs}


def _read_dataset(
    path: Union[str, Path], root: Optional[Union[str, Path]] = None
) -> dict:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -k "get_config or read_dataset" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add _get_config and _read_dataset tool logic"
```

---

### Task 8: `create_server` (FastMCP wiring) + root-containment test

**Files:**
- Modify: `src/mushin/mcp/server.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp/test_server.py`:

```python
def test_describe_outside_root_raises(tmp_path):
    from mushin.mcp.server import RootError, _describe_experiment

    base = _make_experiment(tmp_path / "allowed" / "exp")
    (tmp_path / "allowed").mkdir(exist_ok=True)
    with pytest.raises(RootError):
        _describe_experiment(tmp_path / "elsewhere", root=tmp_path / "allowed")


def test_create_server_registers_tools():
    pytest.importorskip("mcp")  # mcp requires Python >= 3.10
    from mushin.mcp.server import create_server

    server = create_server(root=None)
    assert server.name == "mushin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -k "create_server or outside_root" -q`
Expected: FAIL — `ImportError: cannot import name 'create_server'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/mcp/server.py`, add at the end:

```python
def create_server(root: Optional[Union[str, Path]] = None):
    """Build the FastMCP stdio server. Importing ``mcp`` requires Python >= 3.10."""
    from mcp.server.fastmcp import FastMCP

    rootp = Path(root).expanduser().resolve() if root is not None else None
    mcp = FastMCP("mushin")

    @mcp.tool()
    def list_experiments(root_dir: Optional[str] = None) -> dict:
        """List experiment run directories (those containing a .hydra/ child)."""
        return _list_experiments(root_dir if root_dir else rootp)

    @mcp.tool()
    def describe_experiment(path: str) -> dict:
        """Summarize an experiment: swept params, metric keys, run/ckpt counts."""
        return _describe_experiment(path, rootp)

    @mcp.tool()
    def get_metrics(
        path: str,
        metrics: Optional[list[str]] = None,
        reduce: Optional[str] = None,
    ) -> dict:
        """Per-run metrics, optionally reduced ('mean'/'std') across runs."""
        return _get_metrics(path, metrics, reduce, rootp)

    @mcp.tool()
    def get_config(path: str, job: Optional[int] = None) -> dict:
        """Resolved Hydra config for one run (job index) or all runs."""
        return _get_config(path, job, rootp)

    @mcp.tool()
    def read_dataset(path: str) -> dict:
        """Summarize a saved netCDF dataset: dims, coords, data_vars, stats."""
        return _read_dataset(path, rootp)

    return mcp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -k "create_server or outside_root" -q`
Expected: On Python 3.10+: PASS (2 passed). On Python 3.9: `test_create_server_registers_tools` is SKIPPED, `test_describe_outside_root_raises` PASSES.

- [ ] **Step 5: Run the whole MCP test module**

Run: `python -m pytest tests/test_mcp/ -q`
Expected: All pass (one skip on Python 3.9).

- [ ] **Step 6: Commit**

```bash
git add src/mushin/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add create_server FastMCP wiring"
```

---

### Task 9: `__main__` entry point

**Files:**
- Create: `src/mushin/mcp/__main__.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp/test_server.py`:

```python
def test_main_builds_server_without_running(monkeypatch):
    pytest.importorskip("mcp")
    import mushin.mcp.__main__ as cli

    captured = {}

    class _FakeServer:
        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(cli, "create_server", lambda root: _FakeServer())
    cli.main(["--root", "."])
    assert captured["ran"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -k main_builds -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mushin.mcp.__main__'` (collected as error), or import error inside the test.

- [ ] **Step 3: Write minimal implementation**

Create `src/mushin/mcp/__main__.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Console entry point: ``mushin-mcp`` runs the stdio MCP server."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from mushin.mcp.server import create_server


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mushin-mcp",
        description="Read-only MCP server for analyzing mushin experiments.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Restrict experiment access to this directory (recommended).",
    )
    args = parser.parse_args(argv)
    server = create_server(root=args.root)
    server.run()  # FastMCP defaults to stdio transport


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -k main_builds -q`
Expected: PASS on Python 3.10+; SKIPPED on Python 3.9.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/__main__.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add mushin-mcp console entry point"
```

---

### Task 10: Docs + changelog

**Files:**
- Create: `docs/mcp.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the docs page**

Create `docs/mcp.md`:

```markdown
# Analyze experiments from Claude Code (MCP)

`mushin-mcp` is a read-only [Model Context Protocol](https://modelcontextprotocol.io)
server. Point it at a directory of completed mushin runs and an MCP client
(Claude Code, Claude Desktop, …) can load, summarize, and compare them
conversationally. It never trains, launches sweeps, or loads model weights.

> Requires Python >= 3.10 (the `mcp` SDK does not support 3.9).

## Install

```bash
pip install "mushin-py[mcp]"
```

## Connect Claude Code

```bash
claude mcp add mushin -- mushin-mcp --root ./outputs
```

`--root` restricts the server to one directory; omit it to allow the current
working directory.

## Tools

| Tool | Returns |
|---|---|
| `list_experiments` | Run directories (those containing `.hydra/`) under the root. |
| `describe_experiment` | Swept params, metric keys, run and checkpoint counts. |
| `get_metrics` | Per-run metrics; optional `mean`/`std` reduction across runs. |
| `get_config` | The resolved Hydra config for a run (or all runs). |
| `read_dataset` | Dims, coords, data variables, and basic stats of a saved netCDF. |

## Example prompts

- "List the experiments under ./outputs and tell me what each one swept."
- "Summarize the accuracy metric for the lr sweep, averaged across seeds."
- "Open results.nc and tell me which method scored highest."
```

- [ ] **Step 2: Add a README section**

In `README.md`, immediately after the `## What it provides` list (before the `## Install` heading), insert:

```markdown
## Analyze experiments from Claude Code (MCP)

`mushin` ships an optional read-only [MCP](https://modelcontextprotocol.io)
server so Claude Code (or any MCP client) can load and analyze your completed
runs — list experiments, summarize swept parameters and metrics, and inspect
saved datasets — without launching anything.

```bash
pip install "mushin-py[mcp]"          # requires Python >= 3.10
claude mcp add mushin -- mushin-mcp --root ./outputs
```

See [docs/mcp.md](docs/mcp.md) for the full tool list and example prompts.
```

- [ ] **Step 3: Add a changelog entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add:

```markdown
## [Unreleased]

### Added
- `mushin-mcp` — an optional, read-only [MCP](https://modelcontextprotocol.io)
  server (`pip install "mushin-py[mcp]"`, Python >= 3.10) that lets Claude Code
  and other MCP clients list experiments, summarize swept params and metrics,
  read configs, and inspect saved datasets. No training or sweep launching.
```

- [ ] **Step 4: Verify docs render references resolve**

Run: `python -m pytest tests/test_mcp/ -q && python -c "import mushin.mcp; print('import ok')"`
Expected: tests pass (one skip on 3.9); prints `import ok`.

- [ ] **Step 5: Commit**

```bash
git add docs/mcp.md README.md CHANGELOG.md
git commit -m "docs(mcp): document mushin-mcp server and usage"
```

---

### Task 11: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run lint/format if configured**

Run: `ruff check src/mushin/mcp tests/test_mcp && ruff format --check src/mushin/mcp tests/test_mcp`
Expected: no errors (fix any reported, re-run, then commit with `style: ruff` if changes were needed).

- [ ] **Step 2: Run the complete test suite**

Run: `python -m pytest tests/test_mcp/ -q`
Expected: all MCP tests pass (one skip on Python 3.9).

- [ ] **Step 3: Sanity-check the console script resolves (Python 3.10+ only)**

Run: `python -c "import sys; sys.exit(0 if sys.version_info < (3,10) else 0)"; mushin-mcp --help`
Expected (3.10+): argparse help text for `mushin-mcp`. On 3.9 the extra is not installed, so skip this check.

- [ ] **Step 4: Final commit if any fixups were made**

```bash
git add -A && git commit -m "test(mcp): verification fixups" || echo "nothing to commit"
```

---

## Self-review notes

- **Spec coverage:** all five tools (`list_experiments`, `describe_experiment`,
  `get_metrics`, `get_config`, `read_dataset`) → Tasks 4–8; distribution as the
  `[mcp]` extra + `mushin-mcp` script → Task 1/9; stdio transport + `--root` →
  Tasks 8–9; read-only/no-torch-model-loading → enforced by building only on
  `load_experiment`/`xarray`; error handling → `RootError` + `FileNotFoundError`
  (Tasks 3, 4, 7); testing on a fixture dir → Task 4 builder; docs → Task 10.
  `compare_checkpoints` is intentionally absent per the spec.
- **Python 3.9 vs mcp:** every `mcp`-importing test uses `pytest.importorskip`;
  the extra and dev-dep carry `python_version >= '3.10'` markers.
- **Type/name consistency:** `_as_list`, `_flatten`, `_to_jsonable`, `_resolve`,
  `RootError` are defined once and reused; `create_server` returns a FastMCP
  named `"mushin"`, asserted in Task 8 and used by `__main__` in Task 9.
