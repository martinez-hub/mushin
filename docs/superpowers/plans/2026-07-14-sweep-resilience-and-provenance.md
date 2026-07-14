# Sweep Resilience + Provenance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MultiRunMetricsWorkflow` survive job failures (opt-in), resume a partially-completed sweep by re-running only what's missing, refuse statistics on incomplete data, and auto-capture per-run provenance — shipped as 0.5.0.

**Architecture:** Persist each job's returned metrics to a `mushin_metrics.json` sidecar and track every requested grid cell in a `mushin_sweep_manifest.json`. Replace the fragile order-based `to_xarray` collection with **config-combo-keyed** assembly anchored to the requested grid (missing cells → NaN). Fail-soft branches on Hydra `JobReturn.status`; resume relaunches the full grid with a task wrapper that short-circuits already-completed cells; statistics check a manifest-derived completeness signal.

**Tech Stack:** Python 3.10+, PyTorch Lightning, hydra-zen (`launch`/`zen`), Hydra `JobReturn`, xarray, pytest. Tests: `uv run pytest ...`. Verify NumPy/dep-sensitive changes via `make test-lowest` (Docker) per project policy.

**Spec:** `docs/superpowers/specs/2026-07-14-sweep-resilience-and-provenance-design.md`. **Branch:** `sweep-resilience`.

---

## Key existing code (insertion points)

- `src/mushin/workflows.py`
  - `_task_calls(pre_task, task)` (~line 80): returns `wrapped(cfg)` = `pre_task(cfg); task(cfg)`. The task is `task_fn_wrapper(self.task)` (zen). **This is the wrapper seam** for sidecar/provenance/short-circuit.
  - `BaseWorkflow.run(...)` (~274): the `launch(...)` call (~402) passes `_task_calls(pre_task=..., task=task_fn_wrapper(self.task))`. New kwargs go here.
  - `MultiRunMetricsWorkflow.jobs_post_process` (~716): builds `multirun_working_dirs`, `working_dir`, `cfgs`, then `job_metrics = [j.return_value for j in self.jobs]` (**re-raises on FAILED**) → `_process_metrics` (order-based list append).
  - `to_xarray` (~900): builds coords from `multirun_task_overrides`, then `np.asarray(v).reshape(shape + ...)` per metric — **order-based, assumes complete grid**.
  - `multirun_task_overrides` (~184): parsed `{param: value|multirun([...])}`; the multirun params are the grid dims.
- `src/mushin/benchmark/_stats.py`: `compare_methods(ds, test, alpha)` (~115) — significance entry point.
- `src/mushin/benchmark/compare.py`: `compare(...)` — evaluates models → dataset → `compare_methods`.
- `src/mushin/study/_study.py`: `Study.run()` — training sweep → `compare`.

---

## Task 1: Combo keys + metrics sidecar (new `_sweep_io.py`)

**Files:** Create `src/mushin/_sweep_io.py`; Test `tests/test_sweep_io.py`.

New module holds the on-disk primitives (combo canonicalization, sidecar + manifest I/O) so `workflows.py` stays focused.

- [ ] **Step 1: Failing test**

```python
# tests/test_sweep_io.py
import json
from mushin._sweep_io import combo_key, write_metrics_sidecar, read_metrics_sidecar


def test_combo_key_is_canonical_and_order_stable():
    assert combo_key({"lr": 0.1, "seed": 2}) == combo_key({"seed": 2, "lr": 0.1})
    assert combo_key({"lr": 0.1, "seed": 2}) == "lr=0.1,seed=2"


def test_metrics_sidecar_roundtrip(tmp_path):
    write_metrics_sidecar(tmp_path, {"accuracy": 0.9, "loss": 0.1})
    assert read_metrics_sidecar(tmp_path) == {"accuracy": 0.9, "loss": 0.1}
    assert (tmp_path / "mushin_metrics.json").exists()
    assert read_metrics_sidecar(tmp_path / "nope") is None  # absent -> None


def test_metrics_sidecar_coerces_numpy_and_tensors(tmp_path):
    import numpy as np
    write_metrics_sidecar(tmp_path, {"a": np.float32(0.5), "b": np.array([1, 2])})
    got = read_metrics_sidecar(tmp_path)
    assert got == {"a": 0.5, "b": [1, 2]}
```

- [ ] **Step 2: Run — expect FAIL** (`uv run pytest tests/test_sweep_io.py -v`; module missing).

- [ ] **Step 3: Implement `src/mushin/_sweep_io.py`**

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""On-disk primitives for resilient/resumable sweeps: canonical combo keys and
the per-job metrics sidecar + sweep manifest (see the sweep-resilience design)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICS_FILE = "mushin_metrics.json"
MANIFEST_FILE = "mushin_sweep_manifest.json"


def combo_key(combo: dict[str, Any]) -> str:
    """Canonical, order-stable key for a swept-parameter combination."""
    return ",".join(f"{k}={_scalar(combo[k])}" for k in sorted(combo))


def _scalar(v: Any) -> Any:
    """Best-effort convert numpy/torch scalars & arrays to JSON-native values."""
    if hasattr(v, "tolist"):  # numpy scalar/array, torch tensor
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    return v


def write_metrics_sidecar(job_dir, metrics: dict[str, Any]) -> None:
    payload = {k: _scalar(v) for k, v in metrics.items()}
    _atomic_write_json(Path(job_dir) / METRICS_FILE, payload)


def read_metrics_sidecar(job_dir) -> dict | None:
    p = Path(job_dir) / METRICS_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)  # atomic on POSIX/Windows
```

- [ ] **Step 4: Run — expect PASS**; then `uv run ruff check src/mushin/_sweep_io.py tests/test_sweep_io.py` and `ruff format`.

- [ ] **Step 5: Commit** — `git commit -m "feat(sweep): combo keys + metrics sidecar I/O"`

---

## Task 2: Sweep manifest (extend `_sweep_io.py`)

**Files:** Modify `src/mushin/_sweep_io.py`; Test `tests/test_sweep_io.py`.

- [ ] **Step 1: Failing test** (append)

```python
from mushin._sweep_io import Manifest


def test_manifest_tracks_and_replaces_cells(tmp_path):
    m = Manifest.load_or_new(tmp_path, params=["lr", "seed"])
    m.mark({"lr": 0.1, "seed": 0}, dir="0", status="completed")
    m.mark({"lr": 0.1, "seed": 2}, dir="5", status="failed", error="OOM")
    m.save()

    m2 = Manifest.load_or_new(tmp_path, params=["lr", "seed"])
    assert m2.status({"lr": 0.1, "seed": 0}) == "completed"
    assert m2.status({"lr": 0.1, "seed": 2}) == "failed"
    assert m2.status({"lr": 1.0, "seed": 0}) == "pending"  # unseen -> pending
    # a re-run REPLACES in place (no duplicate entry)
    m2.mark({"lr": 0.1, "seed": 2}, dir="9", status="completed")
    assert m2.status({"lr": 0.1, "seed": 2}) == "completed"
    assert m2.dir({"lr": 0.1, "seed": 2}) == "9"
    assert m2.failed_cells() == []
    m2.mark({"lr": 1.0, "seed": 1}, dir="3", status="failed", error="boom")
    assert not m2.is_complete()
    assert {"lr=1.0,seed=1"} == {c["key"] for c in m2.failed_cells()}
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement `Manifest`** (add to `_sweep_io.py`)

```python
class Manifest:
    """Tracks each requested grid cell's status in <working_dir>/mushin_sweep_manifest.json."""

    SCHEMA = 1

    def __init__(self, root: Path, params: list[str], cells: dict | None = None):
        self.root = Path(root)
        self.params = list(params)
        self.cells: dict[str, dict] = cells or {}

    @classmethod
    def load_or_new(cls, root, params: list[str]) -> "Manifest":
        p = Path(root) / MANIFEST_FILE
        if p.exists():
            d = json.loads(p.read_text())
            return cls(root, d.get("params", params), d.get("cells", {}))
        return cls(root, params)

    def status(self, combo: dict) -> str:
        return self.cells.get(combo_key(combo), {}).get("status", "pending")

    def dir(self, combo: dict) -> str | None:
        return self.cells.get(combo_key(combo), {}).get("dir")

    def mark(self, combo: dict, *, dir: str, status: str, error: str | None = None) -> None:
        entry = {"dir": str(dir), "status": status}
        if error is not None:
            entry["error"] = error
        self.cells[combo_key(combo)] = entry  # replace in place

    def failed_cells(self) -> list[dict]:
        return [
            {"key": k, **v} for k, v in self.cells.items() if v.get("status") == "failed"
        ]

    def is_complete(self) -> bool:
        return all(v.get("status") == "completed" for v in self.cells.values())

    def save(self) -> None:
        _atomic_write_json(
            self.root / MANIFEST_FILE,
            {"schema": self.SCHEMA, "params": self.params, "cells": self.cells},
        )
```

- [ ] **Step 4: Run — PASS**; ruff. **Step 5: Commit** — `feat(sweep): sweep manifest`

---

## Task 3: Config-keyed assembly (refactor collection + `to_xarray`)

**Files:** Modify `src/mushin/workflows.py`; Test `tests/test_workflows.py`.

This is the core change: collect metrics **keyed by each job's combo** (derived from its `cfg`), and emit them in grid order with **NaN for missing cells**, so a short/failed/duplicate set of jobs no longer breaks the reshape.

- [ ] **Step 1: Failing test** — a workflow whose task returns nothing for one combo (simulate a hole) still yields a full-shaped dataset with NaN:

```python
def test_to_xarray_nan_fills_missing_combo(monkeypatch, tmp_path):
    import numpy as np
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class Holey(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            # emulate a missing result for (a=2,b=1): return None (no metrics)
            if a == 2 and b == 1:
                return None
            return dict(val=float(a * 10 + b))

    wf = Holey()
    wf.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=str(tmp_path / "s"))
    ds = wf.to_xarray()
    assert ds.sizes == {"a": 2, "b": 2}
    assert np.isnan(float(ds["val"].sel(a=2, b=1)))       # hole -> NaN
    assert float(ds["val"].sel(a=1, b=1)) == 11.0          # others intact
```

- [ ] **Step 2: Run — expect FAIL** (today the order-based reshape mis-sizes or errors on the `None`).

- [ ] **Step 3: Implement combo-keyed collection**

In `jobs_post_process`, after `self.cfgs = [j.cfg for j in self.jobs]`, replace the order-based metric extraction with a combo-keyed map. Add a helper that reads each job's swept-param values from its `cfg` (the grid params are `self.multirun_task_overrides` keys that are `multirun(...)`):

```python
    def _swept_param_names(self) -> list[str]:
        from mushin import multirun as _mr
        return [k for k, v in self.multirun_task_overrides.items() if isinstance(v, _mr)]

    def _combo_of_cfg(self, cfg) -> dict:
        names = self._swept_param_names()
        return {n: _unwrap_scalar(cfg[n]) for n in names}  # _unwrap_scalar already exists (workflows.py ~62)
```

Then build `self._metrics_by_combo: dict[str, dict]` from completed jobs (COMPLETED status → its returned dict or `mushin_metrics.json`). Keep `self.metrics` for backward-compat by deriving it from the grid order (Step 4). Store `self.working_dir` before use so sidecars/manifest resolve.

- [ ] **Step 4: Rewrite `to_xarray` data assembly to be grid-ordered with NaN-fill**

Replace the per-metric `np.asarray(v).reshape(shape + ...)` block so that, instead of assuming `self.metrics[k]` is a complete flat list, it iterates the grid combos in row-major order and pulls each combo's value (or `np.nan`):

```python
import itertools, numpy as np
# grid combos in row-major order (matches coords/shape ordering)
grid_names = list(orig_coords)
grid_values = [orig_coords[n] for n in grid_names]
combos = [dict(zip(grid_names, vals)) for vals in itertools.product(*grid_values)]

# union of metric keys across completed cells
keys = sorted({k for m in self._metrics_by_combo.values() for k in m})
data = {}
for k in keys:
    col = []
    for combo in combos:
        m = self._metrics_by_combo.get(combo_key(combo))
        col.append(m[k] if (m is not None and k in m) else np.nan)
    arr = _coerce_list_of_arraylikes(col)
    datum = np.asarray(arr, dtype=float).reshape(shape + np.asarray(arr[0]).shape)
    data[k] = (tuple(coords) + tuple(extra_dims_for(k)), datum)
```

Preserve the existing multi-dim-metric / `coord_from_metrics` / `include_working_subdirs_as_data_var` handling — fold the NaN-fill into that path rather than deleting it. **This is the delicate step; keep the existing helpers (`_coerce_list_of_arraylikes`, `_sanitize_coordinate_for_xarray`) and only change the *source* of each metric column from "job-ordered list" to "grid-combo lookup with NaN default".** Import `combo_key` from `mushin._sweep_io`.

- [ ] **Step 5: Run the new test + the FULL existing `to_xarray` suite** — `uv run pytest tests/test_workflows.py -q`. The clean-sweep tests must still pass **byte-identical** (regression: combo-keyed assembly == order-based for a complete grid). Fix until green.

- [ ] **Step 6: Commit** — `refactor(workflows): config-keyed grid assembly with NaN-fill`

---

## Task 4: Auto-write sidecar + provenance from the task wrapper

**Files:** Modify `src/mushin/workflows.py` (the `_task_calls`/task seam), plus a new `src/mushin/_provenance.py` (Task 7 fills provenance; here just the sidecar).

- [ ] **Step 1: Failing test** — after a normal run, each job dir has `mushin_metrics.json`:

```python
def test_run_writes_metrics_sidecar_per_job(tmp_path):
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow
    from mushin._sweep_io import read_metrics_sidecar

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(y=float(x) * 2)

    wf = W()
    wf.run(x=multirun([1, 2, 3]), working_dir=str(tmp_path / "s"))
    for d in wf.multirun_working_dirs:
        assert read_metrics_sidecar(d) is not None
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Wrap the task to write the sidecar.** Introduce a mushin-owned wrapper composed *inside* `_task_calls` (or where `task=task_fn_wrapper(self.task)` is built), operating at the `(cfg) -> result` level (runs in the per-job cwd = job dir):

```python
def _instrument_task(task, *, write_sidecar=True):
    from pathlib import Path
    from ._sweep_io import write_metrics_sidecar

    def wrapped(cfg):
        result = task(cfg)
        if write_sidecar and isinstance(result, dict):
            write_metrics_sidecar(Path.cwd(), result)  # cwd is Hydra's per-job dir
        return result
    return wrapped
```

Compose it around the zen-wrapped task: `task=_instrument_task(task_fn_wrapper(self.task))`.

- [ ] **Step 4: Run — PASS**; full `tests/test_workflows.py` still green (sidecar is additive). **Step 5: Commit** — `feat(workflows): auto-write per-job metrics sidecar`

---

## Task 5: Fail-soft (`on_error`) + manifest + `failures`/`is_complete`

**Files:** Modify `src/mushin/workflows.py`; Test `tests/test_workflows.py`.

- [ ] **Step 1: Failing tests**

```python
import pytest

def _grid_with_one_failure():
    from mushin.workflows import MultiRunMetricsWorkflow
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            if a == 2 and b == 1:
                raise RuntimeError("boom")
            return dict(val=float(a * 10 + b))
    return W

def test_on_error_raise_is_default(tmp_path):
    from mushin import multirun
    with pytest.raises(Exception):
        _grid_with_one_failure()().run(a=multirun([1, 2]), b=multirun([0, 1]),
                                       working_dir=str(tmp_path / "s"))

def test_on_error_nan_records_and_continues(tmp_path):
    import numpy as np
    from mushin import multirun
    wf = _grid_with_one_failure()()
    with pytest.warns(UserWarning, match="failed"):
        wf.run(a=multirun([1, 2]), b=multirun([0, 1]),
               working_dir=str(tmp_path / "s"), on_error="nan")
    assert wf.is_complete is False
    assert any("a=2" in f["combo"] for f in wf.failures)
    ds = wf.to_xarray()
    assert np.isnan(float(ds["val"].sel(a=2, b=1)))
    assert ds.attrs["mushin_failures"]  # non-empty
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.**
  - Add `on_error: str = "raise"` to `run()` (validate `in {"raise", "nan"}`). Thread it into `jobs_post_process`.
  - In `jobs_post_process`, replace `job_metrics = [j.return_value for j in self.jobs]` with a status-aware loop:

    ```python
    from hydra.core.utils import JobStatus
    self.failures = []
    manifest = Manifest.load_or_new(self.working_dir, self._swept_param_names())
    self._metrics_by_combo = {}
    for job, cfg, wdir in zip(self.jobs, self.cfgs, self.multirun_working_dirs):
        combo = self._combo_of_cfg(cfg)
        if job.status == JobStatus.COMPLETED:
            self._metrics_by_combo[combo_key(combo)] = job._return_value
            manifest.mark(combo, dir=wdir.name, status="completed")
        else:
            if on_error == "raise":
                raise job._return_value
            self.failures.append({"combo": combo_key(combo),
                                  "exception": repr(job._return_value),
                                  "working_dir": str(wdir)})
            manifest.mark(combo, dir=wdir.name, status="failed",
                          error=repr(job._return_value))
    manifest.save()
    self._manifest = manifest
    ```
  - Add properties `failures` (default `[]`) and `is_complete` (`self._manifest.is_complete()`).
  - In `to_xarray`, set `ds.attrs["mushin_failures"] = [f["combo"] for f in self.failures]`.
  - Emit a loud `warnings.warn(f"{len(self.failures)} run(s) failed: {combos}; ...", UserWarning)` when non-empty.

  Note: Hydra's basic launcher captures a failing job in `JobReturn(status=FAILED, _return_value=exc)` and continues; confirm during implementation that `launch(...)` returns FAILED jobs rather than raising (if it raises for `hydra.mode`/launcher reasons, set the appropriate `hydra.job.env_set`/`hydra/mode` so failures are captured — resolve in this task).

- [ ] **Step 4: Run — PASS**; full suite green. **Step 5: Commit** — `feat(workflows): on_error='nan' fail-soft with manifest + failures`

---

## Task 6: Statistics refuse on incomplete data (`IncompleteSweepError`)

**Files:** Modify `src/mushin/benchmark/_stats.py`, `src/mushin/benchmark/compare.py`, `src/mushin/study/_study.py`; new exception; Test `tests/test_benchmark/…` + `tests/test_study/…`.

- [ ] **Step 1: Failing test**

```python
def test_compare_refuses_incomplete_sweep():
    import numpy as np, xarray as xr, pytest
    from mushin.benchmark import compare_methods
    from mushin.benchmark._stats import IncompleteSweepError

    ds = xr.Dataset({"acc": (("method", "seed"), np.random.rand(2, 3))},
                    coords={"method": ["a", "b"], "seed": [0, 1, 2]})
    ds.attrs["mushin_failures"] = ["method=a,seed=1"]  # marks incompleteness
    with pytest.raises(IncompleteSweepError, match="failed"):
        compare_methods(ds)

    del ds.attrs["mushin_failures"]
    compare_methods(ds)  # complete -> runs fine
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.**
  - In `_stats.py`: define `class IncompleteSweepError(RuntimeError): ...`. At the top of `compare_methods(ds, ...)`, check `ds.attrs.get("mushin_failures")`; if truthy, `raise IncompleteSweepError(f"{len(...)} run(s) failed ({...}); fix the cause and re-run with resume=True to complete the sweep before comparing.")`. A dataset without the attr (a plain user dataset) is unaffected.
  - Export `IncompleteSweepError` from `mushin.benchmark` (`benchmark/__init__.py`).
  - `compare(...)` builds its dataset itself — ensure that when its inputs derive from a failed workflow the attr propagates (Study path below); a direct `compare(methods=..., data=...)` over models has no sweep failures, so it's unaffected.
  - `Study.run()`: after the training sweep, if the underlying workflow `is_complete` is False, raise `IncompleteSweepError` with the same actionable message (point at `resume=True`) instead of proceeding to `compare`.

- [ ] **Step 4: Run — PASS**; full suite green. **Step 5: Commit** — `feat(benchmark): refuse statistics on incomplete sweeps`

---

## Task 7: Resume (short-circuit) + provenance capture

**Files:** Modify `src/mushin/workflows.py`; new `src/mushin/_provenance.py`; Test `tests/test_workflows.py`, `tests/test_provenance.py`.

### 7a. Resume via short-circuit wrapper

- [ ] **Step 1: Failing test** — inject a failure, resume with a fix, assert completed cells **don't re-execute**:

```python
def test_resume_reruns_only_failed_cell(tmp_path):
    import numpy as np
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    CALLS = {"n": 0}
    class W(MultiRunMetricsWorkflow):
        FAIL = True
        @staticmethod
        def task(a, b):
            CALLS["n"] += 1
            if a == 2 and b == 1 and W.FAIL:
                raise RuntimeError("boom")
            return dict(val=float(a * 10 + b))

    wf = W(); wd = str(tmp_path / "s")
    wf.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=wd, on_error="nan")
    first_calls = CALLS["n"]                 # 4 attempts (1 failed)
    W.FAIL = False; CALLS["n"] = 0
    wf2 = W()
    wf2.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=wd, resume=True)
    assert CALLS["n"] == 1                    # only the previously-failed cell ran
    assert wf2.is_complete
    ds = wf2.to_xarray()
    assert float(ds["val"].sel(a=2, b=1)) == 21.0   # cell now filled in place
    assert ds.sizes == {"a": 2, "b": 2}             # same shape, no growth
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.**
  - Add `resume: bool = False` to `run()`; require `working_dir` when `resume=True`.
  - When `resume`, load the prior `Manifest` from `working_dir` and compose a short-circuit into the task wrapper (extend Task 4's `_instrument_task`):

    ```python
    def _instrument_task(task, *, manifest=None, write_sidecar=True):
        from pathlib import Path
        from ._sweep_io import write_metrics_sidecar, read_metrics_sidecar, combo_key
        def wrapped(cfg):
            if manifest is not None:
                combo = _combo_from_cfg(cfg)      # same swept-name extraction
                if manifest.status(combo) == "completed":
                    cached = read_metrics_sidecar(Path(manifest.root) / manifest.dir(combo))
                    if cached is not None:
                        return cached             # short-circuit: no training
            result = task(cfg)
            if write_sidecar and isinstance(result, dict):
                write_metrics_sidecar(Path.cwd(), result)
            return result
        return wrapped
    ```

  - Note the combo→dir lookup uses the manifest's recorded `dir` (the *prior* sweep's cell), so `read_metrics_sidecar` reads the old completed sidecar; the current relaunch's own dir is only written for cells that actually run.
  - After the relaunch, `jobs_post_process` rebuilds `_metrics_by_combo` and updates the manifest as usual (completed cells whose task short-circuited still report COMPLETED with their returned cached dict → marked completed).

- [ ] **Step 4: Run — PASS**; full suite green. **Step 5: Commit** — `feat(workflows): resume via full-grid relaunch + short-circuit`

### 7b. Provenance

- [ ] **Step 6: Failing test** (`tests/test_provenance.py`)

```python
def test_provenance_written_per_job(tmp_path):
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow
    import json

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(y=float(x))

    wf = W()
    wf.run(x=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    d = wf.multirun_working_dirs[0]
    prov = json.loads((d / "mushin_provenance.json").read_text())
    assert "python" in prov and "packages" in prov and "git" in prov
    assert prov["packages"]["mushin-py"]  # version string
    # git may be None outside a repo — must not crash; key present
    assert "sha" in prov["git"]
```

- [ ] **Step 7: Implement `src/mushin/_provenance.py`**

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Per-run provenance capture (git, versions, config); graceful without git."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_PKGS = ("mushin-py", "torch", "numpy", "pytorch-lightning", "hydra-core", "hydra-zen")


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True,
                                  timeout=5).stdout.strip() or None
        except Exception:
            return None
    sha = run("git", "rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None, "branch": None}
    dirty = bool(run("git", "status", "--porcelain"))
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    return {"sha": sha, "dirty": dirty, "branch": branch}


def _versions() -> dict:
    out = {}
    for p in _PKGS:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def capture(config: Any = None) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git(),
        "packages": _versions(),
        "config": _to_plain(config) if config is not None else None,
    }


def write_provenance(job_dir, config: Any = None) -> None:
    (Path(job_dir) / "mushin_provenance.json").write_text(json.dumps(capture(config), indent=2))


def _to_plain(cfg):
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return None
```

  Wire `write_provenance(Path.cwd(), cfg)` into `_instrument_task` (always-on). Add `capture_env: bool = False` to `run()`; when set, write `mushin_env.txt` once in `working_dir` via `uv export`/`uv pip freeze` (subprocess, fall back to `importlib.metadata` dump). Aggregate into `ds.attrs["provenance"]` in `to_xarray` and expose `wf.provenance`.

- [ ] **Step 8: Run — PASS** (incl. a non-git tmp dir → `git.sha is None`, no crash). **Step 9: Commit** — `feat: per-run provenance capture (+ capture_env)`

---

## Task 8: Integration test, docs, changelog

**Files:** `tests/test_sweep_resilience_integration.py`, `docs/guides/resilience.md`, `mkdocs.yml`, `changes/+sweep-resilience.added.md`.

- [ ] **Step 1: End-to-end integration test** — a real multirun with an injected failure → `on_error="nan"` yields NaN + `IncompleteSweepError` from `compare`/`Study`; `resume=True` (fixed) completes; the same dataset then compares cleanly; provenance + manifest files exist. Run `uv run pytest tests/test_sweep_resilience_integration.py -v`.

- [ ] **Step 2: Guide** — `docs/guides/resilience.md` (fail-soft, resume, the "fix→resume→compare" loop, provenance); add to `mkdocs.yml` nav; `uv run --group docs mkdocs build --strict`.

- [ ] **Step 3: Changelog fragment** — `changes/+sweep-resilience.added.md`:

```markdown
Sweep resilience and provenance: `run(on_error="nan")` records failed grid cells as
NaN (default stays `"raise"`); `run(working_dir=..., resume=True)` re-runs only the
failed/missing cells and fills them in place; `compare`/`Study` refuse statistics on
an incomplete sweep (`IncompleteSweepError`) until you resume; every run writes
per-job provenance (`mushin_provenance.json`: git SHA, versions, config), with an
opt-in `capture_env=True` full dependency snapshot.
```

- [ ] **Step 4: Full verification** — `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, and **`make test-lowest`** (Docker, NumPy-2 floors) since this touches the core collection path. **Step 5: Commit + open PR** (base `main`, target 0.5.0). Then read Codex/CI before merging.

---

## Self-Review

**Spec coverage:** shared sidecar+manifest → Tasks 1,2,4; config-keyed assembly (§3) → Task 3; fail-soft `on_error` (§1a) → Task 5; stats-refuse (§1b) → Task 6; resume short-circuit (§1c, Option C) → Task 7a; provenance (§2) → Task 7b; docs/changelog/verify → Task 8. ✅

**Placeholder scan:** the `to_xarray` rewrite (Task 3 Step 4) is the one step given as a code *skeleton + precise integration rules* rather than a full drop-in, because it must fold into the existing multi-dim/`coord_from_metrics` reshape logic — flagged explicitly with the regression gate (clean sweep must stay byte-identical). No `TBD`/`TODO`.

**Type/name consistency:** `combo_key`, `Manifest` (`load_or_new`/`status`/`dir`/`mark`/`failed_cells`/`is_complete`/`save`), `write_metrics_sidecar`/`read_metrics_sidecar`, `_instrument_task`, `_combo_of_cfg`/`_swept_param_names`, `IncompleteSweepError`, `write_provenance`/`capture`, and the `on_error`/`resume`/`capture_env` kwargs are used consistently across tasks. `_metrics_by_combo` is the single collection structure feeding `to_xarray`.

**Ordering risk:** Task 3 (assembly) lands before Tasks 4–7 so every later feature writes into the combo-keyed structure; the clean-sweep regression test in Task 3 Step 5 protects the existing behavior before any resilience logic is added.
