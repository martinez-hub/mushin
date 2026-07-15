# Resume Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MultiRunMetricsWorkflow` sweeps survive a hard process kill without recomputing finished cells, and let a re-executed cell resume its own training from a checkpoint.

**Architecture:** A durable per-cell `mushin_cell_status.json` written from inside each job (`running`→`completed`/`failed`) makes completion survive SIGKILL; resume reconstructs completion by scanning those sidecars (keyed by the recorded combo, so it is independent of dir naming). A `ResumeContext` is injected into a task that opts in via a `mushin_resume` parameter — kept invisible to hydra-zen by stripping it from the signature `zen` sees and injecting from a contextvar. Directories stay Hydra's positional numeric dirs; a **combo-match guard** ensures a re-executed cell only reuses a checkpoint from a prior attempt of the *same* combo.

**Tech Stack:** Python, hydra/hydra-zen (unchanged — built on top of), mushin. New module `src/mushin/_resume.py`; edits to `src/mushin/workflows.py` and `src/mushin/_sweep_io.py`.

**Spec:** `docs/superpowers/specs/2026-07-14-resume-hardening-design.md`

**Branch:** `resume-hardening` (already created off `main`; includes the merged chdir fix).

---

## Key facts established before writing (do not re-derive)

- `run()` composes the task call:
  `task_call = _task_calls(pre_task=pre_task_fn_wrapper(self.pre_task), task=_instrument_task(task_fn_wrapper(self.task)))`; then `if on_error=="nan": task_call = _fail_soft(task_call)`; then `if resume: task_call = _resume_short_circuit(task_call, prior_manifest)`. Call order outer→inner: resume → fail_soft → task_calls(pre_task, instrument(zen(task))).
- `task_fn_wrapper` defaults to hydra-zen `zen`, which reads the wrapped fn's signature (honoring `__signature__`) to decide which cfg fields to pass. A task param with no cfg entry (`mushin_resume`) makes `zen` raise — so it MUST be stripped before `zen` sees the task.
- `_instrument_task(task)` returns `wrapped(cfg)`; runs inside the job's chdir'd cwd; writes `mushin_provenance.json` then (on a dict result) `mushin_metrics.json` via `write_metrics_sidecar(Path.cwd(), result)`.
- `jobs_post_process` (workflows.py ~955-1021) marks a fresh `Manifest` per combo: `manifest.mark(swept_combo, dir=wdir.name, status=...)`, then `manifest.save()`. `Manifest` (`_sweep_io.py`) keys cells by `combo_key(combo)`, stores `{"dir": basename, "status": ...}`; `Manifest.load_or_new` reads the manifest file. `_resume_short_circuit` reads `manifest.status(combo)`/`manifest.dir(combo)` → `root/dir` → `read_metrics_sidecar`.
- `self._combo_of_cfg(cfg)` → the sanitized swept combo dict (via `_sanitize_coordinate_for_xarray`); this is the SAME formatting `jobs_post_process` and `_resume_short_circuit` use, so keys align. Reuse it everywhere; never introduce a second combo formatting.
- Dirs stay numeric (`working_dir/0,1,2…`, Hydra default `${hydra.job.num}`). No subdir override is added. Correctness across grid changes comes from the combo-match guard, not dir naming.
- `_atomic_write_json(path, payload)` exists in `_sweep_io.py` (temp + atomic rename).

---

## Task 1: `ResumeContext` + status/checkpoint IO (`_resume.py`)

Pure, dependency-free primitives — no Hydra, fully unit-testable.

**Files:**
- Create: `src/mushin/_resume.py`
- Test: `tests/test_resume.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resume.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
from mushin._resume import (
    ResumeContext,
    STATUS_FILE,
    discover_last_ckpt,
    read_cell_status,
    write_cell_status,
)


def test_write_then_read_cell_status(tmp_path):
    write_cell_status(tmp_path, status="running", combo={"seed": 0}, attempt=1)
    got = read_cell_status(tmp_path)
    assert got["status"] == "running"
    assert got["attempt"] == 1
    assert got["combo"] == {"seed": 0}
    assert (tmp_path / STATUS_FILE).exists()


def test_read_cell_status_missing_or_corrupt_returns_none(tmp_path):
    assert read_cell_status(tmp_path) is None
    (tmp_path / STATUS_FILE).write_text("{not json")
    assert read_cell_status(tmp_path) is None


def test_discover_last_ckpt_prefers_last_then_newest(tmp_path):
    assert discover_last_ckpt(tmp_path) is None
    (tmp_path / "epoch=0.ckpt").write_text("a")
    (tmp_path / "epoch=1.ckpt").write_text("b")
    newest = discover_last_ckpt(tmp_path)
    assert newest is not None and newest.suffix == ".ckpt"
    (tmp_path / "last.ckpt").write_text("c")
    assert discover_last_ckpt(tmp_path).name == "last.ckpt"


def test_resume_context_is_frozen():
    rc = ResumeContext(dir=None, is_resume=False, last_ckpt=None, attempt=1)
    try:
        rc.is_resume = True  # type: ignore[misc]
    except Exception as e:
        assert "cannot assign" in str(e).lower() or "frozen" in str(e).lower()
    else:  # pragma: no cover
        raise AssertionError("ResumeContext must be frozen")
```

- [ ] **Step 2: Run — verify it fails**

Run: `uv run pytest tests/test_resume.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'mushin._resume'`).

- [ ] **Step 3: Implement `_resume.py`**

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Primitives for kill-durable, resumable sweeps: the per-cell status sidecar,
best-effort checkpoint discovery, and the ResumeContext handed to a task that
opts in via a ``mushin_resume`` parameter (see the resume-hardening design)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._sweep_io import _atomic_write_json

STATUS_FILE = "mushin_cell_status.json"


@dataclass(frozen=True)
class ResumeContext:
    """Handed to a task that declares a ``mushin_resume`` parameter.

    ``dir`` is the cell's working directory (already the cwd when the task runs).
    ``is_resume`` is True when a prior attempt of the SAME combo left artifacts
    here. ``last_ckpt`` is the newest checkpoint in ``dir`` (or None). ``attempt``
    is 1 on the first run and increments on each re-execution of this combo."""

    dir: Path | None
    is_resume: bool
    last_ckpt: Path | None
    attempt: int


def write_cell_status(
    cell_dir, *, status: str, combo: dict[str, Any], attempt: int
) -> None:
    """Atomically write this cell's status sidecar into its own dir."""
    _atomic_write_json(
        Path(cell_dir) / STATUS_FILE,
        {"status": status, "combo": combo, "attempt": int(attempt)},
    )


def read_cell_status(cell_dir) -> dict | None:
    """Read a cell status sidecar; a missing/corrupt one reads as None."""
    p = Path(cell_dir) / STATUS_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def discover_last_ckpt(cell_dir) -> Path | None:
    """Best-effort: the checkpoint a resuming task should load. Prefers an exact
    ``last.ckpt``; otherwise the most-recently-modified ``*.ckpt`` in ``cell_dir``.
    Returns None if there is none."""
    d = Path(cell_dir)
    exact = d / "last.ckpt"
    if exact.exists():
        return exact
    ckpts = [p for p in d.glob("*.ckpt") if p.is_file()]
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)
```

- [ ] **Step 4: Run — verify pass**

Run: `uv run pytest tests/test_resume.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/_resume.py tests/test_resume.py
git commit -m "feat: ResumeContext + per-cell status/checkpoint IO primitives"
```

---

## Task 2: Durable per-cell status sidecar

Write `running`→`completed`/`failed` from inside each job so completion survives a kill. (No dir-naming change — the sidecar lives in the cell's existing numeric dir.)

**Files:**
- Modify: `src/mushin/workflows.py` (`_instrument_task`, and its call site in `run()`)
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_workflows.py`:

```python
def test_cell_status_sidecar_written_completed(tmp_path):
    from mushin._resume import read_cell_status

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    wd = str(tmp_path / "s")
    W().run(seed=multirun([0, 1]), working_dir=wd)
    statuses = [
        read_cell_status(d)
        for d in Path(wd).iterdir()
        if d.is_dir() and (d / "mushin_cell_status.json").exists()
    ]
    assert statuses and all(s["status"] == "completed" for s in statuses)


def test_cell_status_sidecar_written_failed_under_fail_soft(tmp_path):
    from mushin._resume import read_cell_status

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            if seed == 1:
                raise RuntimeError("boom")
            return dict(val=float(seed))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="fail"):
        W().run(seed=multirun([0, 1]), working_dir=wd, on_error="nan")
    got = {
        read_cell_status(d)["combo"]["seed"]: read_cell_status(d)["status"]
        for d in Path(wd).iterdir()
        if d.is_dir() and (d / "mushin_cell_status.json").exists()
    }
    assert got == {0: "completed", 1: "failed"}
```

- [ ] **Step 2: Run — verify fails**

Run: `uv run pytest tests/test_workflows.py::test_cell_status_sidecar_written_completed -q`
Expected: FAIL (no status sidecar exists).

- [ ] **Step 3: Implement — enhance `_instrument_task`**

Replace the existing `_instrument_task` (workflows.py ~117-137) with:

```python
def _instrument_task(task, combo_of_cfg=None):
    """Wrap a (cfg)->result task so its returned dict is written to a
    mushin_metrics.json sidecar in the per-job working dir (cwd), and a durable
    mushin_cell_status.json records running->completed/failed from inside the job
    (so completion survives a hard process kill). Per-run provenance is written
    first, so a failing task still leaves a provenance record behind."""
    from pathlib import Path

    from ._provenance import write_provenance
    from ._resume import read_cell_status, write_cell_status
    from ._sweep_io import write_metrics_sidecar

    def wrapped(cfg):
        cwd = Path.cwd()
        combo = combo_of_cfg(cfg) if combo_of_cfg is not None else {}
        prior = read_cell_status(cwd)
        # attempt increments only for a prior attempt of the SAME combo (a numeric
        # dir reused by a different combo after a grid change resets to 1).
        attempt = (prior["attempt"] + 1) if (prior and prior.get("combo") == combo) else 1
        try:
            write_provenance(cwd, cfg)
        except Exception:  # noqa: BLE001 - provenance is best-effort
            pass
        write_cell_status(cwd, status="running", combo=combo, attempt=attempt)
        try:
            result = task(cfg)
        except Exception:  # noqa: BLE001 - record failure durably, then re-raise
            write_cell_status(cwd, status="failed", combo=combo, attempt=attempt)
            raise
        if isinstance(result, dict):
            write_metrics_sidecar(cwd, result)
        write_cell_status(cwd, status="completed", combo=combo, attempt=attempt)
        return result

    return wrapped
```

At the call site in `run()` (search `task=_instrument_task(`), pass the combo helper:

```python
        task_call = _task_calls(
            pre_task=pre_task_fn_wrapper(self.pre_task),
            task=_instrument_task(
                task_fn_wrapper(self.task), combo_of_cfg=self._combo_of_cfg
            ),
        )
```

- [ ] **Step 4: Run — verify pass**

Run: `uv run pytest tests/test_workflows.py::test_cell_status_sidecar_written_completed tests/test_workflows.py::test_cell_status_sidecar_written_failed_under_fail_soft -q`
Expected: PASS (2 passed). (`self._combo_of_cfg` yields JSON-native sanitized scalars, so the recorded `combo` round-trips through JSON.)

- [ ] **Step 5: Commit**

```bash
git add src/mushin/workflows.py tests/test_workflows.py
git commit -m "feat: durable per-cell status sidecar (running->completed/failed)"
```

---

## Task 3: Manifest as a kill-durable read view + resume off status sidecars

Make `resume=True` skip completed cells based on the durable sidecars, so a hard kill mid-sweep never recomputes finished cells.

**Files:**
- Modify: `src/mushin/_sweep_io.py` (`Manifest.from_cell_status`)
- Modify: `src/mushin/workflows.py` (`run()` resume block)
- Test: `tests/test_sweep_resilience_integration.py`

- [ ] **Step 1: Write the failing test (simulated hard kill)**

Append to `tests/test_sweep_resilience_integration.py` (add `import subprocess, sys, textwrap, time` and `from pathlib import Path` at the top if absent):

```python
def test_resume_after_hard_kill_skips_completed_cells(tmp_path):
    # A sweep is SIGKILLed after some cells finish (no manifest write happens).
    # Resume must skip the finished cells using the durable per-cell sidecars.
    import subprocess
    import sys
    import textwrap
    import time
    from pathlib import Path

    from mushin._resume import read_cell_status

    wd = tmp_path / "s"
    marker = tmp_path / "ran.log"
    script = tmp_path / "sweep.py"
    script.write_text(
        textwrap.dedent(f"""
        import time
        from mushin import multirun
        from mushin.workflows import MultiRunMetricsWorkflow
        class W(MultiRunMetricsWorkflow):
            @staticmethod
            def task(seed):
                open(r"{marker}", "a").write(f"{{seed}}\\n")
                if seed == 2:
                    time.sleep(30)   # hang so the parent can SIGKILL mid-cell
                return dict(val=float(seed))
        W().run(seed=multirun([0,1,2,3]), working_dir=r"{wd}")
        """)
    )
    p = subprocess.Popen([sys.executable, str(script)])
    done = set()
    for _ in range(600):
        for d in (list(wd.glob("*")) if wd.exists() else []):
            s = read_cell_status(d) if d.is_dir() else None
            if s and s["status"] == "completed":
                done.add(s["combo"]["seed"])
        if {0, 1} <= done:
            break
        time.sleep(0.1)
    p.kill()
    p.wait()
    assert {0, 1} <= done  # at least these two finished before the kill

    marker.write_text("")  # reset the ran log
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W2(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            open(str(marker), "a").write(f"{seed}\n")
            return dict(val=float(seed))

    wf = W2()
    wf.run(seed=multirun([0, 1, 2, 3]), working_dir=str(wd), resume=True)
    reran = {int(x) for x in marker.read_text().split()}
    assert 0 not in reran and 1 not in reran  # durable completion survived the kill
    assert wf.is_complete


def test_resume_of_legacy_sweep_without_status_sidecars(tmp_path):
    # Backward compat: a sweep dir created by pre-feature mushin has an end-of-run
    # manifest + metrics sidecars but NO per-cell status sidecars. Resume must
    # still skip completed cells (via the legacy-manifest seed), not recompute all.
    from pathlib import Path

    from mushin import multirun
    from mushin._resume import STATUS_FILE
    from mushin.workflows import MultiRunMetricsWorkflow

    calls = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            calls["n"] += 1
            return dict(val=float(seed))

    wd = tmp_path / "s"
    W().run(seed=multirun([0, 1, 2]), working_dir=str(wd))
    # simulate a legacy on-disk state: remove every per-cell status sidecar,
    # leaving the manifest + metrics sidecars in place
    for p in wd.rglob(STATUS_FILE):
        p.unlink()

    calls["n"] = 0
    wf = W()
    wf.run(seed=multirun([0, 1, 2]), working_dir=str(wd), resume=True)
    assert calls["n"] == 0  # all three completed cells skipped via the legacy manifest
    assert wf.is_complete
```

- [ ] **Step 2: Run — verify it fails**

Run: `uv run pytest tests/test_sweep_resilience_integration.py::test_resume_after_hard_kill_skips_completed_cells -q`
Expected: FAIL (resume re-runs 0 and 1 — completion was read from the end-of-run manifest, which the kill prevented).

- [ ] **Step 3: Implement `Manifest.from_cell_status`**

In `src/mushin/_sweep_io.py`, add a classmethod to `Manifest` (after `load_or_new`):

```python
    @classmethod
    def from_cell_status(cls, root, params: list[str]) -> Manifest:
        """Reconstruct a manifest, kill-durably, by scanning per-cell status
        sidecars under ``root/*/`` (each written from inside its own job, so a
        mid-sweep process kill cannot lose completed cells).

        Backward compatible: seeds from the legacy end-of-run manifest first, so a
        sweep dir created before per-cell sidecars existed still resumes; per-cell
        sidecars (when present) are authoritative and overlay the seed."""
        from ._resume import read_cell_status

        root = Path(root)
        # Seed from the legacy manifest (empty if none) -> pre-upgrade sweeps still
        # resume their completed cells.
        m = cls.load_or_new(root, params)
        if not root.exists():
            return m
        for d in root.iterdir():
            if not d.is_dir():
                continue
            s = read_cell_status(d)
            if s is None or "combo" not in s:
                continue
            m.cells[combo_key(s["combo"])] = {
                "dir": d.name,
                "status": s.get("status", "pending"),
            }
        return m
```

- [ ] **Step 4: Implement — resume reads the durable status**

In `src/mushin/workflows.py` `run()`, in the `if resume:` block, replace
`prior_manifest = Manifest.load_or_new(Path(working_dir).resolve(), [])` with:

```python
            # Kill-durable: reconstruct prior completion from the per-cell status
            # sidecars (written from inside each job), not the end-of-run manifest
            # (which a hard kill would have prevented).
            prior_manifest = Manifest.from_cell_status(
                Path(working_dir).resolve(), []
            )
```

- [ ] **Step 5: Run — verify pass (durability + backward compat)**

Run: `uv run pytest tests/test_sweep_resilience_integration.py::test_resume_after_hard_kill_skips_completed_cells tests/test_sweep_resilience_integration.py::test_resume_of_legacy_sweep_without_status_sidecars -q`
Expected: PASS (2 passed). (`_resume_short_circuit` reads `manifest.status(combo)=="completed"` → returns cached metrics from `manifest.dir(combo)`; the durable manifest supplies both for new sweeps, and the legacy-manifest seed supplies them for pre-upgrade sweeps.)

- [ ] **Step 6: Run the resilience + io suites (existing resume paths must still pass)**

Run: `uv run pytest tests/test_sweep_resilience_integration.py tests/test_sweep_io.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/mushin/_sweep_io.py src/mushin/workflows.py tests/test_sweep_resilience_integration.py
git commit -m "feat: kill-durable resume via per-cell status sidecars"
```

---

## Task 4: `ResumeContext` injection via introspected `mushin_resume` kwarg (with combo-match guard)

**Files:**
- Modify: `src/mushin/_resume.py` (contextvar, accessor, `build_resume_context` with the guard)
- Modify: `src/mushin/workflows.py` (`_bind_resume_kwarg`; `run()` wiring; `_instrument_task` sets the contextvar)
- Test: `tests/test_workflows.py`, `tests/test_resume.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_resume.py` (unit-level guard test):

```python
def test_build_resume_context_combo_match_guard(tmp_path):
    from mushin._resume import build_resume_context, write_cell_status

    # first attempt of combo {"seed": 0}: fresh
    rc = build_resume_context(tmp_path, {"seed": 0})
    assert rc.is_resume is False and rc.attempt == 1 and rc.last_ckpt is None

    # a prior attempt of the SAME combo left a checkpoint -> resume it
    write_cell_status(tmp_path, status="failed", combo={"seed": 0}, attempt=1)
    (tmp_path / "last.ckpt").write_text("state")
    rc = build_resume_context(tmp_path, {"seed": 0})
    assert rc.is_resume is True and rc.attempt == 2
    assert rc.last_ckpt is not None and rc.last_ckpt.name == "last.ckpt"

    # SAME dir now queried for a DIFFERENT combo (numeric dir reused after a grid
    # change) -> must NOT resume or surface the other cell's checkpoint
    rc = build_resume_context(tmp_path, {"seed": 9})
    assert rc.is_resume is False and rc.attempt == 1 and rc.last_ckpt is None
```

Append to `tests/test_workflows.py` (integration):

```python
def test_task_receives_resume_context_on_reexecution(tmp_path):
    seen = {}

    class W(MultiRunMetricsWorkflow):
        FAIL = True

        @staticmethod
        def task(seed, mushin_resume=None):
            seen[seed] = mushin_resume
            if mushin_resume is not None and mushin_resume.dir is not None:
                (mushin_resume.dir / "last.ckpt").write_text("state")
            if seed == 0 and W.FAIL:
                raise RuntimeError("boom")
            return dict(val=float(seed))

    wd = str(tmp_path / "s")
    W.FAIL = True
    with pytest.warns(UserWarning, match="fail"):
        W().run(seed=multirun([0, 1]), working_dir=wd, on_error="nan")
    assert seen[0] is not None and seen[0].is_resume is False and seen[0].attempt == 1

    W.FAIL = False
    seen.clear()
    wf = W()
    wf.run(seed=multirun([0, 1]), working_dir=wd, resume=True)
    assert 1 not in seen  # seed 1 completed -> short-circuited (task not called)
    rc = seen[0]
    assert rc.is_resume is True
    assert rc.attempt == 2
    assert rc.last_ckpt is not None and rc.last_ckpt.name == "last.ckpt"
    assert wf.is_complete


def test_task_without_mushin_resume_param_is_unaffected(tmp_path):
    # Introspection gate: a task NOT declaring mushin_resume is called as today,
    # with no Hydra/zen config error.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    wf = W()
    wf.run(seed=multirun([0, 1]), working_dir=str(tmp_path / "s"))
    assert wf.to_xarray().sizes == {"seed": 2}
```

- [ ] **Step 2: Run — verify it fails**

Run: `uv run pytest tests/test_resume.py::test_build_resume_context_combo_match_guard tests/test_workflows.py::test_task_receives_resume_context_on_reexecution -q`
Expected: FAIL (`build_resume_context` doesn't exist; the task never receives a context, or `zen` raises on `mushin_resume`).

- [ ] **Step 3: Add contextvar + accessor + `build_resume_context` to `_resume.py`**

Append to `src/mushin/_resume.py`:

```python
import contextvars

_CURRENT_RESUME: contextvars.ContextVar[ResumeContext | None] = contextvars.ContextVar(
    "mushin_current_resume", default=None
)


def current_resume() -> ResumeContext | None:
    """The ResumeContext for the cell currently executing, or None."""
    return _CURRENT_RESUME.get()


def build_resume_context(cell_dir, combo: dict[str, Any]) -> ResumeContext:
    """Compute the ResumeContext for a cell about to (re-)execute in ``cell_dir``.

    Combo-match guard: a prior status sidecar is honored ONLY if its recorded
    combo equals ``combo``. This makes numeric-dir reuse safe — if a grid change
    reused this dir for a different cell, we neither resume nor surface that
    cell's checkpoint."""
    cell_dir = Path(cell_dir)
    prior = read_cell_status(cell_dir)
    matches = prior is not None and prior.get("combo") == combo
    last = discover_last_ckpt(cell_dir) if matches else None
    attempt = (prior["attempt"] + 1) if matches else 1
    return ResumeContext(
        dir=cell_dir, is_resume=matches, last_ckpt=last, attempt=attempt
    )
```

- [ ] **Step 4: Run — verify the unit guard test passes**

Run: `uv run pytest tests/test_resume.py::test_build_resume_context_combo_match_guard -q`
Expected: PASS.

- [ ] **Step 5: Add `_bind_resume_kwarg` to `workflows.py`**

Add near the other task wrappers (e.g. after `_instrument_task`):

```python
def _bind_resume_kwarg(task):
    """If ``task`` declares a ``mushin_resume`` parameter, return a wrapper that
    (a) hides that parameter from hydra-zen's `zen` (which would otherwise try to
    resolve it from config) by stripping it from the exposed signature, and
    (b) injects the current cell's ResumeContext from a contextvar at call time.
    Returns ``(task, False)`` unchanged if the task does not opt in."""
    import functools
    import inspect

    sig = inspect.signature(task)
    if "mushin_resume" not in sig.parameters:
        return task, False

    from ._resume import current_resume

    exposed = [p for n, p in sig.parameters.items() if n != "mushin_resume"]

    @functools.wraps(task)
    def _wrapper(*args, **kwargs):
        return task(*args, **kwargs, mushin_resume=current_resume())

    _wrapper.__signature__ = sig.replace(parameters=exposed)
    return _wrapper, True
```

- [ ] **Step 6: Wire it in `run()` and set the contextvar in `_instrument_task`**

(a) In `run()`, wrap `self.task` with `_bind_resume_kwarg` BEFORE `task_fn_wrapper` (zen) sees it. Replace the `task=_instrument_task(...)` block from Task 2 with:

```python
        _task_fn, _wants_resume = _bind_resume_kwarg(self.task)
        task_call = _task_calls(
            pre_task=pre_task_fn_wrapper(self.pre_task),
            task=_instrument_task(
                task_fn_wrapper(_task_fn),
                combo_of_cfg=self._combo_of_cfg,
                inject_resume=_wants_resume,
            ),
        )
```

(b) Update `_instrument_task` to accept `inject_resume`, compute the ResumeContext via `build_resume_context` (which applies the guard and yields `attempt`), set the contextvar around the call, and use `rc.attempt` for the status writes. Replace `_instrument_task` with:

```python
def _instrument_task(task, combo_of_cfg=None, inject_resume=False):
    from pathlib import Path

    from ._provenance import write_provenance
    from ._resume import _CURRENT_RESUME, build_resume_context, write_cell_status
    from ._sweep_io import write_metrics_sidecar

    def wrapped(cfg):
        cwd = Path.cwd()
        combo = combo_of_cfg(cfg) if combo_of_cfg is not None else {}
        # Compute BEFORE the "running" write overwrites the prior status, so
        # is_resume/attempt reflect the previous attempt (combo-match guarded).
        rc = build_resume_context(cwd, combo)
        try:
            write_provenance(cwd, cfg)
        except Exception:  # noqa: BLE001 - provenance is best-effort
            pass
        token = _CURRENT_RESUME.set(rc) if inject_resume else None
        write_cell_status(cwd, status="running", combo=combo, attempt=rc.attempt)
        try:
            result = task(cfg)
        except Exception:  # noqa: BLE001 - record failure durably, then re-raise
            write_cell_status(cwd, status="failed", combo=combo, attempt=rc.attempt)
            raise
        finally:
            if token is not None:
                _CURRENT_RESUME.reset(token)
        if isinstance(result, dict):
            write_metrics_sidecar(cwd, result)
        write_cell_status(cwd, status="completed", combo=combo, attempt=rc.attempt)
        return result

    return wrapped
```

(This supersedes Task 2's `_instrument_task` body — same behavior plus the contextvar and guard-derived `attempt`.)

- [ ] **Step 7: Run — verify pass**

Run: `uv run pytest tests/test_workflows.py::test_task_receives_resume_context_on_reexecution tests/test_workflows.py::test_task_without_mushin_resume_param_is_unaffected -q`
Expected: PASS (2 passed).

- [ ] **Step 8: Run the whole workflow + resilience + resume suites**

Run: `uv run pytest tests/test_workflows.py tests/test_sweep_resilience_integration.py tests/test_resume.py -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/mushin/workflows.py src/mushin/_resume.py tests/test_workflows.py tests/test_resume.py
git commit -m "feat: inject ResumeContext via introspected mushin_resume kwarg (combo-guarded)"
```

---

## Task 5: Public export + docs + changelog

**Files:**
- Modify: `src/mushin/__init__.py` (export `ResumeContext`)
- Modify: `docs/guides/resilience.md`
- Modify: `docs/notebooks/04_resilience.ipynb` (markdown-only note)
- Create: `changes/+feat-resume-hardening.added.md`
- Test: `tests/test_resume.py` (export smoke)

- [ ] **Step 1: Export `ResumeContext`**

In `src/mushin/__init__.py`, mirror the existing export pattern (find how `Study`/`multirun` are imported and added to `__all__`) and add:

```python
from ._resume import ResumeContext
```
and `"ResumeContext"` to `__all__`.

- [ ] **Step 2: Smoke test the export**

Append to `tests/test_resume.py`:

```python
def test_resume_context_is_public():
    import mushin

    assert mushin.ResumeContext is not None
```

Run: `uv run pytest tests/test_resume.py -q`
Expected: PASS.

- [ ] **Step 3: Document it in the resilience guide**

In `docs/guides/resilience.md`, add a section after the resume section (before Provenance):

````markdown
## Surviving a hard kill & resuming mid-training

`resume=True` is durable across a **hard process kill** (OOM, SLURM preemption,
node death), not just handled Python exceptions. Each cell records its status
(`running` → `completed`/`failed`) from inside its own job, so a mid-sweep kill
never loses the cells that already finished — resuming re-runs only the unfinished
ones.

A long-running cell can also resume its **own** training. Declare a
`mushin_resume` parameter on your `task`; mushin injects a `ResumeContext`:

```python
from mushin.workflows import MultiRunMetricsWorkflow

class Train(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr, seed, mushin_resume=None):
        # mushin_resume.dir       -> this cell's directory (write your checkpoint here)
        # mushin_resume.is_resume -> True when a prior attempt of THIS cell left artifacts
        # mushin_resume.last_ckpt -> newest checkpoint in dir, or None
        ckpt = mushin_resume.last_ckpt if mushin_resume else None
        trainer.fit(model, ckpt_path=ckpt)  # Lightning: default_root_dir=mushin_resume.dir
        return dict(accuracy=...)
```

Write your checkpoint into `mushin_resume.dir` (Lightning: set
`default_root_dir=mushin_resume.dir` with `ModelCheckpoint(save_last=True)`).
Re-running the **same** sweep (same grid) reuses each cell's directory, so a
resumed cell finds its own checkpoint; a cell is never handed a checkpoint from a
different cell. Tasks that don't declare `mushin_resume` are unaffected.
````

- [ ] **Step 4: Add a note to notebook 04 (markdown cell only)**

Insert one markdown cell near the end of `docs/notebooks/04_resilience.ipynb` (via NotebookEdit, `edit_mode=insert` after the last content cell, `cell_type=markdown`):

```markdown
## Beyond in-process failures

The loop above recovers from an in-process error. `resume=True` is *also* durable
across a **hard kill** (OOM, SLURM preemption) — completed cells are recorded from
inside each job, so a killed sweep resumes without recomputing them. A long cell
can resume its own training by declaring a `mushin_resume` parameter; see the
[resilience guide](../guides/resilience.md#surviving-a-hard-kill--resuming-mid-training).
```

Do NOT re-execute the notebook (markdown-only change). Then run
`uv run nbstripout --keep-output docs/notebooks/04_resilience.ipynb`.

- [ ] **Step 5: Changelog fragment**

Create `changes/+feat-resume-hardening.added.md`:

```markdown
Resumable sweeps now survive a hard process kill (OOM, SLURM preemption): each cell records its completion durably from inside its own job, so `resume=True` never recomputes finished cells. A task may also declare a `mushin_resume` parameter to receive a `ResumeContext` (the cell's directory, `is_resume`, and the last checkpoint) and resume its own training mid-run.
```

- [ ] **Step 6: Verify docs build + notebook**

Run:
```bash
uv run --group docs mkdocs build --strict
uv run --extra viz pytest --nbmake docs/notebooks/04_resilience.ipynb -p no:cacheprovider
```
Expected: strict build exit 0; `1 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/mushin/__init__.py tests/test_resume.py docs/guides/resilience.md docs/notebooks/04_resilience.ipynb changes/+feat-resume-hardening.added.md
git commit -m "docs: document resume hardening + export ResumeContext"
```

---

## Task 6: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q -p no:cacheprovider`
Expected: all pass. (The Intel-mac `netCDF4` numpy-ABI RuntimeWarning, if present, is the pre-existing local artifact, not a regression.)

- [ ] **Step 2: Lint + format + spell**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes
```
Expected: all clean.

- [ ] **Step 3: Notebooks + strict docs**

Run:
```bash
uv run --extra viz pytest --nbmake docs/notebooks/ -p no:cacheprovider
uv run --group docs mkdocs build --strict
```
Expected: `6 passed`; strict build exit 0.

- [ ] **Step 4: Torch-sensitive sanity in Docker (per project rule)**

The changes touch Hydra launch + per-job IO, not torch/numpy floors, but run the
lowest-version suite in the container (local Intel-mac cannot validate floors):

Run: `make test-lowest`
Expected: pass. (If Docker is unavailable, note it — CI's `min-versions` job covers this.)

- [ ] **Step 5: Push and open the PR** (only when the user asks)

Branch `resume-hardening`. When instructed, push and open a PR summarizing the two
gaps closed and the `ResumeContext` API. Body MUST note the **cluster-gated
validation caveat**: unit tests simulate kills, but true SLURM preemption/requeue
needs HPC hardware sign-off before this is production-ready — it joins the
cluster-gated set (#50/#58/#59). No Claude attribution.

---

## Self-review notes (for the executor)

- **Spec coverage:** component 1 (numeric dirs, no override) → intentionally *no*
  code (Task's key-facts note); component 2 (durable status) → Task 2; component 3
  (ResumeContext + combo guard) → Tasks 1 & 4; component 4 (resume semantics) →
  Task 3; component 5 (`last_ckpt`) → Task 1 (`discover_last_ckpt`).
- **The load-bearing seam** is Task 4's `_bind_resume_kwarg` ↔ `zen`:
  `test_task_without_mushin_resume_param_is_unaffected` guards the non-opt-in path,
  and `test_task_receives_resume_context_on_reexecution` the opt-in path (including
  "zen must not see `mushin_resume`"). If `zen` still errors, confirm the wrapper's
  `__signature__` actually excludes `mushin_resume` (hydra-zen reads
  `inspect.signature`, which honors `__signature__`).
- **Combo-match guard** is the correctness invariant that makes numeric dirs safe;
  `test_build_resume_context_combo_match_guard` is its dedicated unit test. The
  guard compares `self._combo_of_cfg(cfg)` (recorded) to the same helper's output
  (queried) — one combo formatting throughout; do not introduce a second.
- **No directory-naming change** — dirs stay numeric; nothing in this plan adds a
  `hydra.sweep.subdir` override.
- **`_instrument_task` is defined twice across Tasks 2 and 4** by design (Task 4
  supersedes Task 2's body). The final version is Task 4 Step 6(b).
