# Functional Sweep API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `@mushin.sweep` so the core sweep→dataset flow is a decorated function + one call returning the labeled dataset, with the class kept as the power tool.

**Architecture:** `mushin.sweep(fn)` synthesizes a `MultiRunMetricsWorkflow` subclass with `fn` as its static `task` and returns a small `Sweep` handle. `.run(**kwargs)` instantiates the class fresh, forwards to `wf.run(**kwargs)`, and returns `wf.to_xarray()`. The handle exposes `.workflow` (last-run instance) and `.workflow_cls` (synthesized class). A qualname trick makes the decorated function picklable for out-of-process launchers.

**Tech Stack:** Python, `functools`, `type()`. Edits to `src/mushin/` + `__init__.py` + docs.

**Spec:** `docs/superpowers/specs/2026-07-15-functional-sweep-api-design.md`

**Branch:** `sweep-ergonomics`. **Depends on** the [picklable-dispatch plan](2026-07-15-picklable-dispatch.md) ONLY for the out-of-process functional test (Task 4) — the in-process functional API is independent and can land first or second.

---

## Key facts (verified during spec hardening — do not re-derive)

- `type(fn.__name__, (MultiRunMetricsWorkflow,), {"task": staticmethod(fn)})` yields a valid workflow subclass (mushin requires `task` be a `staticmethod`; verified).
- `wf.run(**kwargs)` already splits its named options (`working_dir`/`on_error`/`resume`/`capture_env`/`sweeper`/`launcher`/`overrides`/`version_base`/`task_fn_wrapper`/…) from the `**workflow_overrides` sweep dims, so `Sweep.run` can forward everything transparently.
- `to_xarray()` stamps `ds.attrs["mushin_failures"]` and `ds.attrs["provenance"]`, so a returned dataset carries resilience info.
- **Decorator shadows the name** (`experiment` becomes the handle), so pickle-by-reference can't find `fn` — fixed by `fn.__qualname__ += ".__mushin_task__"` + `self.__mushin_task__ = fn` on the handle (which *is* findable at `module.<name>`). Verified picklable.
- `mushin_resume` works for free (the injection introspects the task signature = the user's fn).

---

## Task 1: `Sweep` handle + `sweep()` decorator

**Files:**
- Create: `src/mushin/_sweep_decorator.py`
- Test: `tests/test_sweep_decorator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sweep_decorator.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def test_decorated_sweep_returns_labeled_dataset(tmp_path):
    @mushin.sweep
    def experiment(a, b):
        return dict(v=float(a + b))

    ds = experiment.run(a=multirun([1, 2]), b=multirun([0, 1]),
                        working_dir=str(tmp_path / "s"))
    assert ds.sizes == {"a": 2, "b": 2}
    assert float(ds["v"].sel(a=2, b=1)) == 3.0


def test_handle_exposes_workflow_and_class(tmp_path):
    @mushin.sweep
    def experiment(seed):
        return dict(v=float(seed))

    assert experiment.workflow is None  # before first run
    experiment.run(seed=multirun([0, 1]), working_dir=str(tmp_path / "s"))
    assert isinstance(experiment.workflow, MultiRunMetricsWorkflow)
    assert experiment.workflow.provenance is not None
    assert issubclass(experiment.workflow_cls, MultiRunMetricsWorkflow)
    # a fresh instance can be constructed from the class
    experiment.workflow_cls().run(seed=multirun([0]), working_dir=str(tmp_path / "s2"))


def test_wraps_preserves_name_and_doc():
    @mushin.sweep
    def experiment(seed):
        "my docstring"
        return dict(v=float(seed))

    assert experiment.__name__ == "experiment"
    assert experiment.__doc__ == "my docstring"


def test_fresh_instance_per_run_no_state_leak(tmp_path):
    import pytest

    @mushin.sweep
    def experiment(seed):
        if seed == 1:
            raise RuntimeError("boom")
        return dict(v=float(seed))

    with pytest.warns(UserWarning, match="fail"):
        experiment.run(seed=multirun([0, 1]), working_dir=str(tmp_path / "a"),
                       on_error="nan")
    assert experiment.workflow.failures  # this run failed
    # a clean run on a fresh dir must not inherit failures
    experiment.run(seed=multirun([0]), working_dir=str(tmp_path / "b"))
    assert experiment.workflow.failures == []
```

- [ ] **Step 2: Run — verify it fails**

Run: `uv run pytest tests/test_sweep_decorator.py -q`
Expected: FAIL (`module 'mushin' has no attribute 'sweep'`).

- [ ] **Step 3: Implement `src/mushin/_sweep_decorator.py`**

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The `@mushin.sweep` decorator: a boilerplate-free entry point that turns a
plain `task`-style function into a runnable sweep, over `MultiRunMetricsWorkflow`."""

from __future__ import annotations

import functools

from .workflows import MultiRunMetricsWorkflow


class Sweep:
    """Handle returned by `@mushin.sweep`. Call `.run(...)` to run the sweep and
    get the labeled `xarray.Dataset`; drop to `.workflow` (last-run instance) or
    `.workflow_cls` (the synthesized class) for power features."""

    def __init__(self, fn, cls):
        functools.wraps(fn)(self)  # name/doc/__wrapped__ (copies fn's qualname)
        # Make `fn` picklable despite the decorator shadowing its module name:
        # re-point its qualname THROUGH this handle (findable at module.<name>) and
        # hang it here, so out-of-process launchers can serialize the task.
        fn.__qualname__ = fn.__qualname__ + ".__mushin_task__"
        self.__mushin_task__ = fn
        self.workflow_cls = cls
        self.workflow = None  # last-run instance (None before the first run)

    def run(self, **kwargs):
        """Run the sweep and return its labeled `xarray.Dataset`. Forwards every
        keyword to `MultiRunMetricsWorkflow.run` (sweep dims via `multirun(...)`
        plus `working_dir` / `on_error` / `resume` / `launcher` / … )."""
        wf = self.workflow_cls()  # fresh per run — no state carryover
        wf.run(**kwargs)
        self.workflow = wf
        return wf.to_xarray()


def sweep(fn):
    """Turn a plain `task(**params) -> dict` function into a runnable `Sweep`."""
    cls = type(fn.__name__, (MultiRunMetricsWorkflow,), {"task": staticmethod(fn)})
    cls.__module__ = fn.__module__
    cls.__qualname__ = fn.__qualname__
    return Sweep(fn, cls)
```

- [ ] **Step 4: Export `sweep` (needed for the test to import `mushin.sweep`)**

In `src/mushin/__init__.py`: add `from ._sweep_decorator import sweep` to the eager
import block (it is lightweight — only imports `functools` + `workflows`, no
Lightning) and add `"sweep"` to `__all__`.

- [ ] **Step 5: Run — verify pass**

Run: `uv run pytest tests/test_sweep_decorator.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Import stays light (regression)**

Run: `uv run pytest tests/test_lazy_imports.py -q`
Expected: PASS — `import mushin` must still NOT pull `pytorch_lightning` (the
decorator module imports only `functools` + `mushin.workflows`, both pl-free).
If it regresses, confirm `_sweep_decorator` imports nothing Lightning-bound.

- [ ] **Step 7: Commit**

```bash
uv run ruff check src/mushin/_sweep_decorator.py src/mushin/__init__.py tests/test_sweep_decorator.py && uv run ruff format src/mushin/_sweep_decorator.py src/mushin/__init__.py tests/test_sweep_decorator.py
git add src/mushin/_sweep_decorator.py src/mushin/__init__.py tests/test_sweep_decorator.py
git commit -m "feat: @mushin.sweep functional sweep decorator"
```

---

## Task 2: Resilience / resume / `mushin_resume` passthrough

Prove the sugar carries every core feature through unchanged.

**Files:**
- Test: `tests/test_sweep_decorator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sweep_decorator.py`:

```python
def test_decorated_sweep_resilience_and_resume(tmp_path):
    import pytest

    FAIL = {"on": True}

    @mushin.sweep
    def experiment(seed):
        if seed == 1 and FAIL["on"]:
            raise RuntimeError("boom")
        return dict(v=float(seed))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="fail"):
        ds = experiment.run(seed=multirun([0, 1, 2]), working_dir=wd, on_error="nan")
    import numpy as np
    assert np.isnan(float(ds["v"].sel(seed=1)))
    assert ds.attrs["mushin_failures"]  # carried on the dataset

    FAIL["on"] = False
    ds2 = experiment.run(seed=multirun([0, 1, 2]), working_dir=wd, resume=True)
    assert float(ds2["v"].sel(seed=1)) == 1.0  # filled on resume
    assert not ds2.attrs.get("mushin_failures")


def test_decorated_sweep_receives_mushin_resume(tmp_path):
    import pytest

    seen = {}
    FAIL = {"on": True}

    @mushin.sweep
    def experiment(seed, mushin_resume=None):
        seen[seed] = mushin_resume
        if mushin_resume is not None and mushin_resume.dir is not None:
            (mushin_resume.dir / "last.ckpt").write_text("state")
        if seed == 0 and FAIL["on"]:
            raise RuntimeError("boom")
        return dict(v=float(seed))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="fail"):
        experiment.run(seed=multirun([0, 1]), working_dir=wd, on_error="nan")
    assert seen[0].is_resume is False

    FAIL["on"] = False
    seen.clear()
    experiment.run(seed=multirun([0, 1]), working_dir=wd, resume=True)
    assert 1 not in seen  # seed 1 completed -> short-circuited
    assert seen[0].is_resume is True and seen[0].last_ckpt.name == "last.ckpt"
```

- [ ] **Step 2: Run — verify pass**

Run: `uv run pytest tests/test_sweep_decorator.py -k "resilience or mushin_resume" -q`
Expected: PASS (2 passed) — these should pass with Task 1's implementation as-is
(the decorator forwards to `wf.run`, and `mushin_resume` injection is signature-based).
If `mushin_resume` isn't received, confirm the synthesized class's `task` is the raw
`fn` (so the signature-based injection sees `mushin_resume`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_sweep_decorator.py
git commit -m "test: @mushin.sweep carries resilience, resume, and mushin_resume"
```

---

## Task 3: Framework-agnostic + full suite

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sweep_decorator.py`:

```python
def test_decorated_sklearn_sweep_no_torch(tmp_path):
    sklearn = __import__("pytest").importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    @mushin.sweep
    def experiment(C, seed):
        x, y = make_classification(n_samples=200, random_state=seed)
        m = LogisticRegression(C=C, max_iter=500).fit(x, y)
        return dict(accuracy=float(m.score(x, y)))

    ds = experiment.run(C=multirun([0.1, 1.0]), seed=multirun([0, 1]),
                        working_dir=str(tmp_path / "s"))
    assert ds.sizes == {"C": 2, "seed": 2}
```

- [ ] **Step 2: Run + full suite**

```bash
uv run pytest tests/test_sweep_decorator.py -q
uv run pytest -q -p no:cacheprovider
```
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sweep_decorator.py
git commit -m "test: @mushin.sweep is framework-agnostic (sklearn)"
```

---

## Task 4: Out-of-process functional form (requires picklable dispatch)

Only meaningful once the [picklable-dispatch plan](2026-07-15-picklable-dispatch.md)
has landed on the branch. If it has not, SKIP this task and note it.

**Files:**
- Test: `tests/test_sweep_decorator.py`

- [ ] **Step 1: Write the test**

The decorated function must be defined at **module level** in the test file (not
nested) so its qualname trick resolves. Append at module top-level:

```python
@mushin.sweep
def _oop_experiment(seed):
    return dict(v=float(seed))


def test_decorated_sweep_out_of_process_joblib(tmp_path):
    import pytest

    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    ds = _oop_experiment.run(seed=multirun([0, 1, 2]),
                             working_dir=str(tmp_path / "s"), launcher="joblib")
    assert ds.sizes == {"seed": 3}
    assert float(ds["v"].sel(seed=2)) == 2.0
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_sweep_decorator.py::test_decorated_sweep_out_of_process_joblib -q`
Expected: PASS (the handle-qualname trick makes `_oop_experiment`'s task picklable;
picklable dispatch ships it to a worker). If `PicklingError`, confirm both the
qualname trick (Task 1) and picklable dispatch are present.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sweep_decorator.py
git commit -m "test: @mushin.sweep runs out-of-process (joblib)"
```

---

## Task 5: Docs + changelog + verification

**Files:**
- Modify: `docs/quickstart.md` (lead with the functional form)
- Create: `changes/+feat-functional-sweep-api.added.md`

- [ ] **Step 1: Make the functional form the primary quickstart**

In `docs/quickstart.md`, change the opening so the decorated form is shown first,
with the class form kept as "going deeper". Replace the intro + first code block:

````markdown
Decorate a function with `@mushin.sweep`, sweep it over a grid, and get results
back as a labeled `xarray.Dataset` — no subclassing, no callbacks:

```python
import mushin

@mushin.sweep
def experiment(lr, seed):
    ...  # train, evaluate
    return dict(accuracy=acc)      # the returned dict becomes dataset variables

ds = experiment.run(
    lr=mushin.multirun([0.01, 0.1, 1.0]),
    seed=mushin.multirun([0, 1, 2]),
)
```

Need the full tool — `.failures`, `.plot()`, provenance, custom `to_xarray`?
Drop to `experiment.workflow` (the last-run instance) or subclass
`MultiRunMetricsWorkflow` directly (shown below).
````

Keep the existing class-form example further down under a "Going deeper: the
workflow class" heading.

- [ ] **Step 2: Changelog fragment**

Create `changes/+feat-functional-sweep-api.added.md`:

```markdown
New `@mushin.sweep` decorator: turn a plain `task`-style function into a runnable sweep with no subclassing — `experiment.run(lr=multirun([...]), seed=multirun([...]))` returns the labeled `xarray.Dataset` directly. Drop to `experiment.workflow` (the last-run `MultiRunMetricsWorkflow`) or `experiment.workflow_cls` for power features; `mushin_resume` and all `run()` resilience options carry through. The `MultiRunMetricsWorkflow` subclass form is unchanged.
```

- [ ] **Step 3: Verify**

```bash
uv run pytest -q -p no:cacheprovider
uv run ruff check . && uv run ruff format --check .
uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes
uv run --group docs mkdocs build --strict
```
Expected: all green; strict docs build exit 0.

- [ ] **Step 4: Commit**

```bash
git add docs/quickstart.md changes/+feat-functional-sweep-api.added.md
git commit -m "docs: lead quickstart with @mushin.sweep + changelog"
```

- [ ] **Step 5: Push + PR** (only when the user asks)

Independent PR (or stacked on picklable-dispatch). Body: the functional API, the
escape hatches, and that the class form is unchanged. No Claude attribution.

---

## Self-review checklist

- [ ] `import mushin` still does not load `pytorch_lightning` (decorator module is pl-free) — `tests/test_lazy_imports.py` green.
- [ ] `mushin.sweep` in `__all__`; `Sweep` is internal (not exported).
- [ ] Fresh instance per `.run()` (no `.failures`/`.provenance` carryover).
- [ ] `experiment.__name__`/`__doc__` preserved; `fn.__qualname__` mangled for pickling; handle holds `__mushin_task__`.
- [ ] Out-of-process test gated on `hydra_plugins.hydra_joblib_launcher` (importorskip) and on picklable dispatch being present.
