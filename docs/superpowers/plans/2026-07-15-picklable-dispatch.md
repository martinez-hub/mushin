# Picklable Task Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MultiRunMetricsWorkflow`'s per-cell task dispatch stdlib-picklable so standard out-of-process Hydra launchers (joblib, submitit) work with a plain sweep — replacing the unpicklable closure chain with one picklable `_TaskRunner` object, behavior byte-for-byte preserved.

**Architecture:** Collapse `_task_calls` / `_instrument_task` / `_fail_soft` / `_resume_short_circuit` / `_bind_resume_kwarg` into a picklable `_ResumeInjector` class (for `mushin_resume`) + a picklable `_TaskRunner` callable holding only picklable state and importing module-level helpers inside `__call__`. `run()` builds one `_TaskRunner` instead of nesting closures.

**Tech Stack:** Python, hydra-zen (`zen`), Hydra launchers. Edits to `src/mushin/workflows.py`; dev-group dep `hydra-joblib-launcher` for the out-of-process test.

**Spec:** `docs/superpowers/specs/2026-07-15-picklable-dispatch-design.md`

**Branch:** `sweep-ergonomics` (holds the specs; do the work here, or branch `picklable-dispatch` off it — this plan assumes work continues on a branch off current `main` that includes #84/#85).

---

## Key facts (verified during spec hardening — do not re-derive)

- Current composition in `run()` (outer→inner):
  `_resume_short_circuit( _fail_soft( _task_calls(pre_task, _instrument_task(zen_task)) ) )`, built at `workflows.py` ~618-668. `_fail_soft` applied only when `on_error=="nan"`; `_resume_short_circuit` only when `resume=True`.
- `zen(module_level_fn)`, `zen(_ResumeInjector(fn))`, `zen(default pre_task)`, `Manifest`, `inspect.Signature`, and the static `_sanitize_coordinate_for_xarray` ref are all **stdlib-picklable** (verified). The **closures** and a captured **`_CURRENT_RESUME` contextvar** are what break pickling.
- `_bind_resume_kwarg`'s `functools.wraps` wrapper is **not** picklable; replace with the `_ResumeInjector` class (verified picklable + zen-invisible + injects).
- Both `task_fn_wrapper` and `pre_task_fn_wrapper` default to `zen` (picklable). A lambda custom wrapper is not picklable → out-of-process fails (documented constraint).
- Module-level helpers referenced inside `__call__`: `write_provenance` (`._provenance`), `_CURRENT_RESUME`/`build_resume_context`/`write_cell_status`/`current_resume` (`._resume`), `read_metrics_sidecar`/`write_metrics_sidecar` (`._sweep_io`), module-level `_unwrap_scalar` and staticmethod `MultiRunMetricsWorkflow._sanitize_coordinate_for_xarray` (both in `workflows.py`).
- **Behavior invariants to preserve** (from the spec's Semantics Mapping): short-circuit uses `prior_manifest.params`; instrument combo uses `swept_names`; short-circuit returns before `pre_task`; `rc` computed before the "running" write; instrument's inner `try/except` writes "failed" then re-raises into fail-soft; `finally` resets the contextvar; fail-soft only under `on_error=="nan"`.
- **BaseWorkflow behavior:** plain `BaseWorkflow` subclasses (no `_sanitize_coordinate_for_xarray`) currently record an **empty combo** (the `_combo_of_cell` guard sets `combo_of_cfg=None` → `{}`). Preserve by passing the runner `swept_names=()` when `self` lacks the sanitizer.

---

## Task 1: `_ResumeInjector` — picklable `mushin_resume` injection

Replaces the unpicklable `_bind_resume_kwarg` closure.

**Files:**
- Modify: `src/mushin/workflows.py`
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflows.py`:

```python
def test_resume_injector_is_picklable_and_hides_param(tmp_path):
    import inspect
    import pickle

    from hydra_zen import zen

    from mushin.workflows import _ResumeInjector, _prepare_task

    def task(seed, mushin_resume=None):
        return {"v": float(seed), "got": mushin_resume}

    prepared, wants = _prepare_task(task)
    assert wants is True
    # hidden from the signature zen inspects:
    assert list(inspect.signature(prepared).parameters) == ["seed"]
    # picklable, and zen(prepared) picklable:
    pickle.loads(pickle.dumps(prepared))
    pickle.loads(pickle.dumps(zen(prepared)))

    # a task WITHOUT mushin_resume is returned unchanged:
    def plain(seed):
        return {"v": float(seed)}

    prepared2, wants2 = _prepare_task(plain)
    assert wants2 is False and prepared2 is plain
```

- [ ] **Step 2: Run — verify it fails**

Run: `uv run pytest tests/test_workflows.py::test_resume_injector_is_picklable_and_hides_param -q`
Expected: FAIL (`cannot import name '_ResumeInjector'` / `_prepare_task`).

- [ ] **Step 3: Implement**

In `src/mushin/workflows.py`, add these (place near `_bind_resume_kwarg`, which you will remove in Task 2):

```python
class _ResumeInjector:
    """Picklable replacement for the old `_bind_resume_kwarg` closure. Hides a
    task's `mushin_resume` parameter from hydra-zen's `zen` (via a stripped
    __signature__, so zen never tries to resolve it from config) and injects the
    current cell's ResumeContext from a contextvar at call time. Unlike a closure,
    an instance holding a module-level task + a Signature is stdlib-picklable."""

    def __init__(self, task):
        import inspect

        self._task = task
        sig = inspect.signature(task)
        self._sig = sig.replace(
            parameters=[
                p for n, p in sig.parameters.items() if n != "mushin_resume"
            ]
        )

    @property
    def __signature__(self):  # inspect.signature / zen read this
        return self._sig

    def __call__(self, *args, **kwargs):
        from ._resume import current_resume

        return self._task(*args, **kwargs, mushin_resume=current_resume())


def _prepare_task(task):
    """If `task` declares a `mushin_resume` parameter, wrap it in a picklable
    `_ResumeInjector`; otherwise return it unchanged. Returns `(prepared, wants)`."""
    import inspect

    if "mushin_resume" in inspect.signature(task).parameters:
        return _ResumeInjector(task), True
    return task, False
```

- [ ] **Step 4: Run — verify pass**

Run: `uv run pytest tests/test_workflows.py::test_resume_injector_is_picklable_and_hides_param -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/mushin/workflows.py tests/test_workflows.py && uv run ruff format src/mushin/workflows.py tests/test_workflows.py
git add src/mushin/workflows.py tests/test_workflows.py
git commit -m "feat: picklable _ResumeInjector for mushin_resume injection"
```

---

## Task 2: `_TaskRunner` — picklable dispatch, wired into `run()`

The core refactor. The full existing resilience/resume/provenance suites are the equivalence oracle; a new pickle test is added.

**Files:**
- Modify: `src/mushin/workflows.py` (add `_TaskRunner`; rewire `run()`; remove `_task_calls`, `_instrument_task`, `_fail_soft`, `_resume_short_circuit`, `_bind_resume_kwarg`)
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Write the failing test (pickle round-trip of the built dispatch)**

Append to `tests/test_workflows.py`. The workflow MUST be module-level (a task
with a nested qualname is not picklable, which would fail the test for the wrong
reason):

```python
class _PicklableRunnerWF(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed):
        return dict(v=float(seed))


def test_task_runner_is_picklable(tmp_path):
    import pickle

    import mushin.workflows as wf_mod

    captured = {}
    orig = wf_mod.launch

    def capture(cfg, task_call, **k):  # launch(cfg, task_call, **kwargs)
        captured["task_call"] = task_call
        return orig(cfg, task_call, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wf_mod, "launch", capture)
        _PicklableRunnerWF().run(seed=multirun([0, 1]),
                                 working_dir=str(tmp_path / "s"), on_error="nan")

    tc = captured["task_call"]
    assert isinstance(tc, wf_mod._TaskRunner)
    pickle.loads(pickle.dumps(tc))  # the whole dispatch object is picklable
```

- [ ] **Step 2: Run — verify it fails**

Run: `uv run pytest tests/test_workflows.py::test_task_runner_is_picklable -q`
Expected: FAIL (`_TaskRunner` doesn't exist; `task_call` is a closure).

- [ ] **Step 3: Implement `_TaskRunner`**

In `src/mushin/workflows.py`, add (keep `_FailedRun` — it is still used):

```python
class _TaskRunner:
    """Picklable per-cell dispatch. Collapses the old closure chain
    (_task_calls / _instrument_task / _fail_soft / _resume_short_circuit) into one
    object so out-of-process launchers (joblib/submitit) can pickle it. Holds only
    picklable state; imports module-level helpers inside __call__ (never captures
    the _CURRENT_RESUME contextvar). Behavior mirrors the previous chain exactly —
    see the spec's Semantics Mapping."""

    def __init__(self, *, task, pre_task, swept_names, base_provenance,
                 on_error, inject_resume, prior_manifest):
        self.task = task                    # zen(_ResumeInjector(fn)) or zen(fn)
        self.pre_task = pre_task            # zen(self.pre_task)
        self.swept_names = tuple(swept_names)
        self.base_provenance = base_provenance
        self.on_error = on_error
        self.inject_resume = inject_resume
        self.prior_manifest = prior_manifest

    @staticmethod
    def _combo(cfg, names):
        from omegaconf import OmegaConf

        combo = {}
        for n in names:
            val = cfg[n]
            if OmegaConf.is_config(val):
                val = OmegaConf.to_container(val, resolve=True)
            combo[n] = MultiRunMetricsWorkflow._sanitize_coordinate_for_xarray(
                _unwrap_scalar(val)
            )
        return combo

    def __call__(self, cfg):
        from pathlib import Path

        from ._provenance import write_provenance
        from ._resume import (
            _CURRENT_RESUME,
            build_resume_context,
            write_cell_status,
        )
        from ._sweep_io import read_metrics_sidecar, write_metrics_sidecar

        # (1) resume short-circuit — before pre_task/instrument, only when resuming
        if self.prior_manifest is not None:
            sc = self._combo(cfg, self.prior_manifest.params)
            if self.prior_manifest.status(sc) == "completed":
                cached = read_metrics_sidecar(
                    Path(self.prior_manifest.root) / (self.prior_manifest.dir(sc) or "")
                )
                if cached is not None:
                    return cached

        # (2) fail-soft wraps everything below, only when on_error == "nan"
        try:
            self.pre_task(cfg)
            cwd = Path.cwd()
            combo = self._combo(cfg, self.swept_names)
            rc = build_resume_context(cwd, combo)
            try:
                write_provenance(cwd, cfg, base=self.base_provenance)
            except Exception:  # noqa: BLE001 - provenance is best-effort
                pass
            token = _CURRENT_RESUME.set(rc) if self.inject_resume else None
            write_cell_status(cwd, status="running", combo=combo, attempt=rc.attempt)
            try:
                result = self.task(cfg)
            except Exception:  # noqa: BLE001 - durable failed status, then re-raise
                write_cell_status(cwd, status="failed", combo=combo, attempt=rc.attempt)
                raise
            finally:
                if token is not None:
                    _CURRENT_RESUME.reset(token)
            if isinstance(result, dict):
                write_metrics_sidecar(cwd, result)
            write_cell_status(cwd, status="completed", combo=combo, attempt=rc.attempt)
            return result
        except Exception as exc:  # noqa: BLE001 - fail-soft sentinel or re-raise
            if self.on_error == "nan":
                return _FailedRun(exc)
            raise
```

- [ ] **Step 4: Rewire `run()`**

In `run()` (`workflows.py` ~597-668), replace the block that computes `_combo_of_cell`, captures base provenance, calls `_bind_resume_kwarg`, builds `task_call` via `_task_calls`, and the two `if on_error=="nan"` / `if resume:` wrapping lines with:

```python
        # Swept dimension names for the per-cell combo (unchanged).
        _swept_names = tuple(
            k
            for k, v in self._parse_overrides(launch_overrides).items()
            if isinstance(v, multirun)
        )
        # BaseWorkflow (no sanitizer) records an empty combo, as before.
        _runner_swept = _swept_names if hasattr(
            self, "_sanitize_coordinate_for_xarray"
        ) else ()

        from ._provenance import capture_base

        _base_provenance = capture_base()

        # Kill-durable resume manifest (only when resuming) — built here so the
        # runner can hold it (was built inside the old `if resume:` block).
        _prior_manifest = None
        if resume:
            _prior_manifest = Manifest.from_cell_status(
                Path(working_dir).resolve(), list(_swept_names)
            )

        _task_fn, _wants_resume = _prepare_task(self.task)
        task_call = _TaskRunner(
            task=task_fn_wrapper(_task_fn),
            pre_task=pre_task_fn_wrapper(self.pre_task),
            swept_names=_runner_swept,
            base_provenance=_base_provenance,
            on_error=on_error,
            inject_resume=_wants_resume,
            prior_manifest=_prior_manifest,
        )
```

Delete the now-unused functions `_task_calls`, `_instrument_task`, `_fail_soft`,
`_resume_short_circuit`, `_bind_resume_kwarg`, and any now-stale `if on_error == "nan": task_call = _fail_soft(...)` / `if resume: ... _resume_short_circuit(...)` lines and the old `Manifest.from_cell_status(...)`/`prior_manifest` assignment inside the former resume block. Ensure `Manifest` is imported where `run()` can see it (it is imported inside `jobs_post_process`; add a top-of-`run()` or module import if needed — check and adjust).

- [ ] **Step 5: Run the new pickle test + the FULL behavior-equivalence suites**

Run:
```bash
uv run pytest tests/test_workflows.py::test_task_runner_is_picklable -q
uv run pytest tests/test_workflows.py tests/test_sweep_resilience_integration.py tests/test_resume.py tests/test_provenance.py -q -p no:cacheprovider
```
Expected: the pickle test PASSES; **every** resilience/resume/provenance/workflow test stays green (this is the equivalence proof — fail-soft NaN-fill, resume short-circuit, durable status, provenance-once, `mushin_resume` injection, combo-match guard, hard-kill durability all still pass). If any fails, the mapping deviated — fix `_TaskRunner.__call__` against the spec's Semantics Mapping; do NOT weaken tests.

- [ ] **Step 6: Commit**

```bash
uv run ruff check src/mushin/workflows.py tests/test_workflows.py && uv run ruff format src/mushin/workflows.py tests/test_workflows.py
git add src/mushin/workflows.py tests/test_workflows.py
git commit -m "refactor: picklable _TaskRunner dispatch (behavior-preserving)"
```

---

## Task 3: Out-of-process launcher works (joblib)

**Files:**
- Modify: `pyproject.toml` (`dev` group: add `hydra-joblib-launcher`)
- Test: `tests/test_sweep_resilience_integration.py`

- [ ] **Step 1: Add the launcher to the dev group**

In `pyproject.toml` `[dependency-groups]` `dev`, add:

```toml
    # Exercises out-of-process dispatch (loky/processes backend) so the picklable
    # _TaskRunner is verified against a real launcher, not just pickle.dumps.
    "hydra-joblib-launcher >= 1.2",
```
Then `uv sync`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_sweep_resilience_integration.py`. The workflow MUST be
module-level — the joblib/loky launcher pickles the task to ship it to a worker,
so a nested-qualname task cannot work (that is the whole point of this test):

```python
class _OOPWorkflow(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed):
        if seed == 1:
            raise RuntimeError("boom")
        return dict(v=float(seed))


def test_sweep_runs_out_of_process_with_joblib(tmp_path):
    # The picklable _TaskRunner must survive being shipped to a separate worker
    # process by the joblib (loky) launcher, and fail-soft must still work.
    import numpy as np
    import pytest

    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")

    wf = _OOPWorkflow()
    with pytest.warns(UserWarning, match="fail"):
        wf.run(seed=multirun([0, 1, 2]), working_dir=str(tmp_path / "s"),
               launcher="joblib", on_error="nan")
    ds = wf.to_xarray()
    assert ds.sizes == {"seed": 3}
    vals = {int(s): float(ds["v"].sel(seed=s)) for s in ds["seed"].values}
    assert np.isnan(vals[1]) and vals[0] == 0.0 and vals[2] == 2.0
```

Ensure `MultiRunMetricsWorkflow` and `multirun` are imported at the top of
`tests/test_sweep_resilience_integration.py` (they are used elsewhere in the
file; add module-level imports if the class definition needs them).

- [ ] **Step 3: Run — verify it fails without the fix / passes with it**

Run: `uv run pytest tests/test_sweep_resilience_integration.py::test_sweep_runs_out_of_process_with_joblib -q`
Expected: PASS with Task 2's `_TaskRunner` in place. (Before Task 2, the same test would raise `PicklingError` — you can confirm the fix mattered by `git stash` of Task 2, but that is optional.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/test_sweep_resilience_integration.py
git commit -m "test: verify out-of-process (joblib) sweep + fail-soft via picklable dispatch"
```

---

## Task 4: Docs, changelog, full verification

**Files:**
- Modify: `docs/guides/workflows.md` (a short "Parallel / out-of-process launchers" note)
- Create: `changes/+feat-picklable-dispatch.added.md`

- [ ] **Step 1: Document out-of-process launchers**

Add a short section to `docs/guides/workflows.md` (after the sweep basics):

````markdown
## Parallel & out-of-process launchers

By default a sweep runs its cells in-process, sequentially (Hydra's `basic`
launcher). Install a Hydra launcher plugin and pass `launcher=` to parallelize
across cores or submit to a scheduler:

```bash
pip install hydra-joblib-launcher     # local multiprocessing
```
```python
wf.run(..., launcher="joblib")        # loky/processes backend
```

Out-of-process launchers pickle each cell's task to ship it to a worker, so your
`task` (and any custom `pre_task`) must be importable (module-level) — the normal
case. Nested/lambda tasks run only in-process. Resilience (`on_error="nan"`,
`resume=True`) and provenance behave identically out-of-process.
````

- [ ] **Step 2: Changelog fragment**

Create `changes/+feat-picklable-dispatch.added.md`:

```markdown
Sweeps can now use out-of-process Hydra launchers (e.g. `hydra-joblib-launcher`, submitit): per-cell dispatch is stdlib-picklable, so `run(..., launcher="joblib")` parallelizes across worker processes. Previously any process-backed launcher failed with a `PicklingError`. Resilience, resume, and provenance semantics are unchanged in-process and preserved out-of-process.
```

- [ ] **Step 3: Full verification**

```bash
uv run pytest -q -p no:cacheprovider
uv run ruff check . && uv run ruff format --check .
uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes
uv run --group docs mkdocs build --strict
```
Expected: all green; strict docs build exit 0.

- [ ] **Step 4: Docker lowest-version + commit**

```bash
make test-lowest       # verify on the min dependency floors (Linux)
git add docs/guides/workflows.md changes/+feat-picklable-dispatch.added.md
git commit -m "docs: document out-of-process launchers + changelog"
```
Expected: `make test-lowest` passes (note if Docker unavailable — CI `min-versions` covers it).

- [ ] **Step 5: Adversarial review before PR** (REQUIRED — sensitive code)

This refactors the #74/#83/#85 dispatch path. Before opening the PR, run the
`adversarial-workflow` review over the branch diff and fix any confirmed findings.
Then push + open the PR only when the user asks; PR body must note the behavior
mapping + that resilience/resume/provenance suites are green in- and out-of-process.

---

## Self-review checklist (run after writing code)

- [ ] Every one of the five spec invariants is reflected in `_TaskRunner.__call__` (two swept-name sources; short-circuit-before-pre_task; `rc`-before-running; failed-then-reraise; fail-soft-only-on-nan).
- [ ] `swept_names=()` path preserves empty-combo behavior for plain `BaseWorkflow`.
- [ ] No leftover references to the deleted `_task_calls`/`_instrument_task`/`_fail_soft`/`_resume_short_circuit`/`_bind_resume_kwarg` (grep).
- [ ] `_TaskRunner` and `_ResumeInjector` hold only picklable state; `__call__` imports helpers, captures nothing.
