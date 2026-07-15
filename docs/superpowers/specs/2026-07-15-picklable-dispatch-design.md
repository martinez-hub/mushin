# Design: Picklable task dispatch (out-of-process launchers)

**Date:** 2026-07-15
**Status:** Approved (brainstorming) — pending implementation plan
**Related:** [functional sweep API](2026-07-15-functional-sweep-api-design.md) (independent; both benefit)

## Problem

mushin is built on Hydra, which supports out-of-process launchers (joblib for
local multiprocessing, submitit for SLURM) — but **mushin cannot use them today**.
Verified empirically: with `hydra-joblib-launcher` installed and
`wf.run(..., launcher="joblib")`, even the plain class form fails before running a
single cell:

```
PicklingError: Could not pickle the task to send it to the workers.
```

Root cause: `run()` hands Hydra a **closure chain**, not a picklable object:

```
task_call = _task_calls(pre_task, _instrument_task(zen(task), ...))
task_call = _fail_soft(task_call)              # if on_error="nan"
task_call = _resume_short_circuit(task_call, prior_manifest)   # if resume
```

Diagnosis of what is unpicklable (measured):
- **stdlib pickle** → `Can't pickle local object '_fail_soft.<locals>.wrapped'`
  (nested closures are not picklable).
- **cloudpickle** (which *can* pickle closures) → `cannot pickle
  '_contextvars.ContextVar' object` — the resume-hardening `_CURRENT_RESUME`
  contextvar is captured in the `_instrument_task` closure.
- A **module-level task** (staticmethod on a module-level class) *is*
  stdlib-picklable — the user's function is not the problem; the closure chain is.

So any process/SLURM-backed launcher dies at dispatch. Out-of-process parallelism
today requires the cluster-gated HPC PRs (#50/#58/#59), which sidestep pickling
with a re-exec launcher — heavyweight, and unavailable to a user who just wants
`launcher="joblib"`.

## Goal

Make mushin's dispatched task **stdlib-picklable** so standard Hydra out-of-process
launchers (joblib, submitit) work with a plain sweep — no cluster-gated machinery,
no cloudpickle dependency. Behavior (resilience, resume, provenance, fail-soft)
must be byte-for-byte preserved.

## Non-goals

- Not adding launcher plugins as dependencies. Users install
  `hydra-joblib-launcher` / `hydra-submitit-launcher` themselves and pass
  `launcher=`; mushin just stops breaking dispatch.
- Not DDP / multi-node / FSDP (that is the cluster-gated HPC PR scope). This is
  about *dispatch serialization*, which unblocks single-node multiprocessing and
  basic SLURM array submission.
- Not changing the in-process `basic` launcher behavior or any public API.

## Decision (from brainstorming)

**Replace the closure chain with a single picklable callable object** (`_TaskRunner`)
holding only picklable state and referencing module-level helpers by import (not by
capture). Chosen over adding cloudpickle because it needs no new dependency, works
with any launcher's serialization, produces a lighter payload, and forces the
contextvar/closure captures to be eliminated properly rather than papered over.

## Design

### `_TaskRunner` — a picklable replacement for the closure chain

A single callable class in `workflows.py` (or a new `_dispatch.py`) that absorbs
what `_task_calls` / `_instrument_task` / `_fail_soft` / `_resume_short_circuit`
(and the `_bind_resume_kwarg` injection) currently do as nested closures:

```python
class _TaskRunner:
    """Picklable per-cell dispatch: provenance + durable status + resume-context
    injection + fail-soft, in one object. All fields are picklable; module-level
    helpers are imported inside __call__, never captured."""

    def __init__(self, *, task, wants_resume, combo_of_cfg, base_provenance,
                 on_error, prior_manifest):
        self.task = task                 # zen-wrapped user task (see below)
        self.wants_resume = wants_resume # bool
        self.combo_of_cfg = combo_of_cfg # picklable combo builder (see below)
        self.base_provenance = base_provenance   # dict
        self.on_error = on_error         # "raise" | "nan"
        self.prior_manifest = prior_manifest     # Manifest | None (picklable)

    def __call__(self, cfg):
        from pathlib import Path
        from ._provenance import write_provenance
        from ._resume import (_CURRENT_RESUME, build_resume_context,
                              write_cell_status)
        from ._sweep_io import read_metrics_sidecar, write_metrics_sidecar
        # 1. resume short-circuit (if prior_manifest says this combo completed)
        # 2. write provenance (base + cfg)
        # 3. build ResumeContext; set _CURRENT_RESUME if wants_resume
        # 4. write "running" status
        # 5. call self.task(cfg); on exception -> "failed" status, then honor
        #    on_error ("raise" re-raises; "nan" returns the _FailedRun sentinel)
        # 6. write metrics sidecar + "completed" status
        ...
```

Each current closure maps to runner state or `__call__` logic:
- `_task_calls(pre_task, ...)` → call `pre_task` first (pre_task is a workflow
  method; if it must be picklable, it is a bound method of a picklable instance, or
  is passed as a module-level reference — see "pre_task" below).
- `_instrument_task` body → steps 2–6, referencing `write_provenance` /
  `write_cell_status` / `write_metrics_sidecar` by import (module-level, picklable
  by qualname).
- `_fail_soft` → step 5's `on_error` branch (a flag, not a wrapper).
- `_resume_short_circuit` → step 1, driven by `self.prior_manifest`
  (`Manifest.from_cell_status(...)` — a plain dataclass-ish object holding a root
  path + a dict; verify it pickles, else store `(root, cells)`).
- `_bind_resume_kwarg` (`mushin_resume` injection + zen invisibility) → see below.

### Sub-parts to make picklable

- **`self.task`** must be picklable. **Verified:** `zen(module_level_fn)` and
  `zen(_ResumeInjector(module_level_fn))` are both stdlib-picklable, so the runner
  holds the zen-wrapped task directly. For a module-level class-form task
  (staticmethod) it already is; for the functional API, `_TaskRunner` holds the
  *user's original module-level function* (not the synthesized class), so it is
  picklable too.
- **`combo_of_cfg`** is currently the `_combo_of_cell` closure (captures `self` +
  `_swept_names`). Replace with picklable state: store `swept_names: tuple[str]` and
  compute the combo in a method using the module-level `_unwrap_scalar` and the
  **staticmethod** `_sanitize_coordinate_for_xarray` (already a staticmethod on the
  class → reference `MultiRunMetricsWorkflow._sanitize_coordinate_for_xarray`,
  picklable by qualname). No closure.
- **`_CURRENT_RESUME`** contextvar → imported inside `__call__` (module global of
  `mushin._resume`), never captured in a cell → not serialized.
- **`mushin_resume` injection** → the current `_bind_resume_kwarg` returns a
  `functools.wraps` **closure**, which is **NOT picklable** (verified:
  `PicklingError: ... not the same object as ...`). It must be replaced by a small
  **picklable callable class** `_ResumeInjector`:

  ```python
  class _ResumeInjector:
      """Hides `mushin_resume` from hydra-zen (stripped __signature__) and injects
      it from the contextvar at call time. Picklable (module-level task + a
      Signature), unlike the old closure."""
      def __init__(self, task):
          self.task = task
          sig = inspect.signature(task)
          self._sig = sig.replace(parameters=[p for n, p in sig.parameters.items()
                                              if n != "mushin_resume"])
      @property
      def __signature__(self):   # inspect.signature / zen read this
          return self._sig
      def __call__(self, *a, **k):
          from ._resume import current_resume
          return self.task(*a, **k, mushin_resume=current_resume())
  ```

  **Verified end-to-end (2026-07-15):** `inspect.signature(_ResumeInjector(task))`
  shows only the real params (so `zen` never tries to bind `mushin_resume`); the
  call injects it; and `pickle.dumps(zen(_ResumeInjector(task)))` succeeds. So the
  runner holds `self.task = zen(_ResumeInjector(user_task))` when the task opts in,
  or `zen(user_task)` when it doesn't — both picklable. `inspect.Signature` and
  `zen(callable_instance)` are both picklable (verified).
- **`pre_task`** — the default is a no-op staticmethod. If a workflow overrides it,
  it must be picklable (bound method of a picklable instance, or module-level).
  Document: custom `pre_task` with unpicklable captures won't work out-of-process
  (in-process unaffected).

### Semantics mapping (behavior-preserving — the critical part)

The current composition (outer→inner) is
`_resume_short_circuit( _fail_soft( _task_calls(pre_task, _instrument_task(zen_task)) ) )`.
`_TaskRunner.__call__` must reproduce it exactly:

```python
def __call__(self, cfg):
    from pathlib import Path
    from omegaconf import OmegaConf
    from ._provenance import write_provenance
    from ._resume import _CURRENT_RESUME, build_resume_context, write_cell_status
    from ._sweep_io import read_metrics_sidecar, write_metrics_sidecar

    # (1) resume short-circuit — BEFORE pre_task/instrument, only when resuming.
    #     Uses prior_manifest.params (NOT swept_names) to match current code exactly.
    if self.prior_manifest is not None:
        sc = self._combo(cfg, self.prior_manifest.params)
        if self.prior_manifest.status(sc) == "completed":
            cached = read_metrics_sidecar(
                Path(self.prior_manifest.root) / (self.prior_manifest.dir(sc) or ""))
            if cached is not None:
                return cached                      # short-circuit: nothing else runs

    # (2) fail_soft wraps EVERYTHING below — but only when on_error == "nan".
    try:
        self.pre_task(cfg)                          # _task_calls: pre_task first
        # --- instrument body ---
        cwd = Path.cwd()
        combo = self._combo(cfg, self.swept_names)  # instrument uses swept_names
        rc = build_resume_context(cwd, combo)       # BEFORE the running-status write
        try:
            write_provenance(cwd, cfg, base=self.base_provenance)
        except Exception:  # noqa: BLE001 - best-effort
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
    except Exception as exc:  # noqa: BLE001
        if self.on_error == "nan":
            return _FailedRun(exc)                  # _fail_soft sentinel
        raise                                       # on_error=="raise": propagate
```

`self._combo(cfg, names)` is the picklable combo projection (see below): project
`cfg` onto `names`, `OmegaConf.to_container` configs, then
`MultiRunMetricsWorkflow._sanitize_coordinate_for_xarray(_unwrap_scalar(v))`.

**Invariants that MUST be preserved exactly (do not "optimize" away):**
- **Two distinct swept-name sources:** short-circuit projects onto
  `prior_manifest.params`; the instrument combo projects onto `swept_names`. The
  current code sources them separately — keep both (they coincide in practice, but
  preserving both avoids any behavior change and matches the resume-hardening
  tests).
- **Short-circuit returns before `pre_task`** — a completed cell runs no pre_task,
  no provenance, no status write.
- **`rc` is computed before the "running" status overwrites the prior sidecar**
  (so `is_resume`/`attempt` reflect the previous attempt — the combo-match guard).
- **Instrument's inner `try/except`** writes the "failed" status then re-raises
  into the fail_soft layer; the **`finally`** resets the contextvar even on failure.
- **fail_soft only applies when `on_error == "nan"`**; under `"raise"` the exception
  propagates (the unified `try/except` re-raises).

### `_TaskRunner` fields (all picklable)

`task` (= `zen(_ResumeInjector(user_task))` or `zen(user_task)`), `pre_task`
(= `zen(self.pre_task)`), `swept_names: tuple[str]`, `base_provenance: dict`,
`on_error: str`, `inject_resume: bool`, `prior_manifest: Manifest | None`.

### What `run()` builds

`run()` constructs a single `_TaskRunner(...)` and passes it to `launch()` in place
of the closure chain. In-process (`basic`) behavior is identical (the runner is
just called directly). Out-of-process, Hydra/the launcher pickles the runner,
ships it to the worker, and calls it — now succeeds.

## Backward compatibility & risk

- **Behavior-preserving.** This is an internal refactor of the dispatch path; the
  resilience (`on_error`/manifest), resume (short-circuit + `ResumeContext`), and
  provenance semantics must be identical. This is the **most sensitive,
  adversarially-reviewed code in the repo** (#74 resilience, #83 resume-hardening,
  #85 provenance) — the full `test_workflows` / `test_sweep_resilience_integration`
  / `test_resume` / `test_provenance` suites must stay green, and the change should
  get an adversarial-workflow review before merge.
- **Constraint (documented):** out-of-process launchers require everything the
  runner holds to be picklable — the user's task, any custom `pre_task`, and any
  custom `task_fn_wrapper`/`pre_task_fn_wrapper`. The **defaults are all picklable**
  (verified: `zen(task)`, `zen(pre_task)`, and both wrappers default to `zen`), so
  the normal path Just Works. Nested/lambda tasks or a lambda custom wrapper still
  run in-process but can't be shipped to a worker (verified `PicklingError`). The
  in-process `basic` launcher has no such constraint (nothing is pickled).
- No public API change; no dependency change.

## Testing

- **Pickle unit test:** `pickle.dumps(_TaskRunner(...))` round-trips; the
  reconstructed runner produces the same result on a cfg.
- **Real out-of-process test:** a class-form (and, once the functional API lands, a
  decorated) sweep with `launcher="joblib"` (loky/processes backend) completes and
  assembles the correct dataset. Gate the test on `hydra-joblib-launcher` being
  importable (add it to the dev group, or `importorskip`).
- **Semantics preserved out-of-process:** fail-soft NaN-fill, resume short-circuit,
  durable status sidecars, and provenance all behave identically under the joblib
  launcher as under `basic`.
- Full existing resilience/resume/provenance suites unchanged and green.

## Rollout

One PR (independent of the functional-API PR, and a good candidate to land first so
the functional form inherits out-of-process support): `_TaskRunner` in
`workflows.py`/`_dispatch.py`, `run()` rewired to build it, `hydra-joblib-launcher`
added to the dev group for the out-of-process test, tests, changelog. Adversarial
review required given the sensitivity. Version bump: minor (new capability:
out-of-process launchers). This also potentially simplifies the cluster-gated HPC
PRs, which currently re-exec to avoid the very pickling problem this removes.
