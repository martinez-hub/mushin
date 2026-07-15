# Design: Functional sweep API (`@mushin.sweep`)

**Date:** 2026-07-15
**Status:** Approved (brainstorming) — pending implementation plan
**Related:** [picklable dispatch spec](2026-07-15-picklable-dispatch-design.md) (independent, but the functional form benefits once dispatch is picklable)

## Problem

mushin's tagline is "boilerplate-free," and its origin concern (the 2026-07-13
refocus) was that it had become *harder to use*. Yet the core sweep→dataset flow
still opens with real ceremony — the quickstart's first sentence is literally
"Subclass `MultiRunMetricsWorkflow` and implement a static `task` method":

```python
class LRSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr, seed):
        ...
        return dict(accuracy=acc)

wf = LRSweep()
wf.run(lr=multirun([0.01, 0.1]), seed=multirun([0, 1]), working_dir="runs")
ds = wf.to_xarray()
```

Three pieces of ceremony — subclass, `@staticmethod`, separate instantiate — plus
a two-step `run()` then `to_xarray()`, just to run a function over a grid. This is
the single biggest remaining boilerplate in the *core* flow.

## Goal

A functional entry point that makes the common case a decorated function + one
call returning the labeled dataset, while keeping the class as the full-power tool:

```python
import mushin

@mushin.sweep
def experiment(lr, seed):
    ...
    return dict(accuracy=acc)

ds = experiment.run(
    lr=mushin.multirun([0.01, 0.1]),
    seed=mushin.multirun([0, 1]),
    working_dir="runs", on_error="nan",
)
```

## Non-goals

- Not replacing `MultiRunMetricsWorkflow` — the decorator is sugar *over* it; the
  class stays for advanced/subclassing use (`RobustnessCurve`, custom
  `pre_task`/`jobs_post_process`, overriding `to_xarray`).
- Not a functional wrapper for `Study` (train+compare) — that is a distinct class
  with a different shape; out of scope for this release.
- Not changing `multirun(...)` — sweep dimensions stay explicitly wrapped (needed
  to disambiguate a swept list from a single list-valued argument).

## Decisions (from brainstorming)

- **`.run()` returns the `xarray.Dataset` directly** ("dataset-first"). Resilience
  info rides on `ds.attrs` (`mushin_failures`, `provenance`), exactly as
  `to_xarray()` already stamps it. Resume is `.run(..., resume=True)`.
- **The class stays the power tool.** The handle does not duplicate the workflow's
  rich surface (`.failures` as a list, `.plot()`, `.provenance` dict, custom
  `to_xarray` options).
- **Escape hatches:** the handle exposes `.workflow` (the underlying
  `MultiRunMetricsWorkflow` instance from the most recent run) AND `.workflow_cls`
  (the synthesized subclass, for instantiating/subclassing fresh).
- **Name:** `@mushin.sweep`.

## Design

### Public surface

```python
@mushin.sweep
def experiment(lr, seed, mushin_resume=None):   # mushin_resume is optional, works free
    ...
    return dict(accuracy=acc)

# common case — one call, returns the dataset:
ds = experiment.run(lr=mushin.multirun([...]), seed=mushin.multirun([...]),
                    working_dir="runs", on_error="nan", resume=False)

# power drop-down, no rewrite:
experiment.workflow            # last-run instance -> .failures / .plot(...) / .provenance / .to_xarray(...)
experiment.workflow_cls        # the synthesized subclass -> instantiate or subclass fresh
```

`experiment` is a small `Sweep` handle. `functools.wraps(fn)` gives it the
function's `__name__`/`__doc__`/`__wrapped__` so it reads as the function.

### Mechanics

```python
def sweep(fn):
    """Turn a plain `task`-style function into a runnable sweep."""
    cls = type(fn.__name__, (MultiRunMetricsWorkflow,), {"task": staticmethod(fn)})
    cls.__module__ = fn.__module__
    cls.__qualname__ = fn.__qualname__
    return Sweep(fn, cls)


class Sweep:
    def __init__(self, fn, cls):
        functools.wraps(fn)(self)      # name/doc/__wrapped__
        # Make `fn` picklable despite the decorator shadowing its name: re-point
        # its qualname THROUGH this handle (which is findable at module.<name>) so
        # out-of-process launchers can serialize the task. See Risks.
        fn.__qualname__ = fn.__qualname__ + ".__mushin_task__"
        self.__mushin_task__ = fn
        self.workflow_cls = cls
        self.workflow = None           # last-run instance (None before first run)

    def run(self, **kwargs):
        wf = self.workflow_cls()       # fresh per run — no state leak across runs
        wf.run(**kwargs)               # wf.run splits its named options from the multirun params
        self.workflow = wf
        return wf.to_xarray()
```

- `wf.run(**kwargs)` already separates its named options (`working_dir`,
  `on_error`, `resume`, `capture_env`, `sweeper`, `launcher`, `overrides`,
  `version_base`, `task_fn_wrapper`, …) from the `**workflow_overrides` sweep
  dimensions, so `Sweep.run` forwards everything transparently — no re-listing of
  `run()`'s signature, no drift.
- **A fresh workflow instance per `run()`** avoids surprising state carryover
  (`.failures`, `.provenance`) between successive runs on the same handle.

### Free wins

- **`mushin_resume` just works.** The resume-hardening injection introspects the
  task's signature; the synthesized class's `task` *is* the user's function, so a
  decorated function that declares `mushin_resume` receives a `ResumeContext` with
  zero extra plumbing.
- **Framework-agnostic.** The task is any function returning a `dict` — torch,
  Lightning, sklearn, or pure Python — identical to the class contract.
- **Custom `to_xarray`** (e.g. `non_multirun_params_as_singleton_dims=True`) is
  reached via `experiment.workflow.to_xarray(...)` — so `.run()` stays a single
  clean call and we don't reinvent options.

### Launcher note (no regression)

The decorator has **no out-of-process/HPC disadvantage vs. the class form** —
verified: base mushin's *class*-form sweep also fails out-of-process with a plain
launcher (`PicklingError`), because mushin's dispatch is closure-based. Both forms
run on the in-process `basic` launcher today. The separate
[picklable-dispatch work](2026-07-15-picklable-dispatch-design.md) fixes
out-of-process for *both* forms; once it lands, a decorated function whose body is
module-level participates just like a class-form task.

## Backward compatibility

- Purely additive. The subclass form is untouched.
- New public name `mushin.sweep` (add to `__all__`; the `Sweep` class is internal).
- No dependency change.

## Testing

- `@mushin.sweep` on a function → `.run(a=multirun([...]), b=multirun([...]))`
  returns a correctly-labeled `xarray.Dataset` with dims `(a, b)`; equals the
  class-form dataset for the same task/grid.
- `.workflow` is the last-run instance (`.failures`, `.provenance`, `.to_xarray`
  work); `.workflow_cls` is a `MultiRunMetricsWorkflow` subclass that can be
  instantiated fresh.
- Fresh-instance-per-run: two successive `.run()`s don't leak `.failures`.
- Resilience passthrough: `.run(..., on_error="nan")` NaN-fills + `ds.attrs`
  carries `mushin_failures`; `.run(..., resume=True)` completes a prior partial.
- `mushin_resume` free win: a decorated function declaring `mushin_resume` receives
  a `ResumeContext` on a resumed cell.
- `functools.wraps`: `experiment.__name__ == "experiment"`, doc preserved.
- Framework-agnostic: a sklearn task (no torch) works.

## Risks / open questions

- **Handle vs. callable:** `experiment` is a `Sweep` handle, not itself callable as
  the original function. Acceptable (you call `.run(...)`), and `.__wrapped__`
  exposes the original if needed. Confirm no doc/example implies `experiment(...)`
  direct-call semantics.
- **`.workflow` is `None` before the first run** — document; accessing power
  features before running raises a clear `AttributeError`-style message (or returns
  None) rather than surprising behavior.
- **Name collision:** `mushin.sweep` must not clash with anything (none today).
- **[RESOLVED, iter 2] Out-of-process picklability of the functional form.** The
  decorator *shadows* the function name (`experiment` becomes the `Sweep` handle),
  so pickle-by-reference initially cannot find `fn` (`module.experiment` is the
  handle) — verified `PicklingError`. **Fix (namedtuple-style, verified):** in the
  decorator, re-point the function's qualname *through the handle* and hang it
  there —

  ```python
  fn.__qualname__ = fn.__qualname__ + ".__mushin_task__"
  self.__mushin_task__ = fn   # on the Sweep handle
  ```

  Because the handle *is* findable at `module.experiment`, pickle resolves
  `module.experiment.__mushin_task__` → `fn`. Verified: `pickle.dumps(zen(fn))`
  round-trips. No module-namespace pollution (unlike registering a top-level
  mangled name). Minor cosmetic cost: `fn`'s repr/tracebacks show the suffixed
  qualname. Net: once [picklable dispatch](2026-07-15-picklable-dispatch-design.md)
  lands, the functional form has **no out-of-process disadvantage** vs the class
  form. In-process `basic` launcher is unaffected regardless.

## Rollout

One additive PR: `src/mushin/_sweep_decorator.py` (or fold into `workflows.py`),
export `sweep` in `__init__.py` + `__all__`, tests, a quickstart/docs update
showing the functional form as the primary path (class form kept as "going
deeper"), changelog. Version bump: minor (new public API). Independent of the
picklable-dispatch PR; can ship in either order.
