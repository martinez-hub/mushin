# Public Task API (Spec 1 of 2) — Design

**Status:** Approved for planning
**Date:** 2026-06-27
**Issue/origin:** "We should have classification/detection/segmentation on our side, plus a
function where the user can add metrics or design their own tasks." Decomposed into two
specs; this is **Spec 1**.

## Context

mushin already ships three task batteries — `classification`, `segmentation`, `detection`
— wired through a **private** registry:

- `src/mushin/benchmark/_metrics.py` — `classification_battery`, `segmentation_battery`,
  `detection_battery` (factories `(num_classes, ignore_index) -> dict[str, Metric]`).
- `src/mushin/benchmark/_tasks.py` — `TaskSpec` frozen dataclass + `_TASKS` dict +
  `get_task_spec(name)`.
- `src/mushin/benchmark/compare.py` — `compare(...)` looks the task up by string name and
  already supports per-call overrides (`metrics=`, `predict_fn=`, `prob_metrics=`).

The gap: **everything reusable is private.** A user can override metrics for a single
`compare(...)` call, but cannot define a task once and reuse it by name, cannot inspect
which tasks exist, and cannot import a built-in battery to tweak it. torchmetrics spans
~14 domains (regression, audio, image-quality, retrieval, text, clustering, …); we will
never ship a first-class battery for all of them, so a **public extension seam is the
product** and the built-in batteries are curated conveniences on top of it.

This spec makes tasks first-class and registerable. It does **not** add new batteries
(that is Spec 2) and does **not** add the `update_fn` generalization hook (deferred to
Spec 2, designed against retrieval — see "Deferred").

## Goals

1. Promote the private `TaskSpec` to a public, exported `Task` dataclass.
2. Let `compare(task=...)` and `Study(task=...)` accept **either** a `Task` object **or** a
   registered string name.
3. Add an opt-in naming layer: `register_task`, `get_task`, `list_tasks`.
4. Export the three built-in batteries so users can import and tweak one.
5. Zero breaking changes to the existing `compare`/`Study` call surface.

## Non-Goals (explicit)

- **New batteries** (regression, audio, image-quality, retrieval) — Spec 2.
- **The `update_fn` hook / non-`(preds, target)` update signatures** (e.g. retrieval's
  `indexes`) — Spec 2, designed with its first real consumer.
- **Distribution-level metrics** (FID, KID, Inception Score) — these need an
  "accumulate two sets, compute once" model, not per-sample streaming. Out of scope for
  both specs for now; documented as unsupported.

## Architecture

### The `Task` primitive

Promote `TaskSpec` to a public frozen dataclass named `Task`, in `_tasks.py`:

```python
@dataclass(frozen=True)
class Task:
    battery: Callable[..., dict[str, Metric]]
    predict_fn: PredictFn
    prob_metrics: frozenset[str] = frozenset()
    requires_num_classes: bool = True
    description: str = ""
```

Changes from today's `TaskSpec`:
- Renamed `TaskSpec` -> `Task`.
- `prob_metrics` gains a default (`frozenset()`) so trivial tasks are a one-liner.
- New `description: str = ""` field, surfaced by `list_tasks()` for discoverability.
- **No `update_fn` field** (deferred — see Deferred section). Adding it later is a purely
  additive, non-breaking change to a frozen dataclass.

Keep a module-level alias `TaskSpec = Task` so any internal/external references to the old
name keep working for one release. The built-in `Task` instances gain short `description`s.

### The registry API

Three public functions over the existing `_TASKS` dict (replacing the private
`get_task_spec`):

```python
def register_task(name: str, task: Task, *, overwrite: bool = False) -> None: ...
def get_task(name: str) -> Task: ...
def list_tasks() -> dict[str, str]: ...   # {name: description}
```

- `register_task` validates: `name` is a non-empty `str`; `task` is a `Task` instance;
  `name` not already present unless `overwrite=True`. Raises `ValueError` / `TypeError`
  accordingly.
- `get_task` raises `NotImplementedError` (preserving today's behavior/message from
  `get_task_spec`) listing available names when the name is unknown.
- `list_tasks` returns a name -> description mapping (sorted), for users and docs.
- The three built-ins are registered at import time, exactly as today.

`get_task_spec` is kept as a thin deprecated alias delegating to `get_task` for one release
(internal callers migrate to `get_task`).

### `compare` / `Study` accept `Task` or `str`

`compare(task=...)` resolves the task at the top of the function:

```python
spec = task if isinstance(task, Task) else get_task(task)
```

Everything downstream (`spec.battery`, `spec.predict_fn`, `spec.prob_metrics`,
`spec.requires_num_classes`) is unchanged. The `task` parameter type widens from `str` to
`str | Task`; default stays `"classification"`.

The same one-line resolution is threaded through `Study` (`_study.py` stores `self._task`;
`_load.evaluate_checkpoints` forwards `task` to `compare`). Because resolution happens
inside `compare`, `Study` and `evaluate_checkpoints` only need their `task` type widened to
`str | Task` — no logic change.

### Public exports

From `src/mushin/benchmark/__init__.py`, additionally export:
`Task`, `register_task`, `get_task`, `list_tasks`,
`classification_battery`, `segmentation_battery`, `detection_battery`.

From `src/mushin/__init__.py`, re-export the same names at top level so
`from mushin import Task, register_task, list_tasks, classification_battery` works
(matching how `compare`/`BenchmarkResult` are surfaced today via the `benchmark` subpackage;
top-level re-export is additive).

### Extension seam for Spec 2 (design-for-extension note)

Spec 2 will add a per-task `update_fn` that owns the `metric.update(...)` call (for
retrieval's `indexes` and any non-paired signature). Spec 1 must not block this:
- Keep the metric-update dispatch in `_inference.evaluate` in **one** clearly localized
  place (it already is: the `prob_metrics`-based branch). Do not scatter it. This is the
  single seam Spec 2 will parameterize by `update_fn`. No code change required in Spec 1 —
  just a comment marking it as the extension point so Spec 2 doesn't rework Spec 1.

## Data flow (unchanged)

`compare(methods, data, task)` -> resolve `Task` -> build battery (or use `metrics=`
override) -> `evaluate(model, data, battery, predict_fn, prob_metrics, device)` per model
-> `to_dataset` -> `compare_methods` (Holm-corrected significance) -> `BenchmarkResult`.
The only new step is "resolve `Task`," a no-op when a string is passed.

## Error handling

- `register_task("", task)` -> `ValueError` (empty name).
- `register_task("classification", task)` without `overwrite=True` -> `ValueError`
  (already registered).
- `register_task("x", not_a_task)` -> `TypeError`.
- `compare(task="bogus")` -> `NotImplementedError` listing available tasks (unchanged
  message contract).
- `compare(task=Task(...))` with a `Task` whose `requires_num_classes=True` and no
  `num_classes` and no `metrics=` -> existing `ValueError` ("`num_classes` is required …").

## Testing

New `tests/test_benchmark/test_tasks.py` (or extend the existing benchmark tests):

1. `Task` is importable from `mushin` and from `mushin.benchmark`; is frozen.
2. `list_tasks()` returns the three built-ins with non-empty descriptions.
3. `register_task("toy", Task(...))` then `compare(task="toy", ...)` runs end-to-end on a
   tiny synthetic battery and produces a `BenchmarkResult` with the expected data vars.
4. `compare(task=Task(...), ...)` (inline object, never registered) runs end-to-end.
5. Re-registering a name without `overwrite` raises `ValueError`; with `overwrite=True`
   succeeds and replaces.
6. `register_task` with a bad name (`""`) or non-`Task` raises the right error type.
7. `get_task("bogus")` raises `NotImplementedError` listing available names.
8. Backward-compat: `compare(task="classification", num_classes=...)` still works
   identically (regression guard); `TaskSpec` alias still importable internally.
9. `Study(task=Task(...))` / `evaluate_checkpoints(task=Task(...))` resolve correctly.

All tests use tiny synthetic tensors/models (no real data, no GPU), consistent with the
existing hermetic benchmark tests.

## Docs

- Update `docs/guides/custom.md`: add a "Define a reusable task" section showing `Task`,
  `register_task`, `list_tasks`, and importing/tweaking a built-in battery. Keep the
  existing per-call `metrics=`/`predict_fn=` section as the quick path.
- Add a `changes/+public-task-api.added.md` towncrier fragment.
- Reference docs (`reference/benchmark.md`) pick up the new public symbols via mkdocstrings.

## Build order

This is Spec 1 of 2. Spec 2 (new batteries: regression, image-quality paired, audio,
retrieval; plus the `update_fn` hook designed against retrieval) is a separate
spec -> plan -> implementation cycle that builds on this one.
