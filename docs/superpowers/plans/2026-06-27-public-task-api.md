# Public Task API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make mushin's tasks first-class and user-registerable — promote the private `TaskSpec` to a public `Task`, add `register_task`/`get_task`/`list_tasks`, let `compare`/`Study` accept a `Task` object or a string name, and export the built-in batteries.

**Architecture:** All changes are additive over the existing private registry in `src/mushin/benchmark/_tasks.py`. `Task` is a renamed/extended `TaskSpec` (keeps a `TaskSpec = Task` alias). `compare` resolves `task` to a `Task` with a one-liner (`task if isinstance(task, Task) else get_task(task)`); everything downstream is unchanged. New public symbols are re-exported from `mushin.benchmark` and `mushin`.

**Tech Stack:** Python 3.9+, torchmetrics, pytest, towncrier, uv. Tests are hermetic (tiny synthetic tensors/models, no GPU, no real data).

**Reference (read once before starting):**
- Spec: `docs/superpowers/specs/2026-06-27-public-task-api-design.md`
- Current registry: `src/mushin/benchmark/_tasks.py` (the `TaskSpec` dataclass, `_TASKS` dict, `get_task_spec`)
- `compare`: `src/mushin/benchmark/compare.py:18-71` (task lookup at line 45)
- `evaluate` + metric-update dispatch: `src/mushin/benchmark/_inference.py:67-100` (the seam at line 95)
- Existing task tests: `tests/test_benchmark/test_tasks.py`
- Test idioms (synthetic loader/model): `tests/test_benchmark/test_compare.py:1-60`

**Conventions to follow:**
- Every source file starts with the two-line MIT copyright header (see any existing file).
- Run lint/format after edits: `uv run ruff check <paths>` and `uv run ruff format <paths>`.
- Commit messages: imperative mood, **no Claude attribution / no `Co-Authored-By` trailer**.
- Run tests with `uv run pytest`.

---

### Task 1: Promote `TaskSpec` to a public `Task` dataclass

Rename the frozen dataclass to `Task`, give `prob_metrics` a default and add a `description` field, and keep a `TaskSpec = Task` backward-compat alias. Built-in specs gain descriptions. Do **not** add `update_fn` (deferred to Spec 2).

**Files:**
- Modify: `src/mushin/benchmark/_tasks.py`
- Test: `tests/test_benchmark/test_tasks.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmark/test_tasks.py`:

```python
def test_task_is_public_and_frozen():
    from dataclasses import FrozenInstanceError

    from mushin.benchmark import Task

    t = Task(
        battery=lambda num_classes, ignore_index=None: {},
        predict_fn=lambda model, x: (x, x),
    )
    # prob_metrics and requires_num_classes have defaults; description too
    assert t.prob_metrics == frozenset()
    assert t.requires_num_classes is True
    assert t.description == ""
    with pytest.raises(FrozenInstanceError):
        t.description = "nope"


def test_taskspec_alias_still_works():
    from mushin.benchmark import Task
    from mushin.benchmark._tasks import TaskSpec

    assert TaskSpec is Task


def test_builtins_have_descriptions():
    from mushin.benchmark._tasks import _TASKS

    for name, spec in _TASKS.items():
        assert spec.description, f"{name} should have a non-empty description"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_tasks.py::test_task_is_public_and_frozen tests/test_benchmark/test_tasks.py::test_taskspec_alias_still_works tests/test_benchmark/test_tasks.py::test_builtins_have_descriptions -v`
Expected: FAIL — `ImportError: cannot import name 'Task' from 'mushin.benchmark'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/benchmark/_tasks.py`, replace the `TaskSpec` class and `_TASKS` dict (lines 21-46) with:

```python
@dataclass(frozen=True)
class Task:
    """A reusable evaluation task: a metric ``battery`` factory, a ``predict_fn``
    that extracts ``(predictions, probabilities)`` from a model, the subset of
    metric names that consume probabilities, and whether the battery needs
    ``num_classes``. ``description`` is shown by :func:`list_tasks`."""

    battery: Callable[..., dict[str, Metric]]
    predict_fn: PredictFn
    prob_metrics: frozenset[str] = frozenset()
    requires_num_classes: bool = True
    description: str = ""


# Backward-compat alias (deprecated; removed in a future release).
TaskSpec = Task


_TASKS: dict[str, Task] = {
    "classification": Task(
        classification_battery,
        default_classification_predict_fn,
        frozenset({"auroc", "ece"}),
        description="Multiclass classification (accuracy, f1, precision, "
        "recall, auroc, ece).",
    ),
    "segmentation": Task(
        segmentation_battery,
        default_segmentation_predict_fn,
        frozenset(),
        description="Semantic segmentation (miou, dice, pixel_acc, precision, "
        "recall).",
    ),
    "detection": Task(
        detection_battery,
        default_detection_predict_fn,
        frozenset(),
        requires_num_classes=False,
        description="Object detection (mAP/mAR family + IoU variants).",
    ),
}
```

Leave `get_task_spec` as-is for now (Task 2 replaces it).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_tasks.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones that use `get_task_spec`).

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/_tasks.py tests/test_benchmark/test_tasks.py
uv run ruff format src/mushin/benchmark/_tasks.py tests/test_benchmark/test_tasks.py
git add src/mushin/benchmark/_tasks.py tests/test_benchmark/test_tasks.py
git commit -m "feat: promote TaskSpec to public Task dataclass with description"
```

---

### Task 2: Registry API — `register_task`, `get_task`, `list_tasks`

Add the three public registry functions over `_TASKS`. `get_task` replaces `get_task_spec` (which becomes a thin deprecated alias). Built-ins are already registered via the `_TASKS` literal in Task 1.

**Files:**
- Modify: `src/mushin/benchmark/_tasks.py`
- Test: `tests/test_benchmark/test_tasks.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmark/test_tasks.py`:

```python
def _toy_task():
    from mushin.benchmark import Task

    return Task(
        battery=lambda num_classes, ignore_index=None: {},
        predict_fn=lambda model, x: (x, x),
        description="toy",
    )


def test_list_tasks_returns_builtins_with_descriptions():
    from mushin.benchmark import list_tasks

    tasks = list_tasks()
    assert set(tasks) >= {"classification", "segmentation", "detection"}
    assert all(desc for desc in tasks.values())


def test_register_and_get_task():
    from mushin.benchmark import get_task, register_task

    register_task("toy_reg", _toy_task())
    assert get_task("toy_reg").description == "toy"


def test_register_duplicate_requires_overwrite():
    from mushin.benchmark import register_task

    register_task("toy_dup", _toy_task())
    with pytest.raises(ValueError, match="already registered"):
        register_task("toy_dup", _toy_task())
    # overwrite=True replaces it
    register_task("toy_dup", _toy_task(), overwrite=True)


def test_register_validates_inputs():
    from mushin.benchmark import register_task

    with pytest.raises(ValueError, match="non-empty"):
        register_task("", _toy_task())
    with pytest.raises(TypeError, match="Task"):
        register_task("bad", object())


def test_get_task_unknown_raises():
    from mushin.benchmark import get_task

    with pytest.raises(NotImplementedError, match="not supported"):
        get_task("nope_not_a_task")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_tasks.py -k "register or list_tasks or get_task_unknown" -v`
Expected: FAIL — `ImportError: cannot import name 'register_task' from 'mushin.benchmark'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/benchmark/_tasks.py`, replace `get_task_spec` (the current lines 49-54) with:

```python
def register_task(name: str, task: Task, *, overwrite: bool = False) -> None:
    """Register ``task`` under ``name`` so ``compare(task=name)`` and
    ``Study(task=name)`` can look it up. Set ``overwrite=True`` to replace an
    existing entry."""
    if not isinstance(name, str) or not name:
        raise ValueError("`name` must be a non-empty string")
    if not isinstance(task, Task):
        raise TypeError(f"`task` must be a Task, got {type(task).__name__}")
    if name in _TASKS and not overwrite:
        raise ValueError(
            f"task {name!r} is already registered; pass overwrite=True to replace it"
        )
    _TASKS[name] = task


def get_task(task: str) -> Task:
    """Look up a registered task by name."""
    if task not in _TASKS:
        raise NotImplementedError(
            f"task={task!r} is not supported; choose from {sorted(_TASKS)}"
        )
    return _TASKS[task]


def list_tasks() -> dict[str, str]:
    """Return ``{name: description}`` for every registered task, name-sorted."""
    return {name: _TASKS[name].description for name in sorted(_TASKS)}


# Backward-compat alias (deprecated; use get_task).
get_task_spec = get_task
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_tasks.py -v`
Expected: PASS (new tests plus the pre-existing `get_task_spec` tests via the alias).

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/_tasks.py tests/test_benchmark/test_tasks.py
uv run ruff format src/mushin/benchmark/_tasks.py tests/test_benchmark/test_tasks.py
git add src/mushin/benchmark/_tasks.py tests/test_benchmark/test_tasks.py
git commit -m "feat: add register_task/get_task/list_tasks registry API"
```

---

### Task 3: `compare` accepts a `Task` object or a string name

Resolve `task` to a `Task` at the top of `compare`, and migrate its internal call from `get_task_spec` to `get_task`. Widen the `task` parameter type to `str | Task`.

**Files:**
- Modify: `src/mushin/benchmark/compare.py:15` (import) and `:18-58` (signature + resolution)
- Test: `tests/test_benchmark/test_compare.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmark/test_compare.py` (reuses `_loader`/`_Perfect` already in that file):

```python
def test_compare_accepts_task_object():
    from torchmetrics.classification import MulticlassAccuracy

    from mushin.benchmark import Task, compare

    data = _loader(seed=0)
    good = [_Perfect(data) for _ in range(3)]
    bad = [torch.nn.Linear(4, 3) for _ in range(3)]

    task = Task(
        battery=lambda num_classes, ignore_index=None: {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro")
        },
        predict_fn=lambda model, x: (model(x).argmax(dim=-1), model(x).softmax(dim=-1)),
        description="acc-only classification",
    )
    result = compare(
        methods={"good": good, "bad": bad},
        data=data,
        task=task,
        num_classes=3,
    )
    assert isinstance(result, BenchmarkResult)
    assert "accuracy" in result.data


def test_compare_accepts_registered_task_name():
    from torchmetrics.classification import MulticlassAccuracy

    from mushin.benchmark import Task, compare, register_task

    register_task(
        "acc_only",
        Task(
            battery=lambda num_classes, ignore_index=None: {
                "accuracy": MulticlassAccuracy(
                    num_classes=num_classes, average="micro"
                )
            },
            predict_fn=lambda model, x: (
                model(x).argmax(dim=-1),
                model(x).softmax(dim=-1),
            ),
        ),
        overwrite=True,
    )
    data = _loader(seed=1)
    models = [_Perfect(data) for _ in range(3)]
    result = compare(methods={"m": models}, data=data, task="acc_only", num_classes=3)
    assert "accuracy" in result.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_compare.py::test_compare_accepts_task_object -v`
Expected: FAIL — `NotImplementedError` (a `Task` object is passed where a string name is expected, so `get_task_spec(task)` rejects it / errors).

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/benchmark/compare.py`:

Change the import on line 15 from:

```python
from ._tasks import get_task_spec
```

to:

```python
from ._tasks import Task, get_task
```

Change the signature line 21 from:

```python
    task: str = "classification",
```

to:

```python
    task: str | Task = "classification",
```

Change the resolution on line 45 from:

```python
    spec = get_task_spec(task)
```

to:

```python
    spec = task if isinstance(task, Task) else get_task(task)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_compare.py -v`
Expected: PASS (new tests plus all pre-existing compare tests).

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/compare.py tests/test_benchmark/test_compare.py
uv run ruff format src/mushin/benchmark/compare.py tests/test_benchmark/test_compare.py
git add src/mushin/benchmark/compare.py tests/test_benchmark/test_compare.py
git commit -m "feat: compare() accepts a Task object or a registered name"
```

---

### Task 4: `Study` accepts a `Task` object or a string name

Widen the `task` type on `Study.__init__`, `Study.from_checkpoints`, and `evaluate_checkpoints` to `str | Task`. No logic change — resolution already happens inside `compare` (Task 3).

**Files:**
- Modify: `src/mushin/study/_study.py:41` and `:61` (the two `task: str = "classification"` params)
- Modify: `src/mushin/study/_load.py:13-22` (the `task` param type)
- Test: `tests/test_study/test_study.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_study/test_study.py` (it uses tiny synthetic checkpoints via an in-memory `load_fn`):

```python
def test_study_from_checkpoints_accepts_task_object(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchmetrics.classification import MulticlassAccuracy

    from mushin import Study
    from mushin.benchmark import Task

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 4, generator=g)
    y = torch.randint(0, 3, (32,), generator=g)
    data = DataLoader(TensorDataset(x, y), batch_size=16)

    # load_fn ignores the path and returns a fresh linear model
    def load_fn(_path):
        return torch.nn.Linear(4, 3)

    task = Task(
        battery=lambda num_classes, ignore_index=None: {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro")
        },
        predict_fn=lambda model, x: (
            model(x).argmax(dim=-1),
            model(x).softmax(dim=-1),
        ),
    )
    study = Study.from_checkpoints(
        checkpoints={"m": ["a.ckpt", "b.ckpt", "c.ckpt"]},
        load_fn=load_fn,
        data=data,
        num_classes=3,
        task=task,
    )
    result = study.run()
    assert "accuracy" in result.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_study/test_study.py -k accepts_task_object -v`
Expected: FAIL — a type-checker/`NotImplementedError` path or, if `task` is forwarded fine, it actually passes. If it already passes, that confirms only the type annotations need widening (still do Step 3 for the annotations and keep the test as a regression guard).

- [ ] **Step 3: Write minimal implementation**

In `src/mushin/study/_study.py`, change **both** occurrences of:

```python
        task: str = "classification",
```

to:

```python
        task: "str | Task" = "classification",
```

and add the import near the top (after line 11, the existing `from mushin.benchmark import BenchmarkResult`):

```python
from mushin.benchmark import BenchmarkResult, Task
```

(Combine into the existing import line rather than adding a second one.)

In `src/mushin/study/_load.py`, change the `evaluate_checkpoints` signature param (line 17) from:

```python
    task: str,
```

to:

```python
    task,  # str | Task — resolved inside compare()
```

(Leave the body unchanged; it forwards `task` to `compare`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_study/test_study.py -v`
Expected: PASS (new test plus pre-existing study tests).

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/study/_study.py src/mushin/study/_load.py tests/test_study/test_study.py
uv run ruff format src/mushin/study/_study.py src/mushin/study/_load.py tests/test_study/test_study.py
git add src/mushin/study/_study.py src/mushin/study/_load.py tests/test_study/test_study.py
git commit -m "feat: Study accepts a Task object or a registered name"
```

---

### Task 5: Export public symbols from `mushin.benchmark` and `mushin`

Surface `Task`, `register_task`, `get_task`, `list_tasks`, and the three batteries from `mushin.benchmark`, and re-export them at the top level `mushin`.

**Files:**
- Modify: `src/mushin/benchmark/__init__.py`
- Modify: `src/mushin/__init__.py`
- Test: `tests/test_benchmark/test_import.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmark/test_import.py`:

```python
def test_public_task_api_exports():
    import mushin
    from mushin import (
        Task,
        classification_battery,
        detection_battery,
        get_task,
        list_tasks,
        register_task,
        segmentation_battery,
    )
    from mushin import benchmark

    for name in [
        "Task",
        "register_task",
        "get_task",
        "list_tasks",
        "classification_battery",
        "segmentation_battery",
        "detection_battery",
    ]:
        assert name in mushin.__all__, f"{name} missing from mushin.__all__"
        assert name in benchmark.__all__, f"{name} missing from benchmark.__all__"
    assert callable(classification_battery)
    assert "classification" in list_tasks()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_import.py::test_public_task_api_exports -v`
Expected: FAIL — `ImportError: cannot import name 'Task' from 'mushin.benchmark'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/mushin/benchmark/__init__.py` with:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard evaluation protocols: run a benchmark, get a labeled dataset back."""

from ._metrics import (
    classification_battery,
    detection_battery,
    segmentation_battery,
)
from ._result import BenchmarkResult
from ._tasks import Task, get_task, list_tasks, register_task
from .compare import compare

__all__ = [
    "compare",
    "BenchmarkResult",
    "Task",
    "register_task",
    "get_task",
    "list_tasks",
    "classification_battery",
    "segmentation_battery",
    "detection_battery",
]
```

In `src/mushin/__init__.py`, add an import of the benchmark symbols. Place it **before** the `from .study import Study` line (study imports from benchmark, so importing benchmark first respects the existing ordering note). Insert after the `from ._utils import ...` line:

```python
from .benchmark import (
    BenchmarkResult,
    Task,
    classification_battery,
    compare,
    detection_battery,
    get_task,
    list_tasks,
    register_task,
    segmentation_battery,
)
```

Then extend `__all__` in `src/mushin/__init__.py` by adding these entries (keep the existing ones):

```python
    "compare",
    "BenchmarkResult",
    "Task",
    "register_task",
    "get_task",
    "list_tasks",
    "classification_battery",
    "segmentation_battery",
    "detection_battery",
```

- [ ] **Step 2b: Guard against circular import**

Run: `uv run python -c "import mushin; print(mushin.Task, mushin.list_tasks())"`
Expected: prints the `Task` class and the dict of built-in tasks, with **no** `ImportError`/`ImportError: partially initialized module`. If a circular import appears, move the `from .benchmark import ...` line to the very top of the import block (benchmark depends only on torch/torchmetrics, not on mushin top-level).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_import.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_import.py
uv run ruff format src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_import.py
git add src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_import.py
git commit -m "feat: export Task API and batteries from mushin and mushin.benchmark"
```

---

### Task 6: Mark the extension seam for Spec 2

Add a one-line comment at the metric-update dispatch in `evaluate` marking it as the single point Spec 2's `update_fn` will parameterize. No behavior change. (Per the spec's "design-for-extension" note.)

**Files:**
- Modify: `src/mushin/benchmark/_inference.py:94-95`

- [ ] **Step 1: Add the comment**

In `src/mushin/benchmark/_inference.py`, change the inner update loop (lines 94-95) from:

```python
            for name, metric in battery.items():
                metric.update(probs if name in prob_metrics else preds, y)
```

to:

```python
            for name, metric in battery.items():
                # Extension seam: a future per-Task ``update_fn`` (Spec 2) will
                # own this call to support non-(preds, target) signatures such as
                # retrieval's ``indexes``. Keep the dispatch here, in one place.
                metric.update(probs if name in prob_metrics else preds, y)
```

- [ ] **Step 2: Verify nothing broke**

Run: `uv run pytest tests/test_benchmark/test_inference.py -v`
Expected: PASS (comment-only change).

- [ ] **Step 3: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/_inference.py
uv run ruff format src/mushin/benchmark/_inference.py
git add src/mushin/benchmark/_inference.py
git commit -m "docs: mark the metric-update dispatch as the Spec 2 update_fn seam"
```

---

### Task 7: Documentation + changelog

Add a "Define a reusable task" section to the custom guide and a towncrier fragment.

**Files:**
- Modify: `docs/guides/custom.md`
- Create: `changes/+public-task-api.added.md`

- [ ] **Step 1: Add the guide section**

Append to `docs/guides/custom.md` (after the existing per-call `metrics=`/`predict_fn=` content):

````markdown
## Define a reusable task

The per-call `metrics=` / `predict_fn=` overrides are the quick path. To reuse a
configuration across many `compare(...)` calls, build a `Task` and (optionally)
register it under a name:

```python
from torchmetrics.classification import MulticlassAccuracy

from mushin import Task, compare, register_task, list_tasks

acc_only = Task(
    battery=lambda num_classes, ignore_index=None: {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
    },
    predict_fn=lambda model, x: (model(x).argmax(-1), model(x).softmax(-1)),
    prob_metrics=frozenset(),          # which metric names consume probabilities
    description="accuracy-only classification",
)

# Use it inline (no global state):
compare(methods=..., data=..., task=acc_only, num_classes=3)

# Or name it once and reuse by string:
register_task("acc_only", acc_only)
compare(methods=..., data=..., task="acc_only", num_classes=3)

list_tasks()   # {"classification": "...", ..., "acc_only": "accuracy-only ..."}
```

You can also import a built-in battery and tweak it:

```python
from mushin import classification_battery

battery = classification_battery(num_classes=10)
del battery["ece"]                     # drop a metric you do not want
compare(methods=..., data=..., metrics=battery)
```

torchmetrics covers many more domains (regression, audio, image quality,
retrieval, …). Any of them works through a `Task`: put the relevant
`torchmetrics.Metric` instances in the `battery` and return the right tensors
from `predict_fn`. Distribution-level metrics (FID, KID, Inception Score) are not
supported by the streaming `compare` loop.
````

- [ ] **Step 2: Create the changelog fragment**

Create `changes/+public-task-api.added.md` with:

```markdown
Public task API: `Task` dataclass plus `register_task`, `get_task`, and
`list_tasks` make evaluation tasks first-class and reusable. `compare(...)` and
`Study(...)` now accept either a `Task` object or a registered task name, and the
built-in batteries (`classification_battery`, `segmentation_battery`,
`detection_battery`) are exported from `mushin`.
```

- [ ] **Step 3: Verify docs build**

Run: `uv run mkdocs build --strict`
Expected: builds with no warnings/errors.

- [ ] **Step 4: Commit**

```bash
git add docs/guides/custom.md changes/+public-task-api.added.md
git commit -m "docs: reusable-task guide section and changelog fragment"
```

---

### Task 8: Full-suite verification

Run the whole suite and linters to confirm no regressions across benchmark, study, and imports.

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: all pass; deselected `real_data`/`cluster` as usual. No new failures vs. `main`.

- [ ] **Step 2: Lint + format check + spelling**

```bash
uv run ruff check .
uv run ruff format --check .
uv run codespell
```
Expected: all clean.

- [ ] **Step 3: Import smoke test**

Run: `uv run python -c "import mushin; print(sorted(mushin.list_tasks()))"`
Expected: `['classification', 'detection', 'segmentation']` (plus any registered during the session — a fresh process shows only the three built-ins).

- [ ] **Step 4: Final commit (if any formatting changed)**

```bash
git add -A
git commit -m "chore: formatting/lint pass for public Task API" || echo "nothing to commit"
```

---

## Done criteria

- `from mushin import Task, register_task, get_task, list_tasks, classification_battery` works.
- `compare(task=Task(...))` and `compare(task="registered_name")` both run end-to-end.
- `Study(task=Task(...))` runs end-to-end.
- `TaskSpec`/`get_task_spec` aliases still importable (no internal breakage).
- Full suite green; ruff/format/codespell/mkdocs `--strict` clean.
- Extension seam comment present in `_inference.evaluate`.

Spec 2 (regression/image-quality/audio/retrieval batteries + the `update_fn` hook against retrieval) is a separate spec → plan → implementation cycle building on this.
