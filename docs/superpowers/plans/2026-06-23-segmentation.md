# Segmentation support for `compare` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `task="segmentation"` to `mushin.benchmark.compare` (and `Study`) via a task registry, refactoring the shared eval step to stream metrics per batch.

**Architecture:** A `TaskSpec` registry maps each task to its battery factory, predict_fn, and prob-metric names. A streaming `evaluate(model, data, battery, predict_fn, prob_metrics, device)` replaces the old collect-then-compute path (`run_inference` + `compute_metrics`). A new `segmentation_battery` (torchmetrics, all confusion-matrix metrics) and `default_segmentation_predict_fn` slot in. `compare` dispatches via the registry and gains `ignore_index`; `Study` forwards it.

**Tech Stack:** torchmetrics (metrics), torch, the existing `compare`/`Study` machinery, pytest.

**Governance:** `main` is branch-protected. Work on a branch (e.g. `feat-segmentation`), open a PR. **No AI/Claude attribution in commits.** Use **uv**. Repo is at the current checkout.

**Spec:** `docs/superpowers/specs/2026-06-23-segmentation-design.md`.

---

## Verified API facts (rely on these — confirmed against the installed torchmetrics)

- Segmentation metrics over **integer class-label** preds `(N, H, W)` and targets
  `(N, H, W)`: `MulticlassJaccardIndex` (mIoU), `MulticlassF1Score(average="macro")`
  (**= Dice**, portable across versions), `MulticlassAccuracy(average="micro")`
  (pixel accuracy), `MulticlassPrecision/Recall(average="macro")`. All give `1.0`
  on perfect preds; all accept `ignore_index`.
- **Streaming equals one-shot**: `metric.update(batch)` per batch then
  `metric.compute()` gives the identical result to one full call — confirmed.

## File structure

- Create: `src/mushin/benchmark/_tasks.py` — `TaskSpec` + `_TASKS` + `get_task_spec`.
- Modify: `src/mushin/benchmark/_metrics.py` — add `segmentation_battery`; give
  `classification_battery` an (ignored) `ignore_index` kwarg; remove the now-dead
  `compute_metrics` + `_PROB_METRICS`.
- Modify: `src/mushin/benchmark/_predict.py` — add `default_segmentation_predict_fn`.
- Modify: `src/mushin/benchmark/_inference.py` — add streaming `evaluate`; remove
  `run_inference`.
- Modify: `src/mushin/benchmark/compare.py` — dispatch via registry; use
  `evaluate`; add `ignore_index`.
- Modify: `src/mushin/study/_study.py`, `src/mushin/study/_load.py` — `ignore_index`
  pass-through.
- Tests under `tests/test_benchmark/` and `tests/test_study/`.

---

### Task 1: Segmentation metric battery

**Files:**
- Modify: `src/mushin/benchmark/_metrics.py`
- Test: `tests/test_benchmark/test_metrics.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_benchmark/test_metrics.py`:

```python
def test_segmentation_battery_perfect_masks():
    from mushin.benchmark._metrics import compute_battery, segmentation_battery

    battery = segmentation_battery(num_classes=3)
    assert set(battery) == {"miou", "dice", "pixel_acc", "precision", "recall"}
    preds = torch.randint(0, 3, (2, 8, 8))
    out = compute_battery(battery, preds, preds, prob_metrics=frozenset())
    assert out["miou"] == 1.0
    assert out["dice"] == 1.0
    assert out["pixel_acc"] == 1.0


def test_segmentation_battery_ignore_index():
    from mushin.benchmark._metrics import compute_battery, segmentation_battery

    battery = segmentation_battery(num_classes=3, ignore_index=255)
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    tgt = torch.zeros(1, 4, 4, dtype=torch.long)
    tgt[0, 0, 0] = 255  # one void pixel, excluded
    out = compute_battery(battery, pred, tgt, prob_metrics=frozenset())
    assert out["pixel_acc"] == 1.0
```

NOTE: `compute_battery` is a tiny one-shot helper this task adds (used by tests
and reused by the streaming evaluator in Task 3). It resets, updates once, and
computes.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_metrics.py -k segmentation -q`
Expected: FAIL with `ImportError: cannot import name 'segmentation_battery'`.

- [ ] **Step 3: Implement** — replace the body of `src/mushin/benchmark/_metrics.py` with the following. This KEEPS the existing `compute_metrics` and `_PROB_METRICS` (still imported by `compare.py` until Task 5) and ADDS `segmentation_battery`, `compute_battery`, and the `ignore_index` kwarg on `classification_battery`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Metric batteries (classification + segmentation), delegated to torchmetrics."""

from __future__ import annotations

from collections.abc import Collection
from typing import Optional

import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

# Metrics that require probabilities rather than hard class predictions.
# (Used by the legacy compute_metrics; removed in Task 5 once compare streams.)
_PROB_METRICS = frozenset({"auroc", "ece"})


def classification_battery(
    num_classes: int, ignore_index: Optional[int] = None
) -> dict[str, Metric]:
    """The standard multiclass classification battery. ``ignore_index`` is
    accepted for a uniform task interface but is not applied here (the battery's
    AUROC/ECE do not support it)."""
    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "auroc": MulticlassAUROC(num_classes=num_classes),
        "ece": MulticlassCalibrationError(num_classes=num_classes),
    }


def segmentation_battery(
    num_classes: int, ignore_index: Optional[int] = None
) -> dict[str, Metric]:
    """Semantic-segmentation battery over per-pixel class labels. ``dice`` is the
    macro F1 (the Dice coefficient); all metrics are confusion-matrix based, so
    streaming evaluation uses O(C^2) memory."""
    return {
        "miou": MulticlassJaccardIndex(num_classes, ignore_index=ignore_index),
        "dice": MulticlassF1Score(
            num_classes, average="macro", ignore_index=ignore_index
        ),
        "pixel_acc": MulticlassAccuracy(
            num_classes, average="micro", ignore_index=ignore_index
        ),
        "precision": MulticlassPrecision(
            num_classes, average="macro", ignore_index=ignore_index
        ),
        "recall": MulticlassRecall(
            num_classes, average="macro", ignore_index=ignore_index
        ),
    }


def compute_battery(
    battery: dict[str, Metric],
    preds: torch.Tensor,
    targets: torch.Tensor,
    prob_metrics: Collection[str],
    probs: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """One-shot metric computation: reset, update once, compute. Metrics named in
    ``prob_metrics`` are fed ``probs`` (required if non-empty); the rest ``preds``."""
    out: dict[str, float] = {}
    for name, metric in battery.items():
        metric.reset()
        inp = probs if name in prob_metrics else preds
        out[name] = float(metric(inp, targets))
    return out


def compute_metrics(
    preds: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor,
    battery: dict[str, Metric],
) -> dict[str, float]:
    """Legacy one-shot helper (removed in Task 5). Resets every metric first."""
    out: dict[str, float] = {}
    for name, metric in battery.items():
        metric.reset()
        inp = probs if name in _PROB_METRICS else preds
        out[name] = float(metric(inp, targets))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_metrics.py -k segmentation -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_metrics.py tests/test_benchmark/test_metrics.py
git commit -m "Add segmentation metric battery"
```

---

### Task 2: Segmentation predict function

**Files:**
- Modify: `src/mushin/benchmark/_predict.py`
- Test: `tests/test_benchmark/test_predict.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_benchmark/test_predict.py`:

```python
def test_segmentation_predict_returns_pixel_preds_and_probs():
    import torch

    from mushin.benchmark._predict import default_segmentation_predict_fn

    class Seg(torch.nn.Module):
        def forward(self, x):  # x: (N, 1, H, W) -> logits (N, C, H, W)
            return torch.randn(x.shape[0], 3, x.shape[2], x.shape[3])

    x = torch.randn(2, 1, 8, 8)
    preds, probs = default_segmentation_predict_fn(Seg(), x)
    assert preds.shape == (2, 8, 8)
    assert probs.shape == (2, 3, 8, 8)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, 8, 8), atol=1e-5)
    assert torch.equal(preds, probs.argmax(dim=1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_predict.py -k segmentation -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement** — append to `src/mushin/benchmark/_predict.py`:

```python
def default_segmentation_predict_fn(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a segmentation model on ``x`` and return ``(preds, probs)``.

    Assumes ``model(x)`` returns per-pixel logits of shape ``(N, C, H, W)``.
    ``probs`` is the softmax over the channel dim; ``preds`` is its argmax,
    shape ``(N, H, W)``.
    """
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    return preds, probs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_predict.py -k segmentation -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_predict.py tests/test_benchmark/test_predict.py
git commit -m "Add segmentation predict function"
```

---

### Task 3: Streaming evaluator

**Files:**
- Modify: `src/mushin/benchmark/_inference.py`
- Test: `tests/test_benchmark/test_inference.py`

This adds `evaluate` alongside the existing `run_inference` (which Task 5 removes).

- [ ] **Step 1: Write the failing test** — append to `tests/test_benchmark/test_inference.py`:

```python
def test_evaluate_streams_and_matches_one_shot():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery, compute_battery
    from mushin.benchmark._predict import default_classification_predict_fn

    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 4, generator=g)
    y = torch.randint(0, 3, (20,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = torch.nn.Linear(4, 3)

    battery = classification_battery(3)
    streamed = evaluate(
        model, loader, battery, default_classification_predict_fn,
        prob_metrics=frozenset({"auroc", "ece"}),
    )
    # one-shot reference on the whole set
    with torch.no_grad():
        preds, probs = default_classification_predict_fn(model, x)
    one_shot = compute_battery(
        classification_battery(3), preds, y, frozenset({"auroc", "ece"}), probs=probs
    )
    assert streamed.keys() == one_shot.keys()
    for k in streamed:
        assert abs(streamed[k] - one_shot[k]) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_inference.py -k evaluate -q`
Expected: FAIL with `ImportError: cannot import name 'evaluate'`.

- [ ] **Step 3: Implement** — add to `src/mushin/benchmark/_inference.py` (keep the
existing `run_inference` and `PredictFn` for now). Add these imports at the top if
not present: `from collections.abc import Collection`, `from torchmetrics import Metric`.

```python
def evaluate(
    model: torch.nn.Module,
    data: Iterable,
    battery: dict[str, Metric],
    predict_fn: PredictFn,
    prob_metrics: Collection[str],
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Stream ``data`` through ``model``, updating each metric in ``battery`` per
    batch, and return ``{name: value}``. Metrics named in ``prob_metrics`` are fed
    probabilities; the rest hard predictions. O(C^2) memory for confusion-matrix
    metrics."""
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")

    model = model.to(device)
    model.eval()
    for metric in battery.values():
        metric.reset()
        metric.to(device)

    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            preds, probs = predict_fn(model, x)
            for name, metric in battery.items():
                metric.update(probs if name in prob_metrics else preds, y)

    return {name: float(metric.compute()) for name, metric in battery.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_inference.py -k evaluate -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_inference.py tests/test_benchmark/test_inference.py
git commit -m "Add streaming evaluate"
```

---

### Task 4: Task registry

**Files:**
- Create: `src/mushin/benchmark/_tasks.py`
- Test: `tests/test_benchmark/test_tasks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark/test_tasks.py
import pytest

from mushin.benchmark._tasks import get_task_spec


def test_known_tasks():
    assert get_task_spec("classification").prob_metrics == frozenset({"auroc", "ece"})
    assert get_task_spec("segmentation").prob_metrics == frozenset()
    assert callable(get_task_spec("segmentation").battery)
    assert callable(get_task_spec("segmentation").predict_fn)


def test_unknown_task_raises():
    with pytest.raises(NotImplementedError, match="not supported"):
        get_task_spec("detection")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_tasks.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — create `src/mushin/benchmark/_tasks.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Registry mapping a task name to its battery, predict_fn, and prob-metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torchmetrics import Metric

from ._inference import PredictFn
from ._metrics import classification_battery, segmentation_battery
from ._predict import (
    default_classification_predict_fn,
    default_segmentation_predict_fn,
)


@dataclass(frozen=True)
class TaskSpec:
    battery: Callable[..., dict[str, Metric]]
    predict_fn: PredictFn
    prob_metrics: frozenset[str]


_TASKS: dict[str, TaskSpec] = {
    "classification": TaskSpec(
        classification_battery,
        default_classification_predict_fn,
        frozenset({"auroc", "ece"}),
    ),
    "segmentation": TaskSpec(
        segmentation_battery,
        default_segmentation_predict_fn,
        frozenset(),
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    if task not in _TASKS:
        raise NotImplementedError(
            f"task={task!r} is not supported; choose from {sorted(_TASKS)}"
        )
    return _TASKS[task]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_tasks.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_tasks.py tests/test_benchmark/test_tasks.py
git commit -m "Add task registry"
```

---

### Task 5: Wire compare to the registry + streaming; remove dead code

**Files:**
- Modify: `src/mushin/benchmark/compare.py`
- Modify: `src/mushin/benchmark/_inference.py` (remove `run_inference`)
- Modify: `src/mushin/benchmark/_metrics.py` (already done in Task 1; nothing here)
- Test: `tests/test_benchmark/test_compare.py`, `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_benchmark/test_compare.py`:

```python
def test_compare_segmentation_end_to_end():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    # data: inputs (N,1,8,8), targets are per-pixel masks (N,8,8)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 1, 8, 8, generator=g)
    masks = torch.randint(0, 3, (12, 8, 8), generator=g)
    loader = DataLoader(TensorDataset(x, masks), batch_size=4)

    class Perfect(torch.nn.Module):
        # one-hot logits of the true mask -> perfect predictions
        def __init__(self, masks):
            super().__init__()
            self._m = {tuple(xi.flatten().tolist()): mi for xi, mi in zip(x, masks)}

        def forward(self, xb):
            out = []
            for xi in xb:
                m = self._m[tuple(xi.flatten().tolist())]
                out.append(torch.nn.functional.one_hot(m, 3).permute(2, 0, 1).float() * 10)
            return torch.stack(out)

    class Bad(torch.nn.Module):
        def forward(self, xb):
            return torch.zeros(xb.shape[0], 3, 8, 8)

    result = compare(
        methods={"good": [Perfect(masks) for _ in range(3)], "bad": [Bad() for _ in range(3)]},
        data=loader, task="segmentation", num_classes=3, test="welch",
    )
    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert "miou" in result.data.data_vars
    assert float(result.data["miou"].sel({"method": "good"}).mean()) == 1.0


def test_compare_rejects_unknown_task():
    import pytest

    from mushin.benchmark import compare

    with pytest.raises(NotImplementedError, match="not supported"):
        compare(methods={"a": []}, data=[], task="detection", num_classes=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_compare.py -k "segmentation or unknown_task" -q`
Expected: FAIL (segmentation not dispatched yet; the existing `task != "classification"` guard rejects it).

- [ ] **Step 3: Rewire `compare`** — replace the full body of
`src/mushin/benchmark/compare.py` with:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The ``compare`` facade: evaluate methods on a task battery and report significance."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from ._aggregate import to_dataset
from ._inference import PredictFn, evaluate
from ._result import BenchmarkResult
from ._stats import compare_methods
from ._tasks import get_task_spec


def compare(
    methods: dict[str, Sequence[torch.nn.Module]],
    data: Iterable,
    task: str = "classification",
    *,
    num_classes: int | None = None,
    predict_fn: PredictFn | None = None,
    metrics: dict | None = None,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    ignore_index: int | None = None,
    device: torch.device | None = None,
) -> BenchmarkResult:
    """Compare methods on a standard battery and report significance.

    Parameters
    ----------
    task : str
        ``"classification"`` or ``"segmentation"``.
    num_classes : int or None
        Required when ``metrics`` is not provided.
    ignore_index : int or None
        Label to exclude from segmentation metrics (e.g. a void/boundary class).
    """
    spec = get_task_spec(task)

    if metrics is not None:
        battery = metrics
    else:
        if num_classes is None:
            raise ValueError("`num_classes` is required when `metrics` is not provided")
        battery = spec.battery(num_classes, ignore_index=ignore_index)

    fn = predict_fn or spec.predict_fn

    results: dict[str, list[dict[str, float]]] = {}
    for name, models in methods.items():
        results[name] = [
            evaluate(model, data, battery, fn, spec.prob_metrics, device)
            for model in models
        ]

    ds = to_dataset(results)
    comparisons = compare_methods(ds, test=test, alpha=alpha)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
```

- [ ] **Step 4: Remove the now-dead `run_inference`** from
`src/mushin/benchmark/_inference.py` (delete the `run_inference` function and the
`default_classification_predict_fn` import if it becomes unused; keep `PredictFn`
and `evaluate`). Also delete `tests/test_benchmark/test_inference.py`'s
`test_run_inference_*` functions (keep `test_evaluate_*`).

- [ ] **Step 5: Remove the now-dead `compute_metrics` + `_PROB_METRICS`** — they
were kept through Task 1–4 so the package stayed importable; now that `compare`
streams, nothing uses them.
  - In `src/mushin/benchmark/_metrics.py`, delete the `compute_metrics` function
    and the `_PROB_METRICS` constant (keep `compute_battery`, the batteries).
  - In `tests/test_benchmark/test_metrics.py`, delete the tests that call
    `compute_metrics` (the perfect-classifier and state-leak tests that used it).
    The state-leak guarantee is now covered by `evaluate` (resets before the loop)
    and the streaming==one-shot test in Task 3. Keep `test_battery_has_expected_metrics`
    and the segmentation tests.

- [ ] **Step 6: Run the affected tests**

Run: `uv run pytest tests/test_benchmark -q`
Expected: PASS — the new segmentation/unknown-task tests plus the existing
classification suite (regression). If a classification metric value differs, STOP
and report (it should be identical).

- [ ] **Step 7: Commit**

```bash
git add src/mushin/benchmark/compare.py src/mushin/benchmark/_inference.py tests/test_benchmark
git commit -m "Dispatch compare via task registry with streaming eval; segmentation support"
```

---

### Task 6: Study `ignore_index` pass-through

**Files:**
- Modify: `src/mushin/study/_study.py`, `src/mushin/study/_load.py`
- Test: `tests/test_study/test_study.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_study/test_study.py`:

```python
def test_study_forwards_ignore_index_for_segmentation(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin import Study

    x = torch.randn(8, 1, 8, 8)
    masks = torch.randint(0, 3, (8, 8, 8))
    masks[:, 0, 0] = 255  # void pixels everywhere at (0,0)
    loader = DataLoader(TensorDataset(x, masks), batch_size=4)

    class Perfect(torch.nn.Module):
        def __init__(self, masks):
            super().__init__()
            self._m = {tuple(xi.flatten().tolist()): mi for xi, mi in zip(x, masks)}

        def forward(self, xb):
            out = []
            for xi in xb:
                m = self._m[tuple(xi.flatten().tolist())].clamp(max=2)
                out.append(torch.nn.functional.one_hot(m, 3).permute(2, 0, 1).float() * 10)
            return torch.stack(out)

    ckpts = {}
    for name in ("a", "b"):
        paths = []
        for s in range(2):
            p = tmp_path / f"{name}_{s}.pt"
            torch.save(Perfect(masks), p)
            paths.append(str(p))
        ckpts[name] = paths

    study = Study.from_checkpoints(
        checkpoints=ckpts,
        load_fn=lambda p: torch.load(p, weights_only=False),
        data=loader, task="segmentation", num_classes=3, test="welch",
        ignore_index=255,
    )
    result = study.run()
    # void pixels excluded -> perfect pixel accuracy
    assert float(result.data["pixel_acc"].mean()) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_study/test_study.py -k ignore_index -q`
Expected: FAIL with `TypeError: from_checkpoints() got an unexpected keyword argument 'ignore_index'`.

- [ ] **Step 3: Add `ignore_index` to `_load.evaluate_checkpoints`** — change its
signature and the `compare` call in `src/mushin/study/_load.py`:

```python
def evaluate_checkpoints(
    checkpoints: dict[str, Sequence[str]],
    load_fn: Callable[[str], Any],
    data,
    task: str,
    num_classes: int,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    ignore_index: int | None = None,
) -> BenchmarkResult:
```
and pass `ignore_index=ignore_index` into the `compare(...)` call.

- [ ] **Step 4: Thread `ignore_index` through `Study`** in
`src/mushin/study/_study.py`:
  - add `ignore_index` to `_init_common(self, load_fn, data, num_classes, task, test, alpha, ignore_index)` and store `self._ignore_index = ignore_index`;
  - add `ignore_index: int | None = None` (keyword-only) to both `__init__` and
    `from_checkpoints`, and pass it into `_init_common`;
  - in `run()`, pass `self._ignore_index` as the `ignore_index=` argument to
    `evaluate_checkpoints(...)`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_study/test_study.py -q`
Expected: PASS (all study tests).

- [ ] **Step 6: Commit**

```bash
git add src/mushin/study/_study.py src/mushin/study/_load.py tests/test_study/test_study.py
git commit -m "Thread ignore_index through Study to compare"
```

---

### Task 7: Full gate

**Files:** none (verification).

- [ ] **Step 1: Run the full gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run codespell src tests examples README.md CHANGELOG.md && uv run pytest tests/ --hypothesis-profile fast -p no:cacheprovider -q`
Expected: ruff clean, format clean, codespell clean, all tests pass (existing
suite + new segmentation/registry/evaluate/study tests). If `ruff format` wants
changes, run `uv run ruff format .` and commit them.

- [ ] **Step 2: Commit any formatting**

```bash
git add -A && git commit -m "Format" || echo "nothing to format"
```

- [ ] **Step 3: Leave the branch for the controller** to open the PR after the
final whole-branch review.

---

## Self-review notes

- **Spec coverage:** battery (Task 1), seg predict (Task 2), streaming evaluate
  (Task 3), task registry (Task 4), compare dispatch + ignore_index + dead-code
  removal + seg e2e + classification regression (Task 5), Study ignore_index
  (Task 6), gate (Task 7). `Dice = MulticlassF1Score(macro)` and ignore_index
  verified against the installed torchmetrics.
- **Out of scope (intentional):** instance/panoptic segmentation, detection,
  per-class IoU vectors.
- **Type/name consistency:** `segmentation_battery(num_classes, ignore_index=None)`;
  `classification_battery(num_classes, ignore_index=None)`; `compute_battery(
  battery, preds, targets, prob_metrics, probs=None)`;
  `default_segmentation_predict_fn(model, x) -> (preds, probs)`;
  `evaluate(model, data, battery, predict_fn, prob_metrics, device=None) ->
  dict[str, float]`; `TaskSpec(battery, predict_fn, prob_metrics)`;
  `get_task_spec(task)`; `compare(..., ignore_index=None, ...)`;
  `evaluate_checkpoints(..., ignore_index=None)`; `Study(..., ignore_index=None)`.
  Consistent across tasks.
