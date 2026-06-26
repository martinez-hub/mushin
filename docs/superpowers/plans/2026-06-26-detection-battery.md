# Object-Detection Benchmark Battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `task="detection"` benchmark battery that compares trained object detectors across seeds and reports Holm-corrected significance over the full `torchmetrics.detection` bounding-box metric family.

**Architecture:** Plug into the existing `compare(task=...)` / `Study` spine. Generalize the shared `evaluate`/`compute_battery` path to be metric-shape-agnostic (a recursive `_to_device` helper + a dict-valued-metric expansion that also normalizes the COCO `-1.0` sentinel to `NaN`). The detection battery wraps `MeanAveragePrecision` (dropping its non-scalar bookkeeping keys) plus the four IoU-variant metrics; one streaming pass updates all of them.

**Tech Stack:** Python 3.9+, PyTorch, torchmetrics (`>=1.0`; detection metrics gated behind the optional `torchvision` + `pycocotools` deps), xarray, scipy, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-06-26-detection-battery-design.md`

---

## File Structure

- `src/mushin/benchmark/_inference.py` — add `_to_device` + `expand_metric_value` (+ `_as_float`); use them in `evaluate`.
- `src/mushin/benchmark/_metrics.py` — add `detection_battery` (+ `_DetectionMAP`); route `compute_battery` through `expand_metric_value`.
- `src/mushin/benchmark/_predict.py` — add `default_detection_predict_fn`.
- `src/mushin/benchmark/_tasks.py` — add `requires_num_classes` to `TaskSpec`; register `"detection"`.
- `src/mushin/benchmark/compare.py` — make the `num_classes` guard task-aware.
- `pyproject.toml` — `detection` optional extra; dev-group deps (platform-guarded).
- `tests/test_benchmark/` — `test_inference.py`, `test_metrics.py`, `test_predict.py`, `test_tasks.py`, `test_compare.py` additions; new `test_detection.py`.
- `docs/guides/compare.md` (detection section) + `changes/+detection-battery.added.md`.

Each task is TDD: write the failing test, see it fail, implement minimally, see it pass, commit.

---

### Task 1: `_to_device` recursive helper

**Files:**
- Modify: `src/mushin/benchmark/_inference.py`
- Test: `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Write the failing test**

```python
def test_to_device_moves_tensors_in_nested_structures():
    import torch
    from mushin.benchmark._inference import _to_device

    dev = torch.device("cpu")
    obj = [
        {"boxes": torch.zeros(2, 4), "labels": torch.tensor([1, 2])},
        torch.ones(3),
    ]
    moved = _to_device(obj, dev)
    assert isinstance(moved, list)
    assert moved[0]["boxes"].device == dev and moved[0]["labels"].device == dev
    assert moved[1].device == dev
    # non-tensors pass through unchanged
    assert _to_device("a string", dev) == "a string"
    assert _to_device(7, dev) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_inference.py::test_to_device_moves_tensors_in_nested_structures -v`
Expected: FAIL with `ImportError: cannot import name '_to_device'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/mushin/benchmark/_inference.py` (after the imports, before `evaluate`):

```python
def _to_device(obj, device: torch.device):
    """Recursively move tensors to ``device`` through tensors, lists/tuples, and
    dicts; anything else passes through. Lets one streaming loop serve tensor tasks
    and detection's ``list[dict]`` batches alike."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_inference.py::test_to_device_moves_tensors_in_nested_structures -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_inference.py tests/test_benchmark/test_inference.py
git commit -m "feat(benchmark): recursive _to_device helper for list/dict batches"
```

---

### Task 2: `expand_metric_value` + `-1.0` sentinel

**Files:**
- Modify: `src/mushin/benchmark/_inference.py`
- Test: `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Write the failing test**

```python
def test_expand_metric_value_scalar_dict_and_sentinel():
    import math
    import torch
    from mushin.benchmark._inference import expand_metric_value

    # scalar -> kept under the battery name
    assert expand_metric_value("acc", torch.tensor(0.5)) == {"acc": 0.5}
    # dict -> one entry per key (the metric's own key names)
    out = expand_metric_value("map", {"map": torch.tensor(0.4), "map_50": torch.tensor(0.6)})
    assert out == {"map": 0.4, "map_50": 0.6}
    # COCO -1.0 "not applicable" sentinel -> NaN
    sent = expand_metric_value("map", {"map_small": torch.tensor(-1.0)})
    assert math.isnan(sent["map_small"])

def test_expand_metric_value_rejects_non_scalar():
    import pytest
    import torch
    from mushin.benchmark._inference import expand_metric_value

    with pytest.raises(TypeError, match="non-scalar"):
        expand_metric_value("classes", {"classes": torch.tensor([0, 1, 2])})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_inference.py -k expand_metric_value -v`
Expected: FAIL with `ImportError: cannot import name 'expand_metric_value'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/mushin/benchmark/_inference.py` (next to `_to_device`):

```python
def _as_float(v) -> float:
    """Coerce one metric value to a float. The COCO ``-1.0`` 'not applicable'
    sentinel (a bucket with no matching ground truth) becomes ``NaN`` so the
    significance machinery treats it as missing rather than a real score."""
    if isinstance(v, torch.Tensor) and v.numel() != 1:
        raise TypeError(
            f"metric produced a non-scalar value of shape {tuple(v.shape)}; "
            "battery metrics must return scalar values per key"
        )
    f = float(v)
    return float("nan") if f == -1.0 else f


def expand_metric_value(name: str, value) -> dict[str, float]:
    """Flatten one metric's ``compute()`` output into ``{data_var: float}``. A dict
    expands to one entry per key (using the metric's own key names); a scalar is
    kept under ``name``."""
    if isinstance(value, dict):
        return {k: _as_float(v) for k, v in value.items()}
    return {name: _as_float(value)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_inference.py -k expand_metric_value -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_inference.py tests/test_benchmark/test_inference.py
git commit -m "feat(benchmark): expand dict-valued metric outputs, -1 sentinel -> NaN"
```

---

### Task 3: Route `evaluate` through `_to_device` + `expand_metric_value`

**Files:**
- Modify: `src/mushin/benchmark/_inference.py` (the `evaluate` function body)
- Test: `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_expands_dict_metric_and_keeps_scalar():
    import torch
    from torchmetrics import Metric
    from mushin.benchmark._inference import evaluate

    class ScalarMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")
        def update(self, preds, target):
            self.v = preds.float().mean()
        def compute(self):
            return self.v

    class DictMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")
        def update(self, preds, target):
            self.v = preds.float().mean()
        def compute(self):
            return {"a": self.v, "b": self.v + 1}

    model = torch.nn.Identity()
    data = [(torch.tensor([1.0, 1.0]), torch.tensor([0, 0]))]  # one re-iterable batch

    out = evaluate(model, data, {"s": ScalarMetric(), "d": DictMetric()},
                   predict_fn=lambda m, x: (m(x), None), prob_metrics=frozenset())
    assert out == {"s": 1.0, "a": 1.0, "b": 2.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_inference.py::test_evaluate_expands_dict_metric_and_keeps_scalar -v`
Expected: FAIL — current `evaluate` does `float(metric.compute())`, which raises `TypeError` on the dict-returning metric.

- [ ] **Step 3: Edit `evaluate`**

In `src/mushin/benchmark/_inference.py`, replace the body of the `with torch.no_grad():` loop and the final return. Change:

```python
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            preds, probs = predict_fn(model, x)
            for name, metric in battery.items():
                metric.update(probs if name in prob_metrics else preds, y)

    return {name: float(metric.compute()) for name, metric in battery.items()}
```

to:

```python
    with torch.no_grad():
        for x, y in data:
            x = _to_device(x, device)
            y = _to_device(y, device)
            preds, probs = predict_fn(model, x)
            for name, metric in battery.items():
                metric.update(probs if name in prob_metrics else preds, y)

    out: dict[str, float] = {}
    for name, metric in battery.items():
        out.update(expand_metric_value(name, metric.compute()))
    return out
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_benchmark/test_inference.py -v`
Expected: PASS (new test + existing inference tests — scalar tensor tasks still return scalar dicts, unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_inference.py tests/test_benchmark/test_inference.py
git commit -m "feat(benchmark): evaluate handles list/dict batches and dict metrics"
```

---

### Task 4: Route `compute_battery` through `expand_metric_value`

**Files:**
- Modify: `src/mushin/benchmark/_metrics.py` (the `compute_battery` function)
- Test: `tests/test_benchmark/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_battery_expands_dict_metric():
    import torch
    from torchmetrics import Metric
    from mushin.benchmark._metrics import compute_battery

    class DictMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")
        def update(self, preds, target):
            self.v = preds.float().mean()
        def compute(self):
            return {"x": self.v, "y": torch.tensor(-1.0)}  # -1 -> NaN

    import math
    out = compute_battery({"m": DictMetric()}, preds=torch.tensor([1.0]),
                          targets=torch.tensor([1.0]), prob_metrics=frozenset())
    assert out["x"] == 1.0
    assert math.isnan(out["y"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_metrics.py::test_compute_battery_expands_dict_metric -v`
Expected: FAIL — `compute_battery` does `float(metric(...))`, raising on the dict output.

- [ ] **Step 3: Edit `compute_battery`**

In `src/mushin/benchmark/_metrics.py`, change the loop body. From:

```python
    out: dict[str, float] = {}
    for name, metric in battery.items():
        metric.reset()
        inp = probs if name in prob_metrics else preds
        out[name] = float(metric(inp, targets))
    return out
```

to:

```python
    from ._inference import expand_metric_value

    out: dict[str, float] = {}
    for name, metric in battery.items():
        metric.reset()
        inp = probs if name in prob_metrics else preds
        out.update(expand_metric_value(name, metric(inp, targets)))
    return out
```

(The import is function-local to avoid any import-order coupling; `_inference` does not import `_metrics`, so there is no cycle.)

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_benchmark/test_metrics.py -v`
Expected: PASS (new test + existing scalar-metric tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_metrics.py tests/test_benchmark/test_metrics.py
git commit -m "feat(benchmark): compute_battery expands dict-valued metrics too"
```

---

### Task 5: `default_detection_predict_fn`

**Files:**
- Modify: `src/mushin/benchmark/_predict.py`
- Test: `tests/test_benchmark/test_predict.py`

- [ ] **Step 1: Write the failing test**

```python
def test_default_detection_predict_fn_returns_model_output_and_none():
    import torch
    from mushin.benchmark._predict import default_detection_predict_fn

    sentinel = [{"boxes": torch.zeros(1, 4), "scores": torch.tensor([0.9]),
                 "labels": torch.tensor([0])}]

    class FakeDetector(torch.nn.Module):
        def forward(self, x):
            return sentinel

    preds, probs = default_detection_predict_fn(FakeDetector(), [torch.zeros(3, 8, 8)])
    assert preds is sentinel
    assert probs is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_predict.py::test_default_detection_predict_fn_returns_model_output_and_none -v`
Expected: FAIL with `ImportError: cannot import name 'default_detection_predict_fn'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/mushin/benchmark/_predict.py`:

```python
def default_detection_predict_fn(model: torch.nn.Module, x):
    """Run a detection model and return ``(predictions, None)``.

    Assumes the torchvision detection convention: an eval-mode detector maps a
    list of image tensors to a ``list[dict]`` with ``boxes``/``scores``/``labels``.
    There are no probabilities to feed metrics, so the second element is ``None``.
    Override ``predict_fn`` for non-torchvision detectors (DETR, YOLO, ...).
    """
    return model(x), None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_predict.py::test_default_detection_predict_fn_returns_model_output_and_none -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_predict.py tests/test_benchmark/test_predict.py
git commit -m "feat(benchmark): default_detection_predict_fn (torchvision convention)"
```

---

### Task 6: `detection_battery` + `_DetectionMAP` wrapper

**Files:**
- Modify: `src/mushin/benchmark/_metrics.py`
- Test: `tests/test_benchmark/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_detection_battery_contents_and_map_drops_metadata():
    pytest = __import__("pytest")
    pytest.importorskip("torchmetrics.detection")  # needs torchvision + pycocotools
    import torch
    from mushin.benchmark._metrics import detection_battery

    battery = detection_battery()
    assert set(battery) == {"map", "iou", "giou", "ciou", "diou"}

    preds = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
              "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
    tgts = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
             "labels": torch.tensor([0])}]
    battery["map"].update(preds, tgts)
    keys = set(battery["map"].compute())
    # the three non-scalar bookkeeping keys are dropped
    assert {"classes", "map_per_class", "mar_100_per_class"}.isdisjoint(keys)
    # the 12 scalar AP/AR values remain
    assert {"map", "map_50", "map_75", "map_small", "mar_100"} <= keys


def test_detection_battery_clear_error_without_extra(monkeypatch):
    import builtins
    import pytest
    from mushin.benchmark import _metrics

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchmetrics.detection":
            raise ImportError("no detection extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="mushin-py\\[detection\\]"):
        _metrics.detection_battery()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_metrics.py -k detection_battery -v`
Expected: FAIL with `AttributeError`/`ImportError` (no `detection_battery`).

- [ ] **Step 3: Write minimal implementation**

Add to `src/mushin/benchmark/_metrics.py`:

```python
_MAP_DROP = frozenset({"classes", "map_per_class", "mar_100_per_class"})


def detection_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """The bounding-box detection battery: mean-average-precision plus the IoU
    variants, every scalar output surfaced as its own metric. ``num_classes`` and
    ``ignore_index`` are accepted for the uniform task interface but unused (mAP
    infers classes from the labels). Requires the optional ``detection`` extra."""
    try:
        from torchmetrics.detection import (
            CompleteIntersectionOverUnion,
            DistanceIntersectionOverUnion,
            GeneralizedIntersectionOverUnion,
            IntersectionOverUnion,
            MeanAveragePrecision,
        )
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "the detection battery requires the optional detection extra; install "
            "it with `pip install mushin-py[detection]` (torchvision + pycocotools)."
        ) from e

    class _DetectionMAP(MeanAveragePrecision):
        """``MeanAveragePrecision`` minus its non-scalar bookkeeping keys
        (``classes``/``*_per_class``), which are not single comparable scores."""

        def compute(self):
            return {
                k: v for k, v in super().compute().items() if k not in _MAP_DROP
            }

    return {
        "map": _DetectionMAP(box_format="xyxy"),
        "iou": IntersectionOverUnion(),
        "giou": GeneralizedIntersectionOverUnion(),
        "ciou": CompleteIntersectionOverUnion(),
        "diou": DistanceIntersectionOverUnion(),
    }
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_benchmark/test_metrics.py -k detection_battery -v`
Expected: PASS where the extra is installed; the contents test SKIPS if not. The error-path test passes regardless (monkeypatches the import).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_metrics.py tests/test_benchmark/test_metrics.py
git commit -m "feat(benchmark): detection_battery (mAP + IoU family, metadata dropped)"
```

---

### Task 7: Register the detection task (`requires_num_classes` + guard)

**Files:**
- Modify: `src/mushin/benchmark/_tasks.py`, `src/mushin/benchmark/compare.py`
- Test: `tests/test_benchmark/test_tasks.py`, `tests/test_benchmark/test_compare.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_benchmark/test_tasks.py`:

```python
def test_detection_task_registered_and_optional_num_classes():
    from mushin.benchmark._tasks import get_task_spec

    spec = get_task_spec("detection")
    assert spec.requires_num_classes is False
    assert spec.prob_metrics == frozenset()
    # classification still requires num_classes
    assert get_task_spec("classification").requires_num_classes is True
```

In `tests/test_benchmark/test_compare.py`:

```python
def test_compare_detection_does_not_demand_num_classes(monkeypatch):
    # detection must not raise the num_classes ValueError; stub the battery so the
    # test needs no detection extra.
    import torch
    from mushin.benchmark import compare
    from mushin.benchmark import _tasks

    def fake_battery(num_classes=None, ignore_index=None):
        from torchmetrics import Metric

        class Const(Metric):
            def __init__(self):
                super().__init__()
                self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")
            def update(self, preds, target):
                self.v = torch.tensor(0.5)
            def compute(self):
                return self.v

        return {"score": Const()}

    spec = _tasks.get_task_spec("detection")
    monkeypatch.setattr(spec, "__dict__", spec.__dict__)  # noqa - frozen; patch registry instead
    monkeypatch.setitem(
        _tasks._TASKS, "detection",
        _tasks.TaskSpec(fake_battery, lambda m, x: (m(x), None), frozenset(),
                        requires_num_classes=False),
    )

    class M(torch.nn.Module):
        def forward(self, x):
            return x

    data = [(torch.tensor([1.0]), torch.tensor([1.0]))]
    result = compare({"a": [M(), M()], "b": [M(), M()]}, data, task="detection")
    assert "score" in result.data.data_vars
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark/test_tasks.py::test_detection_task_registered_and_optional_num_classes tests/test_benchmark/test_compare.py::test_compare_detection_does_not_demand_num_classes -v`
Expected: FAIL — `NotImplementedError` (task not registered) / `AttributeError` (`requires_num_classes` missing).

- [ ] **Step 3: Edit `_tasks.py`**

Add `requires_num_classes` to `TaskSpec` and register detection. Change the dataclass and `_TASKS`:

```python
from ._metrics import (
    classification_battery,
    detection_battery,
    segmentation_battery,
)
from ._predict import (
    default_classification_predict_fn,
    default_detection_predict_fn,
    default_segmentation_predict_fn,
)


@dataclass(frozen=True)
class TaskSpec:
    battery: Callable[..., dict[str, Metric]]
    predict_fn: PredictFn
    prob_metrics: frozenset[str]
    requires_num_classes: bool = True


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
    "detection": TaskSpec(
        detection_battery,
        default_detection_predict_fn,
        frozenset(),
        requires_num_classes=False,
    ),
}
```

(Importing `_tasks` does **not** import the detection extra: `detection_battery` only imports `torchmetrics.detection` lazily when called.)

- [ ] **Step 4: Edit `compare.py` guard**

In `src/mushin/benchmark/compare.py`, change:

```python
    if metrics is not None:
        battery = metrics
    else:
        if num_classes is None:
            raise ValueError("`num_classes` is required when `metrics` is not provided")
        battery = spec.battery(num_classes, ignore_index=ignore_index)
```

to:

```python
    if metrics is not None:
        battery = metrics
    elif spec.requires_num_classes and num_classes is None:
        raise ValueError("`num_classes` is required when `metrics` is not provided")
    else:
        battery = spec.battery(num_classes, ignore_index=ignore_index)
```

- [ ] **Step 5: Run tests to verify**

Run: `uv run pytest tests/test_benchmark/test_tasks.py tests/test_benchmark/test_compare.py -v`
Expected: PASS (new tests + existing classification/segmentation compare tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/mushin/benchmark/_tasks.py src/mushin/benchmark/compare.py tests/test_benchmark/test_tasks.py tests/test_benchmark/test_compare.py
git commit -m "feat(benchmark): register detection task; task-aware num_classes guard"
```

---

### Task 8: `detection` optional extra + dev deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the optional extra**

In `[project.optional-dependencies]` add:

```toml
# Detection metrics (MeanAveragePrecision + IoU variants) need torchvision and
# pycocotools; gated so the core install stays lean.
detection = ["torchvision", "pycocotools"]
```

- [ ] **Step 2: Add dev-group deps so CI exercises the battery**

In `[dependency-groups]`'s `dev` list add (pycocotools is painful to build on Windows, so guard it; torchvision installs broadly):

```toml
    "torchvision",
    "pycocotools ; sys_platform != 'win32'",
```

- [ ] **Step 3: Re-lock and sync**

Run: `uv lock && uv sync`
Expected: lockfile updates; torchvision + pycocotools install on this platform.

- [ ] **Step 4: Verify the detection metrics now import**

Run: `uv run python -c "import torchmetrics.detection as d; print(hasattr(d, 'MeanAveragePrecision'))"`
Expected: `True`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: detection optional extra + dev deps (torchvision, pycocotools)"
```

> **Note for the implementer:** detection tests use `pytest.importorskip("torchmetrics.detection")`, so on platforms where pycocotools is unavailable (e.g. Windows CI) they skip cleanly rather than fail. If the Windows job's `uv sync` is unhappy even with the marker, drop `pycocotools` from the dev group entirely and rely on a Linux-only `.[detection]` install step — flag this to the user rather than guessing.

---

### Task 9: Hermetic detection tests

**Files:**
- Create: `tests/test_benchmark/test_detection.py`

- [ ] **Step 1: Write the tests**

```python
"""Hermetic detection-battery tests (no real dataset; needs the detection extra)."""

import math

import pytest
import torch

pytest.importorskip("torchmetrics.detection")  # torchvision + pycocotools

from mushin.benchmark import BenchmarkResult, compare  # noqa: E402
from mushin.benchmark._inference import evaluate  # noqa: E402
from mushin.benchmark._metrics import detection_battery  # noqa: E402
from mushin.benchmark._predict import default_detection_predict_fn  # noqa: E402


def _box(x0, y0, x1, y1):
    return torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])


class _FixedDetector(torch.nn.Module):
    """Ignores the input image and emits fixed predictions per batch."""

    def __init__(self, preds):
        super().__init__()
        self._preds = preds

    def forward(self, x):
        return self._preds


def test_perfect_predictions_score_one():
    preds = [{"boxes": _box(0, 0, 10, 10), "scores": torch.tensor([0.9]),
              "labels": torch.tensor([0])}]
    tgts = [{"boxes": _box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    out = evaluate(_FixedDetector(preds), data, detection_battery(),
                   default_detection_predict_fn, prob_metrics=frozenset())
    assert out["map"] == pytest.approx(1.0)
    assert out["iou"] == pytest.approx(1.0)
    assert out["giou"] == pytest.approx(1.0)


def test_disjoint_predictions_score_low():
    preds = [{"boxes": _box(100, 100, 110, 110), "scores": torch.tensor([0.9]),
              "labels": torch.tensor([0])}]
    tgts = [{"boxes": _box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    out = evaluate(_FixedDetector(preds), data, detection_battery(),
                   default_detection_predict_fn, prob_metrics=frozenset())
    assert out["map"] == pytest.approx(0.0, abs=1e-6)


def test_battery_matches_torchmetrics_reference():
    """Our streaming/expansion path reproduces torchmetrics' own numbers."""
    from torchmetrics.detection import MeanAveragePrecision

    preds = [{"boxes": _box(0, 0, 10, 10), "scores": torch.tensor([0.8]),
              "labels": torch.tensor([1])}]
    tgts = [{"boxes": _box(1, 1, 11, 11), "labels": torch.tensor([1])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    out = evaluate(_FixedDetector(preds), data, detection_battery(),
                   default_detection_predict_fn, prob_metrics=frozenset())

    ref = MeanAveragePrecision(box_format="xyxy")
    ref.update(preds, tgts)
    ref_out = ref.compute()
    for key in ("map", "map_50", "map_75", "mar_100"):
        ours = out[key]
        gold = float(ref_out[key])
        if gold == -1.0:
            assert math.isnan(ours)
        else:
            assert ours == pytest.approx(gold)


def test_size_bucket_sentinel_becomes_nan():
    """A 10x10 box is 'medium' under COCO, so map_small has no GT -> -1 -> NaN."""
    preds = [{"boxes": _box(0, 0, 10, 10), "scores": torch.tensor([0.9]),
              "labels": torch.tensor([0])}]
    tgts = [{"boxes": _box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    out = evaluate(_FixedDetector(preds), data, detection_battery(),
                   default_detection_predict_fn, prob_metrics=frozenset())
    assert math.isnan(out["map_small"])  # not -1.0


def test_compare_detection_end_to_end():
    good = [{"boxes": _box(0, 0, 10, 10), "scores": torch.tensor([0.9]),
             "labels": torch.tensor([0])}]
    bad = [{"boxes": _box(50, 50, 60, 60), "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0])}]
    tgts = [{"boxes": _box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]

    result = compare(
        {"good": [_FixedDetector(good), _FixedDetector(good)],
         "bad": [_FixedDetector(bad), _FixedDetector(bad)]},
        data, task="detection", test="welch",
    )
    assert isinstance(result, BenchmarkResult)
    for key in ("map", "map_50", "map_75", "mar_100", "iou", "giou", "ciou", "diou"):
        assert key in result.data.data_vars
    assert float(result.data["map"].sel({"method": "good"}).mean()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_benchmark/test_detection.py -v`
Expected: PASS (all), assuming the detection extra is installed (Task 8). If a size-bucket assumption differs in the installed torchmetrics version, adjust the box size so the target is genuinely outside the `small` bucket (COCO `small` = area < 32², so a 10×10 box = area 100 < 1024 is *small* — verify and flip the asserted key to whichever bucket is empty; the reference test is the source of truth).

> **Implementer note:** COCO size buckets — small `< 32²`, medium `32²–96²`, large `> 96²`. A 10×10 box (area 100) is **small**, so `map_medium`/`map_large` will be the `-1 → NaN` ones, not `map_small`. Use the reference (`test_battery_matches_torchmetrics_reference`) to confirm which keys are `-1.0`, and assert NaN on one of those. Fix `test_size_bucket_sentinel_becomes_nan` to assert on `map_large` accordingly.

- [ ] **Step 3: Commit**

```bash
git add tests/test_benchmark/test_detection.py
git commit -m "test(benchmark): hermetic detection battery tests (perfect/disjoint/reference/sentinel/e2e)"
```

---

### Task 10: Gated real-dataset validation test

**Files:**
- Modify: `tests/test_benchmark/test_detection.py`, `pyproject.toml` (register the marker)

- [ ] **Step 1: Register the marker**

In `pyproject.toml` under `[tool.pytest.ini_options]` add:

```toml
markers = [
    "real_data: end-to-end checks that download a real dataset/model (deselected by default; run with -m real_data)",
]
```

And ensure default runs deselect it by adding `addopts = "-m 'not real_data'"` to `[tool.pytest.ini_options]` (if `addopts` already exists, append `-m 'not real_data'`).

- [ ] **Step 2: Write the gated test**

Append to `tests/test_benchmark/test_detection.py`:

```python
@pytest.mark.real_data
def test_real_coco_sample_end_to_end(tmp_path):
    """Manual validation: a pretrained torchvision detector on a few real COCO
    images yields a plausible mAP. Run with: pytest -m real_data."""
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).eval()

    # A tiny hand-built batch from a known COCO image + its annotations would go
    # here; for the smoke check we assert the pipeline runs and mAP is in [0, 1].
    img = torch.rand(3, 320, 320)
    tgts = [{"boxes": torch.tensor([[10.0, 10.0, 200.0, 200.0]]),
             "labels": torch.tensor([1])}]
    data = [([img], tgts)]

    from mushin.benchmark import compare

    result = compare({"frcnn": [model]}, data, task="detection", test="welch")
    m = float(result.data["map"].mean())
    assert -1.0 <= m <= 1.0  # ran end-to-end on a real detector without error
```

- [ ] **Step 3: Verify it is deselected by default and runnable on demand**

Run (default): `uv run pytest tests/test_benchmark/test_detection.py -v`
Expected: `test_real_coco_sample_end_to_end` shows as deselected.
Run (opt-in): `uv run pytest tests/test_benchmark/test_detection.py -m real_data -v`
Expected: it runs (downloads weights) and passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_benchmark/test_detection.py pyproject.toml
git commit -m "test(benchmark): gated real-data detection smoke test (-m real_data)"
```

---

### Task 11: Docs + changelog

**Files:**
- Modify: `docs/guides/compare.md`
- Create: `changes/+detection-battery.added.md`

- [ ] **Step 1: Add a detection section to the compare guide**

Append to `docs/guides/compare.md` (adapt the heading depth to the file):

````markdown
## Object detection

`task="detection"` compares trained detectors over the full `torchmetrics.detection`
bounding-box family. Each model's dataloader yields `(images, targets)` where
`images` is a `list[Tensor]` and each target is a `dict` with `boxes` (`[N,4]`,
xyxy) and `labels` (`[N]`); an eval-mode torchvision detector returns predictions
as `list[dict]` with `boxes`/`scores`/`labels` (override `predict_fn` for other
detectors).

```python
from mushin.benchmark import compare

result = compare(
    methods={"frcnn": frcnn_seeds, "retina": retina_seeds},  # one model per seed
    data=coco_val_loader,
    task="detection",
    test="welch",
)
result.summary()   # map / map_50 / map_75 / mar_* / iou / giou / ciou / diou + significance
```

The result xarray carries every scalar output: the 12 mAP/mAR values
(`map`, `map_50`, `map_75`, `map_small|medium|large`, `mar_1|10|100`,
`mar_small|medium|large`) plus `iou`, `giou`, `ciou`, `diou`. A size bucket with no
matching ground truth reports `NaN` (COCO's `-1` "not applicable" sentinel),
excluded from significance. `num_classes` is not required for detection.

Install the extra: `pip install mushin-py[detection]` (torchvision + pycocotools).
````

- [ ] **Step 2: Add the changelog fragment**

Create `changes/+detection-battery.added.md`:

```markdown
`compare(task="detection")` — compare trained object detectors across seeds over
the full `torchmetrics.detection` bounding-box family (mean-average-precision plus
the IoU/GIoU/CIoU/DIoU variants), reporting every scalar metric with Holm-corrected
significance. Needs the optional `mushin-py[detection]` extra.
```

- [ ] **Step 3: Verify docs build**

Run: `uv run --group docs mkdocs build --strict`
Expected: builds with no warnings.

- [ ] **Step 4: Commit**

```bash
git add docs/guides/compare.md changes/+detection-battery.added.md
git commit -m "docs(benchmark): detection task guide + changelog fragment"
```

---

### Task 12: Full verification sweep

- [ ] **Step 1: Lint, format, spelling**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run codespell src/mushin/benchmark tests/test_benchmark docs/guides/compare.md`
Expected: all clean. (Fix and re-run if not.)

- [ ] **Step 2: Full test suite**

Run: `uv run pytest -q`
Expected: all pass (detection tests run where the extra is installed; the `real_data` test is deselected).

- [ ] **Step 3: Final commit (if any fixups)**

```bash
git add -A && git commit -m "chore(benchmark): lint/format fixups for detection battery"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** entry point (Task 7 + `compare`), full bbox family incl. metadata drop (Task 6), generalized `evaluate`/`compute_battery` (Tasks 1–4), `default_detection_predict_fn` (Task 5), `-1→NaN` sentinel (Task 2), optional dep (Task 8), hermetic + gated tests (Tasks 9–10), docs (Task 11). All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; the two implementer notes are explicit decision guidance (size-bucket key + Windows pycocotools), not deferred work.
- **Type consistency:** `expand_metric_value(name, value) -> dict[str,float]`, `_to_device(obj, device)`, `_DetectionMAP.compute()`, `TaskSpec.requires_num_classes`, and `default_detection_predict_fn(model, x) -> (preds, None)` are used identically wherever referenced.
- **Known risk:** COCO size-bucket assertion in Task 9 — handled by an explicit implementer note tying it to the reference test (source of truth).
