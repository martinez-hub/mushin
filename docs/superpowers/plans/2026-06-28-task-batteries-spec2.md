# More Task Batteries + `update_fn` Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four built-in batteries (regression, image_quality, audio, retrieval) and a per-`Task` `update_fn` hook, reachable through the Spec 1 public Task API.

**Architecture:** Additive over `src/mushin/benchmark/`. A new optional `Task.update_fn` lets a battery own its `.update()` dispatch (the default path stays byte-for-byte identical). Four battery factories join `_metrics.py`, a shared passthrough predict_fn joins `_predict.py`, and the tasks register in `_TASKS`. Dependency-heavy metrics (LPIPS, PESQ, STOI) sit behind optional `[image]`/`[audio]` extras using the exact detection-extra pattern.

**Tech Stack:** Python 3.9+, torchmetrics (regression/image/audio/retrieval), pytest, towncrier, uv. Tests are hermetic (tiny synthetic tensors, no GPU, no real data); optional-dep metrics gated with `pytest.importorskip`.

**Reference (read once before starting):**
- Spec: `docs/superpowers/specs/2026-06-28-task-batteries-spec2-design.md`
- The update seam: `src/mushin/benchmark/_inference.py` — `evaluate` (lines 67-100), `PredictFn` alias (line 12), the per-batch update loop (lines 94-99).
- Task registry: `src/mushin/benchmark/_tasks.py` — `Task` dataclass, `_TASKS`, `register_task`/`get_task`.
- `compare`: `src/mushin/benchmark/compare.py` — task resolution (line 45), the `evaluate(...)` call (lines 64-67).
- Default predict_fns: `src/mushin/benchmark/_predict.py`.
- Battery factories + the detection lazy-import/all-or-nothing pattern: `src/mushin/benchmark/_metrics.py` (`detection_battery`, lines 72-126).
- Optional extras + platform gating: `pyproject.toml` `[project.optional-dependencies]` (lines 65-82).
- CI `--extra` wiring: `.github/workflows/ci.yml` (test job, lines 45-53).
- Test idioms: `tests/test_benchmark/test_compare.py` (synthetic loader/model), `tests/test_benchmark/test_detection.py` (module-level `pytest.importorskip`).

**Conventions:**
- Two-line MIT copyright header on every source file.
- After edits: `uv run ruff check <paths>` and `uv run ruff format <paths>`.
- Commit messages imperative; **no Claude attribution / no `Co-Authored-By` trailer**.
- Tests via `uv run pytest`. For optional-extra batteries use `uv run --extra image pytest ...` / `uv run --extra audio pytest ...`; if the extra cannot install on the local platform, say so and rely on CI.

---

### Task 1: The `update_fn` hook

Add an `UpdateFn` type and an optional `Task.update_fn` field; thread it through `evaluate` and `compare`. Default (`None`) reproduces today's behavior exactly.

**Files:**
- Modify: `src/mushin/benchmark/_inference.py`
- Modify: `src/mushin/benchmark/_tasks.py`
- Modify: `src/mushin/benchmark/compare.py`
- Test: `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark/test_inference.py`:

```python
def test_evaluate_uses_custom_update_fn():
    import torch
    from torchmetrics import MeanMetric

    from mushin.benchmark._inference import evaluate

    # data yields (x, y) where y is a (value, weight) tuple — a shape the default
    # (preds, target) loop could not handle; the custom update_fn unpacks it.
    data = [(torch.zeros(2, 1), (torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0])))]
    model = torch.nn.Identity()
    battery = {"m": MeanMetric()}

    def predict_fn(model, x):
        return torch.tensor([1.0, 3.0]), None

    calls = {"n": 0}

    def update_fn(battery, preds, probs, target):
        calls["n"] += 1
        value, _weight = target
        battery["m"].update(value)

    out = evaluate(model, data, battery, predict_fn, frozenset(), update_fn=update_fn)
    assert calls["n"] == 1
    assert out["m"] == 2.0  # mean of [1.0, 3.0]


def test_evaluate_default_update_fn_unchanged():
    import torch
    from torchmetrics.classification import MulticlassAccuracy

    from mushin.benchmark._inference import evaluate

    data = [(torch.zeros(3, 2), torch.tensor([0, 1, 2]))]
    model = torch.nn.Identity()
    battery = {"acc": MulticlassAccuracy(num_classes=3, average="micro")}

    def predict_fn(model, x):
        return torch.tensor([0, 1, 2]), None  # all correct

    out = evaluate(model, data, battery, predict_fn, frozenset())  # update_fn omitted
    assert out["acc"] == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_inference.py::test_evaluate_uses_custom_update_fn -v`
Expected: FAIL — `evaluate()` got an unexpected keyword argument `update_fn`.

- [ ] **Step 3: Implement the hook in `_inference.py`**

Add the `Optional` import and `UpdateFn` alias near the top of `src/mushin/benchmark/_inference.py` (the file already has `from collections.abc import Callable, Collection, Iterable` and `from __future__ import annotations`). Add:

```python
from typing import Optional
```

and, right after the existing `PredictFn = ...` line (line 12):

```python
# Owns all metric.update() calls for one batch: (battery, preds, probs, target).
# `Optional[...]` (not `... | None`) because this alias is evaluated at runtime,
# and `X | None` is a TypeError under Python 3.9.
UpdateFn = Callable[
    [dict, torch.Tensor, Optional[torch.Tensor], object], None
]
```

Change the `evaluate` signature (line 67-74) to add the parameter:

```python
def evaluate(
    model: torch.nn.Module,
    data: Iterable,
    battery: dict[str, Metric],
    predict_fn: PredictFn,
    prob_metrics: Collection[str],
    device: torch.device | None = None,
    update_fn: UpdateFn | None = None,
) -> dict[str, float]:
```

Inside `evaluate`, after the `model.eval()` / metric reset block and before the `with torch.no_grad():` loop, install the default update_fn:

```python
    if update_fn is None:

        def update_fn(battery, preds, probs, target):
            for name, metric in battery.items():
                metric.update(probs if name in prob_metrics else preds, target)
```

Replace the per-batch inner loop (current lines 94-99, the `for name, metric in battery.items(): ... metric.update(...)` with the seam comment) with a single call:

```python
    with torch.no_grad():
        for x, y in data:
            x = _to_device(x, device)
            y = _to_device(y, device)
            preds, probs = predict_fn(model, x)
            update_fn(battery, preds, probs, y)
```

- [ ] **Step 4: Add the `Task.update_fn` field in `_tasks.py`**

In `src/mushin/benchmark/_tasks.py`, change the import `from ._inference import PredictFn` to:

```python
from ._inference import PredictFn, UpdateFn
```

Add the field to the `Task` dataclass (after `description: str = ""`):

```python
    update_fn: UpdateFn | None = None
```

(`_tasks.py` has `from __future__ import annotations`, so this field annotation is a deferred string — safe on 3.9.)

- [ ] **Step 5: Pass it through `compare`**

In `src/mushin/benchmark/compare.py`, the `evaluate(...)` call (lines 64-67) currently reads:

```python
        results[name] = [
            evaluate(model, data, battery, fn, pm, device) for model in models
        ]
```

Change it to forward the resolved task's `update_fn`:

```python
        results[name] = [
            evaluate(model, data, battery, fn, pm, device, update_fn=spec.update_fn)
            for model in models
        ]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_benchmark/test_inference.py tests/test_benchmark/test_compare.py tests/test_benchmark/test_tasks.py -v`
Expected: PASS (new update_fn tests + all pre-existing benchmark tests — default path unchanged).

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/_inference.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/compare.py tests/test_benchmark/test_inference.py
uv run ruff format src/mushin/benchmark/_inference.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/compare.py tests/test_benchmark/test_inference.py
git add src/mushin/benchmark/_inference.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/compare.py tests/test_benchmark/test_inference.py
git commit -m "feat: add per-Task update_fn hook for custom metric dispatch"
```

---

### Task 2: Regression battery

Add a shared passthrough predict_fn and the regression battery; register and export it.

**Files:**
- Modify: `src/mushin/benchmark/_predict.py`
- Modify: `src/mushin/benchmark/_metrics.py`
- Modify: `src/mushin/benchmark/_tasks.py`
- Modify: `src/mushin/benchmark/__init__.py`, `src/mushin/__init__.py`
- Test: `tests/test_benchmark/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark/test_metrics.py`:

```python
def test_regression_battery_end_to_end():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    class _AffineModel(torch.nn.Module):
        def __init__(self, w, b):
            super().__init__()
            self.w, self.b = w, b

        def forward(self, x):
            return (x[:, 0] * self.w + self.b)  # shape (N,)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, generator=g)
    y = (x[:, 0] * 2.0 + 1.0)  # true relation
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    good = [_AffineModel(2.0, 1.0) for _ in range(3)]   # exact
    bad = [_AffineModel(0.0, 0.0) for _ in range(3)]    # constant 0

    result = compare(methods={"good": good, "bad": bad}, data=loader, task="regression")
    assert isinstance(result, BenchmarkResult)
    for name in ["mse", "mae", "rmse", "r2", "pearson", "spearman"]:
        assert name in result.data
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_metrics.py::test_regression_battery_end_to_end -v`
Expected: FAIL — `ValueError: task='regression' is not a registered task ...`.

- [ ] **Step 3: Add the shared passthrough predict_fn**

Append to `src/mushin/benchmark/_predict.py`:

```python
def default_passthrough_predict_fn(model: torch.nn.Module, x):
    """Return ``(model(x), None)`` — the model's raw output is the prediction and
    there are no probabilities. Used by tasks with no probability metrics
    (regression, image quality, audio, retrieval), where metrics consume the raw
    output directly against the target."""
    return model(x), None
```

- [ ] **Step 4: Add the regression battery**

In `src/mushin/benchmark/_metrics.py`, add the import near the other torchmetrics imports at the top:

```python
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
)
```

Add the factory (place it after `segmentation_battery`, before the detection section):

```python
def regression_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Scalar-regression battery. ``num_classes``/``ignore_index`` are accepted for
    the uniform task interface but unused. Predictions and targets are continuous
    tensors of matching shape (e.g. ``(N,)``)."""
    return {
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError(),
        "rmse": MeanSquaredError(squared=False),
        "r2": R2Score(),
        "pearson": PearsonCorrCoef(),
        "spearman": SpearmanCorrCoef(),
    }
```

- [ ] **Step 5: Register the task**

In `src/mushin/benchmark/_tasks.py`, add to the imports:

```python
from ._metrics import (
    classification_battery,
    detection_battery,
    regression_battery,
    segmentation_battery,
)
from ._predict import (
    default_classification_predict_fn,
    default_detection_predict_fn,
    default_passthrough_predict_fn,
    default_segmentation_predict_fn,
)
```

Add to the `_TASKS` dict (after the `detection` entry):

```python
    "regression": Task(
        regression_battery,
        default_passthrough_predict_fn,
        frozenset(),
        requires_num_classes=False,
        description="Scalar regression (mse, mae, rmse, r2, pearson, spearman).",
    ),
```

- [ ] **Step 6: Export the battery**

In `src/mushin/benchmark/__init__.py`, add `regression_battery` to the `from ._metrics import (...)` block and to `__all__`. In `src/mushin/__init__.py`, add `regression_battery` to the `from .benchmark import (...)` block and to `__all__`.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_benchmark/test_metrics.py::test_regression_battery_end_to_end tests/test_benchmark/test_import.py -v`
Expected: PASS.

- [ ] **Step 8: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/_predict.py src/mushin/benchmark/_metrics.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_metrics.py
uv run ruff format src/mushin/benchmark/_predict.py src/mushin/benchmark/_metrics.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_metrics.py
git add -A
git commit -m "feat: add regression battery"
```

---

### Task 3: Retrieval battery (uses `update_fn`)

Add the retrieval battery, its `_retrieval_update`, register with `update_fn`, export.

**Files:**
- Modify: `src/mushin/benchmark/_metrics.py`
- Modify: `src/mushin/benchmark/_tasks.py`
- Modify: `src/mushin/benchmark/__init__.py`, `src/mushin/__init__.py`
- Test: `tests/test_benchmark/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark/test_metrics.py`:

```python
def test_retrieval_battery_end_to_end():
    import torch
    from torch.utils.data import DataLoader, Dataset

    from mushin.benchmark import BenchmarkResult, compare

    # Two queries, three docs each. y = (relevance, indexes). The model maps x -> a
    # score; here x already IS the score so Identity ranks perfectly.
    class _RetrievalDS(Dataset):
        def __init__(self):
            self.scores = torch.tensor([0.9, 0.1, 0.2, 0.8, 0.3, 0.7])
            self.rel = torch.tensor([1, 0, 0, 1, 0, 1])
            self.idx = torch.tensor([0, 0, 0, 1, 1, 1])

        def __len__(self):
            return 1  # single batch

        def __getitem__(self, _i):
            return self.scores, (self.rel, self.idx)

    def collate(batch):  # one item; pass tensors through unbatched
        return batch[0]

    loader = DataLoader(_RetrievalDS(), batch_size=1, collate_fn=collate)
    models = [torch.nn.Identity() for _ in range(3)]

    result = compare(methods={"m": models}, data=loader, task="retrieval")
    assert isinstance(result, BenchmarkResult)
    for name in ["retrieval_map", "ndcg", "mrr", "precision", "recall"]:
        assert name in result.data
```

(If the chosen torchmetrics retrieval metrics emit a different scalar key than the battery dict key, the battery dict key wins — see `accumulate_metric`. Keep the dict keys exactly as listed.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_metrics.py::test_retrieval_battery_end_to_end -v`
Expected: FAIL — `task='retrieval' is not a registered task`.

- [ ] **Step 3: Add the battery and the update_fn**

In `src/mushin/benchmark/_metrics.py`, add the import:

```python
from torchmetrics.retrieval import (
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)
```

Add the factory and the update_fn (after `regression_battery`):

```python
def retrieval_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Information-retrieval battery over grouped (per-query) predictions.
    ``num_classes``/``ignore_index`` are accepted for the uniform interface but
    unused. Batches must yield ``y = (relevance, indexes)``; see
    ``retrieval_update``."""
    return {
        "retrieval_map": RetrievalMAP(),
        "ndcg": RetrievalNormalizedDCG(),
        "mrr": RetrievalMRR(),
        "precision": RetrievalPrecision(),
        "recall": RetrievalRecall(),
    }


def retrieval_update(battery, preds, probs, target):
    """update_fn for the retrieval task: ``target`` is a ``(relevance, indexes)``
    tuple, and every retrieval metric takes ``(preds, relevance, indexes=...)``."""
    relevance, indexes = target
    for metric in battery.values():
        metric.update(preds, relevance, indexes=indexes)
```

- [ ] **Step 4: Register the task with update_fn**

In `src/mushin/benchmark/_tasks.py`, extend the `from ._metrics import (...)` block to include `retrieval_battery` and `retrieval_update`. Add to `_TASKS`:

```python
    "retrieval": Task(
        retrieval_battery,
        default_passthrough_predict_fn,
        frozenset(),
        requires_num_classes=False,
        description="Information retrieval (retrieval_map, ndcg, mrr, precision, recall).",
        update_fn=retrieval_update,
    ),
```

- [ ] **Step 5: Export the battery**

Add `retrieval_battery` to `__all__` and the import blocks in both `src/mushin/benchmark/__init__.py` and `src/mushin/__init__.py` (do NOT export `retrieval_update` — it is an internal helper).

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_benchmark/test_metrics.py::test_retrieval_battery_end_to_end tests/test_benchmark/test_import.py -v`
Expected: PASS. If a retrieval metric requires a non-default constructor kwarg to accept the synthetic shapes, set it in the battery (keep the dict keys as specified) and note it in the commit body.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/_metrics.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_metrics.py
uv run ruff format src/mushin/benchmark/_metrics.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_metrics.py
git add -A
git commit -m "feat: add retrieval battery via update_fn hook"
```

---

### Task 4: Image-quality battery + `[image]` extra

Add the image_quality battery (all-or-nothing on `[image]`), the extra, register, export.

**Files:**
- Modify: `src/mushin/benchmark/_metrics.py`
- Modify: `src/mushin/benchmark/_tasks.py`
- Modify: `src/mushin/benchmark/__init__.py`, `src/mushin/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_benchmark/test_image_quality.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark/test_image_quality.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest
import torch


def test_image_quality_missing_extra_raises(monkeypatch):
    # Force the optional LPIPS import/construction to fail and assert the clear
    # missing-extra error, regardless of whether the extra is installed.
    import torchmetrics.image as tmi

    class _Boom:
        def __init__(self, *a, **k):
            raise ImportError("simulated missing lpips")

    monkeypatch.setattr(tmi, "LearnedPerceptualImagePatchSimilarity", _Boom)

    from mushin.benchmark._metrics import image_quality_battery

    with pytest.raises(ImportError, match=r"mushin-py\[image\]"):
        image_quality_battery()


def test_image_quality_battery_end_to_end():
    pytest.importorskip("torchvision")
    pytest.importorskip("lpips")
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    g = torch.Generator().manual_seed(0)
    # MS-SSIM needs images large enough for its 5 scales; 256x256 is safe. Use a
    # near (not exact) reconstruction so PSNR stays finite (identical -> inf).
    ref = torch.rand(2, 3, 256, 256, generator=g)
    gen = (ref + 0.01 * torch.randn(2, 3, 256, 256, generator=g)).clamp(0, 1)
    loader = DataLoader(TensorDataset(gen, ref), batch_size=2)

    class _Recon(torch.nn.Module):
        def forward(self, x):
            return x  # returns the "generated" image; target is the reference

    result = compare(
        methods={"m": [_Recon() for _ in range(3)]},
        data=loader,
        task="image_quality",
    )
    assert isinstance(result, BenchmarkResult)
    for name in ["ssim", "psnr", "ms_ssim", "lpips"]:
        assert name in result.data
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_image_quality.py::test_image_quality_missing_extra_raises -v`
Expected: FAIL — `cannot import name 'image_quality_battery'`.

- [ ] **Step 3: Add the battery (detection-style all-or-nothing)**

In `src/mushin/benchmark/_metrics.py`, add the factory (after the detection section). Construct all four metrics inside one `try`, so a missing optional dep (LPIPS needs torchvision + lpips) is caught and reported clearly — mirroring `detection_battery`:

```python
def image_quality_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Paired image-quality battery (generated vs reference image). Requires the
    optional ``image`` extra (LPIPS pulls in torchvision + lpips). ``num_classes``/
    ``ignore_index`` are accepted for the uniform interface but unused. Images are
    ``(N, C, H, W)``; ``data_range=1.0`` assumes inputs in ``[0, 1]`` and
    ``LearnedPerceptualImagePatchSimilarity(normalize=True)`` accepts that range."""
    try:
        from torchmetrics.image import (
            LearnedPerceptualImagePatchSimilarity,
            MultiScaleStructuralSimilarityIndexMeasure,
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )

        return {
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
            "psnr": PeakSignalNoiseRatio(data_range=1.0),
            "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0),
            "lpips": LearnedPerceptualImagePatchSimilarity(normalize=True),
        }
    except ImportError as e:
        raise ImportError(
            "the image_quality battery requires the optional image extra; install "
            "it with `pip install mushin-py[image]` (torchvision + lpips)."
        ) from e
```

(If the installed torchmetrics needs different kwargs for the synthetic test to pass — e.g. LPIPS `net_type` — adjust the constructor kwargs to make the hermetic test pass while keeping the four dict keys exactly as listed.)

- [ ] **Step 4: Add the `[image]` extra**

In `pyproject.toml` `[project.optional-dependencies]`, after the `detection = [...]` block, add:

```toml
# Image-quality LPIPS needs torchvision (backbone) + the lpips package; SSIM/PSNR/
# MS-SSIM are core torchmetrics but the battery is all-or-nothing, so it requires
# this extra as a whole. torchvision floors pair with the platform-split torch
# floors above.
image = [
    "torchvision >= 0.17,<0.18 ; sys_platform == 'darwin' and platform_machine == 'x86_64'",
    "torchvision >= 0.19 ; sys_platform != 'darwin' or platform_machine != 'x86_64'",
    "lpips >= 0.1.4",
]
```

- [ ] **Step 5: Register + export**

In `src/mushin/benchmark/_tasks.py`, add `image_quality_battery` to the `from ._metrics import (...)` block and add to `_TASKS`:

```python
    "image_quality": Task(
        image_quality_battery,
        default_passthrough_predict_fn,
        frozenset(),
        requires_num_classes=False,
        description="Paired image quality (ssim, psnr, ms_ssim, lpips).",
    ),
```

Export `image_quality_battery` from both `__init__.py` files (`__all__` + import blocks).

- [ ] **Step 6: Run tests**

Run: `uv run --extra image pytest tests/test_benchmark/test_image_quality.py tests/test_benchmark/test_import.py -v`
Expected: the missing-extra test PASSES always; the end-to-end test PASSES if `torchvision`+`lpips` installed, else SKIPS. If the `[image]` extra cannot install on the local platform, run without `--extra` (end-to-end skips) and note that CI will exercise it.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/_metrics.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_image_quality.py
uv run ruff format src/mushin/benchmark/_metrics.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_image_quality.py
git add -A
git commit -m "feat: add image_quality battery behind the [image] extra"
```

---

### Task 5: Audio battery + `[audio]` extra

Add the audio battery (all-or-nothing on `[audio]`), the extra, register, export.

**Files:**
- Modify: `src/mushin/benchmark/_metrics.py`
- Modify: `src/mushin/benchmark/_tasks.py`
- Modify: `src/mushin/benchmark/__init__.py`, `src/mushin/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_benchmark/test_audio.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark/test_audio.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest
import torch


def test_audio_missing_extra_raises(monkeypatch):
    import torchmetrics.audio as tma

    class _Boom:
        def __init__(self, *a, **k):
            raise ImportError("simulated missing pesq")

    monkeypatch.setattr(tma, "PerceptualEvaluationSpeechQuality", _Boom)

    from mushin.benchmark._metrics import audio_battery

    with pytest.raises(ImportError, match=r"mushin-py\[audio\]"):
        audio_battery()


def test_audio_battery_end_to_end():
    pytest.importorskip("pesq")
    pytest.importorskip("pystoi")
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    g = torch.Generator().manual_seed(0)
    # PESQ wideband expects 16 kHz; give ~1 s of audio so it is long enough. Use a
    # near (not exact) reconstruction so SI-SDR/SI-SNR stay finite (identical -> inf).
    ref = torch.randn(2, 16000, generator=g)
    est = ref + 0.01 * torch.randn(2, 16000, generator=g)
    loader = DataLoader(TensorDataset(est, ref), batch_size=2)

    class _Enh(torch.nn.Module):
        def forward(self, x):
            return x

    result = compare(
        methods={"m": [_Enh() for _ in range(3)]},
        data=loader,
        task="audio",
    )
    assert isinstance(result, BenchmarkResult)
    for name in ["si_sdr", "si_snr", "pesq", "stoi"]:
        assert name in result.data
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_audio.py::test_audio_missing_extra_raises -v`
Expected: FAIL — `cannot import name 'audio_battery'`.

- [ ] **Step 3: Add the battery**

In `src/mushin/benchmark/_metrics.py`, add (after `image_quality_battery`):

```python
def audio_battery(
    num_classes: int | None = None, ignore_index: int | None = None
) -> dict[str, Metric]:
    """Speech/audio battery (estimated vs reference waveform). Requires the optional
    ``audio`` extra (PESQ needs ``pesq``; STOI needs ``pystoi``). SI-SDR/SI-SNR are
    core, but the battery is all-or-nothing. ``num_classes``/``ignore_index`` are
    accepted for the uniform interface but unused. Waveforms are ``(N, T)``; PESQ/
    STOI assume a 16 kHz sample rate (override via a custom Task for other rates)."""
    try:
        from torchmetrics.audio import (
            PerceptualEvaluationSpeechQuality,
            ScaleInvariantSignalDistortionRatio,
            ScaleInvariantSignalNoiseRatio,
            ShortTimeObjectiveIntelligibility,
        )

        return {
            "si_sdr": ScaleInvariantSignalDistortionRatio(),
            "si_snr": ScaleInvariantSignalNoiseRatio(),
            "pesq": PerceptualEvaluationSpeechQuality(fs=16000, mode="wb"),
            "stoi": ShortTimeObjectiveIntelligibility(fs=16000),
        }
    except ImportError as e:
        raise ImportError(
            "the audio battery requires the optional audio extra; install it with "
            "`pip install mushin-py[audio]` (pesq + pystoi)."
        ) from e
```

(If `pesq`/`pystoi` constructors need different kwargs for the synthetic test to pass, adjust to make it pass while keeping the four dict keys.)

- [ ] **Step 4: Add the `[audio]` extra**

In `pyproject.toml`, after the `image = [...]` block:

```toml
# Audio PESQ/STOI need the pesq + pystoi C/native packages; SI-SDR/SI-SNR are core
# but the battery is all-or-nothing. Both are gated off win32 where the C build is
# painful (the battery raises a clear error if used without them).
audio = [
    "pesq >= 0.0.4 ; sys_platform != 'win32'",
    "pystoi >= 0.3.3 ; sys_platform != 'win32'",
]
```

- [ ] **Step 5: Register + export**

In `src/mushin/benchmark/_tasks.py`, add `audio_battery` to the `from ._metrics import (...)` block and to `_TASKS`:

```python
    "audio": Task(
        audio_battery,
        default_passthrough_predict_fn,
        frozenset(),
        requires_num_classes=False,
        description="Speech/audio quality (si_sdr, si_snr, pesq, stoi).",
    ),
```

Export `audio_battery` from both `__init__.py` files.

- [ ] **Step 6: Run tests**

Run: `uv run --extra audio pytest tests/test_benchmark/test_audio.py tests/test_benchmark/test_import.py -v`
Expected: missing-extra test PASSES always; end-to-end PASSES if `pesq`+`pystoi` installed, else SKIPS. If the extra cannot install locally, run without `--extra` and rely on CI.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check src/mushin/benchmark/_metrics.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_audio.py
uv run ruff format src/mushin/benchmark/_metrics.py src/mushin/benchmark/_tasks.py src/mushin/benchmark/__init__.py src/mushin/__init__.py tests/test_benchmark/test_audio.py
git add -A
git commit -m "feat: add audio battery behind the [audio] extra"
```

---

### Task 6: Wire the new extras into CI

Exercise the `[image]`/`[audio]` batteries in the `test` CI job, alongside `detection`.

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Update the test job**

In `.github/workflows/ci.yml`, the test job's sync step (line 49) is:

```yaml
        run: uv sync --extra detection --python ${{ matrix.python-version }}
```

Change to:

```yaml
        run: uv sync --extra detection --extra image --extra audio --python ${{ matrix.python-version }}
```

And the run-tests step (line 53):

```yaml
        run: uv run --extra detection pytest tests/ --hypothesis-profile fast -p no:cacheprovider
```

Change to:

```yaml
        run: uv run --extra detection --extra image --extra audio pytest tests/ --hypothesis-profile fast -p no:cacheprovider
```

Also update the comment above line 49 to mention image/audio are exercised on platforms where their deps install (pesq/pystoi are win32-gated, so audio tests skip on Windows via `importorskip`, exactly like pycocotools/detection).

- [ ] **Step 2: Validate the workflow file**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: exercise image and audio batteries in the test matrix"
```

---

### Task 7: Docs + changelog

Document the new batteries, their data contracts, and the `update_fn` extension point.

**Files:**
- Modify: `docs/guides/custom.md`
- Create: `changes/+task-batteries.added.md`

- [ ] **Step 1: Add a "More built-in tasks" section**

Append to `docs/guides/custom.md` (before the `## See also` section — keep "See also" last):

````markdown
## More built-in tasks

Beyond `classification`, `segmentation`, and `detection`, mushin ships these
batteries. Each is `requires_num_classes=False`; the default `predict_fn` returns
`(model(x), None)` and metrics consume the model output against the target.

| task | default metrics | `target` (the `y` in each batch) |
|---|---|---|
| `regression` | mse, mae, rmse, r2, pearson, spearman | continuous tensor, same shape as the output |
| `image_quality` | ssim, psnr, ms_ssim, lpips | reference image `(N, C, H, W)` |
| `audio` | si_sdr, si_snr, pesq, stoi | reference waveform `(N, T)` |
| `retrieval` | retrieval_map, ndcg, mrr, precision, recall | a `(relevance, indexes)` tuple |

```python
from mushin import compare

compare(methods=..., data=regression_loader, task="regression")
```

**Optional extras.** `image_quality` needs `pip install mushin-py[image]`
(torchvision + lpips) and `audio` needs `pip install mushin-py[audio]`
(pesq + pystoi). These batteries are all-or-nothing: they raise a clear error if
the extra is missing. For just the core metrics (e.g. SSIM alone), pass an
explicit `metrics={...}` or build a custom `Task`. Distribution-level metrics
(FID, KID, Inception Score) are not supported by the streaming `compare` loop.

**Retrieval data contract.** Retrieval metrics score documents grouped by query,
so each batch yields `y = (relevance, indexes)`: `relevance` is the per-document
target and `indexes` assigns each document to a query. This is wired through the
task's `update_fn`.

### Custom update step (`update_fn`)

Most tasks update metrics as `metric.update(preds, target)`. When a metric needs a
different call (like retrieval's `indexes`), give the `Task` an `update_fn` that
owns the per-batch dispatch:

```python
from mushin import Task

def my_update(battery, preds, probs, target):
    for metric in battery.values():
        metric.update(preds, target)   # or any signature your metrics need

task = Task(battery=..., predict_fn=..., update_fn=my_update)
```

`update_fn(battery, preds, probs, target)` is called once per batch; when it is
`None` (the default), mushin uses the standard `(preds, target)` loop.
````

- [ ] **Step 2: Create the changelog fragment**

Create `changes/+task-batteries.added.md`:

```markdown
Four new built-in task batteries — `regression`, `image_quality`, `audio`, and
`retrieval` — plus a per-`Task` `update_fn` hook for metrics whose update step is
not `(preds, target)` (used by `retrieval`). LPIPS and PESQ/STOI sit behind the
optional `[image]` and `[audio]` extras. Each battery is exported from `mushin`.
```

- [ ] **Step 3: Verify docs build**

Run: `uv run mkdocs build --strict && rm -rf site`
Expected: builds with no warnings/errors.

- [ ] **Step 4: Commit**

```bash
git add docs/guides/custom.md changes/+task-batteries.added.md
git commit -m "docs: document the new batteries and the update_fn hook"
```

---

### Task 8: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite (with extras where installable)**

Run: `uv run --extra detection --extra image --extra audio pytest -q`
Expected: all pass; optional-dep tests run if their deps installed, else skip cleanly. No new failures vs. `main`.

If the image/audio extras cannot install on the local platform, run `uv run --extra detection pytest -q` instead and note that CI exercises image/audio.

- [ ] **Step 2: Lint, format check, spelling**

```bash
uv run ruff check .
uv run ruff format --check .
uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes
```
Expected: all clean. (codespell is scoped to these paths in CI; do not run it over `docs/` or `site/`.)

- [ ] **Step 3: Import smoke test**

Run: `uv run python -c "import mushin; print(sorted(mushin.list_tasks()))"`
Expected: includes `audio`, `classification`, `detection`, `image_quality`, `regression`, `retrieval`, `segmentation`.

- [ ] **Step 4: Final formatting commit (if needed)**

```bash
git add -A
git commit -m "chore: formatting/lint pass for the new batteries" || echo "nothing to commit"
```

---

## Done criteria

- `compare(task="regression"|"image_quality"|"audio"|"retrieval", ...)` all run end-to-end (image/audio when their extra is installed).
- `Task.update_fn` works; `update_fn=None` is byte-for-byte the old behavior.
- `image_quality`/`audio` raise a clear `mushin-py[image]`/`[audio]` error without the extra.
- `[image]`/`[audio]` extras defined and exercised in CI; `min-versions` job unchanged.
- `list_tasks()` shows all seven tasks; the four new battery factories are exported from `mushin`.
- Full suite green; ruff/format/codespell/mkdocs `--strict` clean.

This is Spec 2 of 2; with it merged, the public Task API effort is complete.
