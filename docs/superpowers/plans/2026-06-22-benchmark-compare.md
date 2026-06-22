# Benchmark Comparison (`compare`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mushin.benchmark.compare(...)` — run a standard classification benchmark across user-supplied trained seeds, aggregate into a labeled `xarray.Dataset`, and report significance — so scientists stop re-wiring evaluation boilerplate.

**Architecture:** A `compare()` facade over small, independently-testable units: an inference runner, a torchmetrics metric battery, an xarray aggregator, a scipy-backed statistics layer, and a `BenchmarkResult`. Metrics are delegated to torchmetrics, statistics to scipy; mushin owns only the protocol.

**Tech Stack:** Python, PyTorch, torchmetrics (metrics), scipy.stats (tests), xarray + pandas (results), pytest.

**Governance:** `main` is branch-protected (PR + green CI required). Work on a branch (e.g. `feat-benchmark-compare`) and open a PR. "Commit" steps create commits on that branch. **No AI/Claude attribution in commit messages.** Use **uv** for all commands.

**Spec:** `docs/superpowers/specs/2026-06-22-benchmark-compare-design.md`.

---

## Key facts the implementer must know

- **torchmetrics metrics are stateful.** Calling `metric(preds, targets)` both
  updates internal state and returns a value. Reusing a metric object across
  evaluations accumulates state and corrupts results. Always `metric.reset()`
  before each computation. (Task 4 has a test that catches this.)
- Class-prediction metrics (accuracy/F1/precision/recall) take **integer class
  predictions**; AUROC and ECE take **probabilities**.
- The dataloader is assumed to yield `(x, y)` tuples (`x` = inputs, `y` = integer
  targets). This is documented in the public API.

## File Structure

- `src/mushin/benchmark/__init__.py` — exports `compare`, `BenchmarkResult`.
- `src/mushin/benchmark/_predict.py` — default classification predict fn.
- `src/mushin/benchmark/_inference.py` — inference runner.
- `src/mushin/benchmark/_metrics.py` — torchmetrics battery + compute.
- `src/mushin/benchmark/_aggregate.py` — records → `xarray.Dataset`.
- `src/mushin/benchmark/_stats.py` — CIs, effect size, Holm, test registry, pairwise compare.
- `src/mushin/benchmark/_result.py` — `BenchmarkResult` + `summary()`.
- `src/mushin/benchmark/compare.py` — the facade.
- `tests/test_benchmark/test_*.py` — one test module per unit.

---

### Task 1: Dependencies and package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/mushin/benchmark/__init__.py`
- Test: `tests/test_benchmark/test_import.py`

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

In the `[project]` `dependencies` list, add these three entries (torchmetrics and
pandas are promoted from transitive to explicit; scipy is new):

```toml
    "torchmetrics >= 1.0",
    "scipy >= 1.10",
    "pandas >= 1.5",
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves and installs scipy (torchmetrics/pandas already present).

- [ ] **Step 3: Create the package init**

Create `src/mushin/benchmark/__init__.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard evaluation protocols: run a benchmark, get a labeled dataset back."""

from .compare import compare
from ._result import BenchmarkResult

__all__ = ["compare", "BenchmarkResult"]
```

NOTE: this imports `compare` and `_result`, which do not exist yet — the import
test below will fail until Task 8. So for Task 1, temporarily make `__init__.py`
empty of those imports:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard evaluation protocols: run a benchmark, get a labeled dataset back."""
```

(The exports are added in Task 8.)

- [ ] **Step 4: Write the import test**

Create `tests/test_benchmark/test_import.py`:

```python
def test_benchmark_package_imports():
    import mushin.benchmark  # noqa: F401


def test_third_party_deps_available():
    import scipy.stats  # noqa: F401
    import torchmetrics  # noqa: F401
    import pandas  # noqa: F401
```

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/test_benchmark/test_import.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/mushin/benchmark/__init__.py tests/test_benchmark/test_import.py
git commit -m "Add benchmark package skeleton and deps (torchmetrics, scipy, pandas)"
```

---

### Task 2: Default classification predict function

**Files:**
- Create: `src/mushin/benchmark/_predict.py`
- Test: `tests/test_benchmark/test_predict.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark/test_predict.py
import torch

from mushin.benchmark._predict import default_classification_predict_fn


def test_predict_returns_preds_and_probs():
    model = torch.nn.Linear(4, 3)
    x = torch.randn(5, 4)
    preds, probs = default_classification_predict_fn(model, x)

    assert preds.shape == (5,)
    assert probs.shape == (5, 3)
    # probs are a valid distribution per row
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    # preds are the argmax of probs
    assert torch.equal(preds, probs.argmax(dim=-1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_predict.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mushin.benchmark._predict'`.

- [ ] **Step 3: Implement**

Create `src/mushin/benchmark/_predict.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Default predict step for classification models."""

from __future__ import annotations

import torch


def default_classification_predict_fn(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a classification model on ``x`` and return ``(preds, probs)``.

    Assumes ``model(x)`` returns class logits of shape ``(N, num_classes)``.
    ``probs`` is the softmax over the last dim; ``preds`` is its argmax.
    """
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return preds, probs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_predict.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_predict.py tests/test_benchmark/test_predict.py
git commit -m "Add default classification predict function"
```

---

### Task 3: Inference runner

**Files:**
- Create: `src/mushin/benchmark/_inference.py`
- Test: `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark/test_inference.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from mushin.benchmark._inference import run_inference


def _loader(n=10, d=4):
    x = torch.randn(n, d)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_run_inference_collects_full_dataset():
    model = torch.nn.Linear(4, 3)
    data = _loader(n=10)

    preds, probs, targets = run_inference(model, data)

    assert preds.shape == (10,)
    assert probs.shape == (10, 3)
    assert targets.shape == (10,)
    # targets must match what the loader holds, in order
    expected = torch.cat([y for _, y in data])
    assert torch.equal(targets, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_inference.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/mushin/benchmark/_inference.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Run a model over a dataloader and collect predictions and targets."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch

from ._predict import default_classification_predict_fn

PredictFn = Callable[[torch.nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def run_inference(
    model: torch.nn.Module,
    data: Iterable,
    predict_fn: Optional[PredictFn] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate ``model`` over ``data`` (yielding ``(x, y)``) and return
    ``(preds, probs, targets)`` concatenated across all batches (on CPU)."""
    if predict_fn is None:
        predict_fn = default_classification_predict_fn
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")

    model = model.to(device)
    model.eval()

    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            preds, probs = predict_fn(model, x)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

    return torch.cat(all_preds), torch.cat(all_probs), torch.cat(all_targets)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_inference.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_inference.py tests/test_benchmark/test_inference.py
git commit -m "Add inference runner"
```

---

### Task 4: Classification metric battery

**Files:**
- Create: `src/mushin/benchmark/_metrics.py`
- Test: `tests/test_benchmark/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark/test_metrics.py
import torch

from mushin.benchmark._metrics import classification_battery, compute_metrics


def test_battery_has_expected_metrics():
    battery = classification_battery(num_classes=3)
    assert set(battery) == {"accuracy", "f1", "precision", "recall", "auroc", "ece"}


def test_perfect_classifier_scores():
    battery = classification_battery(num_classes=3)
    preds = torch.tensor([0, 1, 2, 0, 1, 2])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    # confident, correct probabilities
    probs = torch.nn.functional.one_hot(preds, num_classes=3).float()

    out = compute_metrics(preds, probs, targets, battery)
    assert out["accuracy"] == 1.0
    assert out["f1"] == 1.0


def test_metrics_do_not_carry_state_across_calls():
    # reusing the same battery must not accumulate state
    battery = classification_battery(num_classes=3)
    good_preds = torch.tensor([0, 1, 2])
    good_probs = torch.nn.functional.one_hot(good_preds, num_classes=3).float()
    targets = torch.tensor([0, 1, 2])

    first = compute_metrics(good_preds, good_probs, targets, battery)
    second = compute_metrics(good_preds, good_probs, targets, battery)
    assert first["accuracy"] == second["accuracy"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/mushin/benchmark/_metrics.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard classification metric battery, delegated to torchmetrics."""

from __future__ import annotations

import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

# Metrics that require probabilities rather than hard class predictions.
_PROB_METRICS = frozenset({"auroc", "ece"})


def classification_battery(num_classes: int) -> dict[str, Metric]:
    """The standard multiclass battery. F1/precision/recall use macro averaging;
    accuracy uses micro (overall) averaging."""
    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "auroc": MulticlassAUROC(num_classes=num_classes),
        "ece": MulticlassCalibrationError(num_classes=num_classes),
    }


def compute_metrics(
    preds: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor,
    battery: dict[str, Metric],
) -> dict[str, float]:
    """Compute each metric in ``battery``. Resets every metric first so a shared
    battery cannot accumulate state across evaluations."""
    out: dict[str, float] = {}
    for name, metric in battery.items():
        metric.reset()
        inp = probs if name in _PROB_METRICS else preds
        out[name] = float(metric(inp, targets))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_metrics.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_metrics.py tests/test_benchmark/test_metrics.py
git commit -m "Add classification metric battery (torchmetrics)"
```

---

### Task 5: Aggregate records into an xarray Dataset

**Files:**
- Create: `src/mushin/benchmark/_aggregate.py`
- Test: `tests/test_benchmark/test_aggregate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark/test_aggregate.py
from mushin.benchmark._aggregate import to_dataset


def test_to_dataset_shape_and_values():
    results = {
        "ours": [{"accuracy": 0.9, "f1": 0.8}, {"accuracy": 0.92, "f1": 0.81}],
        "base": [{"accuracy": 0.7, "f1": 0.6}, {"accuracy": 0.72, "f1": 0.61}],
    }
    ds = to_dataset(results)

    assert set(ds.dims) == {"method", "seed"}
    assert ds.sizes == {"method": 2, "seed": 2}
    assert set(ds.data_vars) == {"accuracy", "f1"}
    assert list(ds["method"].values) == ["ours", "base"]
    assert float(ds["accuracy"].sel(method="ours").isel(seed=0)) == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_aggregate.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/mushin/benchmark/_aggregate.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Stack per-(method, seed) metric dicts into a labeled xarray Dataset."""

from __future__ import annotations

import numpy as np
import xarray as xr


def to_dataset(results: dict[str, list[dict[str, float]]]) -> xr.Dataset:
    """``results`` maps method name -> list (over seeds) of metric dicts.

    Returns a Dataset with dims ``(method, seed)`` and one data variable per
    metric. All methods must have the same seeds and metric keys."""
    methods = list(results)
    if not methods:
        raise ValueError("`results` is empty")

    n_seeds = len(results[methods[0]])
    metric_names = list(results[methods[0]][0])

    data_vars = {}
    for metric in metric_names:
        arr = np.array(
            [[results[m][s][metric] for s in range(n_seeds)] for m in methods],
            dtype=float,
        )
        data_vars[metric] = (("method", "seed"), arr)

    return xr.Dataset(
        data_vars,
        coords={"method": methods, "seed": np.arange(n_seeds)},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_aggregate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_aggregate.py tests/test_benchmark/test_aggregate.py
git commit -m "Add xarray aggregation of benchmark results"
```

---

### Task 6: Statistics layer

**Files:**
- Create: `src/mushin/benchmark/_stats.py`
- Test: `tests/test_benchmark/test_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark/test_stats.py
import numpy as np

from mushin.benchmark._aggregate import to_dataset
from mushin.benchmark._stats import (
    cohens_d,
    compare_methods,
    confidence_interval,
    holm_correction,
)


def test_confidence_interval_brackets_mean():
    mean, lo, hi = confidence_interval([0.8, 0.82, 0.79, 0.81, 0.80])
    assert lo < mean < hi
    assert abs(mean - 0.804) < 1e-6


def test_cohens_d_zero_for_identical():
    assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_holm_is_monotone_and_capped():
    corrected = holm_correction([0.01, 0.04, 0.03])
    assert all(0.0 <= c <= 1.0 for c in corrected)
    # smallest raw p gets multiplied by the most (m=3)
    assert corrected[0] >= 0.01 * 3 - 1e-9


def test_compare_flags_clear_difference_as_significant():
    rng = np.random.default_rng(0)
    results = {
        "ours": [{"accuracy": float(v)} for v in 0.90 + 0.01 * rng.standard_normal(8)],
        "base": [{"accuracy": float(v)} for v in 0.70 + 0.01 * rng.standard_normal(8)],
    }
    ds = to_dataset(results)
    df = compare_methods(ds, test="wilcoxon", alpha=0.05)

    row = df[df["metric"] == "accuracy"].iloc[0]
    assert row["significant"]
    assert row["mean_diff"] > 0  # ours - base


def test_welch_test_runs():
    results = {
        "a": [{"accuracy": 0.9}, {"accuracy": 0.92}, {"accuracy": 0.88}],
        "b": [{"accuracy": 0.7}, {"accuracy": 0.72}, {"accuracy": 0.68}],
    }
    ds = to_dataset(results)
    df = compare_methods(ds, test="welch", alpha=0.05)
    assert "p_value" in df.columns and len(df) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_stats.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/mushin/benchmark/_stats.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Statistics for benchmark comparison, delegated to scipy.stats."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# name -> (callable returning (statistic, pvalue), is_paired)
_TESTS = {
    "wilcoxon": (lambda a, b: stats.wilcoxon(a, b), True),
    "ttest_rel": (lambda a, b: stats.ttest_rel(a, b), True),
    "welch": (lambda a, b: stats.ttest_ind(a, b, equal_var=False), False),
    "ttest_ind": (lambda a, b: stats.ttest_ind(a, b, equal_var=True), False),
    "mannwhitney": (lambda a, b: stats.mannwhitneyu(a, b), False),
}


def available_tests() -> list[str]:
    return list(_TESTS)


def confidence_interval(
    values, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` using a Student-t interval."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, mean, mean
    half = float(stats.sem(values) * stats.t.ppf(1 - alpha / 2, n - 1))
    return mean, mean - half, mean + half


def cohens_d(a, b) -> float:
    """Pooled-variance Cohen's d for two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_sd = float(np.sqrt(pooled_var))
    if pooled_sd == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def holm_correction(pvalues) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns corrected p-values in the
    original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    corrected = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        corrected[idx] = min(running, 1.0)
    return [float(c) for c in corrected]


def compare_methods(
    ds: xr.Dataset, test: str = "wilcoxon", alpha: float = 0.05
) -> pd.DataFrame:
    """Pairwise comparison of methods for every metric in ``ds``.

    Returns a tidy DataFrame with columns: metric, method_a, method_b,
    mean_diff, effect_size, p_value, p_corrected, significant. Holm correction is
    applied per metric across the method pairs."""
    if test not in _TESTS:
        raise ValueError(f"unknown test {test!r}; choose from {available_tests()}")
    func, _ = _TESTS[test]
    methods = [str(m) for m in ds["method"].values]

    rows = []
    for metric in ds.data_vars:
        recs, pvals = [], []
        for a, b in itertools.combinations(methods, 2):
            va = ds[metric].sel(method=a).values
            vb = ds[metric].sel(method=b).values
            _, p = func(va, vb)
            recs.append(
                {
                    "metric": str(metric),
                    "method_a": a,
                    "method_b": b,
                    "mean_diff": float(np.mean(va) - np.mean(vb)),
                    "effect_size": cohens_d(va, vb),
                    "p_value": float(p),
                }
            )
            pvals.append(float(p))
        corrected = holm_correction(pvals) if len(pvals) > 1 else pvals
        for rec, pc in zip(recs, corrected):
            rec["p_corrected"] = float(pc)
            rec["significant"] = bool(pc < alpha)
            rows.append(rec)

    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "method_a",
            "method_b",
            "mean_diff",
            "effect_size",
            "p_value",
            "p_corrected",
            "significant",
        ],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_stats.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_stats.py tests/test_benchmark/test_stats.py
git commit -m "Add scipy-backed statistics layer (CIs, effect size, Holm, test registry)"
```

---

### Task 7: BenchmarkResult and summary table

**Files:**
- Create: `src/mushin/benchmark/_result.py`
- Test: `tests/test_benchmark/test_result.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark/test_result.py
from mushin.benchmark._aggregate import to_dataset
from mushin.benchmark._result import BenchmarkResult
from mushin.benchmark._stats import compare_methods


def _result():
    results = {
        "ours": [{"accuracy": 0.90}, {"accuracy": 0.91}, {"accuracy": 0.92}],
        "base": [{"accuracy": 0.70}, {"accuracy": 0.71}, {"accuracy": 0.72}],
    }
    ds = to_dataset(results)
    comparisons = compare_methods(ds, test="welch", alpha=0.05)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=0.05)


def test_summary_has_row_per_method_metric():
    summary = _result().summary()
    # 2 methods x 1 metric
    assert len(summary) == 2
    assert set(summary.columns) >= {
        "method", "metric", "mean", "ci_low", "ci_high", "significant_vs_ref"
    }


def test_summary_marks_significant_against_reference():
    summary = _result().summary()  # reference defaults to first method ("ours")
    base_row = summary[summary["method"] == "base"].iloc[0]
    assert base_row["significant_vs_ref"] == "*"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_result.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/mushin/benchmark/_result.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The object returned by ``compare``: dataset + comparisons + summary table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import xarray as xr

from ._stats import confidence_interval


@dataclass
class BenchmarkResult:
    """Holds benchmark results.

    Attributes
    ----------
    data : xarray.Dataset
        Dims ``(method, seed)``, one data variable per metric.
    comparisons : pandas.DataFrame
        Pairwise significance results (see ``_stats.compare_methods``).
    alpha : float
        Significance level used.
    """

    data: xr.Dataset
    comparisons: pd.DataFrame
    alpha: float = 0.05

    def summary(self, reference: Optional[str] = None) -> pd.DataFrame:
        """Publication-ready table: per method/metric ``mean`` and CI, with a
        ``"*"`` marker when the method differs significantly from ``reference``
        (default: the first method in ``data``)."""
        methods = [str(m) for m in self.data["method"].values]
        ref = reference if reference is not None else methods[0]

        rows = []
        for method in methods:
            for metric in self.data.data_vars:
                vals = self.data[metric].sel(method=method).values
                mean, lo, hi = confidence_interval(vals, self.alpha)
                marker = ""
                if method != ref:
                    match = self.comparisons[
                        (self.comparisons["metric"] == str(metric))
                        & (
                            (
                                (self.comparisons["method_a"] == method)
                                & (self.comparisons["method_b"] == ref)
                            )
                            | (
                                (self.comparisons["method_a"] == ref)
                                & (self.comparisons["method_b"] == method)
                            )
                        )
                    ]
                    if len(match) and bool(match.iloc[0]["significant"]):
                        marker = "*"
                rows.append(
                    {
                        "method": method,
                        "metric": str(metric),
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "significant_vs_ref": marker,
                    }
                )
        return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_result.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/benchmark/_result.py tests/test_benchmark/test_result.py
git commit -m "Add BenchmarkResult with publication-ready summary table"
```

---

### Task 8: The `compare` facade

**Files:**
- Create: `src/mushin/benchmark/compare.py`
- Modify: `src/mushin/benchmark/__init__.py`
- Test: `tests/test_benchmark/test_compare.py`

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/test_benchmark/test_compare.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from mushin.benchmark import BenchmarkResult, compare


def _loader(seed, n=64, d=4, num_classes=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


class _Perfect(torch.nn.Module):
    """Cheats: reads the target off a fixed mapping -> always correct."""

    def __init__(self, loader, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self._map = {tuple(x.tolist()): int(y) for xb, yb in loader for x, y in zip(xb, yb)}

    def forward(self, x):
        idx = torch.tensor([self._map[tuple(row.tolist())] for row in x])
        return torch.nn.functional.one_hot(idx, self.num_classes).float() * 10.0


def test_compare_end_to_end():
    data = _loader(seed=0)
    good = [_Perfect(data) for _ in range(3)]
    bad = [torch.nn.Linear(4, 3) for _ in range(3)]

    result = compare(
        methods={"good": good, "bad": bad},
        data=data,
        task="classification",
        num_classes=3,
        test="welch",
    )

    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert result.data.sizes == {"method": 2, "seed": 3}
    assert "accuracy" in result.data.data_vars
    # the perfect method is perfect
    assert float(result.data["accuracy"].sel(method="good").mean()) == 1.0
    # summary has a row per method/metric
    assert len(result.summary()) == 2 * len(result.data.data_vars)


def test_compare_rejects_unknown_task():
    import pytest

    with pytest.raises(NotImplementedError):
        compare(methods={"a": []}, data=[], task="regression", num_classes=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark/test_compare.py -q`
Expected: FAIL with `ImportError` (cannot import `compare`/`BenchmarkResult`).

- [ ] **Step 3: Implement the facade**

Create `src/mushin/benchmark/compare.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The ``compare`` facade: run a benchmark across methods and seeds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import torch

from ._aggregate import to_dataset
from ._inference import PredictFn, run_inference
from ._metrics import classification_battery, compute_metrics
from ._result import BenchmarkResult
from ._stats import compare_methods


def compare(
    methods: dict[str, Sequence[torch.nn.Module]],
    data: Iterable,
    task: str = "classification",
    *,
    num_classes: int,
    predict_fn: Optional[PredictFn] = None,
    metrics: Optional[dict] = None,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    device: Optional[torch.device] = None,
) -> BenchmarkResult:
    """Compare methods on a standard benchmark and report significance.

    Parameters
    ----------
    methods : dict[str, Sequence[Module]]
        Method name -> one trained model per seed.
    data : Iterable
        A dataloader yielding ``(x, y)`` batches.
    task : str
        Only ``"classification"`` is supported in this version.
    num_classes : int
        Number of classes (keyword-only).
    test : str
        Significance test key (default ``"wilcoxon"``). See
        ``mushin.benchmark._stats.available_tests``.

    Returns
    -------
    BenchmarkResult
    """
    if task != "classification":
        raise NotImplementedError(
            f"task={task!r} is not supported; only 'classification' in this version"
        )

    battery = metrics if metrics is not None else classification_battery(num_classes)

    results: dict[str, list[dict[str, float]]] = {}
    for name, models in methods.items():
        per_seed = []
        for model in models:
            preds, probs, targets = run_inference(model, data, predict_fn, device)
            per_seed.append(compute_metrics(preds, probs, targets, battery))
        results[name] = per_seed

    ds = to_dataset(results)
    comparisons = compare_methods(ds, test=test, alpha=alpha)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
```

- [ ] **Step 4: Wire up the package exports**

Replace `src/mushin/benchmark/__init__.py` with:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard evaluation protocols: run a benchmark, get a labeled dataset back."""

from ._result import BenchmarkResult
from .compare import compare

__all__ = ["compare", "BenchmarkResult"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark/test_compare.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run codespell src tests examples README.md CHANGELOG.md && uv run pytest tests/ --hypothesis-profile fast -p no:cacheprovider -q`
Expected: ruff clean, format clean, codespell clean, all tests pass (the prior suite plus the new `tests/test_benchmark/` modules). If ruff format reports changes, run `uv run ruff format .` and include them.

- [ ] **Step 7: Commit**

```bash
git add src/mushin/benchmark/compare.py src/mushin/benchmark/__init__.py tests/test_benchmark/test_compare.py
git commit -m "Add compare facade and wire up benchmark package exports"
```

- [ ] **Step 8: Open the PR** (controller handles after final review)

Leave the branch ready; the controller opens the PR after the whole-implementation review.

---

## Self-review notes

- **Spec coverage:** public API (Task 8), components — predict (2), inference (3),
  metrics battery via torchmetrics (4), aggregate→xarray (5), stats with scipy
  registry incl. Wilcoxon default + Welch (6), BenchmarkResult/.data/.comparisons
  /.summary (7), new deps (1). Statefulness caveat covered by a test in Task 4.
  Honest small-N caveat is surfaced via CIs in `summary()` (Task 7).
- **Out of spec, intentionally deferred:** multi-dataset benchmarking; non-
  classification batteries (the `task=` guard rejects them with NotImplementedError).
- **Type/name consistency:** `default_classification_predict_fn(model, x) ->
  (preds, probs)`; `run_inference(...) -> (preds, probs, targets)`;
  `classification_battery(num_classes)`; `compute_metrics(preds, probs, targets,
  battery)`; `to_dataset(results)`; `compare_methods(ds, test, alpha)`;
  `confidence_interval`, `cohens_d`, `holm_correction`; `BenchmarkResult(data,
  comparisons, alpha)`; `compare(methods, data, task, *, num_classes, ...)`.
  Names are used consistently across tasks.
