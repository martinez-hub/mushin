# Benchmark Comparison (`compare`) — Design

*Date: 2026-06-22*

## Goal

Remove the most repeated boilerplate in a scientist's evaluate-and-report loop:
wiring up a standard benchmark, running it across seeds, and doing the
statistical testing by hand. mushin should let a researcher compare trained
methods on a standard battery and get back a labeled `xarray` dataset plus
significance results — in one call.

This is the first feature in mushin's **owned** territory (evaluate + report),
chosen from a real dogfooding pain point. See
[`2026-06-22-mushin-positioning-design.md`](2026-06-22-mushin-positioning-design.md)
for the north star and the owned/inherited/seam boundary this respects.

## Design decisions

| Element | Decision |
| --- | --- |
| Beachhead task type | **Classification** only for v1. Other task types are added later behind the same `task=` seam. |
| Core job | **Compare methods and report significance** ("is my method actually better than the baseline?"). One method is the degenerate case. |
| Variance source | **Multiple trained seeds per method** (the user supplies one trained model per seed). Variance = training randomness; this is the publishable claim. mushin does **not** train. |
| Inference | **mushin runs inference** (model + dataloader → predictions) with a `predict_fn` escape hatch for non-standard models. |
| Metrics | **Delegated to torchmetrics** — mushin never reimplements a metric. |
| Statistics | **Delegated to scipy.stats** — a small registry of paired and independent tests; **Wilcoxon signed-rank is the default**. |

## Owned vs. delegated (north-star check)

- **Owned:** the protocol — the inference runner, the metric battery wiring,
  cross-seed aggregation into a labeled dataset, the statistics layer, and the
  report-ready result object.
- **Delegated (seams):** metric computation → `torchmetrics`; statistical tests
  → `scipy.stats`; training → the user.

This keeps mushin off the integration treadmill: no metric or test is
maintained in-tree.

## Public API

```python
from mushin.benchmark import compare

result = compare(
    methods={
        "ours":     [m0, m1, m2, m3, m4],   # one trained model per seed
        "baseline": [b0, b1, b2, b3, b4],
    },
    data=test_dataloader,                    # torch DataLoader yielding (x, y)
    task="classification",                   # selects the metric battery
    num_classes=10,
    # optional knobs / escape hatches:
    predict_fn=None,        # default: model(x) -> logits -> (preds, probs)
    metrics=None,           # default battery; override to add/subset
    test="wilcoxon",        # significance test (see registry below)
    alpha=0.05,
    device=None,            # default: infer from first model's parameters
)
```

`methods` maps a method name to a sequence of trained models (one per seed).
Methods should share the same number of seeds for paired tests; the registry's
independent tests tolerate unequal counts.

## Components

Each is a small, independently testable unit. Proposed file layout
(`src/mushin/benchmark/`):

- `_predict.py` — `default_classification_predict_fn(model, x) ->
  (preds, probs)`: `model(x)` → logits → `softmax` (probs) and `argmax` (preds).
  Targets are *not* produced here — they come from the dataloader. The
  `predict_fn` knob overrides this for non-standard model outputs.
- `_inference.py` — `run_inference(model, data, predict_fn, device) ->
  (preds, probs, targets)`: set `eval()` + `no_grad()`, move the model and each
  batch to `device`, iterate the dataloader, apply `predict_fn` to each batch's
  `x`, collect the batch targets `y`, and concatenate across batches. Owns device
  handling.
- `_metrics.py` — `classification_battery(num_classes) -> dict[str, Metric]`
  using **torchmetrics** (`task="multiclass"`): accuracy, F1, precision, recall,
  AUROC, ECE (calibration). F1/precision/recall use **macro** averaging by
  default. `compute_metrics(preds, probs, targets, battery) -> dict[str, float]`.
- `_aggregate.py` — `to_dataset(records) -> xarray.Dataset`: stack per-`(method,
  seed)` metric dicts into a Dataset with dims `(method, seed)` and one data
  variable per metric. Same "results as a labeled dataset" contract as the rest
  of mushin.
- `_stats.py` — the statistics layer (see below).
- `_result.py` — `BenchmarkResult` (see Return value).
- `compare.py` — the `compare(...)` facade wiring the above.
- `__init__.py` — exports `compare`, `BenchmarkResult`.

## Statistics layer (`_stats.py`)

**Across seeds (per method, per metric):** mean and a 95% confidence interval
(t-interval over seeds). Reported for every method/metric.

**Between methods (pairwise, per metric):** a significance test chosen from a
**scipy-backed registry**, plus an effect size (Cohen's d) and, when more than
two methods are compared, **Holm–Bonferroni** correction across the pairwise
family.

Test registry (`test=` accepts these keys), each mapping to a `scipy.stats`
function:

| key | scipy function | paired? | notes |
| --- | --- | --- | --- |
| `"wilcoxon"` *(default)* | `wilcoxon` | paired | non-parametric, safe for small N |
| `"ttest_rel"` | `ttest_rel` | paired | parametric paired t-test |
| `"welch"` | `ttest_ind(equal_var=False)` | independent | unequal variances |
| `"ttest_ind"` | `ttest_ind(equal_var=True)` | independent | Student's t-test |
| `"mannwhitney"` | `mannwhitneyu` | independent | non-parametric independent |

Paired tests require aligned, equal-length seed vectors per method (same seeds);
independent tests do not. The registry is a dict so adding a test is a one-line
seam, not a code change to the facade.

**Honest caveats (must surface in `summary()`):** with a typical N≈5 seeds,
significance tests are low-power. `summary()` therefore presents CIs prominently
and never reports a bare p-value without the effect size and seed count.

## Return value

A small `BenchmarkResult`:

- `.data` → the `xarray.Dataset` (`method × seed`, one variable per metric) — the
  raw, sliceable substrate.
- `.comparisons` → tidy `pandas.DataFrame`:
  `(metric, method_a, method_b) -> mean_diff, effect_size, p_value, p_corrected,
  significant`.
- `.summary()` → publication-ready `pandas.DataFrame`: per method/metric
  `mean ± CI`, with significance markers relative to a reference method (the
  first method listed in `methods` by default; overridable).

## New dependencies

- `torchmetrics` — promote from transitive (it ships with pytorch-lightning) to
  an explicit direct dependency.
- `scipy` — new direct dependency.
- `pandas` — promote to an explicit direct dependency (already pulled in by
  `xarray`).

## Testing strategy

Unit tests per component, with tiny deterministic synthetic inputs:

- `_predict`: output shapes/types for a 2-layer model on a 1-batch loader.
- `_metrics`: battery values against hand-computed cases (e.g. a perfect
  classifier → accuracy 1.0; a known confusion → known F1).
- `_aggregate`: dims `(method, seed)` and the expected metric variables present.
- `_stats`: known distributions — identical inputs → p ≈ 1 and effect ≈ 0; a
  clearly-shifted distribution → significant; Holm correction reduces the number
  of "significant" flags vs. raw p-values.

One end-to-end test: two trivial models (one deliberately better) × 3 seeds on a
small synthetic classification loader → assert `.data` dims `(method, seed)`,
the battery variables exist, `.summary()` has a row per method, and the better
model is flagged significant on accuracy.

## Scope / non-goals (v1)

- **Classification only.** Regression/robustness are future task batteries behind
  the same `task=` seam.
- **No training, no orchestration, no dashboards/storage.** Those are the user's
  or other tools' (Hydra, W&B, DVC).
- **No multi-dataset benchmarking** in v1 (single `data` loader). A future
  version may accept multiple datasets as an extra dimension.
- mushin does not own any metric or statistical test implementation — only the
  protocol that wires them.
