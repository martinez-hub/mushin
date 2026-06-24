# Comparing methods

`mushin.benchmark.compare` evaluates a set of trained models on a standard
metric battery and runs pairwise significance tests across methods.

## Basic usage

```python
from mushin.benchmark import compare

result = compare(
    methods={"ours": [m0, m1, m2], "baseline": [b0, b1, b2]},
    data=test_loader,
    task="classification",
    num_classes=10,
    test="welch",
)
```

`methods` maps a name to a list of trained models — one per seed. All seeds
for all methods are evaluated on `data`.

## The metric battery

For `task="classification"`, the default battery includes accuracy, macro F1,
and AUROC (computed via torchmetrics). For `task="segmentation"`, it includes
mean IoU and pixel accuracy. You can pass a custom `metrics` dict to override
the defaults.

## Statistical tests

The `test` parameter selects the pairwise significance test:

| `test=` | When to use |
|---|---|
| `"welch"` | Continuous metrics with roughly Gaussian distributions; unequal variance assumed. |
| `"wilcoxon"` | Non-normal distributions or ordinal metrics (default). |
| `"mannwhitney"` | Like Wilcoxon but for independent (unpaired) samples. |

All pairwise comparisons are corrected for multiple testing with the Holm
procedure at significance level `alpha` (default `0.05`).

!!! warning "Single-seed behavior"
    If any method has only one seed, statistical tests cannot run (no
    variance to compare). `compare` will emit a warning and skip significance
    testing for that method rather than producing false significance.

## Reading the result

```python
result.summary()
# prints: mean ± 95% CI per method, with significance markers — paper-ready

result.comparisons
# tidy DataFrame: all pairwise (method_a, method_b, metric) → p-value, effect size

result.data
# xarray.Dataset with dims (method, seed), one variable per metric
```

## Alpha and Holm correction

```python
result = compare(..., test="welch", alpha=0.01)
```

Setting `alpha=0.01` applies a stricter threshold. Holm correction adjusts
the per-comparison threshold so the family-wise error rate stays at `alpha`
across all comparisons.

## See also

- [Segmentation guide](segmentation.md) — `task="segmentation"` and
  `ignore_index`
- [Studies guide](study.md) — run training + compare in one call
- [API Reference — benchmark](../reference/benchmark.md)
