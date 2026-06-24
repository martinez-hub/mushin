# Comparing methods

`mushin.benchmark.compare` evaluates a set of trained models on a standard
metric battery and runs pairwise significance tests across methods. Instead of
writing evaluation loops and statistical boilerplate yourself, you hand mushin
a dict of model lists — one list per method, one model per seed — and get back
a publication-ready result.

## Runnable example

The following example trains a small CNN and MLP across seeds on MNIST and
compares them:

```python
--8<-- "examples/compare_classifiers.py:run"
```

`methods` maps a name to a list of trained models — one per seed. All seeds
for all methods are evaluated on `data`. `test="welch"` uses Welch's t-test
(see [Understanding the statistics](statistics.md) for when to choose each
test).

## The metric battery

For `task="classification"`, the default battery includes:

| Metric | Notes |
|---|---|
| `accuracy` | Micro-averaged multiclass accuracy |
| `f1` | Macro-averaged F1 score |
| `precision` | Macro-averaged precision |
| `recall` | Macro-averaged recall |
| `auroc` | Multiclass AUROC |
| `ece` | Expected calibration error |

For `task="segmentation"`, the battery swaps to: mean IoU (`miou`), Dice
(`dice`), pixel accuracy (`pixel_acc`), and macro precision/recall — AUROC and
ECE are not computed. All metrics are computed via torchmetrics. Pass a custom
`metrics` dict to override the defaults (see [Custom metrics](custom.md)).

## Reading the result

```python
result.summary()
# pandas DataFrame: method, metric, mean, ci_low, ci_high, significant_vs_ref
# the first method is the reference; "*" marks a significant difference

result.comparisons
# tidy DataFrame: all pairwise (method_a, method_b, metric)
# columns: mean_diff, effect_size, p_value, p_corrected, significant

result.data
# xarray.Dataset with dims (method, seed), one data variable per metric
# e.g. result.data["accuracy"].sel(method="cnn")  →  per-seed values
```

The `data` attribute is an `xarray.Dataset` with dimensions `method` and
`seed`. You can slice it, compute means over seeds, or export it to NetCDF
for later analysis.

## Statistical tests

The `test` parameter selects the pairwise significance test:

| `test=` | When to use |
|---|---|
| `"welch"` | Continuous metrics with roughly Gaussian distributions; unequal variance assumed. |
| `"wilcoxon"` | Non-normal distributions or ordinal metrics (default). |
| `"mannwhitney"` | Like Wilcoxon but for independent (unpaired) samples. |
| `"ttest_rel"` | Paired t-test; equal variance assumed. |
| `"ttest_ind"` | Independent t-test; equal variance assumed. |

All pairwise comparisons are corrected for multiple testing with the Holm–Bonferroni
procedure at significance level `alpha` (default `0.05`). See
[Understanding the statistics](statistics.md) for full details.

!!! warning "Single-seed behavior"
    With only one seed per method there is no within-method variance, so a
    parametric test returns a NaN p-value and the comparison is reported as
    **not** significant rather than producing a false positive. `compare` also
    warns when the chosen test cannot reach `alpha` at the given seed count
    (e.g. Wilcoxon with very few seeds). Use several seeds for meaningful
    significance.

## Setting alpha

```python
result = compare(..., test="welch", alpha=0.01)
```

Setting `alpha=0.01` applies a stricter threshold. Holm correction adjusts
the per-comparison threshold so the family-wise error rate stays at `alpha`
across all method pairs.

!!! tip "Pitfalls"
    - **Too few seeds:** Wilcoxon cannot reach p < 0.25 with only 3 seeds.
      Use `test="welch"` or increase seed count.
    - **Reusing the test loader:** `data` must be re-iterable (a `DataLoader`,
      not a bare generator) — mushin iterates it once per model.
    - **Forgetting `num_classes`:** Required when `metrics` is not provided.
    - **Mixing tasks:** Pass `task="segmentation"` for pixel-label outputs;
      the default classification battery will give wrong results otherwise.

## See also

- [Segmentation guide](segmentation.md) — `task="segmentation"` and `ignore_index`
- [Studies guide](study.md) — run training + compare in one call
- [Custom metrics & predict_fn](custom.md) — override the metric battery
- [Understanding the statistics](statistics.md) — tests, Holm correction, effect size
- [API Reference — benchmark](../reference/benchmark.md)
