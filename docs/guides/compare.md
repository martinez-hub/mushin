# Comparing methods

!!! note "Requires the `eval` extra"
    `compare` and the metric batteries are mushin's optional evaluation layer —
    install them with `pip install "mushin-py[eval]"`. Importing them without it
    raises a clear install hint. See [Installation](../install.md#optional-extras).

`mushin.benchmark.compare` evaluates a set of trained models on a standard
metric battery and runs pairwise significance tests across methods. Instead of
writing evaluation loops and statistical boilerplate yourself, you hand mushin
a dict of model lists — one list per method, one model per seed — and get back
a publication-ready result.

> **Prefer to follow along?** [Notebook 02 — Compare & batteries](../notebooks/02_compare_and_batteries.ipynb)
> runs a full `compare` example end to end with outputs and a CI bar chart.

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

## See also

- [Built-in batteries](batteries.md) — all seven registered tasks (classification, segmentation, detection, regression, retrieval, image_quality, audio) with real-model recipes and runnable toys
- [Segmentation guide](segmentation.md) — `task="segmentation"` and `ignore_index`
- [Studies guide](study.md) — run training + compare in one call
- [Custom metrics & predict_fn](custom.md) — override the metric battery
- [Understanding the statistics](statistics.md) — tests, Holm correction, effect size
- [API Reference — benchmark](../reference/benchmark.md)
