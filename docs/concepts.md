# Core concepts

## Workflows: sweep → dataset

A mushin **workflow** is a sweep-and-collect pattern. You define your experiment
as a `task(...)` method and run it across a grid of hyperparameters. mushin uses
Hydra under the hood to launch one job per configuration, each in its own
directory, and assembles the returned metrics into a labeled `xarray.Dataset`.

The key insight: the dataset dimensions are your swept parameters (e.g. `lr`,
`seed`), and the data variables are whatever your `task` returns (e.g.
`accuracy`, `loss`). This gives you a structured, labeled result rather than a
list of floats.

See [Workflows & sweeps](guides/workflows.md) and the
[API Reference — workflows](reference/workflows.md).

## The (method × seed) dataset

Reproducible comparison requires running each method across multiple seeds.
mushin structures the evaluation results as an `xarray.Dataset` with dimensions
`method` and `seed`:

```
<xarray.Dataset> Dimensions: (method: 2, seed: 3)
  Coordinates:
    method  (method)  object  'cnn'  'mlp'
    seed    (seed)    int64   0  1  2
  Data variables:
    accuracy   (method, seed)  float64  ...
    f1         (method, seed)  float64  ...
```

This structure makes it natural to:
- Compute per-method means: `ds["accuracy"].mean("seed")`
- Slice a single method: `ds.sel(method="cnn")`
- Export to NetCDF for later analysis: `ds.to_netcdf("results.nc")`

## Statistical comparison: why seeds + significance

Training a model is stochastic (random initialization, data shuffling). A single
run can produce an outlier. By running each method with multiple seeds, mushin
captures the natural variance of training and uses it to answer the question: *is
the observed difference likely to hold up on a new seed?*

mushin applies a pairwise significance test (Welch, Wilcoxon, or Mann-Whitney U)
and corrects for multiple comparisons with the Holm–Bonferroni procedure. The
result tells you not just *which method scored higher on average*, but *whether
that difference is statistically reliable*.

See [Understanding the statistics](guides/statistics.md) for details on test
selection and the Holm correction.

## The task registry seam

`compare` and `Study` accept a `task=` parameter that selects the metric battery
and the default prediction logic:

| `task=` | Battery | Default predict_fn |
|---|---|---|
| `"classification"` | accuracy, f1, precision, recall, auroc, ece | argmax + softmax |
| `"segmentation"` | miou, dice, pixel_acc, precision, recall | argmax + softmax over spatial dims |

You can override either end:
- Pass `metrics=` to replace the battery entirely.
- Pass `predict_fn=` to adapt models that return dicts or non-standard tensors.

See [Custom metrics & predict_fn](guides/custom.md).

## Working directories

!!! warning "Relative paths inside `task()`"
    Each run executes inside Hydra's own per-job output directory, **not** the
    directory you launched from. A relative path like `open("data/train.csv")`
    will silently resolve against the wrong place. Anchor paths to the launch
    directory with [`mushin.original_cwd()`][mushin.original_cwd]:

    ```python
    import mushin
    path = mushin.original_cwd() / "data" / "train.csv"
    ```
