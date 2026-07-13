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

## Frameworks: Lightning-first, sweep layer agnostic

mushin is built on [PyTorch Lightning](https://lightning.ai/) and hydra-zen, and
that's its first-class path. But the two layers differ in how tied to Lightning
they are:

**The sweep layer is framework-agnostic.** `MultiRunMetricsWorkflow` never
inspects your model — it only sweeps configurations and collects the `dict` your
`task` returns. Whatever you train inside `task` is your business, so you can
sweep scikit-learn, XGBoost, JAX, or plain NumPy and still get the labeled
`xarray.Dataset` back:

```python
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

class RidgeSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(alpha: float, seed: int) -> dict:
        from sklearn.linear_model import Ridge  # nothing here requires torch
        model = Ridge(alpha=alpha, random_state=seed).fit(X_train, y_train)
        return dict(r2=model.score(X_val, y_val))

wf = RidgeSweep()
wf.run(alpha=multirun([0.1, 1.0, 10.0]), seed=multirun([0, 1, 2]))
ds = wf.to_xarray()  # dims (alpha, seed), data var r2
```

**The convenience and evaluation layers are PyTorch/Lightning-specific.** These
assume torch models and won't apply to a scikit-learn estimator:

- `HydraDDP` and `MetricsCallback` — Lightning strategy/callback.
- Auto-tuning (`tune_batch_size` / `tune_learning_rate`) — drives Lightning's `Tuner`.
- `compare` and the batteries (`classification`, `segmentation`, `detection`,
  `regression`, `retrieval`, `image_quality`, `audio`) — take
  `torch.nn.Module` models and score them with `torchmetrics`.

So: use the generic sweep→dataset workflow with **any** framework; reach for the
Lightning conveniences and the statistical `compare` batteries when you're
training torch models. There is no scikit-learn *integration* — only the
framework-neutral workflow that happily wraps it.

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
