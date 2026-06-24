# Tutorial

This tutorial walks you through the full mushin workflow end to end: define a
sweep, collect a labeled dataset, compare methods with statistical significance,
and interpret the result.

## Step 1: Define a sweep

mushin workflows are subclasses of `MultiRunMetricsWorkflow`. You implement a
`task(...)` method that returns a dict of metrics, then call `.run(...)` with
swept parameters:

```python
--8<-- "examples/sweep_to_dataset.py:workflow"
```

The `multirun(...)` wrapper tells Hydra to create one job per value. Here the
sweep creates 3 × 3 = 9 jobs (three learning rates × three seeds). Each job
runs in its own output directory; the returned dict is collected automatically.

## Step 2: Collect the dataset

After `.run(...)` completes, call `.to_xarray()` to get a labeled `xarray.Dataset`:

```python
ds = wf.to_xarray()
# <xarray.Dataset> Dimensions: (lr: 3, seed: 3)
#   Data variables: accuracy (lr, seed)

ds["accuracy"].mean("seed")   # average accuracy per learning rate
ds.sel(lr=0.1)                # slice to a single lr
```

The dimensions come from the swept parameters; the data variables come from the
dict your `task` returned.

## Step 3: Compare methods with statistics

Once you have trained models (one list per method, one model per seed), pass
them to `compare`:

```python
--8<-- "examples/compare_classifiers.py:run"
```

`compare` evaluates every model on `data`, assembles an `(method × seed)`
xarray Dataset of metrics, and runs pairwise Holm-corrected significance tests.

## Step 4: Read the statistics

```python
result.summary()
# method | metric   | mean  | ci_low | ci_high | significant_vs_ref
# cnn    | accuracy | 0.963 | 0.951  | 0.975   |
# mlp    | accuracy | 0.941 | 0.928  | 0.954   | *

result.data           # xarray.Dataset, dims (method, seed)
result.comparisons    # tidy DataFrame with p-values and effect sizes
```

`"*"` in `significant_vs_ref` means the method differs significantly from the
reference (first method listed) after Holm correction at `alpha=0.05`.

## Next steps

- [Core concepts](concepts.md) — the mental model behind mushin
- [Comparing methods](guides/compare.md) — deeper coverage of `compare`
- [Studies](guides/study.md) — combine training and comparison in one call
- [Understanding the statistics](guides/statistics.md) — which test to choose
