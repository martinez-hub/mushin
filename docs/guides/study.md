# Studies

`Study` combines a multi-seed training sweep with `compare` into a single call:
define your training functions once, specify seeds and evaluation data, and get
back a `BenchmarkResult` — no intermediate bookkeeping, no manual checkpoint
management.

## Full motion: train + compare

The example below trains a CNN and MLP across seeds using `Study`, writing
checkpoints to disk and comparing the loaded models:

### Defining train functions

```python
--8<-- "examples/study_mnist.py:train_fn"
```

Each `train_fn(seed: int) -> str` trains a model for the given seed, saves it,
and returns the checkpoint path. `Study` calls every `train_fn` for every seed
and stores the resulting paths.

### Running the study

```python
--8<-- "examples/study_mnist.py:run"
```

`Study` parameters:

| Parameter | Description |
|---|---|
| `methods` | Dict mapping method name to a `train_fn(seed: int) -> str` (returns checkpoint path). |
| `load_fn` | Callable that loads a checkpoint path into a `torch.nn.Module`. |
| `seeds` | List of integer seeds to train each method on. |
| `data` | Re-iterable data loader for evaluation. |
| `num_classes` | Number of classes (required — `Study` evaluates with the default battery for its `task`). |
| `task` | `"classification"` (default) or `"segmentation"`. |
| `test` | Statistical test: `"welch"`, `"wilcoxon"`, `"mannwhitney"`, etc. |
| `alpha` | Significance threshold (default `0.05`). |
| `ignore_index` | For segmentation: label to exclude (e.g. void class). |
| `working_dir` | Directory for Hydra sweep outputs (default: current directory). |
| `on_error` | `"raise"` (default) crashes on a failed training run; `"nan"` records it and keeps going. |
| `resume` | `True` re-runs only the failed/missing `(method, seed)` runs in `working_dir` and reuses the rest. |
| `capture_env` | `True` writes a full dependency snapshot alongside the per-run provenance. |

After `study.run()`, the checkpoint paths are stored at `study.checkpoints`
(`dict[str, list[str]]`) and `study.working_dir` records the resolved directory.

### Resilient and resumable studies

A `Study` runs a real training sweep, so long runs can die partway. The same
resilience the workflows have applies here: with `on_error="nan"` a failed
training run is recorded rather than crashing the whole study, and
`Study.run()` then raises `IncompleteSweepError` — you fix the cause and re-run
with `resume=True` (same `working_dir`) to train only what's missing, then it
proceeds to `compare`. Statistics never run on an incomplete study. See the
[Resilient & resumable sweeps guide](resilience.md) for the full loop.

## Annotated output

```python
result.summary()
# method | metric    | mean   | ci_low | ci_high | significant_vs_ref
# cnn    | accuracy  | 0.963  | 0.951  | 0.975   |
# mlp    | accuracy  | 0.941  | 0.928  | 0.954   | *

result.data
# xarray.Dataset, dims (method, seed): one variable per metric
result.comparisons
# tidy DataFrame: pairwise p-values, effect sizes, Holm-corrected significance
```

`"*"` in `significant_vs_ref` means the method differs significantly from the
reference (the first method listed) after Holm correction.

## Eval-only: from_checkpoints

Already have checkpoints? Skip training entirely:

```python
Study.from_checkpoints(
    checkpoints={
        "cnn": ["cnn_seed0.pt", "cnn_seed1.pt", "cnn_seed2.pt"],
        "mlp": ["mlp_seed0.pt", "mlp_seed1.pt", "mlp_seed2.pt"],
    },
    load_fn=lambda p: torch.load(p, weights_only=False),
    data=val_loader,
    num_classes=10,
    test="welch",
).run()
```

`Study.from_checkpoints` takes the same evaluation parameters as `Study.__init__`
but accepts a pre-built `checkpoints` dict instead of `methods` and `seeds`.

!!! tip "Pitfalls"
    - **train_fn must return a path:** If it returns `None`, `Study` raises
      a `ValueError`. Always return the saved checkpoint path.
    - **Re-iterable data:** `data` must be a `DataLoader`, not a one-shot
      iterator — it is evaluated once per model.
    - **working_dir and Hydra:** `Study` runs a Hydra sweep internally; if
      Hydra's working-directory change behavior conflicts with your setup,
      pass an explicit `working_dir`.

## See also

- [Comparing methods guide](compare.md) — details on statistical tests and results
- [API Reference — study](../reference/study.md)
