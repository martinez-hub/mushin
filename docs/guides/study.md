# Studies

`Study` combines the multi-seed training sweep with `compare` into a single
call: define your training functions, specify seeds and evaluation data, and
get back a `BenchmarkResult` — no intermediate bookkeeping.

## Full motion: train + compare

```python
from mushin import Study

study = Study(
    methods={"cnn": train_cnn, "mlp": train_mlp},   # train_fn(seed) -> checkpoint path
    load_fn=LitClassifier.load_from_checkpoint,       # path -> model
    seeds=[0, 1, 2],
    data=test_loader,
    num_classes=10,
    test="welch",
)
result = study.run()   # -> BenchmarkResult
result.summary()
```

`Study.__init__` parameters:

| Parameter | Description |
|---|---|
| `methods` | Dict mapping method name to a `train_fn(seed: int) -> str` (returns checkpoint path). |
| `load_fn` | Callable that loads a checkpoint path into a `torch.nn.Module`. |
| `seeds` | List of integer seeds to train each method on. |
| `data` | Re-iterable data loader for evaluation. |
| `num_classes` | Number of classes (required unless passing custom `metrics` to compare). |
| `task` | `"classification"` (default) or `"segmentation"`. |
| `test` | Statistical test: `"welch"`, `"wilcoxon"`, or `"mannwhitney"`. |
| `alpha` | Significance threshold (default `0.05`). |
| `ignore_index` | For segmentation: label to exclude (e.g. void class). |
| `working_dir` | Directory for Hydra sweep outputs (default: current directory). |

After `study.run()`, the checkpoint paths are stored at `study.checkpoints`
(a `dict[str, list[str]]`) so you can reload them later.

## Eval-only: from_checkpoints

If you already have checkpoints and just want to run the comparison, skip
training entirely:

```python
Study.from_checkpoints(
    checkpoints={
        "cnn": ["cnn_seed0.ckpt", "cnn_seed1.ckpt", "cnn_seed2.ckpt"],
        "mlp": ["mlp_seed0.ckpt", "mlp_seed1.ckpt", "mlp_seed2.ckpt"],
    },
    load_fn=LitClassifier.load_from_checkpoint,
    data=test_loader,
    num_classes=10,
    test="welch",
).run()
```

`Study.from_checkpoints` takes the same evaluation parameters as `Study.__init__`
but accepts a pre-built `checkpoints` dict instead of `methods` and `seeds`.

## See also

- [Comparing methods guide](compare.md) — details on statistical tests and results
- [API Reference — study](../reference/study.md)
