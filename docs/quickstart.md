# Quickstart

This page walks through the flagship example: run a parameter sweep and get the
results back as a labeled `xarray.Dataset`.

The full runnable script is at `examples/sweep_to_dataset.py` in the repository.

## Define the workflow

Subclass `MultiRunMetricsWorkflow` and implement a static `task` method.
Whatever the method returns becomes a data variable in the output dataset —
no callbacks, no logging framework, just a plain dict.

```python
import torch as tr
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

LEARNING_RATES = [0.01, 0.1, 1.0]
SEEDS = [0, 1, 2]
POINTS_PER_CLASS = 256


def _make_data(seed: int, n: int = POINTS_PER_CLASS) -> tuple[tr.Tensor, tr.Tensor]:
    g = tr.Generator().manual_seed(seed)
    x0 = tr.randn(n, 2, generator=g) + tr.tensor([2.0, 2.0])
    x1 = tr.randn(n, 2, generator=g) + tr.tensor([-2.0, -2.0])
    x = tr.cat([x0, x1])
    y = tr.cat([tr.zeros(n), tr.ones(n)])
    return x, y


class LRSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr: float, seed: int) -> dict:
        tr.manual_seed(seed)
        x, y = _make_data(seed)
        model = tr.nn.Linear(2, 1)
        opt = tr.optim.SGD(model.parameters(), lr=lr)
        for _ in range(100):
            opt.zero_grad()
            logits = model(x).squeeze(1)
            loss = tr.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
        with tr.no_grad():
            preds = (model(x).squeeze(1) > 0).float()
            acc = (preds == y).float().mean().item()
        # returning the dict is what populates the dataset; saving is optional
        result = dict(accuracy=acc)
        tr.save(result, "metrics.pt")
        return result
```

## Run the sweep

Call `wf.run(...)` with `multirun(...)` wrapped arguments. Hydra launches one
job per combination — 3 learning rates × 3 seeds = 9 runs total.

```python
wf = LRSweep()
wf.run(
    lr=multirun(LEARNING_RATES),
    seed=multirun(SEEDS),
)
```

> **Heads up:** your `task()` runs in a per-job directory. If it reads or writes
> files by relative path, wrap them with `mushin.original_cwd() / "..."` — see
> [Concepts](concepts.md#working-directories).

## Get results as a dataset

```python
ds = wf.to_xarray()
print(ds)
```

Expected output:

```
<xarray.Dataset>
Dimensions:   (lr: 3, seed: 3)
Coordinates:
  * lr        (lr) float64 0.01 0.1 1.0
  * seed      (seed) int64 0 1 2
Data variables:
    accuracy  (lr, seed) float64 ...
```

From there, standard xarray/pandas operations apply:

```python
# average accuracy across seeds, per learning rate
mean_acc = ds["accuracy"].mean("seed")
print(mean_acc)

# plot
import matplotlib.pyplot as plt
mean_acc.plot.line(x="lr", marker="o")
plt.xscale("log")
plt.savefig("sweep_accuracy.png", dpi=120, bbox_inches="tight")
```

## Run the full example

```bash
uv run python examples/sweep_to_dataset.py
```

## Next steps

- [Workflows & sweeps guide](guides/workflows.md) — more on `BaseWorkflow` and `MultiRunMetricsWorkflow`
- [Comparing methods guide](guides/compare.md) — evaluate trained models with statistics
- [Studies guide](guides/study.md) — combine training + compare in one call
- [API Reference — workflows](reference/workflows.md)
