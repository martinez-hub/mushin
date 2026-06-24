# Workflows & sweeps

mushin workflows are declarative wrappers around Hydra multirun jobs. You
define your experiment as a method, run it once with swept parameters, and
mushin handles config logging, output directories, and assembling results.

## BaseWorkflow

`BaseWorkflow` is the base class for all mushin workflows. It orchestrates
Hydra jobs and exposes the raw results via `.cfgs`, `.metrics`, and `.jobs`
attributes after `.run(...)` completes.

You rarely subclass `BaseWorkflow` directly — use `MultiRunMetricsWorkflow`
instead, which adds the `to_xarray()` result aggregation layer.

## MultiRunMetricsWorkflow

`MultiRunMetricsWorkflow` is the standard sweep workflow:

1. Subclass it and implement a static `task(...)` method that returns a dict.
2. Call `.run(...)` with `multirun(...)` wrapped arguments to launch a sweep.
3. Call `.to_xarray()` to get a labeled `xarray.Dataset` keyed by swept dims.

### Defining a workflow

```python
import torch as tr
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

class LRSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr: float, seed: int) -> dict:
        tr.manual_seed(seed)
        # ... train and evaluate ...
        return dict(accuracy=acc, loss=val_loss)
```

The dict returned from `task` becomes the data variables in the output dataset.
Any kwargs passed to `.run(...)` that are **not** wrapped in `multirun(...)` are
treated as fixed overrides for every run.

### Running a sweep

```python
wf = LRSweep()
wf.run(lr=multirun([0.01, 0.1, 1.0]), seed=multirun([0, 1, 2]))
```

This launches 9 Hydra jobs (3 lrs × 3 seeds), each in its own output directory.
Hydra writes the resolved config to `.hydra/` inside each output dir.

### Getting results

```python
ds = wf.to_xarray()
# <xarray.Dataset> Dimensions: (lr: 3, seed: 3)
#   Data variables: accuracy (lr, seed), loss (lr, seed)

ds["accuracy"].mean("seed")     # average over seeds, per learning rate
ds.sel(lr=0.1)                   # slice to a single lr
```

You can also save and reload the dataset as NetCDF (requires the `netcdf` extra):

```python
ds.to_netcdf("results.nc")

import xarray as xr
ds = xr.open_dataset("results.nc")
```

## RobustnessCurve

`RobustnessCurve` is a variant workflow for evaluating model robustness across
perturbation strengths (e.g. noise levels, attack epsilons). It shares the same
sweep-and-aggregate interface as `MultiRunMetricsWorkflow`.

See the [API Reference — workflows](../reference/workflows.md) for full
parameter documentation.

## hydra_list and multirun

`from mushin import multirun, hydra_list`

- `multirun(values)` — wraps a list as a Hydra multirun override; Hydra creates
  one job per value.
- `hydra_list(values)` — wraps a list as a single Hydra list override; all
  values are passed as a list to one job.
