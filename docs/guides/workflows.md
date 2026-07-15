# Workflows & sweeps

mushin workflows are declarative wrappers around Hydra multirun jobs. You
define your experiment as a method, run it once with swept parameters, and
mushin handles config logging, output directories, and assembling results into
a labeled `xarray.Dataset`.

> **Prefer to follow along?** [Notebook 01 — Sweeps → datasets](../notebooks/01_sweep_to_dataset.ipynb)
> builds a sweep end to end and plots the result.

## The quick path: `@mushin.sweep`

For most sweeps, skip the subclass entirely — decorate a `task`-style function and
call `.run(...)`, which returns the labeled dataset in one step:

```python
import mushin

@mushin.sweep
def experiment(lr, seed):
    ...
    return dict(accuracy=acc)

ds = experiment.run(lr=mushin.multirun([0.01, 0.1]), seed=mushin.multirun([0, 1]))
```

`@mushin.sweep` synthesizes the `MultiRunMetricsWorkflow` subclass below for you.
Reach the full workflow via `experiment.workflow` (last-run instance) or
`experiment.workflow_cls`, or subclass directly when you need `pre_task` /
`jobs_post_process` / a custom `to_xarray`. The rest of this guide uses the class
form to show what is happening under the hood.

## The mental model

A mushin workflow has three steps:

1. **Define** — subclass `MultiRunMetricsWorkflow` and implement a `task(...)` method that returns a dict of metrics (or decorate a function with `@mushin.sweep`).
2. **Run** — call `.run(...)` with `multirun(...)` wrapped arguments to launch a Hydra sweep.
3. **Collect** — call `.to_xarray()` to get a labeled dataset keyed by swept dimensions.

## Runnable example

The following example sweeps learning rates and seeds on a synthetic 2-class
dataset:

```python
--8<-- "examples/sweep_to_dataset.py:workflow"
```

The dict returned from `task` becomes the data variables in the output dataset.
Any kwargs passed to `.run(...)` that are **not** wrapped in `multirun(...)` are
treated as fixed overrides for every run.

## Getting results

```python
ds = wf.to_xarray()
# <xarray.Dataset> Dimensions: (lr: 3, seed: 3)
#   Data variables: accuracy (lr, seed)

ds["accuracy"].mean("seed")     # average over seeds, per learning rate
ds.sel(lr=0.1)                  # slice to a single lr
```

You can also save and reload the dataset as NetCDF (requires the `netcdf` extra):

```python
ds.to_netcdf("results.nc")

import xarray as xr
ds = xr.open_dataset("results.nc")
```

## BaseWorkflow

`BaseWorkflow` is the base class for all mushin workflows. It orchestrates
Hydra jobs and exposes the raw results via `.cfgs`, `.metrics`, and `.jobs`
attributes after `.run(...)` completes.

You rarely subclass `BaseWorkflow` directly — use `MultiRunMetricsWorkflow`
instead, which adds the `to_xarray()` result aggregation layer.

## RobustnessCurve

`RobustnessCurve` is a variant workflow for evaluating model robustness across
perturbation strengths (e.g. noise levels, attack epsilons). It shares the same
sweep-and-aggregate interface as `MultiRunMetricsWorkflow`.

See the [API Reference — workflows](../reference/workflows.md) for full
parameter documentation.

!!! note "Import path"
    `MultiRunMetricsWorkflow` is the class most experiments use, and it stays a
    top-level import (`from mushin import MultiRunMetricsWorkflow`). Its base
    `BaseWorkflow` and the `RobustnessCurve` variant now live in
    `mushin.workflows` — import them as
    `from mushin.workflows import BaseWorkflow, RobustnessCurve`. Accessing them
    as `mushin.BaseWorkflow` / `mushin.RobustnessCurve` still works but is
    deprecated and emits a `DeprecationWarning`.

## hydra_list and multirun

```python
from mushin import multirun, hydra_list
```

- `multirun(values)` — wraps a list as a Hydra multirun override; Hydra creates
  one job per value.
- `hydra_list(values)` — wraps a list as a single Hydra list override; all
  values are passed as a list to one job.

!!! tip "Pitfalls"
    - **task must return a dict:** `MultiRunMetricsWorkflow` collects the
      returned dict as metrics. Returning `None` or a non-dict silently breaks
      `to_xarray()`.
    - **Fixed vs swept args:** Only `multirun(...)`-wrapped args become dataset
      dimensions; fixed args are recorded in the Hydra config but not in the
      xarray dims.
    - **Output directories:** Hydra writes each job's output to a timestamped
      subdirectory. Pass `working_dir=...` to control the root.

## Parallel & out-of-process launchers

By default a sweep runs its cells in-process, sequentially (Hydra's `basic`
launcher). Install a Hydra launcher plugin and pass `launcher=` to parallelize
across cores or submit to a scheduler:

```bash
pip install hydra-joblib-launcher     # local multiprocessing
```
```python
wf.run(..., launcher="joblib")        # loky/processes backend
```

Out-of-process launchers serialize each cell's task to ship it to a worker. The
default joblib (loky) and submitit backends use `cloudpickle`, which handles most
tasks — including lambdas and nested functions. Some backends (joblib's
`multiprocessing` backend, or a pickle-based submitit setup) use the standard
library's `pickle`, which requires your `task` (and any custom `pre_task`) to be
importable (module-level). Keeping tasks module-level is the portable choice.
Resilience (`on_error="nan"`, `resume=True`) and provenance behave identically
out-of-process.

## See also

- [Tutorial](../tutorial.md) — end-to-end: sweep → dataset → compare
- [API Reference — workflows](../reference/workflows.md)
