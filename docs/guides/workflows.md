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

## Sweep-axis support and limits

`to_xarray` assembles a **Cartesian grid**: every sweep axis must be a discrete
list of values, and every combination is one cell.

- **Nested (dotted) params** work as axes: `wf.run(**{"model.width":
  multirun([4, 8])})` sweeps a nested config path and becomes a normal
  dimension (select it with `ds.sel({"model.width": 8})`).
- **Config groups** work as axes when the config declares the target field:
  register your options (e.g. `cs.store(group="model", name="small", ...)`),
  give the workflow `eval_task_cfg=make_config(model=None)`, and sweep
  `model=multirun(["small", "big"])`. The dataset coordinate is the chosen
  option *name*.
- **`range(1,5)`-style overrides** (e.g. in a sweep dir you re-load) expand to
  their discrete values.
- **Not supported:** continuous `interval(...)` syntax and adaptive sweeper
  plugins (Optuna/Nevergrad/Ax). Those sample points instead of forming a
  grid, which `to_xarray` cannot assemble; mushin raises a clear error rather
  than producing an all-NaN dataset. To use a searcher, run it as a separate
  step and feed the winners to a mushin grid — see
  [Hyperparameter search](#hyperparameter-search).

## Using mushin alongside your experiment tracker

mushin is *not* a tracker and does not replace one: it owns the sidecar
metrics and the final dataset; W&B/TensorBoard/MLflow own live curves,
system metrics, and collaboration. They compose cleanly — mushin never
touches your `Trainer`'s logger, so attach one inside `task()` as usual:

```python
class Experiment(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr: float, seed: int):
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project="my-sweep", name=f"lr={lr}-seed={seed}")
        trainer = pl.Trainer(logger=logger, ...)
        trainer.fit(model, datamodule=dm)
        wandb.finish()  # one run per sweep cell
        # Return the tracker's run id alongside your metrics: it becomes a
        # data variable, so every dataset cell links back to its W&B run.
        return dict(accuracy=float(acc), wandb_run_id=logger.experiment.id)
```

Each sweep cell runs in its own Hydra job directory (Hydra `chdir`s into it),
so file-based loggers like TensorBoard write per-cell logs there — point
`TensorBoardLogger(save_dir=...)` at a fixed path if you want one aggregate
log dir instead.

## Hyperparameter search

mushin *does* hyperparameter search — as **grid search** (a `multirun` per axis)
or **random search** (`sample=K` over the grid). For a small, discrete space that
is often the whole job, and you get more than the winning config: the full
labeled `xarray` dataset over every cell, `compare_methods` statistics, and
per-cell provenance, all reproducibly.

What mushin does *not* do is **adaptive / Bayesian** search — TPE, CMA-ES,
pruning of unpromising trials, or continuous/conditional spaces. Those steer the
next trial from past ones over a continuous domain, which a fixed grid cannot
express (mushin rejects continuous `interval(...)` syntax; see
[Sweep-axis support](#sweep-axis-support-and-limits) above). That is exactly
where a dedicated optimizer like **Optuna** (or Ax, Nevergrad) wins: a large or
continuous space where you want a good config in far fewer trials than an
exhaustive grid.

So the two are complementary, not exclusive. Reach for mushin's grid/random
search when the space is small and discrete; reach for Optuna when it is large or
continuous. And the strongest combination is a **two-phase workflow** — let the
optimizer *search*, then let mushin run the reproducible *final grid* you report:

**1. Search.** The optimizer owns the adaptive part. Give it a cheap objective
(often a single seed) and let it sample the space:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 1e-2)
    return train_once(lr=lr, weight_decay=wd, seed=0)  # your training function

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best = study.best_params            # e.g. {"lr": 3.1e-4, "weight_decay": 4e-3}
```

**2. Final grid.** Hand the winning config (and a baseline) to mushin and sweep
them over *seeds* for the reproducible, labeled dataset and the statistics you
report. Because each candidate couples several values (`lr` + `weight_decay`),
express the candidates as a [config group](#sweep-axis-support-and-limits) so one
axis selects a whole config:

```python
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

cs = ConfigStore.instance()
cs.store(group="hp", name="baseline", node=make_config(lr=1e-3, weight_decay=0.0))
cs.store(group="hp", name="tuned", node=make_config(**best))

class Final(MultiRunMetricsWorkflow):
    @staticmethod
    def task(hp, seed):
        acc = train_once(lr=hp.lr, weight_decay=hp.weight_decay, seed=seed)
        return dict(accuracy=float(acc))

wf = Final(make_config(hp=None))
wf.run(
    hp=multirun(["baseline", "tuned"]),
    seed=multirun(range(10)),
    working_dir="runs/paper/tuned-vs-baseline",
)
```

`wf.to_xarray()` now has an `hp` dimension and a `seed` dimension, so
`compare_methods(wf.to_xarray().rename(hp="method"))` gives the baseline-vs-tuned
test — a defensible claim over fresh seeds, not the optimizer's optimistic
best-trial number (which suffers the winner's curse; see
[From exploration to a paper](exploration-to-paper.md)).

The same shape works for any searcher — Ax, Nevergrad, a hand-rolled random
search — and adds **no dependency to mushin**: the search lives entirely in your
code, and mushin only ever sees the discrete configs you chose to report.

## Parallel & out-of-process launchers

By default a sweep runs its cells in-process, sequentially (Hydra's `basic`
launcher). Install a Hydra launcher plugin and pass `launcher=` to run the cells
across worker processes — locally with joblib, or on a scheduler with submitit:

```bash
pip install hydra-joblib-launcher     # local multiprocessing (loky backend)
```

```python
--8<-- "examples/parallel_sweep.py:parallel"
```

Run it end to end with
[`examples/parallel_sweep.py`](https://github.com/martinez-hub/mushin/blob/main/examples/parallel_sweep.py)
(`python examples/parallel_sweep.py`); the `run_parallel` docstring shows the
`submitit_slurm` variant for a SLURM cluster.

Out-of-process launchers serialize each cell's task to ship it to a worker. The
default joblib (loky) and submitit backends use `cloudpickle`, which handles most
tasks — including lambdas and nested functions. Some backends (joblib's
`multiprocessing` backend, or a pickle-based submitit setup) use the standard
library's `pickle`, which requires your `task` (and any custom `pre_task`) to be
importable (module-level). Keeping tasks module-level is the portable choice.
Resilience (`on_error="nan"`, `resume=True`) and provenance behave identically
out-of-process.

### Two axes of parallelism: launchers vs. `HydraDDP`

A launcher and [`HydraDDP`](../reference/lightning.md) parallelize different
things:

| | Parallelizes | What it is | Where it goes |
|---|---|---|---|
| `launcher="joblib"` / `submitit` | the sweep's **cells** — each `(lr, seed)` combo runs in its own worker process / node | a Hydra **launcher** | passed to `run(...)` |
| `HydraDDP` / `HydraFSDP` | **one training run** — a single model trained data-parallel (DDP) or sharded (FSDP) across multiple GPUs | a Lightning **`DDPStrategy`** | passed to a `Trainer` **inside your `task`** |

`launcher=` distributes the *grid* (the `parallel_sweep.py` example above shows
only this axis — its toy task does no training). `HydraDDP` / `HydraFSDP` train a
*single* model across GPUs.

!!! warning "`HydraDDP` needs the launcher to provide its ranks"
    `HydraDDP` / `HydraFSDP` do **not** spawn extra GPU workers by themselves from
    an imperative `@mushin.sweep` task. Writing
    `pl.Trainer(strategy=HydraDDP(), devices=2)` and running with the default
    (local) launcher **silently trains on a single GPU** — Lightning reports a
    `1/1` world. The strategy's self-launch path rebuilds each extra rank from a
    Hydra `config.yaml` that must contain declarative `trainer` and `module` keys,
    which an imperative task never writes.

    To actually train across GPUs, launch with **submitit** so the scheduler
    starts one process per GPU. Set the launcher's `tasks_per_node` equal to
    `Trainer(devices=...)` (both equal the GPUs per node) and set `num_nodes`; the
    world size is `nodes × gpus_per_node`. Each SLURM task then becomes one DDP /
    FSDP rank:

    ```python
    import pytorch_lightning as pl
    from mushin import HydraDDP

    @mushin.sweep
    def experiment(seed):
        trainer = pl.Trainer(
            strategy=HydraDDP(),
            devices=1,            # GPUs per node (== launcher tasks_per_node)
            num_nodes=2,
            accelerator="gpu",
            max_epochs=1,
        )
        trainer.fit(model, ...)
        return dict(accuracy=...)

    experiment.run(
        seed=mushin.multirun([0]),
        launcher="submitit_slurm",
        launcher_config=mushin.submitit_slurm_config(nodes=2, gpus_per_node=1),
    )
    ```

    If the launcher's process count doesn't match `num_nodes × devices`, the run
    fails fast with a clear `DDP world size mismatch` error rather than hanging.

Because these strategies need real multi-GPU / multi-node hardware, they aren't
exercised by the CPU-only examples or CI. See the
[lightning reference](../reference/lightning.md) for the fully declarative
`builds(Trainer, module)` form.

## See also

- [Tutorial](../tutorial.md) — end-to-end: sweep → dataset → compare
- [API Reference — workflows](../reference/workflows.md)
