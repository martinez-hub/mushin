# mushin

Boilerplate-free, reproducible ML experiment workflows: a framework-agnostic
sweep engine built on
[hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen), with
first-class [PyTorch Lightning](https://lightning.ai/) integration.

`mushin` is the evaluate-and-report layer sitting on top of hydra-zen — and,
for deep learning, Lightning. Define your experiment as a function, sweep over
parameters with Hydra, and get results back as a labeled `xarray.Dataset` — not
rows in a dashboard you have to export.

!!! tip "Highlights"
    - **`@mushin.sweep`** — the boilerplate-free core: decorate a function and
      `experiment.run(...)` returns the labeled dataset. See the
      [quickstart](quickstart.md).
    - **Resilient & resumable sweeps** — `on_error="nan"` fail-soft plus a durable
      `resume=True` that survives a hard process kill or SLURM preemption without
      recomputing finished cells. See [resilience](guides/resilience.md).
    - **Laptop → cluster, one code path** — out-of-process launchers
      (`launcher="joblib"` / submitit) and multi-GPU / multi-node training
      (`HydraDDP` / `HydraFSDP`, GPU packing), validated on real cluster hardware.
      See [multi-node training](guides/multinode.md).
    - **Lean core, opt-in eval** — the evaluation layer (`compare`, the batteries,
      LLM eval, `Study`) is the optional
      [`eval` extra](install.md#optional-extras); a plain install is just the
      sweep → dataset core.

    See the full [changelog](changelog.md) for every release.

## Three pillars

**Sweeps → datasets.**
`MultiRunMetricsWorkflow` runs a Hydra multirun, collects your returned
metrics, and assembles them into a labeled `xarray.Dataset` keyed by
the swept parameters.

**`compare` with statistics.** *(optional [`eval` extra](install.md#optional-extras):
`pip install "mushin-py[eval]"`)*
`benchmark.compare` evaluates a set of trained models on a standard metric
battery (torchmetrics), then runs pairwise significance tests (scipy) with
multiple-comparison correction (Holm by default; also Bonferroni/FDR/none).
The result is a `BenchmarkResult` with a paper-ready
`.summary()`, tidy `.comparisons` DataFrame, and a labeled `.data` dataset.

**`Study`.**
`Study` orchestrates the full pipeline — multi-seed training sweep via Hydra,
then straight into `compare` — in one call. `Study.from_checkpoints` handles
the eval-only case when you already have checkpoints.

## Quick example

```python
import torch as tr
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

class LRSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr: float, seed: int) -> dict:
        tr.manual_seed(seed)
        # ... train a model, then evaluate it ...
        acc = ...  # your validation accuracy
        return dict(accuracy=acc)

wf = LRSweep()
wf.run(lr=multirun([0.01, 0.1, 1.0]), seed=multirun([0, 1, 2]))  # 9 runs

ds = wf.to_xarray()
# <xarray.Dataset> Dimensions: (lr: 3, seed: 3)
#   Data variables: accuracy (lr, seed)

ds["accuracy"].mean("seed")   # average over seeds, per learning rate
```

## Get started

- [Install](install.md) — pip, extras, and the support matrix
- [Quickstart](quickstart.md) — run the flagship sweep example end-to-end
- [Guides](guides/workflows.md) — workflows, compare, Study, segmentation, MCP
- [API Reference](reference/benchmark.md) — full auto-generated docs
