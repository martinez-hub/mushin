# mushin

Boilerplate-free, reproducible ML experiment workflows built on
[PyTorch Lightning](https://lightning.ai/) and
[hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen).

`mushin` is the evaluate-and-report layer sitting on top of Lightning and
hydra-zen. Define your experiment as a function, sweep over parameters with
Hydra, and get results back as a labeled `xarray.Dataset` — not rows in a
dashboard you have to export.

!!! tip "What's new in 0.4.0"
    - **Auto-tuning** — [`tune_batch_size` / `tune_learning_rate`](guides/auto-tuning.md):
      find the batch size / LR once, pin it to a sidecar file, and reuse it — with an
      exact, hardware-independent effective batch (no drift).
    - **Task API + more batteries** — first-class, reusable evaluation tasks
      (`register_task` / `get_task` / `list_tasks`) plus regression, image-quality,
      audio, retrieval, and detection batteries for [`compare`](guides/compare.md).
    - **LLM evaluation** — [`compare_llms`](guides/llm.md): compare LLM systems across
      reproducible seeds with statistical significance.
    - **Lighter import** — `import mushin` no longer pulls the benchmark/LLM machinery
      until you use it, and the new [`mushin.original_cwd()`](concepts.md#working-directories)
      anchors relative paths inside `task()`.
    - **Deprecation** — `BaseWorkflow` and `RobustnessCurve` moved out of the top-level
      namespace; import them from `mushin.workflows` (they still work, with a warning).

    See the full [changelog](changelog.md) for every change.

## Three pillars

**Sweeps → datasets.**
`MultiRunMetricsWorkflow` runs a Hydra multirun, collects your returned
metrics, and assembles them into a labeled `xarray.Dataset` keyed by
the swept parameters.

**`compare` with statistics.** *(optional [`eval` extra](install.md#optional-extras):
`pip install "mushin-py[eval]"`)*
`benchmark.compare` evaluates a set of trained models on a standard metric
battery (torchmetrics), then runs pairwise significance tests (scipy) with
Holm correction. The result is a `BenchmarkResult` with a paper-ready
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
