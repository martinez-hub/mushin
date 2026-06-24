<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/martinez-hub/mushin/main/logos/mushin-dark.png">
    <img src="https://raw.githubusercontent.com/martinez-hub/mushin/main/logos/mushin-light.png" alt="mushin logo" width="200">
  </picture>
</p>

<h1 align="center">mushin</h1>

[![CI](https://github.com/martinez-hub/mushin/actions/workflows/ci.yml/badge.svg)](https://github.com/martinez-hub/mushin/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mushin-py.svg)](https://pypi.org/project/mushin-py/)
[![Python versions](https://img.shields.io/pypi/pyversions/mushin-py.svg)](https://pypi.org/project/mushin-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

**Docs:** https://martinez-hub.github.io/mushin/

Boilerplate-free, reproducible machine-learning experiment workflows built on
[PyTorch Lightning](https://lightning.ai/) and
[hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen).

`mushin` is a standalone carve-out of the `rai_toolbox.mushin` subpackage from
MIT Lincoln Laboratory's
[responsible-ai-toolbox](https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox).
The upstream toolbox is no longer maintained (last release May 2023), but the
`mushin` workflow layer still works against current versions of its
dependencies. This package extracts just that layer so it can be maintained and
used on its own.

## Quickstart: run a sweep, get a dataset

Define your experiment as a function, sweep over parameters, and get the results
back as a labeled `xarray.Dataset` — not rows in a dashboard you have to export.

```python
import torch as tr
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

class LRSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr: float, seed: int) -> dict:
        tr.manual_seed(seed)
        # ... train a model with this lr/seed, then evaluate it ...
        acc = ...  # your validation accuracy
        return dict(accuracy=acc)  # whatever you return becomes a data variable

wf = LRSweep()
wf.run(lr=multirun([0.01, 0.1, 1.0]), seed=multirun([0, 1, 2]))  # 9 runs

ds = wf.to_xarray()
# <xarray.Dataset> Dimensions: (lr: 3, seed: 3)
#   Data variables: accuracy (lr, seed)

ds["accuracy"].mean("seed")   # average over seeds, per learning rate
```

The full runnable version is in [`examples/sweep_to_dataset.py`](examples/sweep_to_dataset.py):

```bash
uv run python examples/sweep_to_dataset.py
```

## Compare methods, with statistics

Evaluate trained models on a standard battery and get a labeled dataset *plus*
significance — metrics delegated to torchmetrics, statistics to scipy:

```python
from mushin.benchmark import compare

result = compare(
    methods={"ours": [m0, m1, m2], "baseline": [b0, b1, b2]},  # one trained model per seed
    data=test_loader, task="classification", num_classes=10, test="welch",
)

result.summary()       # mean ± CI per method, with significance markers — paper-ready
result.comparisons     # tidy DataFrame: pairwise p-values + effect sizes
result.data            # the labeled xarray (method × seed) to slice and plot
```

Don't have the trained models in memory yet? `Study` runs the multi-seed training
sweep (via Hydra) and feeds the results straight into `compare` — define → train →
evaluate → report in one call:

```python
from mushin import Study

study = Study(
    methods={"cnn": train_cnn, "mlp": train_mlp},   # train_fn(seed) -> checkpoint path
    load_fn=LitClassifier.load_from_checkpoint,       # path -> model
    seeds=[0, 1, 2], data=test_loader, num_classes=10, test="welch",
)
result = study.run()                                  # -> BenchmarkResult

# ...or compare checkpoints you already have, no training:
Study.from_checkpoints(
    checkpoints={"cnn": ["cnn_0.ckpt", ...], "mlp": ["mlp_0.ckpt", ...]},
    load_fn=LitClassifier.load_from_checkpoint,
    data=test_loader, num_classes=10, test="welch",
).run()
```

## What it provides

- `benchmark.compare` — run a standard metric battery (torchmetrics) across
  trained seeds and get a labeled dataset + significance (scipy): `BenchmarkResult`
  with `.summary()`, `.comparisons`, and `.data`.
- `Study` — orchestrate a multi-seed training sweep and route the trained models
  into `compare`, in one call; `Study.from_checkpoints(...)` for eval-only.
- `BaseWorkflow`, `MultiRunMetricsWorkflow`, `RobustnessCurve` — declarative,
  reproducible experiment workflows that record configs, checkpoints, and
  metrics, and load results back as labeled `xarray` datasets.
- `MetricsCallback` — a Lightning callback for capturing metrics.
- `HydraDDP` — a Hydra/Lightning strategy for multi-GPU (DDP) launches.
- `multirun`, `hydra_list`, `load_experiment`, `load_from_checkpoint` — helpers.

## Analyze experiments from Claude Code (MCP)

`mushin` ships an optional read-only [MCP](https://modelcontextprotocol.io)
server so Claude Code (or any MCP client) can load and analyze your completed
runs — list experiments, summarize swept parameters and metrics, and inspect
saved datasets — without launching anything.

```bash
pip install "mushin-py[mcp]"          # requires Python >= 3.10
claude mcp add mushin -- mushin-mcp --root ./outputs
```

See the [MCP guide](https://martinez-hub.github.io/mushin/guides/mcp/) for the full tool list and example prompts.

## Install

```bash
pip install mushin-py
```

Already use [uv](https://docs.astral.sh/uv/)? `uv pip install mushin-py` (or
`uv add mushin-py` inside a project) is faster.

> **Install name vs. import name:** the PyPI distribution is **`mushin-py`**, but
> you `import mushin` (same pattern as `scikit-learn` → `sklearn`).

Optional runtime extras: `viz` (matplotlib, for `RobustnessCurve` plotting) and
`netcdf` (netCDF4) — e.g. `pip install "mushin-py[viz]"`.

For a development environment (runtime deps + dev tooling), this project uses
[uv](https://docs.astral.sh/uv/): `uv sync`.

## Develop

```bash
uv run pytest tests/ --hypothesis-profile fast   # tests (DDP test needs >=2 GPUs)
uv run ruff check .                              # lint
uv run ruff format .                             # format
uv run codespell src tests                       # spell check
```

Or use the `make` shortcuts (`make help` to list them): `make check` runs
lint + format-check + spell + tests (what CI runs); `make test-py PYTHON=3.12`
runs the suite on a specific Python version.

Supported Python versions: 3.9 – 3.14.

## Relationship to upstream

This is a fork/extraction, not a replacement endorsed by MIT-LL. The configuration
engine it depends on, `hydra-zen`, is actively maintained by the same group. See
`LICENSE.txt` for attribution; the original MIT copyright is retained.
