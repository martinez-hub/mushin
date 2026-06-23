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

## What it provides

- `BaseWorkflow`, `MultiRunMetricsWorkflow`, `RobustnessCurve` — declarative,
  reproducible experiment workflows that record configs, checkpoints, and
  metrics, and load results back as labeled `xarray` datasets.
- `MetricsCallback` — a Lightning callback for capturing metrics.
- `HydraDDP` — a Hydra/Lightning strategy for multi-GPU (DDP) launches.
- `multirun`, `hydra_list`, `load_experiment`, `load_from_checkpoint` — helpers.

## Install

From PyPI (the distribution is named `mushin-py`; you still `import mushin`):

```bash
pip install mushin-py
```

For a development environment (runtime deps + dev tooling), this project uses
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optional runtime extras: `viz` (matplotlib, for `RobustnessCurve` plotting) and
`netcdf` (netCDF4) — e.g. `pip install "mushin-py[viz]"`.

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
