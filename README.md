# mushin

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

## What it provides

- `BaseWorkflow`, `MultiRunMetricsWorkflow`, `RobustnessCurve` — declarative,
  reproducible experiment workflows that record configs, checkpoints, and
  metrics, and load results back as labeled `xarray` datasets.
- `MetricsCallback` — a Lightning callback for capturing metrics.
- `HydraDDP` — a Hydra/Lightning strategy for multi-GPU (DDP) launches.
- `multirun`, `hydra_list`, `load_experiment`, `load_from_checkpoint` — helpers.

## Install

This project uses [uv](https://docs.astral.sh/uv/). For a development
environment (runtime deps + dev tooling):

```bash
uv sync
```

Runtime-only install with plain pip also works:

```bash
pip install .
```

Optional runtime extras: `viz` (matplotlib, for `RobustnessCurve` plotting) and
`netcdf` (netCDF4).

## Develop

```bash
uv run pytest tests/ --hypothesis-profile fast   # tests (DDP test needs >=2 GPUs)
uv run ruff check .                              # lint
uv run ruff format .                             # format
uv run codespell src tests                       # spell check
```

Supported Python versions: 3.9 – 3.14.

## Relationship to upstream

This is a fork/extraction, not a replacement endorsed by MIT-LL. The configuration
engine it depends on, `hydra-zen`, is actively maintained by the same group. See
`LICENSE.txt` for attribution; the original MIT copyright is retained.
