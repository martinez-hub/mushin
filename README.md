<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/martinez-hub/mushin/main/logos/mushin-dark.png">
    <img src="https://raw.githubusercontent.com/martinez-hub/mushin/main/logos/mushin-light.png" alt="mushin logo" width="200">
  </picture>
</p>

<h1 align="center">mushin</h1>

<p align="center">
  <a href="https://github.com/martinez-hub/mushin/actions/workflows/ci.yml"><img src="https://github.com/martinez-hub/mushin/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/mushin-py/"><img src="https://img.shields.io/pypi/v/mushin-py.svg" alt="PyPI"></a>
  <a href="https://github.com/martinez-hub/mushin/releases/latest"><img src="https://img.shields.io/github/v/release/martinez-hub/mushin?label=release" alt="Latest release"></a>
  <a href="https://pypi.org/project/mushin-py/"><img src="https://img.shields.io/pypi/pyversions/mushin-py.svg" alt="Python versions"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://doi.org/10.5281/zenodo.21436444"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21436444.svg" alt="DOI"></a>
</p>

<p align="center">
  <b>Boilerplate-free, reproducible machine-learning experiment sweeps</b> —<br>
  a framework-agnostic sweep engine built on
  <a href="https://github.com/mit-ll-responsible-ai/hydra-zen">hydra-zen</a>,
  with first-class <a href="https://lightning.ai/">PyTorch Lightning</a> integration.
</p>

<p align="center"><a href="https://martinez-hub.github.io/mushin/"><b>Documentation</b></a></p>

Decorate an experiment, sweep over parameters, and get the results back as a
labeled `xarray.Dataset` — not rows in a dashboard you have to export.

- **Boilerplate-free** — one decorator, no subclassing or callbacks; results come
  back as a labeled dataset you can slice and plot.
- **Reproducible** — every run captures its config + provenance, sweeps resume
  durably after a hard kill or cluster preemption, and auto-tuning pins a
  hardware-independent effective batch size.
- **Scalable** — the *same* task runs in-process, across cores, or on a multi-node
  SLURM cluster (validated on real GPU hardware); change only the launcher.
- **Framework-agnostic** — your task just returns a `dict`, so scikit-learn,
  XGBoost, or any Python model works, not only PyTorch.

## Quickstart

```python
import mushin

@mushin.sweep
def experiment(lr, seed):
    # ... train a model with this lr/seed, then evaluate it ...
    acc = ...                      # your validation accuracy
    return dict(accuracy=acc)      # whatever you return becomes a data variable

ds = experiment.run(
    lr=mushin.multirun([0.01, 0.1, 1.0]),
    seed=mushin.multirun([0, 1, 2]),
)  # 9 runs, returned as a labeled xarray.Dataset
# <xarray.Dataset> Dimensions: (lr: 3, seed: 3)
#   Data variables: accuracy (lr, seed)

ds["accuracy"].mean("seed")        # average over seeds, per learning rate

# Prefer pandas? One call gives a tidy table — no xarray required:
experiment.workflow.to_dataframe() #    lr  seed  accuracy
                                   # 0  0.01    0      ...
```

The decorator version above runs as
[`examples/parallel_sweep.py`](examples/parallel_sweep.py);
[`examples/sweep_to_dataset.py`](examples/sweep_to_dataset.py) is the same flow
written with the class API. Need the full tool (`.failures`, provenance, custom
analysis)? Drop to `experiment.workflow`, or subclass
`MultiRunMetricsWorkflow`.

## What else it does

Each links to a guide with a runnable example:

- **[Compare methods with statistics](https://martinez-hub.github.io/mushin/guides/compare/)** —
  `benchmark.compare` runs a metric battery (torchmetrics) across seeds and returns
  a labeled dataset *plus* significance (scipy); `Study` runs the training sweep and
  feeds it straight in.
- **[Compare LLM systems](https://martinez-hub.github.io/mushin/guides/llm/)** —
  the same significance spine for LLM evals, with Holm-corrected p-values and an
  optional output cache.
- **[Built-in batteries](https://martinez-hub.github.io/mushin/guides/batteries/)** —
  classification, segmentation, detection, regression, retrieval, image-quality, audio.
- **[Resilient & resumable sweeps](https://martinez-hub.github.io/mushin/guides/resilience/)** —
  `on_error="nan"`, durable `resume=True` across hard kills/preemption, and per-run
  provenance.
- **[Parallel & out-of-process launchers](https://martinez-hub.github.io/mushin/guides/workflows/#parallel--out-of-process-launchers)** —
  `run(..., launcher="joblib")` or submitit; stdlib-picklable dispatch.
- **[Multi-node & sharded training](https://martinez-hub.github.io/mushin/guides/multinode/)** —
  `HydraDDP` / `HydraFSDP` and `pin_gpu_round_robin` GPU packing, validated on a real
  SLURM cluster.
- **[Auto-tuning](https://martinez-hub.github.io/mushin/guides/auto-tuning/)** —
  `tune_batch_size` / `tune_learning_rate`, pinned for reproducibility.
- **[Analyze from Claude Code (MCP)](https://martinez-hub.github.io/mushin/guides/mcp/)** —
  an optional read-only MCP server to load and inspect completed runs.

## Installation

```bash
pip install mushin-py           # the sweep -> dataset core
pip install "mushin-py[eval]"   # + compare, metric batteries, LLM eval, Study
```

The PyPI distribution is **`mushin-py`**, but you `import mushin` (like
`scikit-learn` → `sklearn`). The **`eval`** extra adds the evaluation layer
(`compare`, the metric batteries, LLM evaluation, `Study`) and its heavier
dependencies (torchmetrics, scipy) — keeping the core install lean; accessing
those features without it raises a clear install hint. Other optional extras:
`viz`, `netcdf`, `detection`, `image`, `audio`, `mcp` (the battery extras imply
`eval`) — e.g. `pip install "mushin-py[eval,viz]"`. Supported Python: 3.10 – 3.13.

## Versioning & scope

mushin follows SemVer with pre-1.0 semantics: **a minor bump (0.8 → 0.9) may
contain breaking changes**, always listed with migration notes in the
[changelog](CHANGELOG.md); patch releases never break. The public API is the
top-level `mushin` namespace plus the documented `mushin.benchmark` /
`mushin.llm` symbols — underscore modules are internal. The project's scope is
deliberately narrow: **boilerplate-free, reproducible sweep → dataset**, with
an optional evaluation layer behind the `eval` extra. It complements — and
does not replace — experiment trackers like W&B or TensorBoard (see the
[workflows guide](https://martinez-hub.github.io/mushin/guides/workflows/) for
using both together).

## Citation

If you use mushin in your research, please cite it (the concept DOI always
resolves to the latest release):

```bibtex
@software{martinez_mushin,
  author  = {Martínez-Martínez, Josué},
  title   = {mushin: boilerplate-free, reproducible machine-learning experiment sweeps},
  year    = {2026},
  version = {0.10.1},
  doi     = {10.5281/zenodo.21436444},
  url     = {https://github.com/martinez-hub/mushin}
}
```

GitHub's "Cite this repository" button (from [`CITATION.cff`](CITATION.cff))
generates this for you.

## Contributing

Issues and pull requests are welcome. Local development uses
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest tests/ --hypothesis-profile fast   # tests (DDP test needs >=2 GPUs)
uv run ruff check . && uv run ruff format --check .
```

Or `make check` (lint + format + spell + tests, as CI runs). See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Disclaimer

`mushin` is a maintained, standalone carve-out of the `rai_toolbox.mushin`
subpackage from MIT Lincoln Laboratory's
[responsible-ai-toolbox](https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox)
(no longer maintained). This is a fork/extraction, not a replacement endorsed by
MIT-LL; the original MIT copyright is retained (see [`LICENSE.txt`](LICENSE.txt)).
The configuration engine it builds on, `hydra-zen`, is actively maintained by the
same group.
