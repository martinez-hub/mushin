# Install

## Basic install

```bash
pip install mushin-py
```

Already use [uv](https://docs.astral.sh/uv/)?

```bash
uv pip install mushin-py
# or, inside a uv project:
uv add mushin-py
```

!!! note "Install name vs. import name"
    The PyPI distribution is **`mushin-py`**, but you `import mushin`: the
    same pattern as `scikit-learn` → `sklearn`.

## Optional extras

The core install is the sweep → dataset workflow. The **evaluation layer**
(`compare`, the metric batteries, LLM evaluation, and `Study`) is the `eval`
extra. Accessing those features without it raises a clear install hint.

| Extra | What it adds | Install |
|---|---|---|
| `eval` | `compare`, metric batteries, LLM eval, `Study` (adds torchmetrics, scipy) | `pip install "mushin-py[eval]"` |
| `viz` | matplotlib (for plotting results) | `pip install "mushin-py[viz]"` |
| `netcdf` | netCDF4 (save/load datasets as `.nc` files) | `pip install "mushin-py[netcdf]"` |
| `mcp` | MCP server (`mushin-mcp`) for Claude Code integration | `pip install "mushin-py[mcp]"` |
| `detection` / `image` / `audio` | extra metric batteries (imply `eval`) | `pip install "mushin-py[detection]"` |

Combine extras with commas, e.g. `pip install "mushin-py[eval,viz]"`.

## Support matrix

| Platform | Python | torch | NumPy |
|---|---|---|---|
| Linux / Windows / macOS (Apple Silicon) | 3.11 – 3.14 | ≥ 2.4 | ≥ 2 |

A few notes:

- **pytorch-lightning ≥ 2.4** is required on all platforms.
- **Apple Silicon (M-series)**: fully supported. The CI suite runs on Apple
  Silicon on every pull request. The sweep engine has no device code, and
  training on the M-series GPU goes through Lightning's MPS accelerator: build
  your `Trainer` with `accelerator="mps"` (or `"auto"`, which selects MPS)
  inside your task. Auto-tuning and the eval layer follow your model's device,
  and provenance records the MPS chip. The multi-GPU / cluster features
  (`HydraDDP`/`HydraFSDP`, `pin_gpu_round_robin`, `submitit_slurm_config`) are
  simply not applicable on a single-chip Mac; `launcher="joblib"` still
  parallelizes sweep *cells* across CPU cores as usual.
- **Intel macOS (x86_64)** is not supported: PyTorch publishes no x86_64 macOS
  wheels after torch 2.2, and `mushin` requires torch ≥ 2.4 and NumPy 2. The
  last release series that supports Intel Macs is 0.14.x, which supports them
  on Python 3.10 and 3.11 (newer Pythons are unsupported there, and torch 2.2
  has no wheels for Python 3.13+): `python3.11 -m pip install "mushin-py<0.15"`.
- **Python 3.10** is not supported: it reached end-of-life on 2026-10-01,
  and `mushin` requires Python ≥ 3.11.
- **Python 3.15** is not supported yet. Two upstream fixes are pending:
  hydra-zen 0.17.0 fails to import on 3.15, which the upcoming 0.17.1 fixes
  ([hydra-zen#903](https://github.com/mit-ll-responsible-ai/hydra-zen/pull/903)).
  PyTorch has not published 3.15 wheels to PyPI, and the 3.15 builds on
  PyTorch's own index crash on macOS and Windows; the fix ships in torch 2.15
  ([pytorch#200779](https://github.com/pytorch/pytorch/issues/200779)).
  With hydra-zen's fix and torch from PyTorch's index, mushin's test suite
  already passes on Linux 3.15. Support will be announced in the changelog
  once it is tested in CI.
- These floors are enforced by the `min-versions` CI job on every pull request.
