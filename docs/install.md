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
    The PyPI distribution is **`mushin-py`**, but you `import mushin` —
    the same pattern as `scikit-learn` → `sklearn`.

## Optional extras

The core install is the sweep → dataset workflow. The **evaluation layer**
(`compare`, the metric batteries, LLM evaluation, and `Study`) is the `eval`
extra — accessing those features without it raises a clear install hint.

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
| Linux / Windows / non-Intel macOS | 3.10 – 3.13 | ≥ 2.4 | ≥ 2 |
| Intel macOS (x86_64) | 3.10 – 3.11 | 2.2.x | 1.x |

A few notes:

- **pytorch-lightning ≥ 2.4** is required on all platforms.
- **Intel macOS**: Apple has not shipped PyTorch wheels past 2.2.x for the
  x86_64 architecture. `mushin` supports this platform at torch 2.2.x and
  NumPy 1.x, but Python 3.12+ is not available there because NumPy 2 is
  required for Python 3.12 wheels and is ABI-incompatible with torch 2.2.x.
- **Python 3.9** is not supported (it reached end-of-life in October 2025);
  `mushin` requires Python ≥ 3.10.
- These floors are enforced by the `min-versions` CI job on every pull request.
