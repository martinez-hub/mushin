# `mushin-mcp` — read-only experiment-analysis MCP server

**Date:** 2026-06-23
**Status:** Design approved, pending implementation plan
**Scope:** Option A only (read-only analysis). Launching sweeps and live-model
comparison are explicitly deferred to a possible phase 2.

## Motivation

People want to "plug mushin into Claude Code" (or any MCP client). mushin has
three natural surfaces with very different fit for the Model Context Protocol:

| Surface | MCP fit | Why |
|---|---|---|
| Read / analyze completed runs | excellent | read-only, fast, stateless, JSON-friendly |
| Launch sweeps / studies | poor | long-running, GPU-bound, stateful |
| Compare live models (`benchmark.compare`) | infeasible over MCP | needs in-memory `torch.nn.Module` + a re-iterable `DataLoader`, which cannot cross the JSON boundary |

mushin's *outputs* are a great MCP surface; its *execution* is not. This server
exposes only the analysis surface: an agent that can read and reason over
experiments you have already run.

## Key realism findings (from the code)

- `mushin._utils.load_experiment(exp_path)` works purely from a directory path.
  It reads `.hydra/config.yaml`, checkpoint paths, and a metrics dict, returning
  an `Experiment` (or list of them). Pure JSON in / JSON out — the spine of this
  server.
- `benchmark.compare(...)` requires in-memory `torch.nn.Module` instances and a
  re-iterable `DataLoader`. These cannot be passed through MCP, so a generic
  `compare_checkpoints` tool is **not** part of this design.
- `.to_xarray()` is defined on the *user's* `MultiRunMetricsWorkflow` subclass,
  which the server cannot know about. The server therefore builds on raw
  `load_experiment` output, not subclass reconstruction. For already-saved
  datasets, it reads the netCDF directly with `xarray`.

Net effect: the whole surface requires **zero user code**, loads **no torch
models**, and is **fully stateless**.

## Architecture & distribution

- New subpackage `src/mushin/mcp/` with `server.py` and `__main__.py`.
- Built on the official `mcp` Python SDK (FastMCP), **stdio** transport.
- Shipped as an optional extra so the core install stays lean:
  - `pip install "mushin-py[mcp]"` adds the `mcp` dependency.
  - Console entry point `mushin-mcp` registered under `[project.scripts]`.
- Imports only `load_experiment` and `xarray` at runtime — no torch model
  loading, no training, no user subclasses.
- Optional `--root <dir>` argument scopes and secures where the server will
  look for experiments.

Claude Code integration:

```bash
pip install "mushin-py[mcp]"
claude mcp add mushin -- mushin-mcp --root ./outputs
```

## Tools (all read-only, JSON only)

| Tool | Behavior | Built on |
|---|---|---|
| `list_experiments(root?, glob?)` | Find directories containing `.hydra/config.yaml`; return their paths and run counts. | filesystem glob |
| `describe_experiment(path)` | Return the swept parameters (the config *diff* across jobs), available metric keys, number of runs, and checkpoint paths. | `load_experiment` |
| `get_metrics(path, metrics?, reduce?)` | Return metric values per run, optionally reduced (e.g. mean/std) across runs. | `Experiment.metrics` |
| `get_config(path, job?)` | Return the resolved Hydra config for a run. | `Experiment.cfg` |
| `read_dataset(path)` | For a netCDF the user saved (`to_xarray().to_netcdf(...)` or `BenchmarkResult.data`): return dims, coords, data variables, and summary stats. | `xarray.open_dataset` |

These let an agent answer: what did I sweep, which runs exist, what were the
metrics, and summarize this saved dataset — conversational analysis over
completed work, with no execution.

## Data flow

Claude Code (stdio) -> `mushin-mcp` -> `load_experiment` / `xarray` -> JSON
response. Each call is independent; there is no server-side run registry or
mutable state.

## Error handling

- Path not found, missing `.hydra`, or unreadable netCDF -> structured MCP error
  that names the offending path.
- A path outside `--root` (when set) -> refusal.
- Never surface raw Python tracebacks across the MCP boundary.

## Testing

- One unit test per tool against a small fixture experiment directory (reuse
  `mushin.testing` fixtures plus a saved `.nc`).
- No GPU and no training, so the suite runs in CI unchanged.

## Documentation

- A `docs/` page plus a README section covering: installing the extra, the
  `claude mcp add` snippet, and example prompts (e.g. "summarize the accuracy
  sweep in ./outputs").

## Out of scope (this design)

- Launching sweeps or studies.
- `benchmark.compare` over live models (infeasible over MCP).
- Any stateful, long-running, or GPU-bound operation.

These constitute a possible phase 2 (Option B: analysis + guarded async launch),
to be designed separately only after this read-only server proves out.
