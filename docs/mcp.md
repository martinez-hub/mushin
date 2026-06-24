# Analyze experiments from Claude Code (MCP)

`mushin-mcp` is a read-only [Model Context Protocol](https://modelcontextprotocol.io)
server. Point it at a directory of completed mushin runs and an MCP client
(Claude Code, Claude Desktop, …) can load, summarize, and compare them
conversationally. It never trains, launches sweeps, or loads model weights.

> Requires Python >= 3.10 (the `mcp` SDK does not support 3.9).

## Install

```bash
pip install "mushin-py[mcp]"
```

## Connect Claude Code

```bash
claude mcp add mushin -- mushin-mcp --root ./outputs
```

`--root` restricts the server to one directory; omit it to allow the current
working directory.

## Tools

| Tool | Returns |
|---|---|
| `list_experiments` | Run directories (those containing `.hydra/`) under the root. |
| `describe_experiment` | Swept params, metric keys, run and checkpoint counts. |
| `get_metrics` | Per-run metrics; optional `mean`/`std` reduction across runs. |
| `get_config` | The resolved Hydra config for a run (or all runs). |
| `read_dataset` | Dims, coords, data variables, and basic stats of a saved netCDF. |

## Example prompts

- "List the experiments under ./outputs and tell me what each one swept."
- "Summarize the accuracy metric for the lr sweep, averaged across seeds."
- "Open results.nc and tell me which method scored highest."
