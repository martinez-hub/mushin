# Analyzing from Claude Code (MCP)

`mushin-mcp` is a read-only [Model Context Protocol](https://modelcontextprotocol.io)
server. Point it at a directory of completed mushin runs and an MCP client
(Claude Code, Claude Desktop, or any MCP-compatible tool) can load, summarize,
and compare them conversationally — without training, launching sweeps, or
loading model weights.

## Install

```bash
pip install "mushin-py[mcp]"
```

## Connect Claude Code

```bash
claude mcp add mushin -- mushin-mcp --root ./outputs
```

`--root` restricts the server to one directory tree; omit it to allow the
server to read from the current working directory.

After adding the server, restart Claude Code and confirm it appears in the
active server list. You can also run `mushin-mcp --help` to verify the install.

## Connect Claude Desktop

Add an entry to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mushin": {
      "command": "mushin-mcp",
      "args": ["--root", "/path/to/your/outputs"]
    }
  }
}
```

## Tools

| Tool | What it returns |
|---|---|
| `list_experiments` | Run directories (those containing `.hydra/`) under the root. |
| `describe_experiment` | Swept parameters, metric keys, run and checkpoint counts. |
| `get_metrics` | Per-run metrics; optional `mean`/`std` reduction across runs. |
| `get_config` | The resolved Hydra config for a run (or all runs). |
| `read_dataset` | Dimensions, coordinates, data variables, and basic statistics of a saved netCDF file. |

All tools are read-only. The server never writes files, trains models, or
launches sweeps.

## Example prompts

Once connected, you can ask questions like:

- "List the experiments under ./outputs and tell me what each one swept."
- "Summarize the accuracy metric for the lr sweep, averaged across seeds."
- "Open results.nc and tell me which method scored highest on mean IoU."
- "What were the Hydra config differences between the two best runs?"
- "Plot the accuracy vs. learning rate curve from the sweep dataset."

## Running directly

You can also run the server manually (useful for debugging):

```bash
mushin-mcp --root ./outputs
```

The server starts on stdio (standard MCP transport). Use `--help` for all
options.
