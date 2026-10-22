# For AI agents

mushin ships a usage guide for AI coding agents (Claude Code, Codex, Cursor and
the like) that set up experiments for scientists and engineers. It covers
which API to use for which job, a tested template, the pitfalls that give wrong
results, how to run long sweeps from an agent, and what to check before
reporting. This page shows the guide from the main branch; from mushin 0.16,
`python -m mushin guide` prints the copy that matches the installed version.

## Point your agent at it

Add this to your project's `AGENTS.md` or `CLAUDE.md`:

```text
This project uses mushin. Install it with `pip install "mushin-py[eval]"`
(the `eval` extra adds the statistics; not `pip install mushin`, a PyPI name
that is not this project). Before writing or changing an experiment, run
`python -m mushin guide` and follow it. On mushin 0.10 to 0.15, read
https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md
instead; before 0.10, upgrade first.
```

To let an agent inspect finished runs, see
[Analyzing from Claude Code](guides/mcp.md): `mushin-mcp` is a read-only MCP
server over a directory of runs.

---

--8<-- "src/mushin/agent_guide.md:3"
