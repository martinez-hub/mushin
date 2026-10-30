# For AI agents

mushin ships a usage guide for AI coding agents (Claude Code, Codex, Cursor and
the like) that set up experiments for scientists and engineers. It covers
which API to use for which job, a tested template, the pitfalls that give wrong
results, how to run long sweeps from an agent, and what to check before
reporting. This page shows the guide from the main branch; from mushin 0.16,
`python -m mushin guide` prints the copy that matches the installed version.

## Claude Code

Install the guide as a skill from this repository's plugin marketplace, then
load it with `/mushin:guide` before writing or changing an experiment:

```text
/plugin marketplace add martinez-hub/mushin
/plugin install mushin@mushin
```

On Claude Code 2.1.275 or later, one command does both:
`/plugin install mushin --marketplace martinez-hub/mushin`.

The skill runs `python -m mushin guide` in your project, so it always shows
the guide for the installed mushin (0.16 or later; install it first with
`pip install "mushin-py[eval]"`). With `mushin-mcp` connected (see
[Analyzing from Claude Code](guides/mcp.md)), the guide is also the
`/mushin:guide (MCP)` prompt and the `@mushin:mushin://guide` resource.

## Point your agent at it

For any agent, add this to your project's `AGENTS.md` or `CLAUDE.md`:

```text
This project uses mushin. Install it with `pip install "mushin-py[eval]"`
(the `eval` extra adds the statistics; not `pip install mushin`, a PyPI name
that is not this project). Before writing or changing an experiment, run
`python -m mushin guide` and follow it. On mushin 0.10 to 0.15, read
https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md
instead; before 0.10, upgrade first.
```

To give an agent the whole site at once, point it at
[llms-full.txt](https://martinez-hub.github.io/mushin/llms-full.txt): every
page as one Markdown file, with the example code and the API reference
inlined. [llms.txt](https://martinez-hub.github.io/mushin/llms.txt) is the
short index of the same pages.

To let an agent inspect finished runs, see
[Analyzing from Claude Code](guides/mcp.md): `mushin-mcp` is a read-only MCP
server over a directory of runs.

---

--8<-- "src/mushin/agent_guide.md:3"
