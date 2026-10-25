# Packaging the agent guide for agents: design

Date: 2026-10-24. Status: approved by the maintainer's standing instruction to
take the recommended option; implemented by the plan of the same date.

## Problem

`python -m mushin guide` prints a usage guide for AI coding agents, and
`docs/for-agents.md` tells a human to paste a paragraph into `AGENTS.md` or
`CLAUDE.md` so their agent runs it (#255, 0.16.0). That works for every agent
but puts a manual step between the human and the guide, and an agent that
talks to `mushin-mcp` to inspect runs has no way to reach the guide through
the same connection. The agent-guide spec deferred "vendor-specific packaging
(a Claude Code skill, MCP prompts)" to a follow-up. This is it.

## Goals

1. An MCP client connected to `mushin-mcp` can read the guide that matches the
   installed mushin without leaving the client.
2. A Claude Code user installs the guide as a skill with two slash commands,
   and the skill never goes stale: it prints the installed guide.
3. No new surface that must be kept in sync with the guide's text by hand.

## Non-goals

- Bundling `mushin-mcp` in the plugin. A plugin-provided MCP server starts in
  every project where the plugin is enabled and fails, with a visible error,
  wherever `mushin-py[mcp]` is not installed. The documented
  `claude mcp add mushin -- mushin-mcp --root ./outputs` stays the way to
  connect it, per project.
- Packaging for Codex, Cursor or others. They read `AGENTS.md`; the snippet
  on `docs/for-agents.md` already covers them.
- A `python -m mushin skill` command that writes `.claude/skills/...` into
  the user's project: one more code path for the result the plugin gives.
- New prompts for analysis ("summarize this sweep"). The MCP guide's example
  prompts cover that in prose; a prompt catalogue would be a second guide to
  maintain.

## Facts the design rests on (Claude Code docs, 2026-10-24)

- Skills are discovered from `~/.claude/skills/<name>/SKILL.md`, a project's
  `.claude/skills/`, and plugins (`<plugin>/skills/<name>/SKILL.md`, invoked
  as `/<plugin>:<skill>`). A pip package has no hook to install any of these;
  distribution is by plugin or by files the user puts there.
- A marketplace is a repository with `.claude-plugin/marketplace.json`
  (`name`, `owner`, `plugins[]` with `name` and `source`); a plugin is a
  directory with an optional `.claude-plugin/plugin.json` (only `name`
  required) and `skills/` at its root. `/plugin marketplace add owner/repo`
  then `/plugin install <plugin>@<marketplace>`.
- A SKILL.md body may run shell through `` !`command` ``; the output replaces
  the placeholder before the model reads it. The command goes through the
  permission check, so it is pre-approved with `allowed-tools`. A non-zero
  exit aborts the invocation with an error the agent sees.
- MCP prompts appear in Claude Code as `/<server>:<prompt> (MCP)`; MCP
  resources are read through `@server:uri` or the client's resource tools.
- The `mcp` 2.x SDK the server uses (`MCPServer`) has `prompt`, `resource`
  and `instructions`.

## Design

### 1. `mushin-mcp`: a `guide` prompt, a `mushin://guide` resource, instructions

`mushin.mcp.server.create_server` registers, next to the tools:

- `@mcp.prompt("guide")`: returns the guide text as `python -m mushin guide`
  prints it (the version comment, then the file), read through the same code.
  A new helper `mushin.__main__.guide_text() -> str` is that code, used by
  both the command and the server, so they cannot drift.
- `@mcp.resource("mushin://guide", mime_type="text/markdown")`: the same text.
- `MCPServer(..., instructions=...)`: two sentences saying the tools are
  read-only over a directory of runs and that the `guide` prompt and resource
  hold the guide for writing experiments.

The guide is about 15 KB (under the 25,000-token output limit by a wide
margin), so it is returned whole.

### 2. The repository as a Claude Code marketplace with one plugin

```
.claude-plugin/marketplace.json      name "mushin", owner, one plugin entry
plugins/mushin/.claude-plugin/plugin.json   name "mushin", description, version
plugins/mushin/skills/guide/SKILL.md
```

`SKILL.md` frontmatter: `name: guide`; a `description` that names the trigger
(writing or changing an experiment that uses mushin, `mushin-py`, sweeps,
`compare_methods`, `Study`) and says it loads the guide matching the installed
version. The body is a short heading, one shell line, and one paragraph for
the case the guide did not print: install `mushin-py[eval]` (0.16 or later),
or read the raw-GitHub copy. The skill contains no guide text of its own.

The shell line (revised in review) is
`python -m mushin guide 2>/dev/null || python3 -m mushin guide 2>&1 || true`:
`python3` is the only interpreter name on many machines, and a failing
injected command aborts the whole invocation before the body is seen, so the
line must never exit non-zero (`|| true` is the form the Claude Code docs
give). The last attempt's error prints where the guide would be, so a broken
install is not read as a missing one. `allowed-tools` pre-approves each
subcommand, `true` by exact match.

The marketplace lives at the repository root so the install is
`/plugin marketplace add martinez-hub/mushin` then
`/plugin install mushin@mushin`; the skill is `/mushin:guide`. The plugin's
`version` field is `0.1.0` and changes only when the plugin changes, not with
mushin's releases (the skill content comes from the installed mushin).

### 3. Docs

- `docs/for-agents.md` gains a "Claude Code" section before "Point your agent
  at it": the two slash commands, `/mushin:guide`, and the MCP prompt and
  resource when `mushin-mcp` is connected. The existing `AGENTS.md` snippet
  stays for every other agent.
- `docs/guides/mcp.md` gains a "Prompt and resource" section listing
  `guide` and `mushin://guide`, with the Claude Code invocation
  `/mushin:guide (MCP)`.
- No new nav page, so `docs/llms.txt` is unchanged; its MCP line gains the
  words "and the guide as a prompt".
- A `changes/+agent-packaging.added.md` fragment.

### 4. Tests

- `tests/test_mcp/test_server.py`: over the in-memory client (the pattern of
  `test_server_end_to_end_in_memory`), `list_prompts` has `guide`,
  `get_prompt("guide")` returns the text `guide_text()` returns,
  `list_resources` has `mushin://guide`, `read_resource` returns the same
  text; the server's `instructions` mention the guide. `create_server` is
  skipped without the `mcp` extra, as the existing tests are.
- `tests/test_agent_guide.py` (or a new `tests/test_plugin.py`): the
  marketplace and plugin manifests parse, the marketplace's plugin entry
  names the plugin whose manifest it points to and the source directory
  exists, `SKILL.md` has `name: guide` and a description under 1,024
  characters, its body runs exactly `python -m mushin guide` and
  `allowed-tools` pre-approves that command, and the install commands quoted
  in `docs/for-agents.md` name the marketplace and plugin as the manifests
  do. These are the facts a Claude Code user depends on and nothing in CI
  runs Claude Code itself.
- `claude plugin validate plugins/mushin` is run once by hand before the PR
  (the CLI is not in CI).
- `tests/test_lazy_imports.py` is unaffected: `mushin.__main__` imports only
  the standard library, and `create_server` already imports `mcp` lazily.

### 5. Error handling

- `guide_text()` raises if the guide file is missing from the install; the
  prompt and resource propagate that as an MCP error, which the client shows.
  The existing CI step that checks the built wheel ships the guide keeps that
  from happening in a release.
- The skill's shell line cannot fail (see section 2): when mushin is missing,
  older than 0.16, or not on this shell's `python` or `python3`, the last
  attempt's error stands where the guide would be and the body's fallback
  paragraph is shown.
  The docs section that tells the human how to install repeats it.

## Alternatives considered

1. Docs only (status quo). No work, but the agent still needs the human to
   paste the paragraph, and the MCP connection cannot deliver the guide.
2. MCP prompt and resource only. Reaches every MCP client, but a Claude Code
   user who does not inspect runs has no reason to connect `mushin-mcp`, and
   the guide is for writing experiments, which comes first.
3. This design: both, with the plugin kept to the skill. The MCP additions are
   about 30 lines against an existing test harness; the plugin is three small
   files whose only moving part is the command that prints the guide.
4. Plugin bundling the MCP server as well. Rejected: see Non-goals.
