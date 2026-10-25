# Agent packaging implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the agent guide through `mushin-mcp` (a `guide` prompt and a `mushin://guide` resource) and as a Claude Code plugin whose skill prints the installed guide.

**Architecture:** `mushin.__main__.guide_text()` becomes the one function that renders the guide (version comment plus file); `python -m mushin guide`, the MCP prompt and the MCP resource all call it. The repository root carries a Claude Code marketplace with one plugin, `plugins/mushin`, whose `guide` skill runs `python -m mushin guide` at invocation time, so no copy of the guide exists outside the package. Tests pin the MCP protocol surface through the SDK's in-memory client and the plugin's manifests through plain JSON and text checks.

**Tech Stack:** Python 3.11+, `mcp` 2.x (`MCPServer.prompt`, `.resource`, `instructions`; `mcp.Client` for tests), pytest, mkdocs-material docs, towncrier fragments.

Spec: `docs/superpowers/specs/2026-10-24-agent-packaging-design.md`. Branch: `feat/agent-packaging`. Run every command from the repository root.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `src/mushin/__main__.py` | Modify: add `guide_text()`; `main()` prints it. |
| `src/mushin/mcp/server.py` | Modify: `instructions=` on the server; `guide` prompt and `mushin://guide` resource in `create_server`. |
| `tests/test_agent_guide.py` | Modify: `python -m mushin guide` test also pins `guide_text()`. |
| `tests/test_mcp/test_server.py` | Modify: in-memory test covers prompt, resource, instructions. |
| `.claude-plugin/marketplace.json` | Create: the marketplace (name `mushin`, one plugin). |
| `plugins/mushin/.claude-plugin/plugin.json` | Create: the plugin manifest (name `mushin`). |
| `plugins/mushin/skills/guide/SKILL.md` | Create: the skill that prints the installed guide. |
| `tests/test_plugin.py` | Create: manifests, skill frontmatter and body, docs commands agree. |
| `docs/for-agents.md` | Modify: Claude Code section (plugin, MCP prompt). |
| `docs/guides/mcp.md` | Modify: "Prompt and resource" section. |
| `docs/llms.txt` | Modify: the MCP line mentions the guide prompt. |
| `changes/+agent-packaging.added.md` | Create: the fragment (renamed to the PR number later). |

---

### Task 1: `guide_text()` is the one rendering of the guide

**Files:**
- Modify: `src/mushin/__main__.py`
- Test: `tests/test_agent_guide.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_agent_guide.py`, after `test_python_m_mushin_guide`, add:

```python
def test_guide_text_is_what_the_command_prints(capsys):
    """The MCP prompt and resource return guide_text(); it must be exactly
    what `python -m mushin guide` prints, so the two can never drift."""
    from mushin.__main__ import guide_text, main

    assert main(["guide"]) == 0
    assert capsys.readouterr().out == guide_text()
    assert guide_text().startswith("<!-- mushin ")
    assert guide_text().endswith(_text())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra eval pytest tests/test_agent_guide.py -k guide_text_is_what -p no:cacheprovider -q`
Expected: FAIL with `ImportError: cannot import name 'guide_text'`.

- [ ] **Step 3: Implement**

In `src/mushin/__main__.py`, add after `_version()`:

```python
def guide_text() -> str:
    """The guide as ``python -m mushin guide`` prints it: a version comment
    (valid Markdown, invisible when rendered), then the shipped file. The
    MCP server's ``guide`` prompt and ``mushin://guide`` resource return the
    same string, so every route to the guide shows the installed version."""
    path = resources.files("mushin") / GUIDE
    return f"<!-- mushin {_version()} -->\n" + path.read_text(encoding="utf-8")
```

and replace the end of `main()`:

```python
    path = resources.files("mushin") / GUIDE
    if args.path:
        if not Path(str(path)).is_file():  # e.g. mushin imported from a zip
            print(
                "the guide is not a file on disk here (mushin is imported from "
                "an archive); print it with `python -m mushin guide`",
                file=sys.stderr,
            )
            return 1
        print(path)
        return 0
    sys.stdout.write(guide_text())
    return 0
```

(The two `sys.stdout.write` calls and the comment above them are replaced by the single write.)

- [ ] **Step 4: Run the guide tests**

Run: `uv run --extra eval --group docs pytest tests/test_agent_guide.py -p no:cacheprovider -q`
Expected: all pass (the cp1252 pipe test still passes: the text is ASCII).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/__main__.py tests/test_agent_guide.py
git commit -m "refactor: guide_text() renders the agent guide for every route to it"
```

---

### Task 2: the MCP server serves the guide as a prompt and a resource

**Files:**
- Modify: `src/mushin/mcp/server.py` (`create_server`)
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write the failing test**

Add after `test_server_end_to_end_in_memory`:

```python
def test_server_serves_the_guide_as_prompt_and_resource(tmp_path):
    """An MCP client gets the agent guide, matching the installed mushin,
    without leaving the client: as the `guide` prompt (a slash command in
    Claude Code) and as the `mushin://guide` resource."""
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from mushin.__main__ import guide_text
    from mushin.mcp.server import create_server

    server = create_server(root=tmp_path)
    assert "guide" in server.instructions and "read-only" in server.instructions

    async def scenario():
        async with Client(server, raise_exceptions=True) as client:
            prompts = await client.list_prompts()
            prompt = await client.get_prompt("guide")
            resources = await client.list_resources()
            resource = await client.read_resource("mushin://guide")
            return prompts, prompt, resources, resource

    prompts, prompt, resources, resource = anyio.run(scenario)
    assert [p.name for p in prompts.prompts] == ["guide"]
    assert [m.content.text for m in prompt.messages] == [guide_text()]
    assert [(str(r.uri), r.mime_type) for r in resources.resources] == [
        ("mushin://guide", "text/markdown")
    ]
    assert [c.text for c in resource.contents] == [guide_text()]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra eval pytest tests/test_mcp/test_server.py -k prompt_and_resource -p no:cacheprovider -q`
Expected: FAIL at `server.instructions` (None) or at `assert [p.name ...] == ["guide"]`.

- [ ] **Step 3: Implement**

In `src/mushin/mcp/server.py`, `create_server`, replace `mcp = MCPServer("mushin")` with:

```python
    mcp = MCPServer(
        "mushin",
        instructions=(
            "Read-only tools over a directory of completed mushin runs: list "
            "and describe experiments, read their metrics, configs, failures "
            "and provenance, and summarize saved datasets. The `guide` prompt "
            "and the `mushin://guide` resource hold mushin's usage guide for "
            "writing experiments, matching the installed version."
        ),
    )
```

and before `return mcp` add:

```python
# The guide reaches an agent through the same connection it inspects runs
# with: a prompt (a slash command in Claude Code) and a resource (an
# @-mention). Both return what `python -m mushin guide` prints.
@mcp.prompt("guide")
def guide() -> str:
    """mushin's usage guide for writing experiments, matching the
    installed version (what `python -m mushin guide` prints)."""
    from mushin.__main__ import guide_text

    return guide_text()


@mcp.resource("mushin://guide", mime_type="text/markdown")
def guide_resource() -> str:
    """mushin's usage guide for writing experiments (Markdown)."""
    from mushin.__main__ import guide_text

    return guide_text()
```

- [ ] **Step 4: Run the MCP tests**

Run: `uv run --extra eval pytest tests/test_mcp/test_server.py -p no:cacheprovider -q`
Expected: all pass. If the resource test fails on an attribute name, print the result object once and use the attribute the SDK exposes (`mime_type` on `Resource`; `text` on the read contents).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): serve the agent guide as the guide prompt and the mushin://guide resource"
```

---

### Task 3: the Claude Code marketplace and plugin

**Files:**
- Create: `.claude-plugin/marketplace.json`
- Create: `plugins/mushin/.claude-plugin/plugin.json`
- Create: `plugins/mushin/skills/guide/SKILL.md`
- Test: `tests/test_plugin.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_plugin.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The Claude Code marketplace and plugin at the repository root.

Nothing in CI runs Claude Code, so these pin the facts a user depends on:
the manifests parse and agree, the skill names and pre-approves the one
command it runs, and the install commands in the docs name what the
manifests name."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"
PLUGIN_DIR = REPO / "plugins" / "mushin"
PLUGIN = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
SKILL = PLUGIN_DIR / "skills" / "guide" / "SKILL.md"
FOR_AGENTS = REPO / "docs" / "for-agents.md"

GUIDE_COMMAND = "python -m mushin guide"


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.S)
    assert m, "SKILL.md must start with a YAML frontmatter block"
    fields = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    fields["_body"] = m.group(2)
    return fields


def test_marketplace_and_plugin_manifests_agree():
    market = json.loads(MARKETPLACE.read_text(encoding="utf-8"))
    plugin = json.loads(PLUGIN.read_text(encoding="utf-8"))
    assert market["name"] == "mushin"
    assert market["owner"]["name"]
    [entry] = market["plugins"]
    # The entry must name the plugin whose manifest it points to, or the
    # install fails; relative sources start with ./ and stay in the repo.
    assert entry["name"] == plugin["name"] == "mushin"
    assert entry["source"].startswith("./") and ".." not in entry["source"]
    assert (REPO / entry["source"]).resolve() == PLUGIN_DIR
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", plugin["name"])
    assert plugin["description"] and plugin["version"]


def test_skill_prints_the_installed_guide_and_nothing_else():
    fields = _frontmatter(SKILL.read_text(encoding="utf-8"))
    assert fields["name"] == "guide"
    # Claude Code shows the first 1,536 characters of the description.
    assert 0 < len(fields["description"]) <= 1024
    for word in ("mushin", "experiment", "installed"):
        assert word in fields["description"]
    body = fields["_body"]
    # One shell command, run at invocation and pre-approved, so the skill
    # never holds a copy of the guide that could go stale.
    assert body.count("!`") == 1
    assert f"!`{GUIDE_COMMAND}`" in body
    assert fields["allowed-tools"] == f"Bash({GUIDE_COMMAND}:*)"
    assert "mushin-py[eval]" in body  # the fallback when the command fails
    assert SKILL.read_text(encoding="utf-8").isascii()


def test_docs_install_commands_name_the_manifests():
    market = json.loads(MARKETPLACE.read_text(encoding="utf-8"))
    plugin = json.loads(PLUGIN.read_text(encoding="utf-8"))
    docs = FOR_AGENTS.read_text(encoding="utf-8")
    assert "/plugin marketplace add martinez-hub/mushin" in docs
    assert f"/plugin install {plugin['name']}@{market['name']}" in docs
    assert f"/{plugin['name']}:guide" in docs
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra eval pytest tests/test_plugin.py -p no:cacheprovider -q`
Expected: 3 FAIL with `FileNotFoundError` on `marketplace.json`.

- [ ] **Step 3: Create the marketplace manifest**

`.claude-plugin/marketplace.json`:

```json
{
  "name": "mushin",
  "owner": {
    "name": "Josue Martinez-Martinez",
    "url": "https://github.com/martinez-hub"
  },
  "metadata": {
    "description": "Plugins for mushin, the boilerplate-free sweep engine built on hydra-zen.",
    "homepage": "https://martinez-hub.github.io/mushin/"
  },
  "plugins": [
    {
      "name": "mushin",
      "source": "./plugins/mushin",
      "description": "The mushin usage guide for AI coding agents, as the /mushin:guide skill. Prints the guide that matches the installed mushin (0.16 or later).",
      "category": "development"
    }
  ]
}
```

- [ ] **Step 4: Create the plugin manifest**

`plugins/mushin/.claude-plugin/plugin.json`:

```json
{
  "name": "mushin",
  "version": "0.1.0",
  "description": "mushin's usage guide for AI coding agents, as the /mushin:guide skill. It prints the guide that matches the installed mushin, so it never goes stale.",
  "author": {
    "name": "Josue Martinez-Martinez",
    "url": "https://github.com/martinez-hub"
  },
  "homepage": "https://martinez-hub.github.io/mushin/for-agents/",
  "repository": "https://github.com/martinez-hub/mushin",
  "license": "MIT",
  "keywords": ["mushin", "hydra-zen", "experiments", "sweeps", "machine-learning"]
}
```

- [ ] **Step 5: Create the skill**

`plugins/mushin/skills/guide/SKILL.md`:

```markdown
---
name: guide
description: Use before writing or changing an experiment that uses mushin (mushin-py; sweeps, multirun, compare_methods, Study). Loads mushin's usage guide for AI coding agents, matching the installed version, with the rules that prevent wrong results and wasted compute.
allowed-tools: Bash(python -m mushin guide:*)
---

# mushin usage guide

The guide below is printed by the installed mushin, so it matches the
version this project uses. Follow it before writing or changing an
experiment.

!`python -m mushin guide`

If the command above failed, mushin is missing, older than 0.16, or not on
this `python`. Install it with `pip install "mushin-py[eval]"` into the
project's environment (not `pip install mushin`, a PyPI name that is not this
project), or on mushin 0.10 to 0.15 read
https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md
instead.
```

- [ ] **Step 6: Run the plugin tests**

Run: `uv run --extra eval pytest tests/test_plugin.py -p no:cacheprovider -q`
Expected: the first two pass; `test_docs_install_commands_name_the_manifests` fails until Task 4 (the docs). That is expected at this step.

- [ ] **Step 7: Validate the plugin with the Claude Code CLI (by hand, not in CI)**

Run: `claude plugin validate plugins/mushin` and `claude plugin validate .`
Expected: both report a valid plugin / marketplace. If `claude` is not on PATH, skip this step and say so in the PR body.

- [ ] **Step 8: Commit**

```bash
git add .claude-plugin/marketplace.json plugins/mushin tests/test_plugin.py
git commit -m "feat: a Claude Code marketplace and plugin whose guide skill prints the installed agent guide"
```

---

### Task 4: docs, llms.txt and the fragment

**Files:**
- Modify: `docs/for-agents.md`
- Modify: `docs/guides/mcp.md`
- Modify: `docs/llms.txt`
- Create: `changes/+agent-packaging.added.md`

- [ ] **Step 1: The For AI agents page**

In `docs/for-agents.md`, insert before `## Point your agent at it`:

```markdown
## Claude Code

Install the guide as a skill from this repository's plugin marketplace, then
load it with `/mushin:guide` before writing or changing an experiment:

```text
/plugin marketplace add martinez-hub/mushin
/plugin install mushin@mushin
```

The skill runs `python -m mushin guide` in your project, so it always shows
the guide for the installed mushin (0.16 or later; install it first with
`pip install "mushin-py[eval]"`). With `mushin-mcp` connected (see
[Analyzing from Claude Code](guides/mcp.md)), the guide is also the
`/mushin:guide (MCP)` prompt and the `@mushin:mushin://guide` resource.

```

and change the opening of the following section's first sentence from
`Add this to your project's `AGENTS.md` or `CLAUDE.md`:` to
`For any agent, add this to your project's `AGENTS.md` or `CLAUDE.md`:`.

- [ ] **Step 2: The MCP guide**

In `docs/guides/mcp.md`, after the sentence `All tools are read-only. ...launches sweeps.`, insert:

```markdown
## Prompt and resource

The server also serves mushin's usage guide for AI coding agents, matching
the installed version (what `python -m mushin guide` prints):

| | Name | In Claude Code |
|---|---|---|
| Prompt | `guide` | `/mushin:guide (MCP)` |
| Resource | `mushin://guide` | `@mushin:mushin://guide` |

Use it before asking the agent to write or change an experiment. The
[For AI agents](../for-agents.md) page has the same guide as a plugin skill
that does not need the server.

```

- [ ] **Step 3: llms.txt**

In `docs/llms.txt`, change the MCP line to:

```text
- [Analyzing from Claude Code](https://martinez-hub.github.io/mushin/guides/mcp/): the read-only `mushin-mcp` server for inspecting completed runs from an MCP client; it also serves the agent guide as the `guide` prompt and the `mushin://guide` resource
```

- [ ] **Step 4: The fragment**

`changes/+agent-packaging.added.md`:

```markdown
The agent guide reaches agents inside their tools. `mushin-mcp` serves it as the `guide` prompt (`/mushin:guide (MCP)` in Claude Code) and the `mushin://guide` resource, and its instructions say so. The repository is a Claude Code plugin marketplace: `/plugin marketplace add martinez-hub/mushin` then `/plugin install mushin@mushin` installs the `/mushin:guide` skill, which prints the guide of the installed mushin. `python -m mushin guide`, the prompt and the resource all render through one function, so they cannot drift.
```

- [ ] **Step 5: Run the docs checks**

Run:
```bash
uv run --extra eval pytest tests/test_plugin.py -p no:cacheprovider -q
make docs-build
uv run --extra eval --group docs pytest tests/test_llms_txt.py tests/test_docs_claims.py tests/test_agent_guide.py -p no:cacheprovider -q
```
Expected: all pass; the docs build has no warning. `test_llms_txt` checks that every nav page is listed (no page was added) and runs the example.

- [ ] **Step 6: Commit**

```bash
git add docs/for-agents.md docs/guides/mcp.md docs/llms.txt changes/+agent-packaging.added.md
git commit -m "docs: install the agent guide as a Claude Code plugin or read it over MCP"
```

---

### Task 5: full check, PR, reviews

- [ ] **Step 1: Everything CI runs**

Run: `make check`
Expected: all pass (lint, format, spell, tests). codespell covers `src tests ... changes`; the plugin files are outside its scope, so also run `uv run codespell plugins .claude-plugin docs/for-agents.md docs/guides/mcp.md`.

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin feat/agent-packaging
gh pr create --base main --title "feat: the agent guide as an MCP prompt and a Claude Code plugin" --body-file <PR body>
```

The PR body: what, why, the install commands, what was validated by hand (`claude plugin validate`), what the tests pin. Rename the fragment to `<number>.added.md` and push.

- [ ] **Step 3: Reviews before merging**

Run `/code-review <number>` and the adversarial workflow on the head; fix what they confirm; re-run on a fix commit; merge when CI and Codex are clean. Update the page's checklist item and "What shipped" table.
