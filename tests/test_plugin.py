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

import yaml

REPO = Path(__file__).resolve().parents[1]
MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"
PLUGIN_DIR = REPO / "plugins" / "mushin"
PLUGIN = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
SKILL = PLUGIN_DIR / "skills" / "guide" / "SKILL.md"
FOR_AGENTS = REPO / "docs" / "for-agents.md"
README = REPO / "README.md"

GUIDE_COMMAND = "python -m mushin guide"
# The Agent Skills specification (agentskills.io) caps `description` at 1,024
# characters, and Claude Code lists the first 1,536 of description plus
# when_to_use; the lower bound keeps the skill valid for every host.
DESCRIPTION_LIMIT = 1024


def _frontmatter(text: str) -> dict:
    m = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.S)
    assert m, "SKILL.md must start with a YAML frontmatter block"
    fields = yaml.safe_load(m.group(1))  # PyYAML comes with omegaconf
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
    assert 0 < len(fields["description"]) <= DESCRIPTION_LIMIT
    for word in ("mushin", "experiment", "installed"):
        assert word in fields["description"]
    body = fields["_body"]
    # One shell line, run at invocation and pre-approved, so the skill never
    # holds a copy of the guide that could go stale. It tries `python`, then
    # `python3` (the only interpreter name on many boxes), then `true`, so the
    # line never exits non-zero: a failed injected command aborts the whole
    # invocation and the fallback paragraph would never be seen (`|| true` is
    # the form the Claude Code docs give). Each subcommand is pre-approved, or
    # the invocation stops at a prompt; `true` is matched exactly.
    assert body.count("!`") == 1
    [command] = re.findall(r"!`([^`]*)`", body)
    parts = [part.strip() for part in command.split("||")]
    attempts = [part.split(" 2>")[0] for part in parts[:-1]]
    assert attempts == [GUIDE_COMMAND, GUIDE_COMMAND.replace("python", "python3")]
    assert parts[-1] == "true"
    allowed = fields["allowed-tools"]
    if isinstance(allowed, str):  # the docs allow a list or a separated string
        allowed = [a.strip() for a in allowed.split(",")]
    assert allowed == [f"Bash({s}:*)" for s in attempts] + ["Bash(true)"]
    assert "mushin-py[eval]" in body  # the fallback paragraph
    assert SKILL.read_text(encoding="utf-8").isascii()


def test_docs_install_commands_name_the_manifests():
    market = json.loads(MARKETPLACE.read_text(encoding="utf-8"))
    plugin = json.loads(PLUGIN.read_text(encoding="utf-8"))
    docs = FOR_AGENTS.read_text(encoding="utf-8")
    # The marketplace is this repository, as the plugin manifest names it.
    source = plugin["repository"].removeprefix("https://github.com/")
    assert re.fullmatch(r"[\w.-]+/[\w.-]+", source), source
    assert market["owner"]["url"] == f"https://github.com/{source.split('/')[0]}"
    assert f"/plugin marketplace add {source}" in docs
    assert f"/plugin install {plugin['name']}@{market['name']}" in docs
    assert f"/{plugin['name']}:guide" in docs
    # The one-step form (Claude Code 2.1.275 or later) is on both pages, and
    # the README names the version it needs and points at the two-step page.
    one_step = f"/plugin install {plugin['name']} --marketplace {source}"
    readme = README.read_text(encoding="utf-8")
    # The README's Claude Code paragraph only: the page link above it predates
    # this plugin and would satisfy a whole-file check.
    paragraph = readme[readme.index("/mushin:guide") : readme.index("## What else")]
    assert one_step in docs and one_step in paragraph
    assert "2.1.275" in docs and "2.1.275" in paragraph
    assert "for-agents/" in paragraph
