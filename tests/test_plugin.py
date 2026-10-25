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
