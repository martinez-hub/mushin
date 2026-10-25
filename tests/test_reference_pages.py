# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Every public name has an API reference entry (docs/reference/*.md).

The reference pages render docstrings through mkdocstrings `:::` directives.
A name added to a public namespace without one is how the backlog that #267
cleared accrued, so the invariant is pinned here, and the docs CI job runs it
on every PR."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "reference"

# Documented in prose on their page rather than through a directive: the llm
# type aliases (mkdocstrings cannot render an alias) and the two hydra-zen
# re-exports, whose own docs the workflows page links to.
PROSE_ONLY = {"System", "Metric", "Example", "multirun", "hydra_list"}
# Subpackages exposed as names, not objects to document.
MODULES = {"export", "llm"}


def _directive_leaves() -> set[str]:
    leaves = set()
    for page in REFERENCE.glob("*.md"):
        for path in re.findall(r"^::: (\S+)$", page.read_text(encoding="utf-8"), re.M):
            leaves.add(path.rsplit(".", 1)[-1])
    return leaves


def test_every_public_name_has_a_reference_entry():
    pytest.importorskip("scipy")  # the eval extra: mushin.benchmark and mushin.llm
    import mushin
    import mushin.benchmark
    import mushin.lightning
    import mushin.llm
    import mushin.workflows

    public = set()
    for module in (
        mushin,
        mushin.benchmark,
        mushin.lightning,
        mushin.llm,
        mushin.workflows,
    ):
        public |= set(module.__all__)
    missing = sorted(public - PROSE_ONLY - MODULES - _directive_leaves())
    assert missing == [], f"public names with no reference entry: {missing}"
    # The prose-only set is still public, so the page does name each one.
    prose = (REFERENCE / "llm.md").read_text() + (
        REFERENCE / "workflows.md"
    ).read_text()
    for name in PROSE_ONLY:
        assert f"`{name}`" in prose or f"`mushin.{name}`" in prose, name
