# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The docs site's nav and URLs, read from mkdocs.yml, for the tests that check
links into the site (`test_llms_txt.py`, `test_agent_guide.py`)."""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class _MkdocsLoader(yaml.SafeLoader):
    """A safe loader that accepts mkdocs' custom tags instead of raising.
    `!ENV` resolves as mkdocs resolves it (the first variable that is set, else
    the default), so an environment-driven `site_url` still yields a URL. Any
    other tag (`!relative`, `!!python/name:...`) loads as its plain value: only
    `site_url` and `nav` are read here."""


def _construct_tagged(loader, tag, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    if tag != "!ENV":
        return value
    *names, default = value if isinstance(value, list) else [value, None]
    return next((os.environ[n] for n in names if n in os.environ), default)


_MkdocsLoader.add_multi_constructor("", _construct_tagged)


@cache
def _mkdocs() -> dict:
    return yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), _MkdocsLoader)


def _site() -> str:
    return _mkdocs()["site_url"].rstrip("/") + "/"


def _nav_pages(node) -> list[str]:
    """Every local page path under mkdocs.yml's `nav`, in order. External links
    (a nav entry may be a URL) are not pages of the site, so they are skipped."""
    if isinstance(node, str):
        return [] if "://" in node else [node]
    if isinstance(node, list):
        return [page for item in node for page in _nav_pages(item)]
    if isinstance(node, dict):
        return [page for value in node.values() for page in _nav_pages(value)]
    return []


def _site_url(page: str) -> str:
    """The URL mkdocs serves a nav page at, with directory URLs (the default):
    `a/b.md` at `a/b/`, and an `index.md` or `README.md` at its directory."""
    parent, _, name = page.rpartition("/")
    stem = name.rsplit(".", 1)[0]
    if stem in ("index", "README"):
        path = f"{parent}/" if parent else ""
    else:
        path = f"{parent}/{stem}/" if parent else f"{stem}/"
    return _site() + path


def _nav_urls() -> set[str]:
    return {_site_url(page) for page in _nav_pages(_mkdocs()["nav"])}


def markdown_links(text: str) -> list[str]:
    """The targets of the Markdown links `[text](target)` in `text`."""
    return re.findall(r"\]\(([^)\s]+)\)", text)


def python_blocks(text: str) -> list[str]:
    """The bodies of the ```python fences that start a line, in order."""
    return re.findall(r"^```python\n(.*?)^```$", text, re.DOTALL | re.MULTILINE)
