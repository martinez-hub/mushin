# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Keep `docs/llms.txt` in step with the docs site's nav.

`llms.txt` is the index an LLM or coding agent reads instead of crawling the
site, so a page missing from it is invisible to them, and a link to a page that
was renamed or removed sends them to a 404. mkdocs' strict build validates
links between Markdown pages but never reads a `.txt` file, so nothing else
would catch either drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://martinez-hub.github.io/mushin/"
LINK = re.compile(r"\]\((" + re.escape(SITE) + r"[^)]*)\)")


def _nav_pages(node) -> list[str]:
    """Every page path under mkdocs.yml's `nav`, in order."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [page for item in node for page in _nav_pages(item)]
    if isinstance(node, dict):
        return [page for value in node.values() for page in _nav_pages(value)]
    return []


def _site_url(page: str) -> str:
    """The URL mkdocs serves a nav page at (directory URLs, the default)."""
    stem = page.rsplit(".", 1)[0]
    if stem == "index":
        return SITE
    return f"{SITE}{stem}/"


def _nav_urls() -> set[str]:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    return {_site_url(page) for page in _nav_pages(config["nav"])}


def _llms_txt_urls() -> list[str]:
    return LINK.findall((ROOT / "docs" / "llms.txt").read_text(encoding="utf-8"))


def test_llms_txt_follows_the_spec_header():
    """An H1 with the project name, then a blockquote summary (llmstxt.org)."""
    lines = (ROOT / "docs" / "llms.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# mushin"
    assert next(line for line in lines[1:] if line.strip()).startswith("> ")


def test_every_nav_page_is_listed_in_llms_txt():
    missing = _nav_urls() - set(_llms_txt_urls())
    assert not missing, (
        f"pages in mkdocs.yml's nav but not in docs/llms.txt: {sorted(missing)}"
    )


def test_every_llms_txt_site_link_is_a_nav_page():
    stale = set(_llms_txt_urls()) - _nav_urls()
    assert not stale, (
        f"docs/llms.txt links to pages that are not in the nav: {sorted(stale)}"
    )


def test_llms_txt_lists_each_page_once():
    urls = _llms_txt_urls()
    duplicates = {url for url in urls if urls.count(url) > 1}
    assert not duplicates, f"listed more than once: {sorted(duplicates)}"
