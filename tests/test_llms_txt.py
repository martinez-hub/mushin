# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Keep `docs/llms.txt` in step with the docs site and the package.

`llms.txt` is the index an LLM or coding agent reads instead of crawling the
site, so a page missing from it is invisible to them, a link to a page that was
renamed or removed sends them to a 404, and a stale fact (an install name, the
Python range, an extra) ends up in the code they write. mkdocs' strict build
validates links between Markdown pages but never reads a `.txt` file, so nothing
else would catch any of these. CI runs this file in the always-on `docs` job
too, because the `test` job skips the docs-only PRs most likely to break it.
"""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from functools import cache
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
LLMS_TXT = ROOT / "docs" / "llms.txt"


class _MkdocsLoader(yaml.SafeLoader):
    """A safe loader that reads mkdocs' custom tags (`!ENV`, `!relative`,
    `!!python/name:...`) as None instead of raising, since only `site_url` and
    `nav` are needed here and mkdocs itself accepts those tags."""


_MkdocsLoader.add_multi_constructor("", lambda loader, suffix, node: None)


@cache
def _mkdocs() -> dict:
    return yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), _MkdocsLoader)


@cache
def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@cache
def _llms_txt() -> str:
    return LLMS_TXT.read_text(encoding="utf-8")


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


def _llms_txt_urls() -> list[str]:
    return re.findall(r"\]\((" + re.escape(_site()) + r"[^)]*)\)", _llms_txt())


@pytest.mark.parametrize(
    "page, url",
    [
        ("index.md", ""),
        ("guides/compare.md", "guides/compare/"),
        ("guides/index.md", "guides/"),
        ("guides/README.md", "guides/"),
        ("notebooks/01_sweep_to_dataset.ipynb", "notebooks/01_sweep_to_dataset/"),
    ],
)
def test_site_url_matches_mkdocs_directory_urls(page, url):
    assert _site_url(page) == _site() + url


def test_external_nav_links_are_not_pages():
    nav = [{"Home": "index.md"}, {"GitHub": "https://github.com/martinez-hub/mushin"}]
    assert _nav_pages(nav) == ["index.md"]


def test_mkdocs_custom_tags_do_not_break_the_loader():
    config = yaml.load(
        "site_url: !ENV [SITE_URL, 'https://example.org/']\n"
        "emoji_index: !!python/name:material.extensions.emoji.twemoji\n"
        "nav:\n  - Home: index.md\n",
        _MkdocsLoader,
    )
    assert config["nav"] == [{"Home": "index.md"}]


def test_llms_txt_follows_the_spec_header():
    """An H1 with the project name, then a blockquote summary (llmstxt.org)."""
    lines = _llms_txt().splitlines()
    assert lines[0] == "# mushin"
    assert next(line for line in lines[1:] if line.strip()).startswith("> ")


def test_llms_txt_links_use_the_configured_site_url():
    """Every docs link uses mkdocs.yml's `site_url`, so moving the site to a new
    host cannot leave llms.txt pointing at the old one."""
    links = re.findall(r"\]\((https?://[^)]*)\)", _llms_txt())
    docs_links = [link for link in links if "github.io/mushin" in link]
    assert docs_links, "llms.txt has no links to the docs site"
    assert all(link.startswith(_site()) for link in docs_links), (
        f"links not under site_url {_site()}: "
        f"{[link for link in docs_links if not link.startswith(_site())]}"
    )


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
    duplicates = [url for url, n in Counter(_llms_txt_urls()).items() if n > 1]
    assert not duplicates, f"listed more than once: {sorted(duplicates)}"


def test_llms_txt_install_name_matches_pyproject():
    name = _pyproject()["project"]["name"]
    assert f"`pip install {name}`" in _llms_txt()


def test_llms_txt_python_range_matches_the_classifiers():
    match = re.search(r"Supported Python: (3\.\d+) to (3\.\d+)\.", _llms_txt())
    assert match, "could not find the 'Supported Python: X to Y.' line"
    classified = sorted(
        (int(c.rsplit(".", 1)[1]), c.rsplit(" ", 1)[1])
        for c in _pyproject()["project"]["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    )
    assert match.groups() == (classified[0][1], classified[-1][1])


def test_llms_txt_lists_every_extra():
    match = re.search(r"Other extras: ([^.]+)\.", _llms_txt())
    assert match, "could not find the 'Other extras: ...' list"
    listed = set(re.findall(r"`([^`]+)`", match.group(1)))
    extras = set(_pyproject()["project"]["optional-dependencies"]) - {"eval"}
    assert listed == extras
    assert '`pip install "mushin-py[eval]"`' in _llms_txt()


def test_llms_txt_example_runs(cleandir):
    """The example is the code an agent is most likely to copy; it must run and
    produce what the comments next to it say."""
    match = re.search(r"```python\n(.*?)```", _llms_txt(), re.DOTALL)
    assert match, "llms.txt has no python example"
    namespace: dict = {}
    exec(compile(match.group(1), str(LLMS_TXT), "exec"), namespace)
    ds = namespace["ds"]
    assert dict(ds.sizes) == {"lr": 3, "seed": 3}
    assert namespace["experiment"].workflow.to_dataframe().shape == (9, 3)
