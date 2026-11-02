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
from urllib.parse import urlsplit

import pytest
import yaml

from _docs_site import (
    ROOT,
    _MkdocsLoader,
    _nav_pages,
    _nav_urls,
    _site,
    _site_url,
    markdown_links,
    python_blocks,
    stray_python_fences,
)

LLMS_TXT = ROOT / "docs" / "llms.txt"

# Hosts llms.txt may link to besides the docs site itself. A link to any other
# host, or to the site's host outside the site, is a typo or a stale address.
EXTERNAL_HOSTS = {"github.com", "raw.githubusercontent.com"}

# Files the docs build writes into the site that are not nav pages:
# scripts/llms_full.py (a hook in mkdocs.yml) writes llms-full.txt.
BUILT_FILES = {"llms-full.txt"}


@cache
def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@cache
def _llms_txt() -> str:
    return LLMS_TXT.read_text(encoding="utf-8")


def _llms_txt_links() -> list[str]:
    return markdown_links(_llms_txt())


def _llms_txt_urls() -> list[str]:
    """The docs-site links in llms.txt, without their #anchors."""
    return [
        url.split("#", 1)[0] for url in _llms_txt_links() if url.startswith(_site())
    ]


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


def test_mkdocs_custom_tags_do_not_break_the_loader(monkeypatch):
    text = (
        "site_url: !ENV [SITE_URL, 'https://example.org/']\n"
        "emoji_index: !!python/name:material.extensions.emoji.twemoji\n"
        "nav:\n  - Home: index.md\n  - Install: !relative install.md\n"
    )
    monkeypatch.delenv("SITE_URL", raising=False)
    config = yaml.load(text, _MkdocsLoader)
    assert config["site_url"] == "https://example.org/"
    assert config["nav"] == [{"Home": "index.md"}, {"Install": "install.md"}]
    monkeypatch.setenv("SITE_URL", "https://docs.example.org/")
    assert yaml.load(text, _MkdocsLoader)["site_url"] == "https://docs.example.org/"


def test_every_llms_txt_link_is_on_the_site_or_a_known_host():
    """Links outside the docs site are not checked against the nav, so a typo in
    the site prefix (`/mushn/`), `http://`, or an old host would slip through."""
    site = urlsplit(_site())
    bad = [
        url
        for url in _llms_txt_links()
        if not url.startswith(_site())
        and (
            urlsplit(url).scheme != "https"
            or urlsplit(url).netloc == site.netloc
            or urlsplit(url).netloc not in EXTERNAL_HOSTS
        )
    ]
    assert not bad, (
        f"docs/llms.txt links outside the site {_site()} to unexpected places: "
        f"{bad} (fix the link, or add its host to EXTERNAL_HOSTS)"
    )


def test_llms_txt_follows_the_spec_header():
    """An H1 with the project name, then a blockquote summary (llmstxt.org)."""
    lines = _llms_txt().splitlines()
    assert lines[0] == "# mushin"
    assert next(line for line in lines[1:] if line.strip()).startswith("> ")


def test_every_nav_page_is_listed_in_llms_txt():
    """Nav URLs are built from mkdocs.yml's `site_url`, so this also fails when
    the site moves to a new host and llms.txt still links to the old one."""
    missing = _nav_urls() - set(_llms_txt_urls())
    assert not missing, (
        f"pages in mkdocs.yml's nav but not in docs/llms.txt: {sorted(missing)}"
    )


def test_every_llms_txt_site_link_is_a_nav_page():
    built = {_site() + name for name in BUILT_FILES}
    stale = set(_llms_txt_urls()) - _nav_urls() - built
    assert not stale, (
        f"docs/llms.txt links to pages that are not in the nav: {sorted(stale)}"
    )


def test_llms_txt_links_llms_full_txt():
    """The one-file copy is only useful if an agent reading the index finds it."""
    assert _site() + "llms-full.txt" in _llms_txt_links()
    hooks = yaml.load((ROOT / "mkdocs.yml").read_text("utf-8"), _MkdocsLoader)
    assert "scripts/llms_full.py" in hooks.get("hooks", [])


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
    """The examples are the code an agent is most likely to copy; every python
    block must run, in order, and produce what the comments next to it say."""
    blocks = python_blocks(_llms_txt())
    assert blocks, "llms.txt has no python example"
    namespace: dict = {}
    exec(compile("\n".join(blocks), str(LLMS_TXT), "exec"), namespace)
    ds = namespace["ds"]
    assert dict(ds.sizes) == {"lr": 3, "seed": 3}
    assert namespace["experiment"].workflow.to_dataframe().shape == (9, 3)


def test_every_llms_txt_python_example_runs():
    """`test_llms_txt_example_runs` runs only fences tagged exactly ```python at
    the start of a line; any other Python fence would ship untested."""
    stray = stray_python_fences(_llms_txt())
    assert not stray, stray
