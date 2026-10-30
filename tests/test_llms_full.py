# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Check `llms-full.txt`, the docs site as one file, as `mkdocs build` writes it.

The hook in `scripts/llms_full.py` builds the file from the nav pages on every
build, so it cannot go stale, but it can silently go wrong: a page dropped, an
include or an API directive left as a bare directive, a notebook's base64 plots
pasted into an agent's context, or a relative link that means nothing outside
the site. The strict build never reads the file, so these tests do. They need
the docs dependency group (mkdocs, pymdownx and griffe); CI runs them in the
`docs` job on every PR.
"""

from __future__ import annotations

import importlib.util
import json
import re
from functools import cache

import pytest

pytest.importorskip("pymdownx.snippets")
pytest.importorskip("griffe")
mkdocs_config = pytest.importorskip("mkdocs.config")

from _docs_site import ROOT, _nav_pages, _site, _site_url  # noqa: E402

HOOK = ROOT / "scripts" / "llms_full.py"
# Anything that looks like a snippet marker, however it was respaced.
_MARKER_LIKE = re.compile(r"^[\s>]*;*-+\s*8\s*<\s*-+", re.MULTILINE)


@cache
def _hook():
    spec = importlib.util.spec_from_file_location("llms_full", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def _config():
    return mkdocs_config.load_config(str(ROOT / "mkdocs.yml"))


@cache
def _full() -> str:
    config = _config()
    return _hook().build(
        docs_dir=ROOT / "docs",
        nav=config["nav"],
        site_url=config["site_url"],
        snippet_config=config["mdx_configs"]["pymdownx.snippets"],
        root=ROOT,
    )


def _llms_txt() -> str:
    return (ROOT / "docs" / "llms.txt").read_text(encoding="utf-8")


def _optional() -> set[str]:
    return _hook()._optional_urls(_llms_txt())


def _sources() -> list[str]:
    return re.findall(r"^Source: (\S+)$", _full(), re.MULTILINE)


def test_mkdocs_runs_the_hook():
    # mkdocs resolves each hook path against mkdocs.yml's directory.
    assert HOOK in [ROOT / path for path in _config()["hooks"]]


def test_it_opens_with_llms_txt_up_to_its_first_section():
    header = _llms_txt().split("\n## ", 1)[0].rstrip("\n")
    assert _full().startswith(header + "\n")
    assert _full().startswith("# mushin\n")


def test_every_page_in_nav_order_except_optional():
    optional = _optional()
    # Guards the section parsing: skipping nothing would also pass otherwise.
    assert _site() + "changelog/" in optional
    expected = [
        url
        for url in (_site_url(page) for page in _nav_pages(_config()["nav"]))
        if url not in optional
    ]
    assert _sources() == expected


def test_each_page_is_one_section_with_a_title():
    sections = _full().split("\n---\n\nSource: ")[1:]
    assert len(sections) == len(_sources())
    for section in sections:
        url, _, body = section.partition("\n\n")
        assert body.startswith("# "), f"{url} has no H1"


def test_no_include_or_api_directive_is_left_unresolved():
    assert not _MARKER_LIKE.findall(_full())
    assert not re.findall(r"^::: ", _full(), re.MULTILINE)


def test_the_guides_example_code_is_inlined():
    """The block-form includes pull code from examples/; check one per page."""
    pages = sorted((ROOT / "docs").glob("**/*.md"))
    included = set()
    for page in pages:
        if "superpowers" in page.parts:
            continue
        included |= set(
            re.findall(
                r"^--8<--\n(examples/[\w.]+\.py):(\w+)\n--8<--$",
                page.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )
    assert included
    for script, section in sorted(included):
        source = (ROOT / script).read_text(encoding="utf-8")
        body = source.split(f"# --8<-- [start:{section}]\n", 1)[1]
        first = next(line for line in body.splitlines() if line.strip())
        assert first.strip() in _full(), f"{script}:{section} not inlined"


def test_the_api_reference_is_rendered():
    names = []
    for page in sorted((ROOT / "docs" / "reference").glob("*.md")):
        names += re.findall(r"^::: (\S+)", page.read_text("utf-8"), re.MULTILINE)
    assert names
    missing = [
        name
        for name in names
        if not re.search(rf"^#+ `{re.escape(name)}`$", _full(), re.MULTILINE)
    ]
    assert not missing, missing
    # A signature, not only a heading.
    assert "```python\nshow(root, *," in _full()


def test_notebooks_bring_their_code_and_not_their_outputs():
    full = _full()
    assert "data:image/" not in full
    assert '"output_type"' not in full
    for path in sorted((ROOT / "docs" / "notebooks").glob("*.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        code = next(
            "".join(c["source"]).strip("\n")
            for c in cells
            if c["cell_type"] == "code" and "".join(c["source"]).strip()
        )
        assert f"```python\n{code}\n```" in full, path.name


def test_links_between_pages_are_site_urls():
    relative = re.findall(
        r"\]\((?![a-z][a-z0-9+.-]*:|#)([^)\s]+\.(?:md|ipynb))", _full()
    )
    assert not relative, relative
    # A rewritten link lands on a real page.
    assert "](https://martinez-hub.github.io/mushin/guides/exploration-to-paper/)" in (
        _full()
    )


def test_the_hook_writes_the_file(tmp_path):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    config = mkdocs_config.load_config(str(ROOT / "mkdocs.yml"), site_dir=str(site_dir))
    _hook().on_post_build(config)
    assert (site_dir / "llms-full.txt").read_text(encoding="utf-8") == _full()
