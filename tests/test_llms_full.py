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
import textwrap
from functools import cache
from urllib.parse import urlsplit

import pytest

pytest.importorskip("pymdownx.snippets")
pytest.importorskip("griffe")
mkdocs_config = pytest.importorskip("mkdocs.config")

from _docs_site import ROOT, _nav_pages, _site, _site_url, markdown_links  # noqa: E402

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
def _full(site_url: str | None = None) -> str:
    config = _config()
    return _hook().build(
        docs_dir=ROOT / "docs",
        nav=config["nav"],
        site_url=site_url or config["site_url"],
        snippet_config=config["mdx_configs"]["pymdownx.snippets"],
        root=ROOT,
    )


@cache
def _package():
    import griffe

    return griffe.load(
        "mushin",
        search_paths=[str(ROOT / "src")],
        resolve_aliases=True,
        resolve_external=False,
    )


def _directives() -> list[tuple[str, int]]:
    """``(name, heading level)`` for every directive on the reference pages."""
    found = []
    for page in sorted((ROOT / "docs" / "reference").glob("*.md")):
        text = page.read_text("utf-8")
        for match in re.finditer(r"^::: (\S+)\n((?:[ \t]+\S.*\n)*)", text, re.M):
            level = re.search(r"heading_level:\s*(\d+)", match[2])
            found.append((match[1], int(level[1]) if level else 2))
    return found


def _api_section(name: str) -> str:
    """The text from ``name``'s heading to the next heading of any level."""
    match = re.search(
        rf"^#+ `{re.escape(name)}`\n(.*?)(?=^#+ |\Z)", _full(), re.M | re.S
    )
    assert match, f"no heading for {name}"
    return match[1]


def _llms_txt() -> str:
    return (ROOT / "docs" / "llms.txt").read_text(encoding="utf-8")


def _sources() -> list[str]:
    return re.findall(r"^Source: (\S+)$", _full(), re.MULTILINE)


def test_mkdocs_runs_the_hook():
    # mkdocs resolves each hook path against mkdocs.yml's directory.
    assert HOOK in [ROOT / path for path in _config()["hooks"]]


def test_it_opens_with_llms_txt_up_to_its_first_section():
    header = _llms_txt().split("\n## ", 1)[0].rstrip("\n")
    assert _full().startswith(header + "\n")
    assert _full().startswith("# mushin\n")


def _optional_paths() -> set[str]:
    """The site paths of llms.txt's Optional links, worked out here rather
    than by the hook, without their anchors."""
    section = _llms_txt().partition("\n## Optional\n")[2].split("\n## ", 1)[0]
    return {urlsplit(link).path for link in markdown_links(section)}


def test_every_page_in_nav_order_except_optional():
    optional = _optional_paths()
    # Guards the section parsing: skipping nothing would also pass otherwise.
    assert urlsplit(_site() + "changelog/").path in optional
    expected = [
        url
        for url in (_site_url(page) for page in _nav_pages(_config()["nav"]))
        if urlsplit(url).path not in optional
    ]
    assert _sources() == expected


def test_an_anchored_optional_link_still_leaves_its_page_out():
    hook = _hook()
    text = "## Optional\n\n- [Contributing](https://example.org/mushin/contributing/#setup)\n"
    assert hook._optional_paths(text) == {"/mushin/contributing/"}


def test_mkdocs_serve_writes_the_same_pages():
    """`mkdocs serve` rewrites site_url to a local address: the Optional pages
    are still left out, and the pages are the same."""
    served = _full("http://127.0.0.1:8000/mushin/")
    paths = [urlsplit(url).path for url in re.findall(r"^Source: (\S+)$", served, re.M)]
    assert paths == [urlsplit(url).path for url in _sources()]


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
        body = body.split(f"# --8<-- [end:{section}]", 1)[0]
        # The whole section, not a first line several sections share.
        assert textwrap.dedent(body).strip("\n") in _full(), f"{script}:{section}"


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


def test_each_api_entry_has_its_signature_and_docstring():
    package = _package()
    for name, _ in _directives():
        section = _api_section(name)
        fence = re.match(r"\s*```python\n(.*?)\n```", section, re.S)
        assert fence, f"{name} has no signature fence"
        short = name.rsplit(".", 1)[1]
        assert short in fence[1], f"{name}'s signature names nothing: {fence[1]!r}"
        obj = package[name.removeprefix("mushin.")]
        obj = obj.final_target if obj.is_alias else obj
        if obj.is_class:
            assert fence[1].startswith(f"class {short}"), fence[1]
        if obj.docstring:
            first = obj.docstring.value.strip().splitlines()[0]
            assert first in section, f"{name}'s docstring is missing"


def test_a_class_brings_its_documented_members_and_dunders():
    full = _full()
    assert "`mushin.workflows.MultiRunMetricsWorkflow.to_xarray`" in full
    # mkdocstrings shows a documented dunder by default; a private name not.
    assert "`mushin.workflows.BaseWorkflow.__init__`" in full
    assert not re.search(r"^#+ `mushin\.[\w.]*\._[^_][\w]*`$", full, re.M)


def test_show_bases_false_is_kept():
    """lightning.md hides HydraDDP's bases (a private mixin and Lightning's)."""
    fence = re.match(
        r"\s*```python\n(.*?)\n```", _api_section("mushin.lightning.HydraDDP"), re.S
    )
    assert fence[1] == "class HydraDDP"


def test_directive_heading_levels_are_kept():
    for name, level in _directives():
        assert re.search(rf"^{'#' * level} `{re.escape(name)}`$", _full(), re.M), (
            f"{name} should be at heading level {level}"
        )


def test_no_directive_option_reaches_the_file():
    assert not re.search(r"^[ \t]+(options|heading_level|show_bases):", _full(), re.M)


def test_a_module_or_external_directive_does_not_break_the_build():
    hook = _hook()
    text = hook._render_directives(
        "::: mushin.export\n\n::: hydra_zen.zen\n", _package()
    )
    assert "## `mushin.export`" in text and "table(" in text
    assert "## `hydra_zen.zen`" in text


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
        prose = next(
            "".join(c["source"]).strip("\n")
            for c in cells
            if c["cell_type"] == "markdown" and "".join(c["source"]).strip()
        )
        assert prose in full, f"{path.name}'s Markdown cells are missing"


def test_links_between_pages_are_site_urls():
    relative = re.findall(
        r"\]\((?![a-z][a-z0-9+.-]*:|#)<?([^)\s>]+\.(?:md|ipynb))", _full()
    )
    assert not relative, relative
    definitions = re.findall(
        r"^\[[^\]]+\]:\s*<?(?![a-z][a-z0-9+.-]*:|#)(\S+\.(?:md|ipynb))", _full(), re.M
    )
    assert not definitions, definitions
    # A rewritten link lands on a real page.
    assert "](https://martinez-hub.github.io/mushin/guides/exploration-to-paper/)" in (
        _full()
    )


def test_every_site_url_in_the_file_is_a_page():
    """Every link into the site resolves: a page, its anchors, or the files
    the build writes beside them (llms.txt, a notebook's download)."""
    site = _site()
    pages = {_site_url(page) for page in _nav_pages(_config()["nav"])}
    allowed = pages | {site + "llms.txt", site + "llms-full.txt"}
    found = re.findall(rf"{re.escape(site)}[^)\s>\]]*", _full())
    bad = sorted(
        {
            url
            for url in (u.split("#", 1)[0] for u in found)
            if url not in allowed and not re.search(r"/([\w-]+)/\1\.ipynb$", url)
        }
    )
    assert not bad, bad


def test_links_in_code_are_left_as_written():
    text = "See [a](mcp.md).\n\n```markdown\n[b](mcp.md)\n```\n"
    out = _hook()._absolute_links(text, "guides/x.md", "https://e.org/")
    assert "[a](https://e.org/guides/mcp/)" in out
    assert "[b](mcp.md)" in out


def test_every_link_form_becomes_a_site_url():
    text = (
        '[a](mcp.md "Title") [b](<mcp.md>) ![logo](../assets/logo.png)\n\n'
        "[c]: mcp.md#top\n"
    )
    out = _hook()._absolute_links(text, "guides/x.md", "https://e.org/")
    assert '[a](https://e.org/guides/mcp/ "Title")' in out
    assert "[b](https://e.org/guides/mcp/)" in out
    assert "![logo](https://e.org/assets/logo.png)" in out
    assert "[c]: https://e.org/guides/mcp/#top" in out


def test_api_cross_references_become_links():
    """mkdocs-autorefs writes ``[text][mushin.name]``; in one file it needs
    the URL of the page that documents the name."""
    assert not re.search(r"\]\[mushin\.[\w.]+\]", _full())


def test_the_hook_writes_the_file(tmp_path):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    config = mkdocs_config.load_config(str(ROOT / "mkdocs.yml"), site_dir=str(site_dir))
    _hook().on_post_build(config)
    assert (site_dir / "llms-full.txt").read_text(encoding="utf-8") == _full()
