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
from html.parser import HTMLParser
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


@cache
def _pages():
    """The hook's pages before their links are rewritten, and its anchors."""
    config = _config()
    return _hook().read_pages(
        docs_dir=ROOT / "docs",
        nav=config["nav"],
        site=_site(),
        snippet_config=config["mdx_configs"]["pymdownx.snippets"],
        root=ROOT,
    )


class _Html(HTMLParser):
    """The link targets of rendered HTML, its code text, and its other text."""

    def __init__(self):
        super().__init__()
        self.urls: list[str] = []
        self.code: list[str] = []
        self.text: list[str] = []
        self._in_code = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.urls.append(attrs["href"])
        if tag == "img" and attrs.get("src"):
            self.urls.append(attrs["src"])
        if tag in ("code", "pre"):
            self._in_code += 1

    def handle_endtag(self, tag):
        if tag in ("code", "pre"):
            self._in_code -= 1

    def handle_data(self, data):
        (self.code if self._in_code else self.text).append(data)


def _render(text: str) -> _Html:
    """``text`` as the site's Markdown renders it: the oracle for what is a
    link and what is code, rather than a regex of the tests' own."""
    import markdown

    config = _config()
    # The snippet includes are resolved already; a literal marker in a code
    # block must stay one.
    extensions = [e for e in config["markdown_extensions"] if e != "pymdownx.snippets"]
    html = markdown.markdown(
        text,
        extensions=extensions,
        extension_configs={
            k: v for k, v in config["mdx_configs"].items() if k in extensions
        },
    )
    parser = _Html()
    parser.feed(html)
    return parser


@cache
def _rendered_full() -> _Html:
    return _render(_full())


@pytest.fixture(scope="module")
def built_site(tmp_path_factory):
    """The site as `mkdocs build` writes it, hook included."""
    from mkdocs.commands.build import build

    site_dir = tmp_path_factory.mktemp("site")
    # From the repository, as `mkdocs build` runs: pymdownx.snippets resolves
    # its base_path against the working directory, not mkdocs.yml.
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(ROOT)
        config = mkdocs_config.load_config(
            str(ROOT / "mkdocs.yml"), site_dir=str(site_dir)
        )
        build(config)
    return site_dir


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
    # Outside code: a page may show a directive as an example.
    assert not re.search(r"(?:^|\n)::: \S", "".join(_rendered_full().text))


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


def _signature_of(name: str) -> str:
    return re.match(r"\s*```python\n(.*?)\n```", _api_section(name), re.S)[1]


def test_show_bases_false_is_kept():
    """lightning.md hides HydraDDP's bases (a private mixin and Lightning's);
    a class without the option shows its bases, as on the site."""
    assert _signature_of("mushin.lightning.HydraDDP") == "class HydraDDP"
    assert (
        _signature_of("mushin.benchmark.IncompleteSweepError")
        == "class IncompleteSweepError(RuntimeError)"
    )


def test_show_bases_applies_to_the_whole_entry():
    """mkdocstrings applies an option to the object and its members, so a
    nested class hides its bases too."""
    import griffe

    module = griffe.visit(
        "m",
        filepath=ROOT / "m.py",
        code='class A(Exception):\n    """A."""\n\n    class C(KeyError):\n        """C."""\n',
    )
    hidden = _hook()._api_entry(module["A"], "m.A", 2, False, [])
    assert "```python\nclass A\n```" in hidden and "```python\nclass C\n```" in hidden
    shown = _hook()._api_entry(module["A"], "m.A", 2, True, [])
    assert "class A(Exception)" in shown and "class C(KeyError)" in shown


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
        "::: mushin.export\n\n::: hydra_zen.zen\n\n::: mushin.does_not_exist\n",
        _package(),
        [],
    )
    assert "## `mushin.export`" in text and "table(" in text
    assert "## `hydra_zen.zen`" in text
    assert "## `mushin.does_not_exist`" in text


def test_a_directive_shown_in_a_fence_is_left_as_written():
    hook = _hook()
    example = (
        "```markdown\n::: mushin.export\n    options:\n      heading_level: 3\n```\n"
    )
    entries: list[str] = []
    assert hook._render_directives(example, _package(), entries) == example
    assert entries == []


def test_an_indented_block_after_a_directive_is_the_pages_own():
    """mkdocstrings reads options from the lines right under the directive, or
    from a next block that starts with `options:`; any other indented block
    is the page's code, which YAML could not even parse."""
    hook = _hook()
    code = "    @functools.cache\n    def f(): ...\n"
    text = hook._render_directives(f"::: mushin.export\n\n{code}", _package(), [])
    assert code in text and "## `mushin.export`" in text
    separated = "::: mushin.export\n\n    options:\n      heading_level: 3\n"
    assert "### `mushin.export`" in hook._render_directives(separated, _package(), [])


def test_the_hook_renders_every_mkdocstrings_option_the_docs_use():
    """The hook follows only the options the site uses. Another one (members,
    filters, inherited members) would make the file differ from the site in
    silence, so it fails here until the hook renders it."""
    hook = _hook()
    used = set()
    for page in sorted((ROOT / "docs").rglob("*.md")):
        if "superpowers" in page.parts:
            continue
        for match in hook._DIRECTIVE.finditer(page.read_text(encoding="utf-8")):
            block = match[0].split("\n", 1)[1] if "\n" in match[0] else ""
            used |= set(hook._directive_options(block))
    assert used, "found no directive options: the parsing changed"
    assert used <= {"heading_level", "show_bases"}, used
    plugin = _config()["plugins"]["mkdocstrings"].config
    options = plugin["handlers"]["python"]["options"]
    # These change nothing in the file, or are what the hook assumes.
    known = {
        "docstring_style",
        "show_source",
        "show_root_heading",
        "show_root_full_path",
        "members_order",
        "separate_signature",
        "heading_level",
    }
    assert set(options) <= known, set(options) - known
    assert options.get("members_order") == "source"  # griffe's order
    assert options.get("heading_level", 2) == 2
    assert options.get("show_root_heading") is True


def test_the_api_entries_match_the_built_site(built_site):
    """Each reference page's objects, in order, are the ones mkdocstrings
    renders: members, filters, order and re-exports checked by outcome."""
    pages, _ = _pages()
    checked = 0
    # Every page, so an object the site renders where the hook rendered none
    # (a directive form it does not read) fails too.
    for _, _, url, _, entries in pages:
        html = (built_site / url.removeprefix(_site()) / "index.html").read_text(
            encoding="utf-8"
        )
        assert re.findall(r'<h[1-6] id="(mushin\.[^"]+)"', html) == entries, url
        checked += bool(entries)
    assert checked >= 8


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


def test_every_link_in_the_file_is_absolute():
    """Every link and image the file renders, in any Markdown form, is a URL
    or an anchor: a relative one means nothing outside the site."""
    urls = _rendered_full().urls
    relative = sorted({u for u in urls if not re.match(r"[a-z][a-z0-9+.-]*:|#", u)})
    assert not relative, relative
    assert _site() + "guides/exploration-to-paper/" in urls


def test_every_site_url_in_the_file_is_served(built_site):
    """Every link into the site resolves in the site `mkdocs build` writes: a
    page, or a file beside them (llms.txt, a notebook's download, an image).
    Not a page's source: the build serves no .md, and a notebook's download
    is at notebooks/X/X.ipynb."""
    site = _site()

    def served(url: str) -> bool:
        path = built_site / url.removeprefix(site)
        return path.is_file() or (path / "index.html").is_file()

    urls = {u.split("#", 1)[0] for u in _rendered_full().urls if u.startswith(site)}
    assert site + "guides/exploration-to-paper/" in urls
    bad = sorted(url for url in urls if not served(url))
    assert not bad, bad


def test_code_on_every_page_is_left_as_written():
    """The rewrite changes links only: every page's code, as the site's
    Markdown renders it, is the same before and after."""
    pages, anchors = _pages()
    hook = _hook()
    for _, page, _, text, _ in pages:
        after = hook._absolute_links(text, page, _site(), ROOT / "docs", anchors)
        assert _render(after).code == _render(text).code, page


def test_links_in_code_are_left_as_written():
    text = (
        "See [a](mcp.md), `[b](mcp.md)` and `x[0][mushin.run]`.\n\n"
        "```markdown\n[c](mcp.md)\n```\n\n"
        "````markdown\n```python\n[d](mcp.md)\n```\n[e](mcp.md)\n````\n\n"
        "[f](mcp.md) after.\n"
    )
    anchors = {"mushin.run": "https://e.org/api/#mushin.run"}
    out = _hook()._absolute_links(text, "guides/x.md", "https://e.org/", None, anchors)
    assert "[a](https://e.org/guides/mcp/)" in out
    for kept in ("`[b](mcp.md)`", "`x[0][mushin.run]`", "[c](mcp.md)"):
        assert kept in out
    # A four-backtick fence holds a three-backtick one.
    assert "[d](mcp.md)" in out and "[e](mcp.md)" in out
    assert "[f](https://e.org/guides/mcp/) after." in out


def test_every_link_form_becomes_a_site_url():
    text = (
        '[a](mcp.md "Title") [b](<mcp.md>) ![logo](../assets/logo.png)\n'
        "[g](mcp.md 'Single') [h](mcp.md (Paren))\n\n"
        "[c]: mcp.md#top\n\n"
        "!!! note\n    [i]: mcp.md\n"
    )
    out = _hook()._absolute_links(text, "guides/x.md", "https://e.org/")
    assert '[a](https://e.org/guides/mcp/ "Title")' in out
    assert "[b](https://e.org/guides/mcp/)" in out
    assert "![logo](https://e.org/assets/logo.png)" in out
    assert "[g](https://e.org/guides/mcp/ 'Single')" in out
    assert "[h](https://e.org/guides/mcp/ (Paren))" in out
    assert "[c]: https://e.org/guides/mcp/#top" in out
    assert "    [i]: https://e.org/guides/mcp/" in out


def test_only_a_reference_to_a_documented_object_is_rewritten():
    """An autorefs cross-reference to an entry becomes a link to its anchor,
    a member's too; any other bracket pair is left exactly as written."""
    anchors = {
        "mushin.A": "https://e.org/r/#mushin.A",
        "mushin.A.m": "https://e.org/r/#mushin.A.m",
    }
    text = "[x][mushin.A.m], [`mushin.A`][], [y][mushin.gone], ds[0][a.b], [z][l]\n"
    out = _hook()._absolute_links(text, "x.md", "https://e.org/", None, anchors)
    assert out == (
        "[x](https://e.org/r/#mushin.A.m), [`mushin.A`](https://e.org/r/#mushin.A), "
        "[y][mushin.gone], ds[0][a.b], [z][l]\n"
    )
    # An undotted label is an ordinary reference link with its own definition.
    own = "[mushin][]\n\n[mushin]: https://pypi.org/project/mushin-py/\n"
    pkg = {"mushin": "https://e.org/r/#mushin"}
    assert _hook()._absolute_links(own, "x.md", "https://e.org/", None, pkg) == own


def test_a_wrapped_cross_reference_is_linked():
    anchors = {"mushin.A": "https://e.org/r/#mushin.A"}
    text = "Use [the helper\n`mushin.A`][mushin.A] here.\n\n[not\n\nthis][mushin.A]\n"
    out = _hook()._absolute_links(text, "x.md", "https://e.org/", None, anchors)
    assert "Use [the helper\n`mushin.A`](https://e.org/r/#mushin.A) here." in out
    assert "[not\n\nthis][mushin.A]" in out  # a blank line ends a paragraph


def test_api_cross_references_become_links():
    """mkdocs-autorefs writes ``[text][mushin.name]``; in one file it needs
    the URL of the page that documents the name."""
    link = f"[`mushin.original_cwd()`]({_site()}reference/utils/#mushin.original_cwd)"
    assert link in _full()
    text = "".join(_rendered_full().text)
    assert not re.search(r"\]\[mushin\.[\w.]+\]", text)
    # The collapsed form, `[`mushin.x`][]`, leaves its `][]` outside the code.
    assert "][]" not in text


def test_the_hook_writes_the_file(built_site):
    assert (built_site / "llms-full.txt").read_text(encoding="utf-8") == _full()
