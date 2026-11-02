# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Write `llms-full.txt`: the whole docs site as one Markdown file.

`docs/llms.txt` is an index: an agent that wants the docs in its context still
has to fetch some forty pages one by one, and the guides' example code reaches
the site only through snippet includes, so the raw Markdown on GitHub shows a
directive where the code should be. `llms-full.txt` is the same site in one
fetch, with the includes resolved, the way the site renders them.

mkdocs runs this file as a hook (`hooks:` in `mkdocs.yml`) and writes the
result next to `llms.txt` in the built site, so every build regenerates it and
it cannot drift from the pages. It holds, in this order:

- `llms.txt` up to its first section (the H1, the summary, the facts agents
  must get right and the minimal sweep), so the file stands on its own;
- every nav page in nav order, each after a `Source:` line with its URL, except
  the pages `llms.txt` lists under `## Optional` (llmstxt.org: the section an
  agent may skip when context is short; today the contributing guide and the
  changelog, which is longer than every other page together).

A page's snippet includes are resolved with pymdownx.snippets itself, using the
options in `mkdocs.yml`, so they resolve exactly as on the site. An API page's
`::: mushin.name` directives become the object's signature and docstring, read
with griffe (mkdocstrings' own loader, so nothing is imported) and, for a
class or a module, its members that have a docstring and are not private
(mkdocstrings' default filter, so a documented `__init__` counts), at the
directive's `heading_level` and with its `show_bases`. Those are the only
options the reference pages use; the tests fail on any other, so a new one
gets rendered here before it can silently differ from the site. A name
griffe cannot find becomes a heading, so a reference page never breaks the
build. Relative links (to pages, images, in reference definitions) become
site URLs, since a relative `guides/mcp.md` means nothing in a single file,
and an mkdocs-autorefs cross-reference (`[text][mushin.name]`) by the path a
reference page documents the object under becomes a link to its anchor (not
by another of its paths, an alias, which the tests reject). Code (fences of
any length, inline code spans) is left as written. A page using a form this
does not read as the site does (an indented code block outside an admonition,
an escaped backtick, an unbalanced bracket, a fence superfences closes
differently: it needs exactly the opener's length and drops an unclosed one)
fails the tests rather than shipping a relative link. Notebooks contribute
their Markdown cells and their code cells as ```python fences; their
committed outputs (plots, tables, base64 images) stay out.
`tests/test_llms_full.py` checks all of this against the site mkdocs builds.
"""

from __future__ import annotations

import json
import math
import posixpath
import re
import textwrap
from pathlib import Path
from urllib.parse import urlsplit

LLMS_TXT = "llms.txt"
OUTPUT = "llms-full.txt"

# A relative link target: no scheme, no leading slash or #, in an inline
# link or image `](target "title")` / `](<target>)`, or a reference definition
# `[label]: target`, which may be indented (inside an admonition).
_RELATIVE = r"(?![a-z][a-z0-9+.-]*:|[/#])"
_TITLE = r"""(?:\s+(?:"[^"]*"|'[^']*'|\([^)]*\)))?"""
_INLINE_LINK = re.compile(
    rf"\]\((<?){_RELATIVE}([^)\s>#]+)(#[^)\s>]*)?(>?)({_TITLE})\)"
)
_DEFINITION = re.compile(
    rf"^([ \t]*\[[^\]]+\]:[ \t]*)(<?){_RELATIVE}([^\s>#]+)(#\S*?)?(>?)(?=\s|$)",
    re.M,
)
# An mkdocs-autorefs cross-reference, `[text][identifier]` or
# `[identifier][]`; its text may wrap onto the next line of the paragraph.
_AUTOREF = re.compile(r"\[((?:[^\[\]\n]|\n(?![ \t]*\n))+)\]\[([\w.]*)\]")
# A fence opens on three or more backticks or tildes and closes on a line of
# the same character, at least as many, and nothing else (CommonMark), so a
# four-backtick fence can show a three-backtick one.
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_FENCE_CLOSE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*$")
# An inline code span: a run of backticks to the next run of the same length,
# within one paragraph.
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)\1(?!`)", re.S)


def _nav_entries(node, title: str | None = None) -> list[tuple[str, str]]:
    """(title, page) for every local page under mkdocs' `nav`, in order."""
    if isinstance(node, str):
        return [] if "://" in node else [(title or node, node)]
    if isinstance(node, list):
        return [entry for item in node for entry in _nav_entries(item)]
    if isinstance(node, dict):
        return [e for key, value in node.items() for e in _nav_entries(value, key)]
    return []


def _site_url(site: str, page: str) -> str:
    """The URL mkdocs serves `page` at with directory URLs (the default)."""
    parent, _, name = page.rpartition("/")
    stem = name.rsplit(".", 1)[0]
    if stem in ("index", "README"):
        path = f"{parent}/" if parent else ""
    else:
        path = f"{parent}/{stem}/" if parent else f"{stem}/"
    return site + path


def _optional_paths(llms_txt: str) -> set[str]:
    """The site paths of the links in llms.txt's `## Optional` section, without
    their anchors. Paths, not URLs: `mkdocs serve` rewrites `site_url` to a
    local address, and the pages to leave out are the same. (An external link
    there whose path is a page's path would leave that page out too.)"""
    found = re.search(r"^## Optional\n", llms_txt, re.M)
    if not found:
        return set()
    section = llms_txt[found.end() :].split("\n## ", 1)[0]
    return {urlsplit(u).path for u in re.findall(r"\]\(([^)\s]+)\)", section)}


# A mkdocstrings directive, `::: dotted.path`, and its YAML, read as
# mkdocstrings reads it (its extension's `run`): the indented lines right
# under it, or, when there are none, the next block if that starts with
# `    options:` or `    handler:`. Any other indented block after a blank
# line is the page's own (a code block), not options.
_DIRECTIVE = re.compile(
    r"^::: ?(\S+) *(?:\n|\Z)"
    r"(?:(?:(?: {4}|\t)[ \t]*\S.*(?:\n|\Z))+"
    r"|[ \t]*\n(?=    (?:options|handler):)(?:[ \t]*\S.*(?:\n|\Z))+)?",
    re.M,
)


def _signature(obj, show_bases: bool = True) -> str:
    if obj.is_attribute:
        annotation = f": {obj.annotation}" if obj.annotation is not None else ""
        value = f" = {obj.value}" if obj.value is not None else ""
        return f"{obj.name}{annotation}{value}"
    if obj.is_module:
        return f"import {obj.path}"
    signature = obj.signature()
    if obj.is_class and not signature:
        # griffe gives no signature for a class whose `__init__` it does not
        # see (one inherited from an external base): name it and its bases.
        bases = ", ".join(str(base) for base in obj.bases) if show_bases else ""
        signature = f"{obj.name}({bases})" if bases else obj.name
    return ("class " if obj.is_class else "") + signature


# The members an entry shows, in the order mkdocstrings groups them.
_CATEGORIES = ("attribute", "class", "function")


def _private(name: str) -> bool:
    """mkdocstrings' default member filter (`!^_[^_]`): a single leading
    underscore is private; a documented dunder (`__init__`) is shown."""
    return re.match(r"_[^_]", name) is not None


def _api_entry(
    obj, heading: str, level: int, show_bases: bool, rendered: list[str]
) -> str:
    """``obj``'s entry and its members', each heading's path added to
    ``rendered`` (the anchors mkdocstrings gives them)."""
    obj = obj.final_target if obj.is_alias else obj
    rendered.append(heading)
    signature = _signature(obj, show_bases)
    parts = [f"{'#' * level} `{heading}`", f"```python\n{signature}\n```"]
    if obj.docstring:
        parts.append(obj.docstring.value.strip("\n"))
    shown = []
    if obj.is_class or obj.is_module:
        for name, member in obj.members.items():
            if _private(name):
                continue
            # mkdocstrings' `source` order: an import by the line it is on.
            line = member.alias_lineno if member.is_alias else member.lineno
            if member.is_alias:
                # An import: on a class, an inherited name; in a module, a
                # re-export, shown when it names the package's own object.
                if obj.is_class:
                    continue
                try:
                    member = member.final_target
                except Exception:  # noqa: BLE001 - an external target
                    continue
                if not member.path.startswith(f"{obj.package.name}."):
                    continue
            if not member.docstring or member.kind.value not in _CATEGORIES:
                continue
            category = _CATEGORIES.index(member.kind.value)
            shown.append((category, math.inf if line is None else line, name, member))
    # Grouped by category, each in source order, as mkdocstrings' default
    # `group_by_category` renders them (the sort is stable for equal lines).
    for _, _, name, member in sorted(shown, key=lambda m: m[:2]):
        # An option applies to the whole entry, as in mkdocstrings.
        entry = _api_entry(member, f"{heading}.{name}", level + 1, show_bases, rendered)
        parts.append(entry)
    return "\n\n".join(parts)


def _directive_options(block: str) -> dict:
    """The `options:` mapping under a directive, as mkdocstrings reads it."""
    import yaml

    loaded = yaml.safe_load(textwrap.dedent(block)) if block.strip() else None
    options = loaded.get("options") if isinstance(loaded, dict) else None
    return options if isinstance(options, dict) else {}


def _render_directives(text: str, package, rendered: list[str]) -> str:
    """Each `::: name` directive as the object's entry, at the heading level
    its options give, the paths of the entries added to ``rendered``; a name
    outside the package, or one griffe cannot find, as a heading only, so a
    reference page the site renders never breaks the build."""

    def entry(match: re.Match) -> str:
        path = match[1]
        block = match[0].split("\n", 1)[1] if "\n" in match[0] else ""
        options = _directive_options(block)
        level = options.get("heading_level", 2)
        level = level if isinstance(level, int) and 1 <= level <= 6 else 2
        show_bases = options.get("show_bases", True) is not False
        obj = None
        if path == package.name or path.startswith(f"{package.name}."):
            try:
                obj = (
                    package[path.removeprefix(package.name).lstrip(".")]
                    if (path != package.name)
                    else package
                )
            except Exception:  # noqa: BLE001 - not in the package
                obj = None
        if obj is None:
            return f"{'#' * level} `{path}`\n"
        return _api_entry(obj, path, level, show_bases, rendered) + "\n"

    # mkdocstrings never sees a fence's text, so an example stays one.
    return _outside_fences(text, lambda run: _DIRECTIVE.sub(entry, run))


def _notebook_markdown(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    parts = []
    for cell in cells:
        source = "".join(cell["source"]).strip("\n")
        if not source:
            continue
        if cell["cell_type"] == "markdown":
            parts.append(source)
        elif cell["cell_type"] == "code":
            parts.append(f"```python\n{source}\n```")
    return "\n\n".join(parts)


def _absolute_links(
    text: str,
    page: str,
    site: str,
    docs_dir: Path | None = None,
    anchors: dict[str, str] | None = None,
) -> str:
    """Rewrite `page`'s relative links as the site URLs they serve at: a page
    as its URL, any other file (an image) at its path in the site, in inline
    links and images (with a title or in angle brackets) and in reference
    definitions; and an autorefs cross-reference to an object in `anchors`
    as a link to its anchor. Anything else, and all code (fences, inline
    spans), is left as written. With `docs_dir`, a target that is not a file
    of the site (a path out of the docs) is left as written too."""
    page_dir = posixpath.dirname(page)
    anchors = anchors or {}

    def url_of(target: str) -> str | None:
        path = posixpath.normpath(posixpath.join(page_dir, target))
        if path.startswith("../") or path == "..":
            return None
        if path.endswith((".md", ".ipynb")):
            return _site_url(site, path)
        if docs_dir is not None and not (docs_dir / path).is_file():
            return None
        return site + path

    def inline(match: re.Match) -> str:
        url = url_of(match[2])
        if url is None:
            return match[0]
        return f"]({url}{match[3] or ''}{match[5]})"

    def definition(match: re.Match) -> str:
        url = url_of(match[3])
        return match[0] if url is None else f"{match[1]}{url}{match[4] or ''}"

    def rewrite(prose: str) -> str:
        # Code spans are set aside first, so nothing in them is rewritten and
        # a link's text may hold one (`[`run`][mushin.run]`); the whole run
        # of lines is rewritten at once, so a link may wrap.
        spans: list[str] = []

        def hide(match: re.Match) -> str:
            spans.append(match[0])
            return f"\x00{len(spans) - 1}\x00"

        def show(chunk: str) -> str:
            return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m[1])], chunk)

        def autoref(match: re.Match) -> str:
            target = match[2] or show(match[1]).strip("`")
            # An object path has a dot; an undotted label (`[mushin][]`) is an
            # ordinary reference link, which may have its own definition.
            url = anchors.get(target) if "." in target else None
            return f"[{match[1]}]({url})" if url else match[0]

        prose = _CODE_SPAN.sub(hide, prose)
        prose = _INLINE_LINK.sub(inline, prose)
        prose = _DEFINITION.sub(definition, prose)
        prose = _AUTOREF.sub(autoref, prose)
        return show(prose)

    return _outside_fences(text, rewrite)


def _outside_fences(text: str, rewrite) -> str:
    """``text`` with ``rewrite`` applied to each run of lines outside a code
    fence, the fences as written: a page that shows a link or a `:::`
    directive as an example keeps it."""
    out: list[str] = []
    prose: list[str] = []
    fence = None
    for line in text.split("\n"):
        if fence is None:
            opened = _FENCE.match(line)
            if opened is None:
                prose.append(line)
                continue
            if prose:
                out.append(rewrite("\n".join(prose)))
                prose = []
            fence = opened[1]
        else:
            closed = _FENCE_CLOSE.match(line)
            if closed and closed[1][0] == fence[0] and len(closed[1]) >= len(fence):
                fence = None
        out.append(line)
    if prose:
        out.append(rewrite("\n".join(prose)))
    return "\n".join(out)


def _snippet_preprocessor(snippet_config: dict, root: Path):
    import markdown

    config = dict(snippet_config)
    # mkdocs passes base_path through as written; anchor it at the repository
    # so the result does not depend on the directory the build ran from.
    config["base_path"] = [str(root / p) for p in config.get("base_path", ["."])]
    md = markdown.Markdown(
        extensions=["pymdownx.snippets"],
        extension_configs={"pymdownx.snippets": config},
    )
    return md.preprocessors["snippet"]


def read_pages(
    *, docs_dir: Path, nav: list, site: str, snippet_config: dict, root: Path
) -> tuple[list[tuple[str, str, str, str, list[str]]], dict[str, str]]:
    """Every nav page as ``(title, page, url, text, entries)``, its snippet
    includes resolved and its directives rendered (``entries`` holds the
    object paths they rendered), its links not yet rewritten; and
    ``{object path: URL#anchor}`` for every entry. All pages first, so a
    cross-reference to an object any reference page documents can be
    linked, and only to an entry rendered, so no anchor points at a member
    the site does not show."""
    snippets = _snippet_preprocessor(snippet_config, root)
    import griffe

    package = griffe.load(
        "mushin",
        search_paths=[str(root / "src")],
        resolve_aliases=True,
        resolve_external=False,
    )
    pages, anchors = [], {}
    for title, page in _nav_entries(nav):
        url = _site_url(site, page)
        path = docs_dir / page
        entries: list[str] = []
        if page.endswith(".ipynb"):
            text = _notebook_markdown(path)
        else:
            # Read as mkdocs reads a page: UTF-8, a byte-order mark dropped.
            lines = path.read_text(encoding="utf-8-sig").split("\n")
            text = "\n".join(snippets.run(lines))
            if _DIRECTIVE.search(text):
                text = _render_directives(text, package, entries)
                for name in entries:
                    anchors.setdefault(name, f"{url}#{name}")
        pages.append((title, page, url, text, entries))
    return pages, anchors


def build(
    *, docs_dir: Path, nav: list, site_url: str, snippet_config: dict, root: Path
) -> str:
    """The text of llms-full.txt for the site described by these settings."""
    site = site_url.rstrip("/") + "/"
    llms_txt = (docs_dir / LLMS_TXT).read_text(encoding="utf-8")
    skip = _optional_paths(llms_txt)
    pages, anchors = read_pages(
        docs_dir=docs_dir,
        nav=nav,
        site=site,
        snippet_config=snippet_config,
        root=root,
    )
    header = llms_txt.split("\n## ", 1)[0].rstrip("\n")
    parts = [header]
    for title, page, url, text, _ in pages:
        if urlsplit(url).path in skip:
            continue
        text = _absolute_links(text, page, site, docs_dir, anchors).strip("\n")
        if not text.startswith("# "):
            text = f"# {title}\n\n{text}"
        parts.append(f"---\n\nSource: {url}\n\n{text}")
    return "\n\n".join(parts) + "\n"


def on_post_build(config, **kwargs) -> None:
    """mkdocs hook: write llms-full.txt into the built site."""
    root = Path(config["config_file_path"]).resolve().parent
    text = build(
        docs_dir=Path(config["docs_dir"]),
        nav=config["nav"],
        site_url=config["site_url"],
        snippet_config=config["mdx_configs"].get("pymdownx.snippets", {}),
        root=root,
    )
    (Path(config["site_dir"]) / OUTPUT).write_text(text, encoding="utf-8")
