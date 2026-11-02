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
directive's `heading_level`, with its `members` and `show_bases` options; a
name griffe cannot find becomes a heading, so a reference page never breaks
the build. Relative links (to pages, images, in reference definitions)
become site URLs, since a relative `guides/mcp.md` means nothing in a single
file, and an mkdocs-autorefs cross-reference (`[text][mushin.name]`) a link
to the object's anchor; code fences are left as written. Notebooks contribute
their Markdown cells and their code cells as ```python fences; their
committed outputs (plots, tables, base64 images) stay out.
`tests/test_llms_full.py` checks all of this.
"""

from __future__ import annotations

import json
import posixpath
import re
import textwrap
from pathlib import Path
from urllib.parse import urlsplit

LLMS_TXT = "llms.txt"
OUTPUT = "llms-full.txt"

# A relative link target: no scheme, no leading slash or #, in an inline
# link or image `](target "title")` / `](<target>)`, or a reference definition
# `[label]: target`.
_RELATIVE = r"(?![a-z][a-z0-9+.-]*:|[/#])"
_INLINE_LINK = re.compile(
    rf"\]\((<?){_RELATIVE}([^)\s>#]+)(#[^)\s>]*)?(>?)((?:\s+\"[^\"]*\")?)\)"
)
_DEFINITION = re.compile(
    rf"^(\[[^\]]+\]:[ \t]*)(<?){_RELATIVE}([^\s>#]+)(#\S*?)?(>?)(?=\s|$)", re.M
)
# An mkdocs-autorefs cross-reference: `[text][identifier]` or `[identifier][]`.
_AUTOREF = re.compile(r"\[([^\]\n]+)\]\[([\w.]*)\]")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")


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
    local address, and the pages to leave out are the same."""
    found = re.search(r"^## Optional\n", llms_txt, re.M)
    if not found:
        return set()
    section = llms_txt[found.end() :].split("\n## ", 1)[0]
    return {urlsplit(u).path for u in re.findall(r"\]\(([^)\s]+)\)", section)}


# A mkdocstrings directive, `::: dotted.path`, and the indented options under it.
_DIRECTIVE = re.compile(
    r"^::: (\S+)[ \t]*\n(?:[ \t]+\S.*\n|[ \t]*\n(?=[ \t]+\S))*", re.M
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


def _private(name: str) -> bool:
    """mkdocstrings' default member filter (`!^_[^_]`): a single leading
    underscore is private; a documented dunder (`__init__`) is shown."""
    return re.match(r"_[^_]", name) is not None


def _api_entry(
    obj, heading: str, level: int, members=None, show_bases: bool = True
) -> str:
    obj = obj.final_target if obj.is_alias else obj
    signature = _signature(obj, show_bases)
    parts = [f"{'#' * level} `{heading}`", f"```python\n{signature}\n```"]
    if obj.docstring:
        parts.append(obj.docstring.value.strip("\n"))
    if obj.is_class or obj.is_module:
        for name, member in obj.members.items():
            if (members is not None and name not in members) or _private(name):
                continue
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
            if not member.docstring:
                continue
            if member.is_function or member.is_attribute or member.is_class:
                parts.append(_api_entry(member, f"{heading}.{name}", level + 1))
    return "\n\n".join(parts)


def _directive_options(block: str) -> dict:
    """The `options:` mapping under a directive, as mkdocstrings reads it."""
    import yaml

    loaded = yaml.safe_load(textwrap.dedent(block)) if block.strip() else None
    options = loaded.get("options") if isinstance(loaded, dict) else None
    return options if isinstance(options, dict) else {}


def _render_directives(text: str, package) -> str:
    """Each `::: name` directive as the object's entry, at the heading level
    its options give; a name outside the package, or one griffe cannot find,
    as a heading only, so a reference page the site renders never breaks the
    build."""

    def entry(match: re.Match) -> str:
        path = match[1]
        options = _directive_options(match[0].split("\n", 1)[1])
        level = options.get("heading_level", 2)
        level = level if isinstance(level, int) and 1 <= level <= 6 else 2
        members = options.get("members")
        members = set(members) if isinstance(members, list) else None
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
        return _api_entry(obj, path, level, members, show_bases) + "\n"

    return _DIRECTIVE.sub(entry, text)


def _api_anchors(nav: list, docs_dir: Path, site: str, package) -> dict[str, str]:
    """``{object path: URL#anchor}`` for every object a reference page
    documents, as mkdocstrings anchors it (the path is the id), so an
    autorefs cross-reference on any page can be linked."""
    anchors: dict[str, str] = {}
    for _, page in _nav_entries(nav):
        if not page.endswith(".md"):
            continue
        text = (docs_dir / page).read_text(encoding="utf-8-sig")
        for match in _DIRECTIVE.finditer(text):
            path = match[1]
            if not path.startswith(f"{package.name}."):
                continue
            try:
                obj = package[path.removeprefix(f"{package.name}.")]
            except Exception:  # noqa: BLE001 - not in the package
                continue
            url = _site_url(site, page)
            anchors[path] = f"{url}#{path}"
            obj = obj.final_target if obj.is_alias else obj
            for name in getattr(obj, "members", {}):
                if not _private(name):
                    anchors.setdefault(f"{path}.{name}", f"{url}#{path}.{name}")
    return anchors


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
    definitions; and an autorefs cross-reference to a documented object as a
    link to its anchor, or as its text when nothing documents it. Code fences
    are left as written. With `docs_dir`, a target that is not a file of the
    site (a path out of the docs) is left as written too."""
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

    def autoref(match: re.Match) -> str:
        text_, target = match[1], match[2] or match[1].strip("`")
        if "." not in target:
            return match[0]  # an ordinary reference link, not autorefs'
        url = anchors.get(target)
        return f"[{text_}]({url})" if url else text_

    out, fenced = [], None
    for line in text.split("\n"):
        fence = _FENCE.match(line)
        if fence and (fenced is None or fence[1] == fenced):
            fenced = None if fenced else fence[1]
            out.append(line)
            continue
        if fenced:
            out.append(line)
            continue
        line = _INLINE_LINK.sub(inline, line)
        line = _DEFINITION.sub(definition, line)
        line = _AUTOREF.sub(autoref, line)
        out.append(line)
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


def build(
    *, docs_dir: Path, nav: list, site_url: str, snippet_config: dict, root: Path
) -> str:
    """The text of llms-full.txt for the site described by these settings."""
    site = site_url.rstrip("/") + "/"
    llms_txt = (docs_dir / LLMS_TXT).read_text(encoding="utf-8")
    skip = _optional_paths(llms_txt)
    snippets = _snippet_preprocessor(snippet_config, root)
    import griffe

    package = griffe.load(
        "mushin",
        search_paths=[str(root / "src")],
        resolve_aliases=True,
        resolve_external=False,
    )
    anchors = _api_anchors(nav, docs_dir, site, package)

    header = llms_txt.split("\n## ", 1)[0].rstrip("\n")
    parts = [header]
    for title, page in _nav_entries(nav):
        url = _site_url(site, page)
        if urlsplit(url).path in skip:
            continue
        path = docs_dir / page
        if page.endswith(".ipynb"):
            text = _notebook_markdown(path)
        else:
            # Read as mkdocs reads a page: UTF-8, a byte-order mark dropped.
            lines = path.read_text(encoding="utf-8-sig").split("\n")
            text = "\n".join(snippets.run(lines))
            if _DIRECTIVE.search(text):
                text = _render_directives(text, package)
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
