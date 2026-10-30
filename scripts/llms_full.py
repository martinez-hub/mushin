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
class, its own public members that have a docstring, as mkdocstrings shows
them by default. Links to other
pages become site URLs, since a relative `guides/mcp.md` means nothing in a
single file. Notebooks contribute their Markdown cells and their code cells as
```python fences; their committed outputs (plots, tables, base64 images) stay
out. `tests/test_llms_full.py` checks all of this.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path

LLMS_TXT = "llms.txt"
OUTPUT = "llms-full.txt"

# `[text](target)` where the target is a page of the site: no scheme, no
# leading slash, ending in .md or .ipynb, with an optional #anchor.
_PAGE_LINK = re.compile(
    r"\]\((?![a-z][a-z0-9+.-]*:|/)([^)\s#]+\.(?:md|ipynb))(#[^)\s]*)?\)"
)


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


def _optional_urls(llms_txt: str) -> set[str]:
    """The link targets in llms.txt's `## Optional` section."""
    _, found, rest = llms_txt.partition("\n## Optional\n")
    if not found:
        return set()
    section = rest.split("\n## ", 1)[0]
    return set(re.findall(r"\]\(([^)\s]+)\)", section))


# A mkdocstrings directive, `::: dotted.path`, and the indented options under it.
_DIRECTIVE = re.compile(
    r"^::: (\S+)[ \t]*\n(?:[ \t]+\S.*\n|[ \t]*\n(?=[ \t]+\S))*", re.M
)


def _signature(obj) -> str:
    if obj.is_attribute:
        annotation = f": {obj.annotation}" if obj.annotation is not None else ""
        value = f" = {obj.value}" if obj.value is not None else ""
        return f"{obj.name}{annotation}{value}"
    prefix = "class " if obj.is_class else ""
    return prefix + obj.signature()


def _api_entry(obj, heading: str, level: int) -> str:
    obj = obj.final_target if obj.is_alias else obj
    parts = [f"{'#' * level} `{heading}`", f"```python\n{_signature(obj)}\n```"]
    if obj.docstring:
        parts.append(obj.docstring.value.strip("\n"))
    if obj.is_class:
        for name, member in obj.members.items():
            if name.startswith("_") or member.is_alias or not member.docstring:
                continue
            if member.is_function or member.is_attribute:
                parts.append(_api_entry(member, f"{heading}.{name}", level + 1))
    return "\n\n".join(parts)


def _render_directives(text: str, package) -> str:
    def entry(match: re.Match) -> str:
        path = match[1]
        obj = package[path.removeprefix(f"{package.name}.")]
        return _api_entry(obj, path, 2) + "\n"

    return _DIRECTIVE.sub(entry, text)


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


def _absolute_links(text: str, page: str, site: str) -> str:
    """Rewrite `page`'s links to other pages as the site URLs they serve at."""
    page_dir = posixpath.dirname(page)

    def absolute(match: re.Match) -> str:
        target = posixpath.normpath(posixpath.join(page_dir, match[1]))
        return f"]({_site_url(site, target)}{match[2] or ''})"

    return _PAGE_LINK.sub(absolute, text)


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
    skip = _optional_urls(llms_txt)
    snippets = _snippet_preprocessor(snippet_config, root)
    package = None

    header = llms_txt.split("\n## ", 1)[0].rstrip("\n")
    parts = [header]
    for title, page in _nav_entries(nav):
        url = _site_url(site, page)
        if url in skip:
            continue
        path = docs_dir / page
        if page.endswith(".ipynb"):
            text = _notebook_markdown(path)
        else:
            # Read as mkdocs reads a page: UTF-8, a byte-order mark dropped.
            lines = path.read_text(encoding="utf-8-sig").split("\n")
            text = "\n".join(snippets.run(lines))
            if _DIRECTIVE.search(text):
                if package is None:
                    import griffe

                    package = griffe.load(
                        "mushin",
                        search_paths=[str(root / "src")],
                        resolve_aliases=True,
                        resolve_external=False,
                    )
                text = _render_directives(text, package)
        text = _absolute_links(text, page, site).strip("\n")
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
