# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Keep the docs' example includes in a form ruff cannot break.

The guides show tested code from ``examples/`` through pymdownx.snippets
includes. Inside a fenced Python block, the one-line form
``--8<-- "examples/x.py:run"`` is itself valid Python, so ``ruff format``
(which formats code blocks in Markdown) rewrites it to
``--8 < --"examples/x.py:run"``. Snippets no longer recognizes that,
``mkdocs build --strict`` does not flag it, and the page shows the directive
instead of the code. That happened to all 19 includes inside code fences when
ruff 0.16 arrived (#154), and went unnoticed for months.

Inside a fence, use the block form. A lone ``--8<--`` is not valid Python, so
ruff leaves the block alone:

    --8<--
    examples/x.py:run
    --8<--

The checks use snippets' own patterns, so "recognized" here means what
snippets itself recognizes. pymdown-extensions comes with the docs dependency
group; the CI docs job runs this file on every PR.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

snippets = pytest.importorskip("pymdownx.snippets")

_RE_MARKER = snippets.SnippetPreprocessor.RE_ALL_SNIPPETS
_RE_FILE = snippets.SnippetPreprocessor.RE_SNIPPET_FILE
_RE_SECTION = snippets.SnippetPreprocessor.RE_SNIPPET_SECTION
# Anything that looks like a snippet marker, however it was respaced, and
# also after a blockquote's `>`, where snippets does not look for one.
_MARKER_LIKE = re.compile(r"^[\s>]*;*-+\s*8\s*<\s*-+")
# A code fence line: a run of 3+ backticks or tildes, then an info string.
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*?)\r?$")

ROOT = Path(__file__).resolve().parent.parent
# docs/superpowers is excluded from the site (mkdocs.yml `exclude_docs`).
PAGES = sorted(
    p
    for p in (ROOT / "docs").rglob("*.md")
    if "superpowers" not in p.relative_to(ROOT / "docs").parts
)


def _scan(page: Path):
    """Yield (lineno, line, in_fence) for each line of `page`.

    The page is read as MkDocs reads it (dropping a UTF-8 BOM) and split on
    newlines, as Python-Markdown splits it before snippets runs. A fence closes
    only on a bare run of its own character that is at least as long as the one
    that opened it. A run of backticks followed by a backtick is a code span,
    not a fence (CommonMark).
    """
    fence = None
    text = page.read_text(encoding="utf-8-sig")
    for lineno, line in enumerate(text.split("\n"), 1):
        m = _FENCE.match(line)
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            run, info = m.groups()
            if fence is None:
                fence = run
                continue
            if run[0] == fence[0] and len(run) >= len(fence) and not info.strip():
                fence = None
                continue
        yield lineno, line, fence is not None


def _includes(page: Path):
    """Yield (lineno, spec) for each file `page` includes, in either form.

    This follows snippets' parser, which expands an include wherever it
    appears, fenced or not. An escaped marker is literal text, a spec starting
    with ``;`` is commented out in either form, a blank line inside a block is
    skipped, and so is an inline marker inside a block.
    """
    block = False
    for lineno, line, _ in _scan(page):
        m = _RE_MARKER.match(line)
        if m:
            if m.group("escape"):
                continue
            if m.group("block_marker"):
                block = not block
                continue
            if block:
                continue
            spec = m.group("snippet")[1:-1].strip()
        elif block:
            spec = line.strip()
            if not spec:
                continue
        else:
            continue
        if not spec.startswith(";"):
            yield lineno, spec


def _sections(path: Path) -> set[tuple[str, str]]:
    """The (type, name) section markers snippets would see in `path`."""
    found = set()
    for line in path.read_text(encoding="utf-8").split("\n"):
        m = _RE_SECTION.match(line)
        if m and not m.group("escape"):
            found.add((m.group("type"), m.group("name")))
    return found


def test_pages_were_found():
    assert len(PAGES) > 10, PAGES


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_snippet_markers_are_well_formed(page: Path):
    bad = []
    for lineno, line, in_fence in _scan(page):
        m = _RE_MARKER.match(line)
        if m:
            if m.group("inline_marker") and in_fence and not m.group("escape"):
                bad.append(f"{lineno}: {line!r}: use the block form inside a fence")
            continue
        # Snippets deletes any page line that contains an unescaped section
        # marker. That is harmless for a lone marker, but loses any other text.
        sec = _RE_SECTION.match(line)
        if sec:
            if not sec.group("escape") and (sec["pre"] + sec["post"]).strip(" \t>"):
                bad.append(
                    f"{lineno}: {line!r}: snippets deletes this whole line because"
                    " it contains a section marker; escape the marker with `;`"
                )
            continue
        if _MARKER_LIKE.match(line):
            bad.append(
                f"{lineno}: {line!r}: snippets does not recognize this marker"
                " (reformatted by ruff, inside a blockquote, or stray whitespace?)"
            )
    assert not bad, "\n".join(f"{page.relative_to(ROOT)}:{b}" for b in bad)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_includes_resolve(page: Path):
    missing = []
    for lineno, spec in _includes(page):
        path, _lines, section = _RE_FILE.match(spec).groups()
        path = path.strip()  # snippets strips the path before the suffix
        target = ROOT / path
        where = f"{page.relative_to(ROOT)}:{lineno}: {spec!r}"
        if not target.is_file():
            missing.append(f"{where}: no such file")
            continue
        if section:
            name = section[1:]
            have = _sections(target)
            for kind in ("start", "end"):
                if (kind, name) not in have:
                    missing.append(f"{where}: {path} has no `--8<-- [{kind}:{name}]`")
    assert not missing, "\n".join(missing)
