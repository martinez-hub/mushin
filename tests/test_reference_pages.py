# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Every public name has an API reference entry (docs/reference/*.md).

The reference pages render docstrings through mkdocstrings `:::` directives.
A name added to a public namespace without one is how the backlog that #267
cleared accrued, so the invariant is pinned here, and the docs CI job runs it
on every PR."""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

from _docs_site import ROOT, _mkdocs, _nav_pages

# Documented in prose on the workflows page rather than through a directive:
# the two hydra-zen re-exports, whose own docs the page links to.
PROSE_ONLY = {"multirun", "hydra_list"}
# Subpackages exposed as names, not objects to document.
MODULES = {"export", "llm"}


def _reference_pages() -> list[Path]:
    # Pages in the nav only: a stray page nobody can reach documents nothing.
    pages = [
        ROOT / "docs" / p
        for p in _nav_pages(_mkdocs()["nav"])
        if p.startswith("reference/")
    ]
    assert pages, "no reference pages in the nav"
    return pages


def _directives() -> list[str]:
    paths = []
    for page in _reference_pages():
        paths += re.findall(r"^::: (\S+)$", page.read_text(encoding="utf-8"), re.M)
    return paths


def _resolve(dotted: str):
    """The object a directive renders: the longest importable module prefix,
    then attributes. A directive that names a module renders the module."""
    parts = dotted.split(".")
    for k in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:k]))
        except ImportError:
            continue
        for part in parts[k:]:
            obj = getattr(obj, part)
        return obj
    raise AssertionError(f"cannot resolve {dotted}")


def test_every_public_name_has_a_reference_entry():
    pytest.importorskip("scipy")  # the eval extra: mushin.benchmark and mushin.llm
    import mushin
    import mushin.benchmark
    import mushin.lightning
    import mushin.llm
    import mushin.workflows

    rendered = {}
    for dotted in _directives():
        obj = _resolve(dotted)
        # `mushin.benchmark.compare` is both a function and the module that
        # holds it; a directive at a module renders the module.
        assert not inspect.ismodule(obj), f"{dotted} renders a module, not an object"
        rendered[id(obj)] = dotted
    missing = []
    for module in (
        mushin,
        mushin.benchmark,
        mushin.lightning,
        mushin.llm,
        mushin.workflows,
    ):
        for name in module.__all__:
            if name in PROSE_ONLY or name in MODULES:
                continue
            if id(getattr(module, name)) not in rendered:
                missing.append(f"{module.__name__}.{name}")
    assert missing == [], f"public names with no reference entry: {missing}"
    prose = (ROOT / "docs" / "reference" / "workflows.md").read_text(encoding="utf-8")
    for name in PROSE_ONLY:
        assert f"`mushin.{name}`" in prose, name


def test_docstrings_use_no_sphinx_roles():
    """mkdocstrings renders Sphinx roles (:func:`x`) literally; the reference
    pages are built from these docstrings."""
    role = re.compile(r":(?:func|class|meth|math|mod|attr|obj):`")
    offenders = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "src" / "mushin").rglob("*.py")
        if role.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
