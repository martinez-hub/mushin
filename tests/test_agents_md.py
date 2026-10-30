# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Keep `AGENTS.md` true to the repository it describes.

`AGENTS.md` is what a coding agent reads before changing mushin, and Claude
Code loads it through `CLAUDE.md`. An agent acts on it without checking: it
runs the make targets it names, edits the files it points to and runs the test
list in "Before you push" as the docs gate. A renamed file, a dropped target or
a docs test missing from that list therefore misleads every agent, and nothing
else reads the file. CI runs this file in the always-on `docs` job, because the
`test` job skips the docs-only PRs most likely to break it.
"""

from __future__ import annotations

import re
import tomllib
from functools import cache

import yaml

from _docs_site import ROOT, markdown_links

AGENTS_MD = ROOT / "AGENTS.md"
PACKAGE = ROOT / "src" / "mushin"

# File suffixes that mark a backticked token as a path, beside a `/`.
PATH_SUFFIXES = {
    ".cff",
    ".cfg",
    ".ipynb",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
# Paths AGENTS.md names that a sweep or the docs build creates, so a clean
# checkout lacks them.
RUNTIME_PATHS = {"multirun/", "llms-full.txt"}


@cache
def _text() -> str:
    return AGENTS_MD.read_text(encoding="utf-8")


@cache
def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _section(title: str) -> str:
    """The body of AGENTS.md's `## title` section."""
    _, found, rest = _text().partition(f"\n## {title}\n")
    assert found, f"AGENTS.md has no '## {title}' section"
    return rest.split("\n## ", 1)[0]


def _is_path(token: str) -> bool:
    if re.search(r"[\s<>]|://", token):
        return False
    if token.startswith("."):
        # `.claude-plugin/` is a path; a bare `.md` in prose is a suffix.
        return "/" in token
    return "/" in token or any(token.endswith(s) for s in PATH_SUFFIXES)


def _named_paths() -> list[str]:
    """Backticked tokens that look like repository paths: they contain a `/`
    or end in a file suffix, and hold no spaces, placeholders or URLs."""
    tokens = set(re.findall(r"`([^`\n]+)`", _text()))
    return sorted({token for token in tokens if _is_path(token)} - RUNTIME_PATHS)


def _exists(path: str) -> bool:
    # The Layout table names modules inside the package by their short names
    # (`_show.py`, `study/_study.py`), so a path may be relative to it.
    for base in (ROOT, PACKAGE):
        if "*" in path:
            if any(base.glob(path)):
                return True
        elif (base / path).exists():
            return True
    return False


def test_agents_md_names_paths():
    """Guards the path extraction itself: an empty list passes vacuously."""
    paths = _named_paths()
    assert "src/mushin/__init__.py" in paths
    assert "tests/test_lazy_imports.py" in paths
    assert ".md" not in paths  # a bare suffix in prose is not a path


def test_every_path_agents_md_names_exists():
    missing = [path for path in _named_paths() if not _exists(path)]
    assert not missing, (
        f"AGENTS.md names paths that do not exist: {missing} (fix AGENTS.md, or "
        "add a path a run creates to RUNTIME_PATHS)"
    )


def test_every_relative_link_in_agents_md_resolves():
    links = [link.split("#", 1)[0] for link in markdown_links(_text())]
    broken = [
        link for link in links if "://" not in link and not (ROOT / link).exists()
    ]
    assert not broken, f"AGENTS.md links to missing files: {broken}"


def test_every_make_target_agents_md_names_exists():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([A-Za-z][\w-]*):", makefile, re.MULTILINE))
    named = set(re.findall(r"\bmake ([a-z][\w-]*)", _text()))
    assert named, "found no `make <target>` in AGENTS.md"
    assert named <= targets, f"not Makefile targets: {sorted(named - targets)}"


def test_every_extra_and_group_agents_md_names_exists():
    project = _pyproject()
    extras = set(re.findall(r"--extra (\w+)", _text()))
    groups = set(re.findall(r"--group (\w+)", _text()))
    assert extras <= set(project["project"]["optional-dependencies"]), extras
    assert groups <= set(project["dependency-groups"]), groups


def test_layout_lists_every_subpackage():
    """A new subpackage is where a newcomer needs the map most."""
    layout = _section("Layout")
    missing = [
        f"src/mushin/{pkg.name}/"
        for pkg in sorted(PACKAGE.iterdir())
        if (pkg / "__init__.py").is_file() and f"`src/mushin/{pkg.name}/`" not in layout
    ]
    assert not missing, f"subpackages missing from AGENTS.md's Layout: {missing}"


def test_claude_md_imports_agents_md():
    """Claude Code reads CLAUDE.md, not AGENTS.md; the import is the bridge."""
    lines = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
    assert "@AGENTS.md" in lines


def test_the_docs_gate_matches_the_ci_docs_job():
    """'Before you push' gives the docs-pinning tests to run locally; it must be
    the list CI's docs job runs, or a local pass proves less than it claims."""
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
    runs = [
        step["run"]
        for job in ci["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Docs-pinning tests"
    ]
    assert len(runs) == 1, "expected one 'Docs-pinning tests' step in ci.yml"
    in_ci = set(re.findall(r"tests/\S+\.py", runs[0]))

    command = next(
        line for line in _section("Before you push").splitlines() if "pytest" in line
    )
    in_agents_md = set(re.findall(r"tests/\S+\.py", command))
    assert in_agents_md == in_ci, (
        f"only in AGENTS.md: {sorted(in_agents_md - in_ci)}; "
        f"only in ci.yml: {sorted(in_ci - in_agents_md)}"
    )


def test_fragment_types_match_towncrier():
    match = re.search(r"with `<type>` one of ([^.]+)\.", _text())
    assert match, "could not find the news-fragment type list in AGENTS.md"
    listed = set(re.findall(r"`(\w+)`", match.group(1)))
    configured = {t["directory"] for t in _pyproject()["tool"]["towncrier"]["type"]}
    assert listed == configured


def test_python_floor_and_range_match_pyproject():
    project = _pyproject()
    match = re.search(
        r"\*\*Python (3\.\d+) is the floor\*\* \(`target-version = \"(py3\d+)\"`\); "
        r"(3\.\d+) to (3\.\d+) are",
        _text(),
    )
    assert match, "could not find the Python floor line in AGENTS.md"
    floor, target, low, high = match.groups()
    assert project["project"]["requires-python"] == f">={floor}"
    assert project["tool"]["ruff"]["target-version"] == target
    assert target == "py" + floor.replace(".", "")
    classified = sorted(
        (int(c.rsplit(".", 1)[1]), c.rsplit(" ", 1)[1])
        for c in project["project"]["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    )
    assert (low, high) == (classified[0][1], classified[-1][1])
