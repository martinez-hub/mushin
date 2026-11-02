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
    """A token that names a repository path: a bare file name, or a token with
    a `/` that ends in `/` or a file suffix, or whose first segment is an
    entry of the repository or of the package (the Layout names modules by
    their short names, `study/_study.py`). The shape decides, not existence,
    so a renamed directory is still checked. Not any token with a `/`
    (`actions/checkout`, `and/or`), and not a bare suffix (`.md`)."""
    token = token.rstrip(".,;:)")
    if not token or re.search(r"[\s<>\"'`]|://", token):
        return False
    if "/" not in token:
        return not token.startswith(".") and token.endswith(tuple(PATH_SUFFIXES))
    first, last = token.split("/", 1)[0], token.rsplit("/", 1)[1]
    if not first:
        return False
    return (
        token.endswith("/")
        or last.endswith(tuple(PATH_SUFFIXES))
        or any((base / first).exists() for base in (ROOT, PACKAGE))
    )


def _code_blocks(text: str) -> list[str]:
    """The bodies of the fenced code blocks."""
    return re.findall(r"^```[^\n]*\n(.*?)^```$", text, re.M | re.S)


def _named_paths(text: str | None = None) -> list[str]:
    """The repository paths AGENTS.md names: backticked tokens, and the words
    of its command blocks (`pytest tests/test_show.py`), that look like
    paths and are not ones a run creates."""
    text = _text() if text is None else text
    tokens = set(re.findall(r"`([^`\n]+)`", text))
    for block in _code_blocks(text):
        tokens |= {word for word in block.split() if "/" in word}
    paths = {token.rstrip(".,;:)") for token in tokens if _is_path(token)}
    return sorted(paths - RUNTIME_PATHS)


def _exists(path: str) -> bool:
    # The Layout table names modules inside the package by their short names
    # (`_show.py`, `study/_study.py`), so a path may be relative to it, and
    # the conventions name a test helper bare (`conftest.py`). Those places
    # only: a bare name matched anywhere would let a deleted
    # `src/mushin/__main__.py` pass on another package's `__main__.py`.
    bases = (ROOT, PACKAGE) if "/" in path else (ROOT, PACKAGE, ROOT / "tests")
    for base in bases:
        if "*" in path:
            if any(base.glob(path)):
                return True
        elif (base / path).exists():
            return True
    return False


def _make_targets(text: str) -> set[str]:
    """The make targets a text names in code: `` `make check` `` inline, or a
    command block's `make lint` line. Prose ("make sure") is not one."""
    named = set(re.findall(r"`make ([a-z][\w-]*)", text))
    for block in _code_blocks(text):
        named |= set(re.findall(r"^\s*make ([a-z][\w-]*)", block, re.M))
    return named


def _gate_tests(section: str) -> set[str]:
    """The test files of the pytest commands in a section, with a command
    wrapped over lines (`\\` continuations) read whole."""
    joined = re.sub(r"\\\n\s*", " ", section)
    return {
        test
        for line in joined.splitlines()
        if "pytest" in line
        for test in re.findall(r"tests/\S+\.py", line)
    }


def test_agents_md_names_paths():
    """Guards the path extraction itself: an empty list passes vacuously."""
    paths = _named_paths()
    assert "src/mushin/__init__.py" in paths
    assert "tests/test_lazy_imports.py" in paths
    assert "tests/test_show.py" in paths  # from a command block
    assert ".md" not in paths  # a bare suffix in prose is not a path


def test_a_slash_or_a_file_name_in_prose_is_read_rightly():
    assert not _is_path("actions/checkout")
    assert not _is_path("martinez-hub/mushin")
    assert not _is_path("and/or")
    assert _is_path("src/mushin/_show.py") and _is_path("study/_study.py")
    assert _is_path("conftest.py") and _exists("conftest.py")
    # A path whose directory was renamed is still a path, and so is checked.
    assert _is_path("renamed_dir/") and _is_path("renamed_dir/helpers.py")
    assert not _exists("renamed_dir/helpers.py")
    text = "Use `actions/checkout`.\n\n```bash\npytest tests/test_show.py -q\n```\n"
    assert _named_paths(text) == ["tests/test_show.py"]


def test_a_bare_name_resolves_only_where_agents_md_means_it():
    """A module the Layout names by its package path (`study/_study.py`) is
    not found by its bare name, which would match any file of that name."""
    assert _exists("study/_study.py") and not _exists("_study.py")
    assert _exists("__main__.py")  # the package's own
    assert not _exists("hydra_plugins")  # only under tests/plugins/


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
    named = _make_targets(_text())
    assert named, "found no `make <target>` in AGENTS.md"
    assert named <= targets, f"not Makefile targets: {sorted(named - targets)}"


def test_make_targets_are_read_from_code_only():
    text = "Run `make check`; make sure it passes.\n\n```bash\nmake lint\n```\n"
    assert _make_targets(text) == {"check", "lint"}


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
    # Every pytest step of the docs job: the snippet-include check runs in a
    # step of its own beside 'Docs-pinning tests'.
    in_ci = {
        test
        for step in ci["jobs"]["docs"]["steps"]
        if "pytest" in step.get("run", "")
        for test in re.findall(r"tests/\S+\.py", step["run"])
    }
    assert "tests/test_docs_snippets.py" in in_ci, "the docs job's steps changed"
    in_agents_md = _gate_tests(_section("Before you push"))
    assert in_agents_md == in_ci, (
        f"only in AGENTS.md: {sorted(in_agents_md - in_ci)}; "
        f"only in ci.yml: {sorted(in_ci - in_agents_md)}"
    )


def test_a_wrapped_gate_command_is_read_whole():
    section = "2. Run:\n   `uv run pytest tests/test_a.py \\\n   tests/test_b.py`\n"
    assert _gate_tests(section) == {"tests/test_a.py", "tests/test_b.py"}


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
