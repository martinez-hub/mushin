# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The release script's edits, run against the repository's real files.

`scripts/release.py` rewrites CITATION.cff, the README's BibTeX block and the
CHANGELOG.md footer for the Prepare release workflow, and cuts the GitHub
Release notes out of CHANGELOG.md for the Tag and release workflow. Each edit
is a regex over a file someone may reformat by hand, so these tests run them
on the files as they are now: a change that would make a release edit the
wrong line, or nothing, fails here instead of in the release.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "release.py"


def _load():
    spec = importlib.util.spec_from_file_location("release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = _load()


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _current() -> str:
    return tomllib.loads(_read("pyproject.toml"))["project"]["version"]


def _changed_lines(before: str, after: str) -> list[str]:
    return [b for a, b in zip(before.splitlines(), after.splitlines()) if a != b]


@pytest.mark.parametrize(
    "spec, expected",
    [("patch", "1.4.3"), ("minor", "1.5.0"), ("major", "2.0.0"), ("1.10.0", "1.10.0")],
)
def test_next_version(spec, expected):
    assert release.next_version("1.4.2", spec) == expected


@pytest.mark.parametrize("spec", ["1.4.2", "1.4.1", "1.4", "v1.5.0", "next"])
def test_next_version_refuses_a_version_not_above_the_current_one(spec):
    with pytest.raises(SystemExit):
        release.next_version("1.4.2", spec)


def test_a_release_needs_fragments():
    with pytest.raises(SystemExit, match="nothing to release"):
        release.check_fragments([], "1.4.2", "1.4.3")


def test_a_removal_needs_a_minor_release():
    fragments = ["12.removed.md", "13.fixed.md"]
    with pytest.raises(SystemExit, match="12.removed.md"):
        release.check_fragments(fragments, "1.4.2", "1.4.3")
    release.check_fragments(fragments, "1.4.2", "1.5.0")
    release.check_fragments(["+slug.misc.md"], "1.4.2", "1.4.3")


def test_update_citation_sets_only_version_and_date():
    before = _read("CITATION.cff")
    after = release.update_citation(before, "9.8.7", "2031-02-03")
    assert sorted(_changed_lines(before, after)) == [
        'date-released: "2031-02-03"',
        'version: "9.8.7"',
    ]


def test_update_readme_sets_only_the_bibtex_version_and_year():
    before = _read("README.md")
    after = release.update_readme(before, "9.8.7", "2031")
    changed = _changed_lines(before, after)
    assert any(re.fullmatch(r"\s*version\s*=\s*\{9\.8\.7\},?", c) for c in changed)
    assert all(re.match(r"\s*(version|year)\s*=", c) for c in changed), changed
    assert len(after.splitlines()) == len(before.splitlines())


def test_update_changelog_footer_links_the_new_version():
    current = _current()
    after = release.update_changelog_footer(_read("CHANGELOG.md"), "9.8.7", current)
    url = release.REPO_URL
    assert (
        f"[Unreleased]: {url}/compare/v9.8.7...HEAD\n"
        f"[9.8.7]: {url}/compare/v{current}...v9.8.7\n"
        f"[{current}]: "
    ) in after


def test_update_changelog_footer_refuses_a_footer_it_does_not_recognize():
    with pytest.raises(SystemExit, match="footer"):
        release.update_changelog_footer(_read("CHANGELOG.md"), "9.8.7", "0.0.1")


def test_release_notes_are_one_section_without_its_heading():
    changelog = _read("CHANGELOG.md")
    versions = re.findall(r"(?m)^## \[(\d[^\]]*)\] - ", changelog)
    newest, oldest = versions[0], versions[-1]
    notes = release.release_notes(changelog, newest)
    assert notes.strip() and "## [" not in notes
    assert notes.strip() in changelog
    # The oldest section runs into the link-reference footer, not past it.
    assert "[Unreleased]:" not in release.release_notes(changelog, oldest)


def test_release_notes_refuse_an_unknown_version():
    with pytest.raises(SystemExit, match="no section"):
        release.release_notes(_read("CHANGELOG.md"), "9.8.7")
