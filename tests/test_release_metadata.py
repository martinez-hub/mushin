# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Keep the citable release metadata in step with the package version.

`CITATION.cff` (GitHub's "Cite this repository" button) and the README's BibTeX
block are what researchers copy into papers, and a release bumps them next to
`pyproject.toml`. Nothing reads them at run time, so a missed bump ships
silently and gets cited with the wrong version or date. Between releases the
package version is the last release, which is also the newest `CHANGELOG.md`
section, so every one of these must name it.
"""

from __future__ import annotations

import re
import tomllib
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/martinez-hub/mushin"


@cache
def _version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


@cache
def _citation() -> dict:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


@cache
def _changelog() -> str:
    return (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def _released_sections() -> list[tuple[str, str]]:
    """(version, date) of every `## [X.Y.Z] - YYYY-MM-DD` section, newest first."""
    return re.findall(r"(?m)^## \[(\d[^\]]*)\] - (\d{4}-\d{2}-\d{2})$", _changelog())


def test_newest_changelog_section_is_the_package_version():
    sections = _released_sections()
    assert sections, "no `## [X.Y.Z] - YYYY-MM-DD` section in CHANGELOG.md"
    assert sections[0][0] == _version()


def test_citation_cff_names_the_package_version():
    assert str(_citation()["version"]) == _version()


def test_citation_cff_release_date_is_the_changelog_date():
    dates = dict(_released_sections())
    assert str(_citation()["date-released"]) == dates[_version()], (
        "CITATION.cff's date-released must be the date of this version's "
        "CHANGELOG.md section"
    )


def test_readme_bibtex_names_the_package_version():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"```bibtex\n(.*?)```", readme, re.DOTALL)
    assert block, "no ```bibtex block in README.md"
    match = re.search(r"^\s*version\s*=\s*\{([^}]*)\}", block.group(1), re.MULTILINE)
    assert match, "the README's BibTeX block has no `version = {...}` field"
    assert match.group(1) == _version()


def test_changelog_footer_links_the_package_version():
    """The `[Unreleased]` and `[X.Y.Z]` link references at the foot of
    CHANGELOG.md, which make the section headings clickable."""
    version = _version()
    previous = _released_sections()[1][0]
    footer = _changelog()
    assert f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD" in footer
    assert f"[{version}]: {REPO_URL}/compare/v{previous}...v{version}" in footer
