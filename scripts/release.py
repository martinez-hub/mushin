"""Prepare a release's files, or print a released version's notes.

`RELEASING.md` used to list these as hand edits, each of which went stale
silently when forgotten. The Prepare release workflow runs `prepare`, and the
Tag and release workflow runs `notes`; both work the same way locally.

    python scripts/release.py prepare {patch,minor,major,X.Y.Z}
        Builds the CHANGELOG.md section from changes/ (towncrier, which also
        deletes the fragments), sets the version in pyproject.toml and uv.lock
        (`uv version --no-sync`), and brings CITATION.cff, the README's BibTeX
        block and the CHANGELOG.md footer links to the new version, so
        tests/test_release_metadata.py passes. Prints the new version.

    python scripts/release.py notes X.Y.Z
        Prints that version's CHANGELOG.md section, without its heading, as
        the GitHub Release notes.

`prepare` imports towncrier, so run it in the dev environment; `notes` needs
only the standard library.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/martinez-hub/mushin"
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


def _parse(version: str) -> tuple[int, int, int]:
    if not VERSION_RE.fullmatch(version):
        raise SystemExit(f"not an X.Y.Z version: {version!r}")
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch


def next_version(current: str, spec: str) -> str:
    """The version `spec` names: `patch`, `minor` or `major` bumps `current`,
    and anything else must be an explicit X.Y.Z above it."""
    major, minor, patch = _parse(current)
    bumped = {
        "patch": f"{major}.{minor}.{patch + 1}",
        "minor": f"{major}.{minor + 1}.0",
        "major": f"{major + 1}.0.0",
    }
    new = bumped.get(spec, spec)
    if _parse(new) <= _parse(current):
        raise SystemExit(f"{new} is not above the current version {current}")
    return new


def check_fragments(fragments: list[str], current: str, new: str) -> None:
    """Refuse a release with nothing to announce, and a patch release carrying a
    removal: the project is pre-1.0, so breaking changes take a minor bump
    (RELEASING.md)."""
    if not fragments:
        raise SystemExit("changes/ holds no fragments: there is nothing to release")
    removed = [f for f in fragments if f.split(".")[-2] == "removed"]
    if removed and _parse(new)[:2] == _parse(current)[:2]:
        raise SystemExit(
            f"{', '.join(removed)} removes something, which needs a minor "
            f"release, not {new}"
        )


def update_citation(text: str, version: str, date: str) -> str:
    text, n_version = re.subn(r'(?m)^version: ".*"$', f'version: "{version}"', text)
    text, n_date = re.subn(
        r'(?m)^date-released: ".*"$', f'date-released: "{date}"', text
    )
    if (n_version, n_date) != (1, 1):
        raise SystemExit("CITATION.cff: expected one version: and one date-released:")
    return text


def update_readme(text: str, version: str, year: str) -> str:
    """Set `version` and `year` in the README's BibTeX block, and nowhere else."""
    block = re.search(r"```bibtex\n.*?```", text, re.DOTALL)
    if block is None:
        raise SystemExit("README.md: no ```bibtex block")
    new_block, n_version = re.subn(
        r"(?m)^(\s*version\s*=\s*)\{[^}]*\}", rf"\g<1>{{{version}}}", block.group(0)
    )
    new_block, n_year = re.subn(
        r"(?m)^(\s*year\s*=\s*)\{[^}]*\}", rf"\g<1>{{{year}}}", new_block
    )
    if (n_version, n_year) != (1, 1):
        raise SystemExit("README.md: the BibTeX block needs one version and one year")
    return text[: block.start()] + new_block + text[block.end() :]


def update_changelog_footer(text: str, version: str, previous: str) -> str:
    """Point `[Unreleased]` at the new version and add the new version's link."""
    old = f"[Unreleased]: {REPO_URL}/compare/v{previous}...HEAD"
    if text.count(old) != 1:
        raise SystemExit(f"CHANGELOG.md: expected the footer line {old!r}")
    new = (
        f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD\n"
        f"[{version}]: {REPO_URL}/compare/v{previous}...v{version}"
    )
    return text.replace(old, new)


def section_date(changelog: str, version: str) -> str:
    match = re.search(
        rf"(?m)^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", changelog
    )
    if match is None:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    return match.group(1)


def release_notes(changelog: str, version: str) -> str:
    """The body of `version`'s CHANGELOG.md section, up to the next section or
    the link-reference footer."""
    section_date(changelog, version)  # fails clearly when the section is missing
    start = re.search(rf"(?m)^## \[{re.escape(version)}\] - .*\n", changelog).end()
    end = re.compile(r"(?m)^(## \[|\[Unreleased\]: )").search(changelog, start)
    return changelog[start : end.start() if end else len(changelog)].strip() + "\n"


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _write(name: str, text: str) -> None:
    (ROOT / name).write_text(text, encoding="utf-8")


def prepare(spec: str) -> str:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    current = pyproject["project"]["version"]
    new = next_version(current, spec)
    fragments = sorted(
        p.name for p in (ROOT / "changes").iterdir() if p.name != "README.md"
    )
    check_fragments(fragments, current, new)

    # towncrier writes today's date into the heading and `git rm`s the
    # fragments; CITATION.cff then takes its date from that heading. Their
    # output goes to stderr, so stdout carries only the new version.
    subprocess.run(
        [sys.executable, "-m", "towncrier", "build", "--yes", "--version", new],
        cwd=ROOT,
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["uv", "version", new, "--no-sync"], cwd=ROOT, check=True, stdout=sys.stderr
    )

    changelog = _read("CHANGELOG.md")
    date = section_date(changelog, new)
    _write("CHANGELOG.md", update_changelog_footer(changelog, new, current))
    _write("CITATION.cff", update_citation(_read("CITATION.cff"), new, date))
    _write("README.md", update_readme(_read("README.md"), new, date[:4]))
    return new


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[0] not in ("prepare", "notes"):
        raise SystemExit(__doc__)
    command, arg = argv
    if command == "prepare":
        print(prepare(arg))
    else:
        sys.stdout.write(release_notes(_read("CHANGELOG.md"), arg))


if __name__ == "__main__":
    main(sys.argv[1:])
