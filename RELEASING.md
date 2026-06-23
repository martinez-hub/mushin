# Releasing mushin

mushin (`mushin-py` on PyPI) publishes via **Trusted Publishing** (OIDC) — there
are no API tokens to manage. A release is triggered by publishing a GitHub
Release whose tag matches the project version; `.github/workflows/publish.yml`
verifies `tag == version`, builds the sdist + wheel, and publishes to PyPI.

## Cutting a release

1. Choose the next version `X.Y.Z` (SemVer; `0.x` keeps breaking changes in the
   minor slot).
2. Assemble the changelog from news fragments:
   ```bash
   make changelog VERSION=X.Y.Z
   ```
   This rewrites `CHANGELOG.md` (a new `## [X.Y.Z] - <today>` section under the
   towncrier marker) and deletes the consumed fragments. Review the diff and
   tidy wording if needed. Preview any time without consuming fragments:
   ```bash
   make changelog-draft VERSION=X.Y.Z
   ```
3. Bump the version in `pyproject.toml` to `X.Y.Z`, then `uv sync` to update
   `uv.lock`.
4. Update the `[Unreleased]`/version link-reference footer at the bottom of
   `CHANGELOG.md`.
5. Commit on a branch and open a PR. Because this PR rewrites `CHANGELOG.md`
   directly, apply the **`release`** label so the changelog gate allows it
   (normal PRs must use `changes/` fragments instead). Merge once CI is green.
6. Tag and publish a GitHub Release `vX.Y.Z` on the merge commit (the tag must
   equal the `pyproject` version, prefixed `v`):
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   gh release create vX.Y.Z --title vX.Y.Z --notes-from-tag
   ```
   Publishing the release triggers `publish.yml`.

## Trusted Publishing setup (one-time, already configured)

The PyPI publisher is registered against the `pypi` GitHub Environment. For
reference (or to re-register), the publisher is:

- **PyPI Project Name:** `mushin-py`
- **Owner:** `martinez-hub`
- **Repository name:** `mushin`
- **Workflow name:** `publish.yml`
- **Environment name:** `pypi`

Optionally add protection rules to the `pypi`
[environment](https://github.com/martinez-hub/mushin/settings/environments)
(e.g. required reviewers) for an extra gate before publishing.
