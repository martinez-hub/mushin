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
   `uv.lock`. Also bump `version:`/`date-released:` in `CITATION.cff` and the
   `version = {...}` in the README BibTeX block — these are what researchers
   cite, and they go stale silently.
4. Update the `[Unreleased]`/version link-reference footer at the bottom of
   `CHANGELOG.md`.
5. Commit on a branch and open a PR. Because this PR rewrites `CHANGELOG.md`
   directly, it needs the **`changelog-exempt`** label so the changelog gate
   allows it (normal PRs must use `changes/` fragments instead). Apply the label
   when you create the PR:
   ```bash
   gh pr create --label changelog-exempt ...
   ```
   The gate reads the labels from the event that started the run, and
   re-running a job replays that event, so a label added later is never seen.
   If you forget, close and reopen the PR. Merge once CI is green.
6. Tag the release merge commit and publish a GitHub Release `vX.Y.Z` (the tag
   must equal the `pyproject` version, prefixed `v`). Write the release notes
   to a file first, from the new `CHANGELOG.md` section:
   ```bash
   git switch main && git pull --ff-only   # HEAD is now the release merge commit
   git tag vX.Y.Z
   git push origin vX.Y.Z
   gh release create vX.Y.Z --title vX.Y.Z --notes-file notes.md
   ```
   Don't use `--notes-from-tag`. A plain `git tag` is lightweight, so the notes
   would be the tagged commit's message, `release: X.Y.Z (#N)`.
   Publishing the release triggers `publish.yml`. Don't merge anything else
   into `main` until the release commit's CI run finishes: CI cancels
   in-progress runs on the same branch, which would leave the tagged commit
   with a cancelled run.

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
