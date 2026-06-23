# Releasing mushin

Releases are published to PyPI by `.github/workflows/publish.yml` when a GitHub
Release is published whose tag matches the project version.

## Steps

1. Choose the next version `X.Y.Z` (SemVer; `0.x` keeps breaking changes in the
   minor slot).
2. Assemble the changelog from news fragments:
   ```bash
   make changelog VERSION=X.Y.Z
   ```
   This rewrites `CHANGELOG.md` (a new `## [X.Y.Z] - <today>` section under the
   towncrier marker) and deletes the consumed fragments. Review the diff;
   tidy wording if needed.
3. Bump the version in `pyproject.toml` to `X.Y.Z`, then `uv sync` to update
   `uv.lock`.
4. Update the `[Unreleased]`/version link-reference footer at the bottom of
   `CHANGELOG.md`.
5. Commit on a branch, open a PR, merge once CI is green.
6. Tag and publish a GitHub Release `vX.Y.Z` on the merge commit. `publish.yml`
   verifies `tag == version` and publishes via Trusted Publishing.

Preview the changelog section any time without consuming fragments:

```bash
make changelog-draft VERSION=X.Y.Z
```
