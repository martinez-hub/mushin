# Releasing

mushin publishes to PyPI via **Trusted Publishing** (OIDC) — there are no API
tokens to manage. Releases are triggered by publishing a GitHub Release.

## One-time PyPI setup (maintainer)

Because the `mushin` project does not exist on PyPI yet, register a **pending
publisher** before the first release:

1. Sign in to https://pypi.org and go to **Your account → Publishing**.
2. Under "Add a new pending publisher", fill in:
   - **PyPI Project Name:** `mushin`
   - **Owner:** `martinez-hub`
   - **Repository name:** `mushin`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
3. Save. (After the first successful publish, this becomes a normal publisher.)

Optionally, on GitHub add protection rules to the `pypi`
[environment](https://github.com/martinez-hub/mushin/settings/environments)
(e.g. required reviewers) for an extra gate before publishing.

## Cutting a release

1. Update the version in `pyproject.toml` (e.g. `0.4.0` → `0.5.0`).
2. Move the `## [Unreleased]` notes in `CHANGELOG.md` under a new version
   heading with today's date, and update the compare links at the bottom.
3. Open a PR with those changes; merge once CI is green.
4. Tag and push, then create the GitHub Release:
   ```bash
   git tag v0.5.0          # tag must equal the pyproject version, prefixed "v"
   git push origin v0.5.0
   gh release create v0.5.0 --title v0.5.0 --notes-from-tag
   ```
   (Or create the release from the GitHub UI.)
5. Publishing the release triggers `.github/workflows/publish.yml`, which
   verifies the tag matches the package version, builds the sdist + wheel, and
   publishes to PyPI.

## Versioning

This project follows [Semantic Versioning](https://semver.org/). The release
workflow fails fast if the tag (without the `v`) does not match
`project.version` in `pyproject.toml`.
