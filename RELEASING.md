# Releasing mushin

mushin (`mushin-py` on PyPI) publishes via **Trusted Publishing** (OIDC) — there
are no API tokens to manage. A release is triggered by publishing a GitHub
Release whose tag matches the project version; `.github/workflows/publish.yml`
verifies `tag == version`, builds the sdist + wheel, and publishes to PyPI.

## Cutting a release

1. Choose the bump (SemVer; `0.x` keeps breaking changes in the minor slot, so
   a pending `changes/*.removed.md` fragment means a minor, and the workflow
   refuses a patch release then). Preview the changelog section any time,
   without consuming fragments:
   ```bash
   make changelog-draft VERSION=X.Y.Z
   ```
2. In the Actions tab, run **Prepare release** with `patch`, `minor`, `major`
   or an exact `X.Y.Z`. It runs `scripts/release.py prepare`, which:
   - renames each `+<slug>` fragment after the PR whose squash merge added it,
     so its entry carries the PR number;
   - builds the new `CHANGELOG.md` section from `changes/` (towncrier) and
     deletes the consumed fragments;
   - sets the version in `pyproject.toml` and `uv.lock`;
   - sets `version:` and `date-released:` in `CITATION.cff` and the `version`
     and `year` of the README BibTeX block, which are what researchers cite;
   - points the `[Unreleased]` link at the bottom of `CHANGELOG.md` at the new
     version and adds the new version's link.

   `tests/test_release_metadata.py` fails until all of these name the new
   version.

   It then opens a `release: X.Y.Z` PR with the **`changelog-exempt`** label,
   which the changelog gate needs because the PR rewrites `CHANGELOG.md`.
3. Review the PR, and tidy the changelog wording by pushing to its branch if
   needed. GitHub holds the CI runs of a PR that a workflow opened: approve
   them on the PR. Merge once CI is green.
4. Merging runs **Tag and release**. It tags the merge commit `vX.Y.Z` (even
   if `main` has moved on), creates the GitHub Release with that version's
   `CHANGELOG.md` section as its notes, and starts `publish.yml` on the tag,
   which checks `tag == version`, builds and smoke-tests the wheel, and
   publishes to PyPI. It also starts **Refresh notebook outputs**, which
   re-runs the example notebooks at the tag and opens a `docs: refresh the
   notebook outputs for X.Y.Z` PR; approve its CI runs and it merges itself
   once they pass. That merge, made with the workflow's token, starts no
   workflows, so the docs site picks up the new outputs from **Deploy docs**'s
   daily check rather than at once. Watch the runs in the Actions tab.

### By hand

The workflows only automate these steps, so each can be done locally:

```bash
uv run python scripts/release.py prepare patch   # step 2's edits; prints X.Y.Z
git switch -c release/X.Y.Z && git commit -am "release: X.Y.Z"
gh pr create --title "release: X.Y.Z" --label changelog-exempt ...
```

The changelog gate reads the PR's labels when it runs, so a label added after
the PR opened counts once the `changelog` job is re-run. A PR titled
`release: X.Y.Z` with that label still runs Tag and release when it merges. If
that run fails, tag and publish from the release merge commit yourself:

```bash
sha=$(gh pr view <release-PR-number> --json mergeCommit --jq .mergeCommit.oid)
git tag vX.Y.Z "$sha"   # the release merge commit, even if main has moved on
git push origin vX.Y.Z
python scripts/release.py notes X.Y.Z > notes.md
gh release create vX.Y.Z --title vX.Y.Z --notes-file notes.md
```

Don't use `--notes-from-tag`: a plain `git tag` is lightweight, so the notes
would be the tagged commit's message. A release published by a person triggers
`publish.yml` itself.

## One-time setup

Prepare release needs **Allow GitHub Actions to create and approve pull
requests**, under Settings, Actions, General, Workflow permissions. Without it
the workflow pushes the release branch but fails to open the PR; open it by
hand as above.

### Trusted Publishing (already configured)

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
